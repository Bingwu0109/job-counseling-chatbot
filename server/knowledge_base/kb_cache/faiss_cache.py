from configs import CACHED_VS_NUM, CACHED_MEMO_VS_NUM
from server.knowledge_base.kb_cache.base import *
from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from server.utils import load_local_embeddings
from server.knowledge_base.utils import get_vs_path
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import os
from langchain.schema import Document


# patch FAISS to include doc id in Document.metadata
def _new_ds_search(self, search: str) -> Union[str, Document]:
    '''用于扩展InMemoryDocstore的搜索功能
    search: 一个搜索字符串

    返回值：
        返回类型为Union[str, Document]，意味着可以返回一个字符串或一个Document对象。
    '''
    # 首先检查search字符串是否在内部字典_dict中。如果不在，返回一个提示字符串，指出ID未找到。
    if search not in self._dict:
        return f"ID {search} not found."
    else:
        # 如果找到了对应的条目，检查这个条目是否为Document类型的实例。
        doc = self._dict[search]
        if isinstance(doc, Document):
            # 如果是，就在文档的元数据中添加一个id字段，其值为搜索字符串search。
            doc.metadata["id"] = search
        return doc
# 将InMemoryDocstore类的search方法替换为刚刚定义的_new_ds_search方法。
InMemoryDocstore.search = _new_ds_search


class ThreadSafeFaiss(ThreadSafeObject):
    def __repr__(self) -> str:
        '''定义ThreadSafeFaiss实例的字符串表示形式，方便调试和日志记录。'''
        # 获取当前实例的类名
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"

    def docs_count(self) -> int:
        '''返回存储在FAISS实例中的文档数量'''
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
        '''将FAISS实例及其包含的文档信息保存到指定路径'''
        # 通过acquire方法获取锁，确保线程安全。
        with self.acquire():
            # 如果指定的路径不是一个目录且create_path为True，则创建该路径。
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            # 调用_obj的save_local方法将数据保存到磁盘上的指定路径。
            ret = self._obj.save_local(path)
            logger.info(f"已将向量库 {self.key} 保存到磁盘")
        return ret

    def clear(self):
        '''清空FAISS实例中存储的所有文档信息'''
        ret = []
        # 通过acquire方法获取锁，确保线程安全。
        with self.acquire():
            # 获取所有文档的ID列表
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                # 调用_obj的delete方法删除这些文档
                ret = self._obj.delete(ids)
                # 断言删除操作后，文档存储应为空。
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"已将向量库 {self.key} 清空")
        # 返回删除操作的结果
        return ret


class _FaissPool(CachePool):
    def new_vector_store(
        self,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> FAISS:
        '''创建一个新的 FAISS 向量存储库'''
        # embedding实例，用于将文档转换成向量形式
        embeddings = EmbeddingsFunAdapter(embed_model)
        # 创建一个初始文档 doc，其内容为 "init"，并没有实际的元数据。
        doc = Document(page_content="init", metadata={})
        # 创建一个FAISS实例，输入包含上述初始文档，使用embeddings进行文档到向量的转换，
        # 启用L2正则化，并设置距离策略为内积。
        vector_store = FAISS.from_documents([doc],
                                            embeddings,
                                            normalize_L2=True,
                                            distance_strategy="METRIC_INNER_PRODUCT")
        # 获取创建的向量存储库中所有文档的 ID，并删除这些文档
        ids = list(vector_store.docstore._dict.keys())
        vector_store.delete(ids)
        return vector_store

    def save_vector_store(self, kb_name: str, path: str=None):
        '''将指定名称的 FAISS 向量存储库保存到磁盘'''
        # 使用赋值表达式（:=，被称为海象操作符）尝试从缓存中获取指定名称的向量库实例。
        if cache := self.get(kb_name):
            return cache.save(path)

    def unload_vector_store(self, kb_name: str):
        '''从缓存池中卸载并释放指定名称的FAISS向量存储库'''
        if cache := self.get(kb_name):
            # 如果成功获取到缓存实例，则调用 pop 方法从缓存池中移除该向量库，
            # 并记录一条日志信息，表示成功释放了向量库。
            self.pop(kb_name)
            logger.info(f"成功释放向量库：{kb_name}")


class KBFaissPool(_FaissPool):
    def load_vector_store(
            self,
            kb_name: str, # 知识库名称
            vector_name: str = None, # 向量名称
            create: bool = True, # 表示如果向量存储不存在时是否创建它
            embed_model: str = EMBEDDING_MODEL, # 嵌入模型
            embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        '''负责加载或创建一个向量存储库'''
        # 获取一个原子锁，这通常用于确保在多线程环境下，这个代码块在任何时刻只被一个线程执行，避免数据竞争。
        self.atomic.acquire()
        vector_name = vector_name or embed_model
        cache = self.get((kb_name, vector_name)) # 用元组比拼接字符串好一些
        if cache is None:
            item = ThreadSafeFaiss((kb_name, vector_name), pool=self)
            # 在缓存中设置新创建的 item，以 (kb_name, vector_name) 作为键。
            self.set((kb_name, vector_name), item)
            with item.acquire(msg="初始化"):
                # 释放原子锁
                self.atomic.release()
                # 记录一条信息日志，说明正在从磁盘加载向量存储库。
                logger.info(f"loading vector store in '{kb_name}/vector_store/{vector_name}' from disk.")
                vs_path = get_vs_path(kb_name, vector_name)
                # 检查是否存在名为 index.faiss 的文件
                if os.path.isfile(os.path.join(vs_path, "index.faiss")):
                    # 如果存在，则加载知识库嵌入
                    embeddings = self.load_kb_embeddings(kb_name=kb_name, embed_device=embed_device, default_embed_model=embed_model)
                    vector_store = FAISS.load_local(vs_path, embeddings, normalize_L2=True,distance_strategy="METRIC_INNER_PRODUCT")
                elif create:
                    # 如果不存在但允许创建新的向量存储库，则创建一个空的向量存储库；
                    if not os.path.exists(vs_path):
                        os.makedirs(vs_path)
                    vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                    vector_store.save_local(vs_path)
                else:
                    # 如果不允许创建且向量存储库不存在，则抛出运行时错误。
                    raise RuntimeError(f"knowledge base {kb_name} not exist.")
                # 将加载或创建的向量存储库赋值给 item.obj
                item.obj = vector_store
                # 表示加载完成
                item.finish_loading()
        else:
            #  如果缓存中已存在对应项，则释放原子锁。
            self.atomic.release()
        # 无论是从缓存中直接获取还是新创建了向量存储库，
        # 最后都返回与给定的kb_name和vector_name相对应的ThreadSafeFaiss实例。
        return self.get((kb_name, vector_name))


class MemoFaissPool(_FaissPool):
    def load_vector_store(
        self,
        kb_name: str,
        embed_model: str = EMBEDDING_MODEL,
        embed_device: str = embedding_device(),
    ) -> ThreadSafeFaiss:
        # 获取一个原子锁，这通常用于确保在多线程环境下，这个代码块在任何时刻只被一个线程执行，避免数据竞争。
        self.atomic.acquire()
        cache = self.get(kb_name)
        if cache is None:
            # 将知识库名称和当前 MemoFaissPool 实例传递给它
            item = ThreadSafeFaiss(kb_name, pool=self)
            # 在缓存中设置新创建的item，以kb_name作为键。
            self.set(kb_name, item)
            # 通过一个上下文管理器获取item的锁
            with item.acquire(msg="初始化"):
                # 释放之前获取的原子锁
                self.atomic.release()
                logger.info(f"loading vector store in '{kb_name}' to memory.")
                # 创建一个新的向量存储库
                vector_store = self.new_vector_store(embed_model=embed_model, embed_device=embed_device)
                # 将新创建的向量存储库赋值给item.obj
                item.obj = vector_store
                # 标记item的加载过程完成
                item.finish_loading()
        else:
            # 如果缓存中已存在对应项，释放原子锁。
            self.atomic.release()
        return self.get(kb_name)


kb_faiss_pool = KBFaissPool(cache_num=CACHED_VS_NUM)
memo_faiss_pool = MemoFaissPool(cache_num=CACHED_MEMO_VS_NUM)


if __name__ == "__main__":
    import time, random
    from pprint import pprint
    # 三个模拟的向量存储库名称
    kb_names = ["vs1", "vs2", "vs3"]
    # for name in kb_names:
    #     memo_faiss_pool.load_vector_store(name)

    def worker(vs_name: str = "samples", name: str = None):
        vs_name = vs_name # 向量存储库的名称
        time.sleep(random.randint(1, 5))  # 随机延迟 1 到 5 秒，模拟耗时操作。
        embeddings = load_local_embeddings() # 加载本地嵌入模型
        r = random.randint(1, 3) # 随机选择一个操作：增加文档、搜索文档或删除文档。
        # 确保线程安全地访问和操作向量存储库
        with kb_faiss_pool.load_vector_store(vs_name).acquire(name) as vs:
            if r == 1: # add docs
                ids = vs.add_texts([f"text added by {name}"], embeddings=embeddings)
                pprint(ids)
            elif r == 2: # search docs
                docs = vs.similarity_search_with_score(f"{name}", k=3, score_threshold=1.0)
                pprint(docs)
        if r == 3: # delete docs
            logger.warning(f"清除 {vs_name} by {name}")
            kb_faiss_pool.get(vs_name).clear()

    threads = []
    # 创建并启动30个线程，每个线程执行worker函数，
    # 并传递一个随机选择的向量存储库名称和唯一的工作线程名称。
    for n in range(1, 30):
        t = threading.Thread(target=worker,
                             kwargs={"vs_name": random.choice(kb_names), "name": f"worker {n}"},
                             daemon=True) # 主线程结束时，它不会等待这些守护线程完成。
        t.start()
        threads.append(t)

    # 等待所有启动的线程完成
    for t in threads:
        t.join()
