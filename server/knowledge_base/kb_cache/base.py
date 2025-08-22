from langchain.embeddings.base import Embeddings
from langchain.vectorstores.faiss import FAISS
import threading # 提供对线程的支持
from configs import (EMBEDDING_MODEL, CHUNK_SIZE,
                     logger, log_verbose)
from server.utils import embedding_device, get_model_path, list_online_embed_models
from contextlib import contextmanager # 创建上下文管理器
from collections import OrderedDict # 创建可以记住元素添加顺序的字典
from typing import List, Any, Union, Tuple


class ThreadSafeObject:
    '''它被设计为线程安全的对象包装器，可以在多线程环境中安全地访问和修改被封装的对象。'''
    def __init__(self, key: Union[str, Tuple], obj: Any = None, pool: "CachePool" = None):
        self._obj = obj
        self._key = key
        self._pool = pool
        self._lock = threading.RLock() # 一个重入锁，用于在操作对象时提供线程安全保护。
        self._loaded = threading.Event() # 一个事件，用于控制对象加载状态的同步。

    def __repr__(self) -> str:
        '''定义了对象的字符串表示形式，便于调试和日志记录。'''
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}>"

    @property
    def key(self):
        '''key属性通过@property装饰器定义，使得_key字段可以安全地被读取，但不被直接修改。'''
        return self._key

    @contextmanager
    def acquire(self, owner: str = "", msg: str = "") -> FAISS:
        '''一个上下文管理器，用于在操作封装的对象前获取锁，操作完成后释放锁。'''
        # 操作者的标识，默认为空字符串
        owner = owner or f"thread {threading.get_native_id()}"

        try:
            # 获取重入锁，这确保了在多线程环境中对封装对象的操作是串行化的，防止了并发访问导致的数据不一致问题。
            self._lock.acquire()
            if self._pool is not None:
                # 把当前对象在缓存中移动到最末尾，表明它最近被使用过。
                # 这是一种常见的缓存管理策略，有助于在需要时淘汰最不常用的对象。
                self._pool._cache.move_to_end(self.key)
            if log_verbose:
                logger.info(f"{owner} 开始操作：{self.key}。{msg}")
            # 返回封装的对象
            yield self._obj
        finally:
            # 释放锁并记录日志
            if log_verbose:
                logger.info(f"{owner} 结束操作：{self.key}。{msg}")
            self._lock.release()

    def start_loading(self):
        '''这表示对象的加载过程开始了，其他线程如果调用wait_for_loading方法会被阻塞，直到加载完成。'''
        self._loaded.clear()

    def finish_loading(self):
        '''这表示对象已经加载完成，任何因调用wait_for_loading方法而阻塞的线程将被唤醒。'''
        self._loaded.set()

    def wait_for_loading(self):
        '''这通常用于等待对象的加载完成。如果对象已经加载（即_loaded已经被设置），此方法会立即返回。'''
        self._loaded.wait()

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, val: Any):
        self._obj = val


class CachePool:
    def __init__(self, cache_num: int = -1):
        self._cache_num = cache_num # 限制缓存中可以存储的元素数量
        self._cache = OrderedDict() # 一个记住元素添加顺序的字典
        # 初始化一个可重入锁（RLock）对象赋值给 atomic 属性。
        # 这是为了确保多线程环境下对缓存的访问是线程安全的。
        self.atomic = threading.RLock()

    def keys(self) -> List[str]:
        # 用于获取缓存中所有键的列表
        return list(self._cache.keys())

    def _check_count(self):
        # 用于检查缓存中的元素数量是否超出了限制，并在必要时删除最旧的元素。
        if isinstance(self._cache_num, int) and self._cache_num > 0:
            while len(self._cache) > self._cache_num:
                # 删除缓存中最旧的元素，last=False 表明是从字典的开始（最旧的元素）处删除。
                self._cache.popitem(last=False)

    def get(self, key: str) -> ThreadSafeObject:
        '''根据给定的键从缓存中获取元素'''
        if cache := self._cache.get(key):
            cache.wait_for_loading()
            # 返回从缓存中获取的对象
            return cache

    def set(self, key: str, obj: ThreadSafeObject) -> ThreadSafeObject:
        '''用于向缓存中添加或更新一个键值对'''
        # 在缓存中设置给定的键和值
        self._cache[key] = obj
        # 确保缓存没有超出设置的大小限制
        self._check_count()
        # 返回被添加到缓存中的对象
        return obj

    def pop(self, key: str = None) -> ThreadSafeObject:
        '''用于从缓存中删除并返回一个元素'''
        if key is None:
            # 如果没有提供键，则从缓存中删除并返回最旧的元素。
            return self._cache.popitem(last=False)
        else:
            # 使用提供的键从缓存中删除对应的元素并返回。如果键不存在，则返回 None。
            return self._cache.pop(key, None)

    def acquire(self, key: Union[str, Tuple], owner: str = "", msg: str = ""):
        # 调用get方法尝试根据key从缓存中获取对象。
        cache = self.get(key)
        if cache is None:
            # 如果根据键找不到对象，则抛出运行时错误，提示请求的资源不存在。
            raise RuntimeError(f"请求的资源 {key} 不存在")
        # 如果找到的缓存对象是一个ThreadSafeObject（一个假定的线程安全对象）类型。
        elif isinstance(cache, ThreadSafeObject):
            # 将该键对应的缓存对象移动到OrderedDict的末尾，
            # 这样做通常是为了更新缓存的访问顺序，保证最近使用的对象不会被早期删除。
            self._cache.move_to_end(key)
            return cache.acquire(owner=owner, msg=msg)
        else:
            # 如果找到的缓存对象不是ThreadSafeObject类型，直接返回该对象。
            return cache

    def load_kb_embeddings(
            self,
            kb_name: str, # 知识库名称
            embed_device: str = embedding_device(),
            default_embed_model: str = EMBEDDING_MODEL,
    ) -> Embeddings:
        from server.db.repository.knowledge_base_repository import get_kb_detail
        from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
        # 获取知识库的详细信息
        kb_detail = get_kb_detail(kb_name)
        # 获取嵌入模型名称
        embed_model = kb_detail.get("embed_model", default_embed_model)
        # 查获得的嵌入模型是否在在线嵌入模型列表中
        if embed_model in list_online_embed_models():
            # 返回一个EmbeddingsFunAdapter实例，用指定的嵌入模型进行初始化。
            return EmbeddingsFunAdapter(embed_model)
        else:
            # 如果获得的嵌入模型不在在线列表中，从嵌入池中加载嵌入，使用指定的模型和装置。
            return embeddings_pool.load_embeddings(model=embed_model, device=embed_device)


class EmbeddingsPool(CachePool):
    def load_embeddings(self, model: str = None, device: str = None) -> Embeddings:
        # 在多线程环境中，为了避免竞争条件，这行代码使用atomic锁来同步线程。
        # 这确保了在加载或访问嵌入时不会有其他线程同时修改它们。
        self.atomic.acquire()
        model = model or EMBEDDING_MODEL
        device = embedding_device()
        # 创建一个元组key，包含model和device，用作嵌入缓存的键。
        key = (model, device)
        # 检查是否已有对应key的嵌入缓存。如果没有，则进入代码块进行加载和设置。
        if not self.get(key):
            # 创建一个线程安全对象item，用于在多线程环境下安全地处理嵌入。
            item = ThreadSafeObject(key, pool=self)
            # 在缓存中设置键为key，值为item的条目。
            self.set(key, item)
            # 获取item的锁
            with item.acquire(msg="初始化"):
                # 释放之前获取的atomic锁，允许其他线程进行操作。
                self.atomic.release()
                # 根据模型的类型（model变量的值）来加载不同的嵌入实现
                if model == "text-embedding-ada-002":  # openai text-embedding-ada-002
                    from langchain.embeddings.openai import OpenAIEmbeddings
                    embeddings = OpenAIEmbeddings(model=model,
                                                  openai_api_key=get_model_path(model),
                                                  chunk_size=CHUNK_SIZE)
                elif 'bge-' in model:
                    from langchain.embeddings import HuggingFaceBgeEmbeddings
                    if 'zh' in model:
                        # for chinese model
                        query_instruction = "为这个句子生成表示以用于检索相关文章："
                    elif 'en' in model:
                        # for english model
                        query_instruction = "Represent this sentence for searching relevant passages:"
                    else:
                        # maybe ReRanker or else, just use empty string instead
                        query_instruction = ""
                    embeddings = HuggingFaceBgeEmbeddings(model_name=get_model_path(model),
                                                          model_kwargs={'device': device},
                                                          query_instruction=query_instruction)
                    if model == "bge-large-zh-noinstruct":  # bge large -noinstruct embedding
                        embeddings.query_instruction = ""
                else:
                    # 对于其他模型，使用HuggingFaceEmbeddings
                    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name=get_model_path(model),
                                                       model_kwargs={'device': device})
                # 将加载的嵌入实例赋值给item.obj，这样就可以在缓存中保持嵌入的实例。
                item.obj = embeddings
                # 调用item的finish_loading方法，表示嵌入加载完毕，对象现在可以安全地被其他线程访问。
                item.finish_loading()
        else:
            # 如果缓存中已存在对应key的嵌入，释放之前获取的锁。
            self.atomic.release()
        # 从缓存中获取对应key的嵌入实例，并返回它。
        return self.get(key).obj


embeddings_pool = EmbeddingsPool(cache_num=1)
