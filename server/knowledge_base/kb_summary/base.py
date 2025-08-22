from typing import List

from configs import (
    EMBEDDING_MODEL,
    KB_ROOT_PATH)

from abc import ABC, abstractmethod
from server.knowledge_base.kb_cache.faiss_cache import kb_faiss_pool, ThreadSafeFaiss
import os
import shutil
from server.db.repository.knowledge_metadata_repository import add_summary_to_db, delete_summary_from_db

from langchain.docstore.document import Document


class KBSummaryService(ABC):
    kb_name: str # 知识库的名称
    embed_model: str # 嵌入模型的名称
    vs_path: str # 向量存储的路径
    kb_path: str # 知识库的路径

    def __init__(self,
                 knowledge_base_name: str, # 知识库的名称
                 embed_model: str = EMBEDDING_MODEL
                 ):
        self.kb_name = knowledge_base_name
        self.embed_model = embed_model

        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()

        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)


    def get_vs_path(self):
        '''返回向量存储的完整路径'''
        return os.path.join(self.get_kb_path(), "summary_vector_store")

    def get_kb_path(self):
        '''返回知识库的完整路径'''
        return os.path.join(KB_ROOT_PATH, self.kb_name)

    def load_vector_store(self) -> ThreadSafeFaiss:
        '''创建一个线程安全的向量存储'''
        return kb_faiss_pool.load_vector_store(kb_name=self.kb_name,
                                               vector_name="summary_vector_store",
                                               embed_model=self.embed_model,
                                               create=True)

    def add_kb_summary(self,
                       summary_combine_docs: List[Document] # 代表需要添加到知识库的摘要文档集合
                       ):
        '''向知识库添加摘要'''
        # 获取向量存储的锁
        with self.load_vector_store().acquire() as vs:
            # 将文档添加到向量存储中，返回添加文档的ID列表。
            ids = vs.add_documents(documents=summary_combine_docs)
            # 将更新后的向量存储保存到本地路径
            vs.save_local(self.vs_path)

        summary_infos = [{"summary_context": doc.page_content,
                          "summary_id": id,
                          "doc_ids": doc.metadata.get('doc_ids'),
                          "metadata": doc.metadata} for id, doc in zip(ids, summary_combine_docs)]
        # 将摘要信息添加到数据库中
        status = add_summary_to_db(kb_name=self.kb_name, summary_infos=summary_infos)
        # status : 表明添加摘要到数据库的操作结果
        return status

    def create_kb_summary(self):
        """
        创建知识库chunk summary
        :return:
        """

        if not os.path.exists(self.vs_path):
            # 创建该路径
            os.makedirs(self.vs_path)

    def drop_kb_summary(self):
        """
        删除知识库chunk summary
        :param kb_name:
        :return:
        """
        # 获取一个原子操作的上下文，atomic可能是为了确保删除操作的原子性，
        # 即要么完全执行，要么完全不执行，避免操作过程中发生错误导致的数据不一致。
        with kb_faiss_pool.atomic:
            # 从向量池中移除指定的知识库向量存储
            kb_faiss_pool.pop(self.kb_name)
            # 删除向量存储的本地路径及其所有内容
            shutil.rmtree(self.vs_path)
        # 从数据库中删除摘要信息
        delete_summary_from_db(kb_name=self.kb_name)
