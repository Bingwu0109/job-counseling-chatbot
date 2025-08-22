import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from typing import Any, List, Optional
from sentence_transformers import CrossEncoder
from typing import Optional, Sequence
from langchain_core.documents import Document
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from llama_index.bridge.pydantic import Field, PrivateAttr

# 功能：使用特定的模型（来重新排序（或"压缩"）一系列文档，并根据给定的查询评分，选择最相关的文档。
class LangchainReranker(BaseDocumentCompressor):
    """Document compressor that uses `Cohere Rerank API`."""
    model_name_or_path: str = Field()
    _model: Any = PrivateAttr()
    top_n: int = Field()
    device: str = Field()
    max_length: int = Field()
    batch_size: int = Field()
    # show_progress_bar: bool = None
    num_workers: int = Field()

    # activation_fct = None
    # apply_softmax = False

    def __init__(self,
                 model_name_or_path: str,
                 top_n: int = 3,
                 device: str = "cuda",
                 max_length: int = 1024,
                 batch_size: int = 32,
                 # show_progress_bar: bool = None,
                 num_workers: int = 0, # 工作线程数
                 # activation_fct = None,
                 # apply_softmax = False,
                 ):
        # 创建了一个模型实例
        self._model = CrossEncoder(model_name=model_name_or_path, max_length=1024, device=device)
        super().__init__(
            top_n=top_n,
            model_name_or_path=model_name_or_path,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            # show_progress_bar=show_progress_bar,
            num_workers=num_workers,
            # activation_fct=activation_fct,
            # apply_softmax=apply_softmax
        )

    def compress_documents(
            self,
            documents: Sequence[Document], # 需要压缩的文档序列
            query: str, # 用于压缩文档的查询字符串
            callbacks: Optional[Callbacks] = None, # 在压缩过程中运行的回调函数集
    ) -> Sequence[Document]:
        """
        Compress documents using Cohere's rerank API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        # 将documents序列转换为列表，并从每个Document对象中提取页面内容（page_content），存储在列表_docs中。
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        # 为每个文档和查询字符串创建一对，生成一个包含句子对的列表
        sentence_pairs = [[query, _doc] for _doc in _docs]
        # 使用模型的predict方法对句子对进行预测
        results = self._model.predict(sentences=sentence_pairs,
                                      batch_size=self.batch_size,
                                      #  show_progress_bar=self.show_progress_bar,
                                      num_workers=self.num_workers,
                                      #  activation_fct=self.activation_fct,
                                      #  apply_softmax=self.apply_softmax,
                                      convert_to_tensor=True
                                      )
        #  计算要返回的文档数量top_k
        # 获取前top_k个最相关的文档及其相关性分数
        # 遍历这些分数和索引，为每个选中的文档创建一个新的Document对象，
        # 将相关性分数添加到文档的元数据中，并将该文档添加到最终结果列表中。
        top_k = self.top_n if self.top_n < len(results) else len(results)
        values, indices = results.topk(top_k)
        final_results = []
        for value, index in zip(values, indices):
            doc = doc_list[index]
            doc.metadata["relevance_score"] = value
            final_results.append(doc)
        return final_results


if __name__ == "__main__":
    from configs import (LLM_MODELS,
                         VECTOR_SEARCH_TOP_K,
                         SCORE_THRESHOLD,
                         TEMPERATURE,
                         USE_RERANKER,
                         RERANKER_MODEL,
                         RERANKER_MAX_LENGTH,
                         MODEL_PATH)
    from server.utils import embedding_device

    if USE_RERANKER:
        reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL, "BAAI/bge-reranker-large")
        print("-----------------model path------------------")
        print(reranker_model_path)
        reranker_model = LangchainReranker(top_n=3,
                                           device=embedding_device(),
                                           max_length=RERANKER_MAX_LENGTH,
                                           model_name_or_path=reranker_model_path
                                           )
