from typing import Any, Dict, List, Optional
from langchain.chains.base import Chain
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForChainRun
from server.knowledge_base.kb_service.base import KBServiceFactory, SearchMode
from configs import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD


class HybridRetrievalChain(Chain):
    """
    混合检索链，支持向量检索、BM25检索和混合检索
    """
    
    knowledge_base_name: str
    search_mode: str = SearchMode.HYBRID
    top_k: int = VECTOR_SEARCH_TOP_K
    score_threshold: float = SCORE_THRESHOLD
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60
    
    class Config:
        """Configuration for this pydantic object."""
        extra = "forbid"
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys this chain returns."""
        return ["documents", "source_documents"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """执行混合检索"""
        query = inputs["query"]
        
        # 获取知识库服务
        kb = KBServiceFactory.get_service_by_name(self.knowledge_base_name)
        if kb is None:
            raise ValueError(f"未找到知识库: {self.knowledge_base_name}")
        
        # 执行检索
        try:
            docs_with_scores = kb.search_docs(
                query=query,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
                search_mode=self.search_mode,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
                rrf_k=self.rrf_k
            )
            
            # 提取文档和分数
            documents = [doc for doc, score in docs_with_scores]
            scores = [score for doc, score in docs_with_scores]
            
            # 为文档添加分数信息
            for doc, score in zip(documents, scores):
                doc.metadata["retrieval_score"] = score
                doc.metadata["search_mode"] = self.search_mode
            
            return {
                "documents": documents,
                "source_documents": documents,  # 兼容性
            }
            
        except Exception as e:
            raise ValueError(f"检索过程中出错: {str(e)}")

    @property
    def _chain_type(self) -> str:
        return "hybrid_retrieval_chain"


class AdaptiveRetrievalChain(Chain):
    """
    自适应检索链，根据查询类型自动选择最佳检索模式
    """
    
    knowledge_base_name: str
    top_k: int = VECTOR_SEARCH_TOP_K
    score_threshold: float = SCORE_THRESHOLD
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60
    
    # 关键词检测列表
    keyword_indicators = ["怎么", "如何", "什么是", "定义", "步骤", "方法", "流程"]
    semantic_indicators = ["相似", "类似", "相关", "对比", "区别", "联系"]
    
    class Config:
        """Configuration for this pydantic object."""
        extra = "forbid"
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        """Output keys this chain returns."""
        return ["documents", "source_documents", "search_mode_used"]

    def _detect_query_type(self, query: str) -> str:
        """检测查询类型并返回推荐的检索模式"""
        query_lower = query.lower()
        
        # 检查是否包含关键词指示符
        keyword_score = sum(1 for indicator in self.keyword_indicators 
                          if indicator in query_lower)
        
        # 检查是否包含语义指示符  
        semantic_score = sum(1 for indicator in self.semantic_indicators
                           if indicator in query_lower)
        
        # 查询长度分析
        query_length = len(query)
        
        # 决策逻辑
        if keyword_score > semantic_score and query_length < 20:
            # 短查询且包含关键词指示符 -> BM25
            return SearchMode.BM25_ONLY
        elif semantic_score > keyword_score or query_length > 50:
            # 长查询或包含语义指示符 -> 向量检索
            return SearchMode.VECTOR_ONLY
        else:
            # 其他情况使用混合检索
            return SearchMode.HYBRID

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """执行自适应检索"""
        query = inputs["query"]
        
        # 自动检测最佳检索模式
        search_mode = self._detect_query_type(query)
        
        # 获取知识库服务
        kb = KBServiceFactory.get_service_by_name(self.knowledge_base_name)
        if kb is None:
            raise ValueError(f"未找到知识库: {self.knowledge_base_name}")
        
        # 执行检索
        try:
            docs_with_scores = kb.search_docs(
                query=query,
                top_k=self.top_k,
                score_threshold=self.score_threshold,
                search_mode=search_mode,
                dense_weight=self.dense_weight,
                sparse_weight=self.sparse_weight,
                rrf_k=self.rrf_k
            )
            
            # 提取文档和分数
            documents = [doc for doc, score in docs_with_scores]
            scores = [score for doc, score in docs_with_scores]
            
            # 为文档添加元信息
            for doc, score in zip(documents, scores):
                doc.metadata["retrieval_score"] = score
                doc.metadata["search_mode"] = search_mode
                doc.metadata["adaptive_selection"] = True
            
            return {
                "documents": documents,
                "source_documents": documents,
                "search_mode_used": search_mode,
            }
            
        except Exception as e:
            raise ValueError(f"自适应检索过程中出错: {str(e)}")

    @property
    def _chain_type(self) -> str:
        return "adaptive_retrieval_chain"


# 便捷函数
def create_hybrid_retrieval_chain(
    knowledge_base_name: str,
    search_mode: str = SearchMode.HYBRID,
    top_k: int = VECTOR_SEARCH_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
) -> HybridRetrievalChain:
    """创建混合检索链的便捷函数"""
    return HybridRetrievalChain(
        knowledge_base_name=knowledge_base_name,
        search_mode=search_mode,
        top_k=top_k,
        score_threshold=score_threshold,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k,
    )


def create_adaptive_retrieval_chain(
    knowledge_base_name: str,
    top_k: int = VECTOR_SEARCH_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
) -> AdaptiveRetrievalChain:
    """创建自适应检索链的便捷函数"""
    return AdaptiveRetrievalChain(
        knowledge_base_name=knowledge_base_name,
        top_k=top_k,
        score_threshold=score_threshold,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=rrf_k,
    )


# 使用示例
if __name__ == "__main__":
    # 示例1: 使用混合检索链
    hybrid_chain = create_hybrid_retrieval_chain(
        knowledge_base_name="samples",
        search_mode=SearchMode.HYBRID,
        top_k=5
    )
    
    result = hybrid_chain({"query": "如何使用langchain"})
    print("混合检索结果:")
    for i, doc in enumerate(result["documents"]):
        print(f"{i+1}. {doc.page_content[:100]}...")
        print(f"   分数: {doc.metadata.get('retrieval_score', 'N/A')}")
        print(f"   检索模式: {doc.metadata.get('search_mode', 'N/A')}")
    
    print("\n" + "="*50 + "\n")
    
    # 示例2: 使用自适应检索链
    adaptive_chain = create_adaptive_retrieval_chain(
        knowledge_base_name="samples"
    )
    
    queries = [
        "什么是机器学习",  # 可能选择BM25
        "机器学习和深度学习的相似之处",  # 可能选择向量检索
        "介绍一下人工智能的发展历程"  # 可能选择混合检索
    ]
    
    for query in queries:
        result = adaptive_chain({"query": query})
        print(f"查询: {query}")
        print(f"自动选择的检索模式: {result['search_mode_used']}")
        print(f"检索到的文档数量: {len(result['documents'])}")
        print("-" * 30)
