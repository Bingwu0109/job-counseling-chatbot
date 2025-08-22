import operator
from abc import ABC, abstractmethod
import asyncio
import os
import time
import hashlib
import re
from pathlib import Path
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from server.db.repository.knowledge_base_repository import (
    add_kb_to_db, delete_kb_from_db, list_kbs_from_db, kb_exists,
    load_kb_from_db, get_kb_detail,
)
from server.db.repository.knowledge_file_repository import (
    add_file_to_db, delete_file_from_db, delete_files_from_db, file_exists_in_db,
    count_files_from_db, list_files_from_db, get_file_detail, delete_file_from_db,
    list_docs_from_db,
)

from configs import (kbs_config, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     EMBEDDING_MODEL, KB_INFO, logger, log_verbose)
from server.knowledge_base.utils import (
    get_kb_path, get_doc_path, KnowledgeFile,
    list_kbs_from_folder, list_files_from_folder,
)

from typing import List, Union, Dict, Optional, Tuple

from server.embeddings_api import embed_texts, aembed_texts, embed_documents
from server.knowledge_base.model.kb_document_model import DocumentWithVSId

# 导入RAG-fusion相关配置和工具
try:
    from configs import (
        ENABLE_RAG_FUSION,
        RAG_FUSION_CONFIG, 
        RAG_FUSION_QUERY_COUNT,
        RAG_FUSION_LLM_MODEL,
        RAG_FUSION_SUPPORTED_MODELS,
        RAG_FUSION_TEMPERATURE,
        RAG_FUSION_MAX_TOKENS,
        RAG_FUSION_TIMEOUT,
        RAG_FUSION_QUERY_PROMPT,
    )
    RAG_FUSION_AVAILABLE = ENABLE_RAG_FUSION
except ImportError:
    RAG_FUSION_AVAILABLE = False
    RAG_FUSION_CONFIG = {}
    RAG_FUSION_QUERY_COUNT = 3
    RAG_FUSION_LLM_MODEL = "Qwen1.5-7B-Chat"
    RAG_FUSION_SUPPORTED_MODELS = []
    RAG_FUSION_TEMPERATURE = 0.7
    RAG_FUSION_MAX_TOKENS = 200
    RAG_FUSION_TIMEOUT = 30
    RAG_FUSION_QUERY_PROMPT = """Based on the original query, generate {num_queries} different but related search queries to improve information retrieval.

Original query: {original_query}

Requirements:
1. Generate {num_queries} queries (including the original one)
2. Each query should approach the topic from a different angle
3. Queries should be concise and focused
4. Use synonyms and related terms when appropriate
5. Return only the queries, one per line

Generated queries:"""

# ================= RAG-fusion工具函数实现 =================

# 简单的内存缓存
_rag_fusion_query_cache = {}

def _clean_cache():
    """清理过期的缓存条目"""
    current_time = time.time()
    cache_expire_time = RAG_FUSION_CONFIG.get("cache", {}).get("cache_expire_time", 3600) if RAG_FUSION_CONFIG else 3600
    
    expired_keys = [
        key for key, (timestamp, _) in _rag_fusion_query_cache.items() 
        if current_time - timestamp > cache_expire_time
    ]
    
    for key in expired_keys:
        _rag_fusion_query_cache.pop(key, None)


def _get_cache_key(original_query: str, config: dict) -> str:
    """生成缓存键"""
    cache_str = f"{original_query}_{config.get('query_count', 3)}_{config.get('llm_model', 'default')}"
    return hashlib.md5(cache_str.encode()).hexdigest()


def _calculate_query_similarity(query1: str, query2: str) -> float:
    """计算两个查询之间的简单相似度（基于词汇重叠）"""
    words1 = set(re.findall(r'\b\w+\b', query1.lower()))
    words2 = set(re.findall(r'\b\w+\b', query2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def _filter_similar_queries(queries: List[str], threshold: float = 0.9) -> List[str]:
    """过滤过于相似的查询"""
    filtered_queries = []
    
    for query in queries:
        is_similar = False
        for existing_query in filtered_queries:
            if _calculate_query_similarity(query, existing_query) > threshold:
                is_similar = True
                break
        
        if not is_similar:
            filtered_queries.append(query)
    
    return filtered_queries


def _parse_generated_queries(response_text: str, original_query: str) -> List[str]:
    """解析LLM生成的查询文本"""
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
    
    queries = [original_query]  # 首先添加原始查询
    
    for line in lines:
        # 移除数字编号、破折号等前缀
        clean_line = re.sub(r'^[\d\-\.\)\(]*\s*', '', line).strip()
        # 移除引号
        clean_line = clean_line.strip('"\'')
        
        # 过滤过短或过长的查询
        if 5 <= len(clean_line) <= 100 and clean_line not in queries:
            queries.append(clean_line)
    
    return queries


def generate_fusion_queries(original_query: str, 
                           num_queries: int = 3,
                           model_name: str = None,
                           use_cache: bool = True,
                           timeout: int = 30) -> List[str]:
    """
    生成RAG-fusion查询
    
    Args:
        original_query: 原始查询
        num_queries: 生成查询总数（包括原查询）
        model_name: 使用的LLM模型
        use_cache: 是否使用缓存
        timeout: 超时时间
    
    Returns:
        查询列表，第一个是原查询
    """
    if not RAG_FUSION_AVAILABLE:
        return [original_query]
    
    # 检查缓存
    if use_cache:
        cache_key = _get_cache_key(original_query, {
            "query_count": num_queries, 
            "llm_model": model_name or RAG_FUSION_LLM_MODEL
        })
        
        _clean_cache()  # 清理过期缓存
        
        if cache_key in _rag_fusion_query_cache:
            logger.info("RAG-fusion查询缓存命中")
            cached_data = _rag_fusion_query_cache[cache_key]
            return cached_data[1]  # 返回查询列表
    
    try:
        # 构建提示
        model_name = model_name or RAG_FUSION_LLM_MODEL
        prompt = RAG_FUSION_QUERY_PROMPT.format(
            num_queries=num_queries - 1,  # 减1因为原查询已包含
            original_query=original_query
        )
        
        logger.info(f"使用模型 {model_name} 生成RAG-fusion查询")
        
        # 调用LLM生成查询
        try:
            # 尝试导入并使用LLM API
            try:
                from server.llm_api import get_ChatOpenAI
                llm = get_ChatOpenAI(
                    model_name=model_name,
                    temperature=RAG_FUSION_TEMPERATURE,
                    max_tokens=RAG_FUSION_MAX_TOKENS,
                    request_timeout=timeout
                )
                response = llm.predict(prompt)
                
            except ImportError:
                # 如果无法导入，尝试其他方式
                try:
                    from server.chat.chat import chat_completion_sync
                    response = chat_completion_sync(
                        model_name=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=RAG_FUSION_MAX_TOKENS,
                        temperature=RAG_FUSION_TEMPERATURE,
                        timeout=timeout
                    )
                    response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    
                except ImportError:
                    logger.warning("无法找到可用的LLM API，使用简单的查询变体生成")
                    # 简单的查询变体生成作为fallback
                    response = _generate_simple_query_variants(original_query, num_queries - 1)
                    
        except Exception as e:
            logger.warning(f"LLM调用失败: {e}, 使用简单的查询变体生成")
            response = _generate_simple_query_variants(original_query, num_queries - 1)
        
        # 解析生成的查询
        if isinstance(response, list):
            # 如果是简单变体生成，直接使用
            queries = [original_query] + response
        else:
            queries = _parse_generated_queries(response, original_query)
        
        # 过滤相似查询
        config = RAG_FUSION_CONFIG.get("query_generation", {}) if RAG_FUSION_CONFIG else {}
        if config.get("filter_similar", True):
            threshold = config.get("similarity_threshold", 0.9)
            queries = _filter_similar_queries(queries, threshold)
        
        # 确保查询数量在合理范围内
        max_queries = config.get("max_queries", 5)
        min_queries = config.get("min_queries", 2)
        
        if len(queries) > max_queries:
            queries = queries[:max_queries]
        elif len(queries) < min_queries and len(queries) > 1:
            # 如果查询太少，添加原查询的变体
            queries.extend([f"Related to: {original_query}", f"Information about: {original_query}"])
            queries = queries[:max_queries]
        
        # 缓存结果
        if use_cache:
            cache_key = _get_cache_key(original_query, {
                "query_count": num_queries, 
                "llm_model": model_name
            })
            _rag_fusion_query_cache[cache_key] = (time.time(), queries)
        
        logger.info(f"RAG-fusion查询生成成功: {len(queries)}个查询")
        return queries
        
    except Exception as e:
        logger.error(f"RAG-fusion查询生成失败: {e}")
        return [original_query]


def _generate_simple_query_variants(original_query: str, num_variants: int) -> List[str]:
    """生成简单的查询变体（作为LLM API不可用时的fallback）"""
    variants = []
    
    # 基本变体模板
    templates = [
        "What is {}?",
        "How does {} work?",
        "Explain {}",
        "Information about {}",
        "Details on {}",
        "Learn about {}",
        "Understanding {}",
        "{} explained",
        "Guide to {}",
        "Overview of {}",
    ]
    
    # 提取关键词
    words = original_query.split()
    key_phrase = " ".join(words[:min(3, len(words))])  # 取前3个词作为关键短语
    
    # 生成变体
    for i, template in enumerate(templates[:num_variants]):
        try:
            if "{}" in template:
                variant = template.format(key_phrase)
            else:
                variant = f"{template} {key_phrase}"
            variants.append(variant)
        except:
            variants.append(f"Related to: {original_query}")
    
    # 如果还需要更多变体
    while len(variants) < num_variants:
        variants.append(f"Search for: {original_query}")
        if len(variants) >= num_variants:
            break
        variants.append(f"Find information about: {original_query}")
    
    return variants[:num_variants]


async def generate_fusion_queries_async(original_query: str,
                                        num_queries: int = 3,
                                        model_name: str = None,
                                        use_cache: bool = True,
                                        timeout: int = 30) -> List[str]:
    """异步生成RAG-fusion查询"""
    # 简化实现：使用同步版本在线程池中执行
    import concurrent.futures
    
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, 
            generate_fusion_queries,
            original_query, num_queries, model_name, use_cache, timeout
        )


def calculate_query_similarity(query1: str, query2: str) -> float:
    """计算查询相似度（公开接口）"""
    return _calculate_query_similarity(query1, query2)


def get_rag_fusion_stats() -> Dict:
    """获取RAG-fusion统计信息"""
    return {
        "enabled": RAG_FUSION_AVAILABLE,
        "cache_size": len(_rag_fusion_query_cache),
        "supported_models": RAG_FUSION_SUPPORTED_MODELS if RAG_FUSION_AVAILABLE else [],
        "current_model": RAG_FUSION_LLM_MODEL if RAG_FUSION_AVAILABLE else None,
        "cache_stats": {
            "entries": len(_rag_fusion_query_cache),
            "max_size": RAG_FUSION_CONFIG.get("cache", {}).get("max_cache_size", 1000) if RAG_FUSION_CONFIG else 1000
        }
    }


# ================= 原有的normalize函数 =================

def normalize(embeddings: List[List[float]]) -> np.ndarray:
    '''
    sklearn.preprocessing.normalize 的替代（使用 L2），避免安装 scipy, scikit-learn
    对输入的嵌入向量（embeddings）进行L2归一化
    '''
    norm = np.linalg.norm(embeddings, axis=1)
    norm = np.reshape(norm, (norm.shape[0], 1))
    norm = np.tile(norm, (1, len(embeddings[0])))
    return np.divide(embeddings, norm)


def normalize_v2(embeddings: List[List[float]]) -> np.ndarray:
    '''更高效的L2归一化实现'''
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norm


def reciprocal_rank_fusion(results_list: List[List[Tuple[Document, float]]], 
                          k: int = 60, 
                          top_k: int = 5,
                          normalize_scores: bool = True) -> List[Tuple[Document, float]]:
    """
    实现倒数排名融合算法 (Reciprocal Rank Fusion) - 增强版
    
    Args:
        results_list: 多个检索结果列表
        k: RRF参数，用于平滑排名
        top_k: 返回的最终结果数量
        normalize_scores: 是否标准化分数
    """
    if not results_list or not any(results_list):
        return []
    
    doc_scores = {}
    
    # 第一轮：计算RRF分数
    for results in results_list:
        if not results:
            continue
            
        for rank, (doc, original_score) in enumerate(results):
            # 使用文档内容和来源作为唯一标识
            doc_key = f"{doc.page_content[:100]}_{doc.metadata.get('source', '')}"
            rrf_score = 1.0 / (k + rank + 1)
            
            if doc_key in doc_scores:
                doc_scores[doc_key]["rrf_score"] += rrf_score
                doc_scores[doc_key]["rank_count"] += 1
                # 保持最高的原始分数
                if original_score > doc_scores[doc_key]["original_score"]:
                    doc_scores[doc_key]["original_score"] = original_score
                    doc_scores[doc_key]["doc"] = doc
            else:
                doc_scores[doc_key] = {
                    "doc": doc,
                    "rrf_score": rrf_score,
                    "original_score": original_score,
                    "rank_count": 1
                }
    
    # 第二轮：标准化和最终排序
    if normalize_scores and doc_scores:
        max_rrf = max(item["rrf_score"] for item in doc_scores.values())
        min_rrf = min(item["rrf_score"] for item in doc_scores.values())
        
        if max_rrf > min_rrf:
            for item in doc_scores.values():
                # 标准化到0-1范围，并考虑出现频次
                normalized_rrf = (item["rrf_score"] - min_rrf) / (max_rrf - min_rrf)
                frequency_boost = min(item["rank_count"] / len(results_list), 0.3)
                item["final_score"] = normalized_rrf + frequency_boost
        else:
            for item in doc_scores.values():
                item["final_score"] = item["rrf_score"]
    else:
        for item in doc_scores.values():
            item["final_score"] = item["rrf_score"]
    
    # 按最终分数排序
    sorted_docs = sorted(doc_scores.values(), 
                        key=lambda x: x["final_score"], 
                        reverse=True)
    
    return [(item["doc"], item["final_score"]) for item in sorted_docs[:top_k]]


# ================= 枚举类 =================

class SupportedVSType:
    '''向量搜索类型枚举'''
    FAISS = 'faiss'
    MILVUS = 'milvus'
    DEFAULT = 'default'
    ZILLIZ = 'zilliz'
    PG = 'pg'
    ES = 'es'
    CHROMADB = 'chromadb'


class SearchMode:
    '''搜索模式枚举'''
    VECTOR_ONLY = 'vector'      # 仅向量检索
    BM25_ONLY = 'bm25'         # 仅BM25检索
    HYBRID = 'hybrid'          # 混合检索
    RAG_FUSION = 'rag_fusion'  # RAG-fusion检索
    ADAPTIVE = 'adaptive'      # 自适应检索

    @classmethod
    def get_all_modes(cls):
        """获取所有可用的搜索模式"""
        modes = [cls.VECTOR_ONLY, cls.BM25_ONLY, cls.HYBRID, cls.ADAPTIVE]
        if RAG_FUSION_AVAILABLE:
            modes.append(cls.RAG_FUSION)
        return modes

    @classmethod
    def is_valid_mode(cls, mode: str) -> bool:
        """验证搜索模式是否有效"""
        return mode in cls.get_all_modes()


# ================= 知识库服务基类 =================

class KBService(ABC):
    '''知识库服务的抽象基类'''
    
    def __init__(self,
                 knowledge_base_name: str,
                 embed_model: str = EMBEDDING_MODEL,
                 ):
        self.kb_name = knowledge_base_name
        self.kb_info = KB_INFO.get(knowledge_base_name, f"关于{knowledge_base_name}的知识库")
        self.embed_model = embed_model
        self.kb_path = get_kb_path(self.kb_name)
        self.doc_path = get_doc_path(self.kb_name)
        
        # RAG-fusion相关属性
        self._rag_fusion_enabled = RAG_FUSION_AVAILABLE
        self._rag_fusion_stats = {"queries_generated": 0, "searches_executed": 0, "cache_hits": 0}
        
        self.do_init()

    def __repr__(self) -> str:
        return f"{self.kb_name} @ {self.embed_model}"

    def save_vector_store(self):
        '''保存向量库'''
        pass

    def create_kb(self):
        """创建知识库"""
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)
        self.do_create_kb()
        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)
        return status

    def clear_vs(self):
        """清空向量库中的所有内容"""
        self.do_clear_vs()
        status = delete_files_from_db(self.kb_name)
        return status

    def drop_kb(self):
        """删除知识库"""
        self.do_drop_kb()
        status = delete_kb_from_db(self.kb_name)
        return status

    def _docs_to_embeddings(self, docs: List[Document]) -> Dict:
        '''将文档列表转换为嵌入向量'''
        return embed_documents(docs=docs, embed_model=self.embed_model, to_query=False)

    def add_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """向知识库添加文档"""
        if docs:
            custom_docs = True
            for doc in docs:
                doc.metadata.setdefault("source", kb_file.filename)
        else:
            docs = kb_file.file2text()
            custom_docs = False

        if docs:
            # 将 metadata["source"] 改为相对路径
            for doc in docs:
                try:
                    source = doc.metadata.get("source", "")
                    if os.path.isabs(source):
                        rel_path = Path(source).relative_to(self.doc_path)
                        doc.metadata["source"] = str(rel_path.as_posix().strip("/"))
                except Exception as e:
                    print(f"cannot convert absolute path ({source}) to relative path. error is : {e}")
            
            self.delete_doc(kb_file)
            doc_infos = self.do_add_doc(docs, **kwargs)
            status = add_file_to_db(kb_file,
                                    custom_docs=custom_docs,
                                    docs_count=len(docs),
                                    doc_infos=doc_infos)
        else:
            status = False
        return status

    def delete_doc(self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs):
        """从知识库中删除文档"""
        self.do_delete_doc(kb_file, **kwargs)
        status = delete_file_from_db(kb_file)
        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)
        return status

    def update_info(self, kb_info: str):
        """更新知识库信息"""
        self.kb_info = kb_info
        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)
        return status

    def update_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """更新知识库中的文档内容"""
        if os.path.exists(kb_file.filepath):
            self.delete_doc(kb_file, **kwargs)
            return self.add_doc(kb_file, docs=docs, **kwargs)

    def exist_doc(self, file_name: str):
        '''检查文档是否存在'''
        return file_exists_in_db(KnowledgeFile(knowledge_base_name=self.kb_name,
                                               filename=file_name))

    def list_files(self):
        '''列出知识库中所有文件'''
        return list_files_from_db(self.kb_name)

    def count_files(self):
        '''统计知识库中的文件数量'''
        return count_files_from_db(self.kb_name)

    def search_docs(self,
                    query: str,
                    top_k: int = VECTOR_SEARCH_TOP_K,
                    score_threshold: float = SCORE_THRESHOLD,
                    search_mode: str = SearchMode.VECTOR_ONLY,
                    # params of Hybrid Retrieval
                    dense_weight: float = 0.7,
                    sparse_weight: float = 0.3,
                    rrf_k: int = 60,
                    # params of RAG-fusion
                    rag_fusion_query_count: int = None,
                    rag_fusion_model: str = None,
                    rag_fusion_timeout: int = None,  # 新增：超时参数
                    fusion_search_strategy: str = None,  # 新增：搜索策略参数
                    **kwargs
                    ) -> List[Document]:
        '''
        根据查询字符串返回最相关的文档
        
        Args:
            query: 查询字符串
            top_k: 返回的文档数量
            score_threshold: 分数阈值
            search_mode: 搜索模式
            dense_weight: 稠密检索权重（混合模式）
            sparse_weight: 稀疏检索权重（混合模式）
            rrf_k: RRF算法参数
            rag_fusion_query_count: RAG-fusion查询数量
            rag_fusion_model: RAG-fusion使用的模型
            rag_fusion_timeout: RAG-fusion查询生成超时时间
            fusion_search_strategy: RAG-fusion搜索策略
        '''
        
        # 验证搜索模式
        if not SearchMode.is_valid_mode(search_mode):
            logger.warning(f"不支持的搜索模式 '{search_mode}', 使用向量检索")
            search_mode = SearchMode.VECTOR_ONLY
        
        try:
            if search_mode == SearchMode.VECTOR_ONLY:
                docs = self.do_search(query, top_k, score_threshold)
            elif search_mode == SearchMode.BM25_ONLY:
                docs = self.do_bm25_search(query, top_k, score_threshold)
            elif search_mode == SearchMode.HYBRID:
                docs = self.do_hybrid_search(query, top_k, score_threshold, 
                                           dense_weight, sparse_weight, rrf_k)
            elif search_mode == SearchMode.RAG_FUSION and RAG_FUSION_AVAILABLE:
                docs = self.do_rag_fusion_search(
                    query, top_k, score_threshold,
                    query_count=rag_fusion_query_count or RAG_FUSION_QUERY_COUNT,
                    model_name=rag_fusion_model or RAG_FUSION_LLM_MODEL,
                    timeout=rag_fusion_timeout,  # 传递超时参数
                    fusion_search_strategy=fusion_search_strategy,  # 传递策略参数
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    rrf_k=rrf_k,
                    **kwargs
                )
            elif search_mode == SearchMode.ADAPTIVE:
                docs = self.do_adaptive_search(query, top_k, score_threshold, 
                                             dense_weight, sparse_weight, rrf_k)
            else:
                # 默认使用向量检索
                logger.info(f"搜索模式 '{search_mode}' 不可用，使用向量检索")
                docs = self.do_search(query, top_k, score_threshold)
        except Exception as e:
            logger.error(f"搜索执行错误: {e}, 降级为向量检索")
            docs = self.do_search(query, top_k, score_threshold)
        
        return docs

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        '''根据文档ID检索文档'''
        return []

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        '''通过文档ID删除文档'''
        raise NotImplementedError

    def update_doc_by_ids(self, docs: Dict[str, Document]) -> bool:
        '''根据ID更新文档'''
        self.del_doc_by_ids(list(docs.keys()))
        docs_list = []
        ids = []
        for k, v in docs.items():
            if not v or not v.page_content.strip():
                continue
            ids.append(k)
            docs_list.append(v)
        self.do_add_doc(docs=docs_list, ids=ids)
        return True

    def list_docs(self, file_name: str = None, metadata: Dict = {}) -> List[DocumentWithVSId]:
        '''检索文档列表'''
        doc_infos = list_docs_from_db(kb_name=self.kb_name, file_name=file_name, metadata=metadata)
        docs = []
        for x in doc_infos:
            doc_info = self.get_doc_by_ids([x["id"]])[0]
            if doc_info is not None:
                doc_with_id = DocumentWithVSId(**doc_info.dict(), id=x["id"])
                docs.append(doc_with_id)
        return docs

    def get_relative_source_path(self, filepath: str):
        '''转换为相对路径'''
        relative_path = filepath
        if os.path.isabs(relative_path):
            try:
                relative_path = Path(filepath).relative_to(self.doc_path)
            except Exception as e:
                print(f"cannot convert absolute path ({filepath}) to relative path. error is : {e}")
        relative_path = str(relative_path.as_posix().strip("/"))
        return relative_path

    # ================= RAG-fusion支持方法 =================
    
    def supports_rag_fusion(self) -> bool:
        """检查当前向量存储是否支持RAG-fusion"""
        if not RAG_FUSION_AVAILABLE:
            return False
        
        # 检查知识库配置
        vs_config = kbs_config.get(self.vs_type(), {})
        supports_fusion = vs_config.get("supports_rag_fusion", True)  # 默认支持
        
        return supports_fusion
    
    def get_rag_fusion_config(self) -> Dict:
        """获取RAG-fusion配置"""
        default_config = {
            "query_generation": {
                "max_queries": 5,
                "min_queries": 2,
                "timeout": 30,
            },
            "fusion": {
                "method": "rrf",
                "rrf_k": 60,
                "top_k_per_query": 10,
                "final_top_k": 20,
            },
            "cache": {
                "enable": True,
                "max_size": 1000,
            }
        }
        
        if RAG_FUSION_CONFIG:
            # 递归合并配置
            def merge_configs(default, custom):
                result = default.copy()
                for key, value in custom.items():
                    if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                        result[key] = merge_configs(result[key], value)
                    else:
                        result[key] = value
                return result
            
            return merge_configs(default_config, RAG_FUSION_CONFIG)
        
        return default_config
    
    def do_rag_fusion_search(self,
                            query: str,
                            top_k: int,
                            score_threshold: float = SCORE_THRESHOLD,
                            query_count: int = 3,
                            model_name: str = None,
                            timeout: int = None,  # 新增：超时参数
                            fusion_search_strategy: str = None,  # 新增：搜索策略
                            dense_weight: float = 0.7,
                            sparse_weight: float = 0.3,
                            rrf_k: int = 60,
                            **kwargs
                            ) -> List[Tuple[Document, float]]:
        """
        执行RAG-fusion检索
        默认实现：生成多个查询，然后使用混合检索
        子类可以重写此方法提供更优化的实现
        """
        if not self.supports_rag_fusion():
            # 如果不支持RAG-fusion，降级为混合检索
            logger.info("RAG-fusion不支持，降级为混合检索")
            return self.do_hybrid_search(query, top_k, score_threshold, 
                                       dense_weight, sparse_weight, rrf_k)
        
        try:
            config = self.get_rag_fusion_config()
            
            # 生成融合查询
            query_timeout = timeout or config.get("query_generation", {}).get("timeout", 30)
            
            queries = generate_fusion_queries(
                original_query=query,
                num_queries=query_count,
                model_name=model_name or RAG_FUSION_LLM_MODEL,
                use_cache=config.get("cache", {}).get("enable", True),
                timeout=query_timeout
            )
            
            if not queries or len(queries) < 2:
                # 如果查询生成失败，降级为普通检索
                logger.info("查询生成失败，降级为混合检索")
                return self.do_hybrid_search(query, top_k, score_threshold,
                                           dense_weight, sparse_weight, rrf_k)
            
            # 更新统计
            self._rag_fusion_stats["queries_generated"] += len(queries)
            
            # 对每个查询执行检索
            all_results = []
            fusion_config = config.get("fusion", {})
            per_query_top_k = fusion_config.get("top_k_per_query", top_k)
            
            # 使用传入的搜索策略或默认策略
            search_strategy = fusion_search_strategy or kwargs.get("fusion_search_strategy", "hybrid")
            
            for fusion_query in queries:
                try:
                    # 根据搜索策略选择检索方法
                    if search_strategy == "hybrid":
                        results = self.do_hybrid_search(
                            fusion_query, per_query_top_k, score_threshold,
                            dense_weight, sparse_weight, rrf_k
                        )
                    elif search_strategy == "vector":
                        results = self.do_search(fusion_query, per_query_top_k, score_threshold)
                    elif search_strategy == "bm25":
                        results = self.do_bm25_search(fusion_query, per_query_top_k, score_threshold)
                    else:
                        # 默认使用混合检索
                        results = self.do_hybrid_search(
                            fusion_query, per_query_top_k, score_threshold,
                            dense_weight, sparse_weight, rrf_k
                        )
                    
                    if results:
                        all_results.append(results)
                        
                except Exception as e:
                    logger.warning(f"执行查询 '{fusion_query}' 时出错: {e}")
                    continue
            
            if not all_results:
                return []
            
            # 更新统计
            self._rag_fusion_stats["searches_executed"] += len(all_results)
            
            # 使用RRF融合所有结果
            final_rrf_k = fusion_config.get("rrf_k", rrf_k)
            final_top_k = fusion_config.get("final_top_k", top_k)
            
            fused_results = reciprocal_rank_fusion(
                all_results, 
                k=final_rrf_k, 
                top_k=final_top_k,
                normalize_scores=fusion_config.get("normalize_scores", True)
            )
            
            # 记录日志
            if config.get("logging", {}).get("enabled", False):
                logger.info(f"RAG-fusion检索完成: 使用{len(queries)}个查询, "
                      f"融合{len(all_results)}组结果, 最终返回{len(fused_results)}个文档")
            
            return fused_results
            
        except Exception as e:
            logger.error(f"RAG-fusion检索出错: {e}")
            # 出错时降级为混合检索
            return self.do_hybrid_search(query, top_k, score_threshold,
                                       dense_weight, sparse_weight, rrf_k)

    def do_adaptive_search(self,
                          query: str,
                          top_k: int,
                          score_threshold: float = SCORE_THRESHOLD,
                          dense_weight: float = 0.7,
                          sparse_weight: float = 0.3,
                          rrf_k: int = 60,
                          ) -> List[Tuple[Document, float]]:
        """
        自适应检索：根据查询特征自动选择最佳检索策略
        子类可以重写提供更智能的实现
        """
        try:
            # 查询分析
            query_lower = query.lower()
            query_length = len(query.split())
            
            # 检测查询类型的关键词
            config = RAG_FUSION_CONFIG.get("adaptive", {}) if RAG_FUSION_CONFIG else {}
            
            keyword_indicators = config.get("keyword_indicators", [
                "how", "what", "when", "where", "why", "who", "which", "how many", "how much"
            ])
            semantic_indicators = config.get("semantic_indicators", [
                "similar", "like", "related", "compare", "difference", "relationship", "explain"
            ])
            complex_indicators = config.get("complex_indicators", [
                "analyze", "detailed", "comprehensive", "in-depth", "thoroughly"
            ])
            
            has_keywords = any(indicator in query_lower for indicator in keyword_indicators)
            has_semantic = any(indicator in query_lower for indicator in semantic_indicators)
            has_complex = any(indicator in query_lower for indicator in complex_indicators)
            
            # 决策逻辑
            query_length_threshold = config.get("query_length_threshold", 10)
            complex_query_threshold = config.get("complex_query_threshold", 15)
            
            if has_complex or query_length > complex_query_threshold:
                # 复杂查询 -> RAG-fusion
                if RAG_FUSION_AVAILABLE and self.supports_rag_fusion():
                    return self.do_rag_fusion_search(
                        query, top_k, score_threshold, 
                        query_count=config.get("complex_query_count", 4),
                        dense_weight=dense_weight, sparse_weight=sparse_weight, rrf_k=rrf_k
                    )
            
            if has_keywords and query_length < query_length_threshold:
                # 短关键词查询 -> BM25
                return self.do_bm25_search(query, top_k, score_threshold)
            elif has_semantic or query_length > query_length_threshold:
                # 语义查询或中长查询 -> 向量检索
                return self.do_search(query, top_k, score_threshold)
            else:
                # 默认混合检索
                return self.do_hybrid_search(query, top_k, score_threshold,
                                           dense_weight, sparse_weight, rrf_k)
                
        except Exception as e:
            logger.warning(f"自适应检索出错: {e}, 降级为混合检索")
            return self.do_hybrid_search(query, top_k, score_threshold,
                                       dense_weight, sparse_weight, rrf_k)

    def get_stats(self) -> Dict:
        """获取服务统计信息（增强版）"""
        stats = {
            "kb_name": self.kb_name,
            "vs_type": self.vs_type(),
            "embed_model": self.embed_model,
            "file_count": self.count_files(),
            "supports_rag_fusion": self.supports_rag_fusion(),
            "search_modes_supported": SearchMode.get_all_modes(),
        }
        
        # 添加RAG-fusion统计
        if self.supports_rag_fusion():
            stats.update({
                "rag_fusion_stats": self._rag_fusion_stats.copy(),
                "rag_fusion_config": self.get_rag_fusion_config()
            })
        
        return stats

    # ================= 抽象方法 =================

    @abstractmethod
    def do_create_kb(self):
        """创建知识库的具体实现"""
        pass

    @staticmethod
    def list_kbs_type():
        '''列出所有可用的知识库类型'''
        return list(kbs_config.keys())

    @classmethod
    def list_kbs(cls):
        '''从数据库中检索所有知识库的列表'''
        return list_kbs_from_db()

    def exists(self, kb_name: str = None):
        '''检查知识库是否存在'''
        kb_name = kb_name or self.kb_name
        return kb_exists(kb_name)

    @abstractmethod
    def vs_type(self) -> str:
        '''返回向量搜索类型'''
        pass

    @abstractmethod
    def do_init(self):
        '''初始化操作'''
        pass

    @abstractmethod
    def do_drop_kb(self):
        """删除知识库"""
        pass

    @abstractmethod
    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float,
                  ) -> List[Tuple[Document, float]]:
        """向量检索"""
        pass

    @abstractmethod
    def do_bm25_search(self,
                       query: str,
                       top_k: int,
                       score_threshold: float,
                       ) -> List[Tuple[Document, float]]:
        """BM25检索"""
        pass

    @abstractmethod
    def do_hybrid_search(self,
                         query: str,
                         top_k: int,
                         score_threshold: float,
                         dense_weight: float = 0.7,
                         sparse_weight: float = 0.3,
                         rrf_k: int = 60,
                         ) -> List[Tuple[Document, float]]:
        """混合检索"""
        pass

    @abstractmethod
    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:
        """添加文档"""
        pass

    @abstractmethod
    def do_delete_doc(self,
                      kb_file: KnowledgeFile):
        """删除文档"""
        pass

    @abstractmethod
    def do_clear_vs(self):
        """清空向量库"""
        pass


# ================= 服务工厂 =================

class KBServiceFactory:

    @staticmethod
    def get_service(kb_name: str,
                    vector_store_type: Union[str, SupportedVSType],
                    embed_model: str = EMBEDDING_MODEL,
                    ) -> KBService:
        if isinstance(vector_store_type, str):
            vector_store_type = getattr(SupportedVSType, vector_store_type.upper(), SupportedVSType.FAISS)

        if SupportedVSType.FAISS == vector_store_type:
            from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
            return FaissKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.PG == vector_store_type:
            from server.knowledge_base.kb_service.pg_kb_service import PGKBService
            return PGKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.MILVUS == vector_store_type:
            from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
            return MilvusKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.ZILLIZ == vector_store_type:
            from server.knowledge_base.kb_service.zilliz_kb_service import ZillizKBService
            return ZillizKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.ES == vector_store_type:
            from server.knowledge_base.kb_service.es_kb_service import ESKBService
            return ESKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.CHROMADB == vector_store_type:
            from server.knowledge_base.kb_service.chromadb_kb_service import ChromaKBService
            return ChromaKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.DEFAULT == vector_store_type:
            from server.knowledge_base.kb_service.default_kb_service import DefaultKBService
            return DefaultKBService(kb_name)

    @staticmethod
    def get_service_by_name(kb_name: str) -> KBService:
        _, vs_type, embed_model = load_kb_from_db(kb_name)
        if _ is None:
            return None
        return KBServiceFactory.get_service(kb_name, vs_type, embed_model)

    @staticmethod
    def get_default():
        '''获取默认知识库服务实例'''
        return KBServiceFactory.get_service("default", SupportedVSType.DEFAULT)


# ================= 工具函数 =================

def get_kb_details() -> List[Dict]:
    """获取知识库详情（增强版）"""
    kbs_in_folder = list_kbs_from_folder()
    kbs_in_db = KBService.list_kbs()
    result = {}
    
    for kb in kbs_in_folder:
        result[kb] = {
            "kb_name": kb,
            "vs_type": "",
            "kb_info": "",
            "embed_model": "",
            "file_count": 0,
            "create_time": None,
            "in_folder": True,
            "in_db": False,
            "supports_rag_fusion": False,
            "search_modes": [SearchMode.VECTOR_ONLY]
        }
    
    for kb in kbs_in_db:
        kb_detail = get_kb_detail(kb)
        if kb_detail:
            kb_detail["in_db"] = True
            
            # 尝试获取服务能力信息
            try:
                service = KBServiceFactory.get_service_by_name(kb)
                if service:
                    kb_detail["supports_rag_fusion"] = service.supports_rag_fusion()
                    kb_detail["search_modes"] = SearchMode.get_all_modes() if service.supports_rag_fusion() else [SearchMode.VECTOR_ONLY, SearchMode.HYBRID]
            except:
                kb_detail["supports_rag_fusion"] = False
                kb_detail["search_modes"] = [SearchMode.VECTOR_ONLY]
            
            if kb in result:
                result[kb].update(kb_detail)
            else:
                kb_detail["in_folder"] = False
                result[kb] = kb_detail

    data = []
    for i, v in enumerate(result.values()):
        v['No'] = i + 1
        data.append(v)
    return data


def get_kb_file_details(kb_name: str) -> List[Dict]:
    '''获取知识库文件详情'''
    kb = KBServiceFactory.get_service_by_name(kb_name)
    if kb is None:
        return []
    
    files_in_folder = list_files_from_folder(kb_name)
    files_in_db = kb.list_files()
    result = {}
    
    for doc in files_in_folder:
        result[doc] = {
            "kb_name": kb_name,
            "file_name": doc,
            "file_ext": os.path.splitext(doc)[-1],
            "file_version": 0,
            "document_loader": "",
            "docs_count": 0,
            "text_splitter": "",
            "create_time": None,
            "in_folder": True,
            "in_db": False,
        }
    
    lower_names = {x.lower(): x for x in result}
    for doc in files_in_db:
        doc_detail = get_file_detail(kb_name, doc)
        if doc_detail:
            doc_detail["in_db"] = True
            if doc.lower() in lower_names:
                result[lower_names[doc.lower()]].update(doc_detail)
            else:
                doc_detail["in_folder"] = False
                result[doc] = doc_detail
    
    data = []
    for i, v in enumerate(result.values()):
        v['No'] = i + 1
        data.append(v)
    return data


class EmbeddingsFunAdapter(Embeddings):
    '''嵌入函数适配器'''
    def __init__(self, embed_model: str = EMBEDDING_MODEL):
        self.embed_model = embed_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = embed_texts(texts=texts, embed_model=self.embed_model, to_query=False).data
        return normalize(embeddings).tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = embed_texts(texts=[text], embed_model=self.embed_model, to_query=True).data
        query_embed = embeddings[0]
        query_embed_2d = np.reshape(query_embed, (1, -1))
        normalized_query_embed = normalize(query_embed_2d)
        return normalized_query_embed[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = (await aembed_texts(texts=texts, embed_model=self.embed_model, to_query=False)).data
        return normalize(embeddings).tolist()

    async def aembed_query(self, text: str) -> List[float]:
        embeddings = (await aembed_texts(texts=[text], embed_model=self.embed_model, to_query=True)).data
        query_embed = embeddings[0]
        query_embed_2d = np.reshape(query_embed, (1, -1))
        normalized_query_embed = normalize(query_embed_2d)
        return normalized_query_embed[0].tolist()


def score_threshold_process(score_threshold, k, docs):
    '''分数阈值处理'''
    if score_threshold is not None:
        cmp = operator.le
        docs = [
            (doc, similarity)
            for doc, similarity in docs
            if cmp(similarity, score_threshold)
        ]
    return docs[:k]