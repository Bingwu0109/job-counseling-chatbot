import os
import shutil
import pickle
import re
import time
from typing import List, Dict, Optional, Tuple
from rank_bm25 import BM25Okapi

from configs import SCORE_THRESHOLD
from server.knowledge_base.kb_service.base import (
    KBService, SupportedVSType, EmbeddingsFunAdapter, SearchMode,
    reciprocal_rank_fusion, score_threshold_process
)
from server.knowledge_base.kb_cache.faiss_cache import kb_faiss_pool, ThreadSafeFaiss
from server.knowledge_base.utils import KnowledgeFile, get_kb_path, get_vs_path
from server.utils import torch_gc
from langchain.docstore.document import Document

# 导入RAG-fusion相关配置
try:
    from configs import (
        ENABLE_RAG_FUSION,
        RAG_FUSION_CONFIG,
        RAG_FUSION_QUERY_COUNT,
        RAG_FUSION_LLM_MODEL,
        RAG_FUSION_SUPPORTED_MODELS
    )
    RAG_FUSION_AVAILABLE = ENABLE_RAG_FUSION
except ImportError:
    RAG_FUSION_AVAILABLE = False
    RAG_FUSION_CONFIG = {}
    RAG_FUSION_QUERY_COUNT = 3
    RAG_FUSION_LLM_MODEL = "Qwen1.5-7B-Chat"
    RAG_FUSION_SUPPORTED_MODELS = []

# 导入RAG-fusion工具函数
if RAG_FUSION_AVAILABLE:
    try:
        from server.knowledge_base.utils import (
            generate_fusion_queries,
            calculate_query_similarity,
            get_rag_fusion_stats
        )
    except ImportError:
        try:
            from server.utils import generate_fusion_queries
        except ImportError:
            RAG_FUSION_AVAILABLE = False
            print("警告: 无法导入generate_fusion_queries函数，RAG-fusion功能将被禁用")


class EnglishTextProcessor:
    """英文文本处理器 - 优化版"""
    
    def __init__(self):
        # 英文停用词（扩展版）
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'will', 'with', 'would', 'could', 'should', 'may', 'might', 'can',
            'shall', 'must', 'ought', 'need', 'dare', 'used', 'am', 'i', 'you', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our',
            'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'this', 'these',
            'that', 'those', 'there', 'here', 'when', 'where', 'why', 'what', 'which',
            'who', 'whom', 'whose', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'now', 'then', 'also', 'but', 'or',
            'if', 'because', 'while', 'during', 'before', 'after', 'above', 'below',
            'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        }
        
        # 技术术语词典（不会被过滤）
        self.technical_terms = {
            'api', 'url', 'http', 'https', 'json', 'xml', 'html', 'css', 'js',
            'sql', 'db', 'id', 'ui', 'ux', 'app', 'web', 'server', 'client',
            'config', 'setup', 'install', 'docker', 'git', 'npm', 'pip', 'ai',
            'ml', 'gpu', 'cpu', 'ram', 'ssd', 'hdd', 'os', 'linux', 'windows'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """英文分词处理 - 优化版"""
        if not text:
            return []
        
        # 转换为小写
        text = text.lower()
        
        # 使用更精确的正则表达式
        # 保留字母、数字和常见技术符号
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b|\b\d+\b', text)
        
        # 高级过滤逻辑
        filtered_tokens = []
        for token in tokens:
            # 保留技术术语
            if token in self.technical_terms:
                filtered_tokens.append(token)
                continue
            
            # 基本过滤条件
            if (len(token) >= 2 and  # 至少2个字符
                token not in self.stopwords and  # 不是停用词
                not token.isdigit() and  # 不是纯数字
                not re.match(r'^[0-9]+$', token)):  # 不是数字字符串
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def preprocess_query(self, query: str) -> str:
        """预处理查询文本"""
        # 移除多余的空格和标点
        query = re.sub(r'\s+', ' ', query.strip())
        # 保留重要的标点符号
        query = re.sub(r'[^\w\s\-\.]', ' ', query)
        return query


class FaissKBService(KBService):
    vs_path: str
    kb_path: str
    vector_name: str = None
    bm25_path: str = None
    _bm25_retriever = None
    _corpus_texts = None
    _text_processor = None
    
    # RAG-fusion缓存和统计
    _query_cache = {}
    _cache_max_size = 100
    _rag_fusion_stats = {
        "total_queries": 0,
        "cache_hits": 0,
        "successful_fusions": 0,
        "failed_fusions": 0,
        "average_query_time": 0.0
    }

    def vs_type(self) -> str:
        return SupportedVSType.FAISS

    def get_vs_path(self):
        return get_vs_path(self.kb_name, self.vector_name)

    def get_kb_path(self):
        return get_kb_path(self.kb_name)

    def get_bm25_path(self):
        return os.path.join(self.vs_path, "bm25_index.pkl")

    def load_vector_store(self) -> ThreadSafeFaiss:
        return kb_faiss_pool.load_vector_store(kb_name=self.kb_name,
                                               vector_name=self.vector_name,
                                               embed_model=self.embed_model)

    def save_vector_store(self):
        self.load_vector_store().save(self.vs_path)

    def _init_bm25_retriever(self):
        """初始化BM25检索器"""
        try:
            if not os.path.exists(self.bm25_path):
                print(f"BM25索引文件不存在: {self.bm25_path}，将重新构建")
                self._rebuild_bm25_index()
                return
                
            with open(self.bm25_path, 'rb') as f:
                bm25_data = pickle.load(f)
                self._bm25_retriever = bm25_data.get('bm25')
                self._corpus_texts = bm25_data.get('corpus_texts', [])
                
                if self._bm25_retriever and self._corpus_texts:
                    print(f"BM25索引已从文件加载: {self.bm25_path}, 语料库大小: {len(self._corpus_texts)}")
                else:
                    print("BM25索引数据不完整，重新构建")
                    self._rebuild_bm25_index()
                    
        except (FileNotFoundError, EOFError, pickle.UnpicklingError, Exception) as e:
            print(f"无法加载BM25索引: {e}，将重新构建")
            self._rebuild_bm25_index()

    def _rebuild_bm25_index(self):
        """重新构建BM25索引"""
        print("正在构建BM25索引...")

        try:
            with self.load_vector_store().acquire() as vs:
                if not hasattr(vs, 'docstore') or not vs.docstore or not vs.docstore._dict:
                    print("向量存储为空，BM25索引构建跳过")
                    self._bm25_retriever = None
                    self._corpus_texts = []
                    return
                
                all_docs = []
                for doc_id, doc in vs.docstore._dict.items():
                    if doc and hasattr(doc, 'page_content') and doc.page_content:
                        all_docs.append(doc)

            if not all_docs:
                print("没有找到有效文档，BM25索引构建失败")
                self._bm25_retriever = None
                self._corpus_texts = []
                return

            tokenized_corpus = []
            self._corpus_texts = []

            for doc in all_docs:
                try:
                    tokens = self._text_processor.tokenize(doc.page_content)
                    if tokens:  # 只添加有效的token化文档
                        tokenized_corpus.append(tokens)
                        self._corpus_texts.append(doc)
                except Exception as e:
                    print(f"处理文档时出错: {e}")
                    continue

            if tokenized_corpus and len(tokenized_corpus) > 0:
                self._bm25_retriever = BM25Okapi(tokenized_corpus)
                
                # 保存索引到文件
                try:
                    os.makedirs(os.path.dirname(self.bm25_path), exist_ok=True)
                    with open(self.bm25_path, 'wb') as f:
                        pickle.dump({
                            'bm25': self._bm25_retriever,
                            'corpus_texts': self._corpus_texts,
                            'build_time': time.time(),
                            'corpus_size': len(self._corpus_texts)
                        }, f)
                    print(f"BM25索引已保存到: {self.bm25_path}, 语料库大小: {len(self._corpus_texts)}")
                except Exception as e:
                    print(f"保存BM25索引失败: {e}")
            else:
                print("没有有效的文档内容用于构建BM25索引")
                self._bm25_retriever = None
                self._corpus_texts = []
                
        except Exception as e:
            print(f"构建BM25索引时出错: {e}")
            self._bm25_retriever = None
            self._corpus_texts = []

    def _clear_bm25_index(self):
        """清除BM25索引"""
        self._bm25_retriever = None
        self._corpus_texts = []
        try:
            if os.path.exists(self.bm25_path):
                os.remove(self.bm25_path)
                print(f"BM25索引文件已删除: {self.bm25_path}")
        except Exception as e:
            print(f"删除BM25索引文件失败: {e}")

    def _clean_query_cache(self):
        """清理查询缓存"""
        if len(self._query_cache) > self._cache_max_size:
            # 删除最旧的一半缓存
            items = list(self._query_cache.items())
            items.sort(key=lambda x: x[1].get('timestamp', 0))
            for key, _ in items[:len(items)//2]:
                self._query_cache.pop(key, None)

    def _update_rag_fusion_stats(self, query_time: float, success: bool):
        """更新RAG-fusion统计信息"""
        self._rag_fusion_stats["total_queries"] += 1
        if success:
            self._rag_fusion_stats["successful_fusions"] += 1
        else:
            self._rag_fusion_stats["failed_fusions"] += 1
        
        # 更新平均查询时间
        total = self._rag_fusion_stats["total_queries"]
        current_avg = self._rag_fusion_stats["average_query_time"]
        self._rag_fusion_stats["average_query_time"] = (current_avg * (total - 1) + query_time) / total

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        """根据ID获取文档"""
        try:
            with self.load_vector_store().acquire() as vs:
                if not vs.docstore or not vs.docstore._dict:
                    return [None] * len(ids)
                return [vs.docstore._dict.get(id) for id in ids]
        except Exception as e:
            print(f"根据ID获取文档失败: {e}")
            return [None] * len(ids)

    def del_doc_by_ids(self, ids: List[str]) -> bool:
        """根据ID删除文档"""
        try:
            with self.load_vector_store().acquire() as vs:
                vs.delete(ids)
            return True
        except Exception as e:
            print(f"根据ID删除文档失败: {e}")
            return False

    def do_init(self):
        '''初始化FaissKBService对象'''
        self.vector_name = self.vector_name or self.embed_model
        self.kb_path = self.get_kb_path()
        self.vs_path = self.get_vs_path()
        self.bm25_path = self.get_bm25_path()
        self._text_processor = EnglishTextProcessor()

        # 如果向量存储存在，初始化BM25检索器
        if os.path.exists(self.vs_path):
            self._init_bm25_retriever()
        
        # 设置RAG-fusion缓存大小
        cache_config = RAG_FUSION_CONFIG.get("cache", {}) if RAG_FUSION_CONFIG else {}
        self._cache_max_size = cache_config.get("max_cache_size", 100)

    def do_create_kb(self):
        '''创建知识库'''
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)
        self.load_vector_store()
        # 初始化空的BM25检索器
        self._bm25_retriever = None
        self._corpus_texts = []

    def do_drop_kb(self):
        '''删除知识库'''
        self.clear_vs()
        try:
            shutil.rmtree(self.kb_path)
            print(f"知识库目录已删除: {self.kb_path}")
        except Exception as e:
            print(f"删除知识库目录失败: {e}")

    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float = SCORE_THRESHOLD,
                  ) -> List[Tuple[Document, float]]:
        '''执行向量搜索查询'''
        try:
            embed_func = EmbeddingsFunAdapter(self.embed_model)
            embeddings = embed_func.embed_query(query)
            
            with self.load_vector_store().acquire() as vs:
                docs = vs.similarity_search_with_score_by_vector(
                    embeddings, k=top_k, score_threshold=score_threshold
                )
            return docs
        except Exception as e:
            print(f"向量搜索执行失败: {e}")
            return []

    def do_bm25_search(self,
                       query: str,
                       top_k: int,
                       score_threshold: float = SCORE_THRESHOLD,
                       ) -> List[Tuple[Document, float]]:
        """执行BM25稀疏检索"""
        if self._bm25_retriever is None or not self._corpus_texts:
            print("BM25检索器未初始化或语料库为空")
            return []

        try:
            # 预处理查询
            processed_query = self._text_processor.preprocess_query(query)
            query_tokens = self._text_processor.tokenize(processed_query)

            if not query_tokens:
                print("查询分词后为空")
                return []

            scores = self._bm25_retriever.get_scores(query_tokens)
            
            # 确保分数数组长度与语料库匹配
            if len(scores) != len(self._corpus_texts):
                print(f"分数数组长度({len(scores)})与语料库长度({len(self._corpus_texts)})不匹配")
                return []
            
            doc_scores = list(zip(self._corpus_texts, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # BM25分数阈值处理
            if score_threshold is not None and score_threshold > 0:
                doc_scores = [(doc, score) for doc, score in doc_scores 
                             if score >= score_threshold]

            return doc_scores[:top_k]

        except Exception as e:
            print(f"BM25检索出错: {e}")
            return []

    def do_hybrid_search(self,
                         query: str,
                         top_k: int,
                         score_threshold: float = SCORE_THRESHOLD,
                         dense_weight: float = 0.7,
                         sparse_weight: float = 0.3,
                         rrf_k: int = 60,
                         ) -> List[Tuple[Document, float]]:
        """执行混合检索（结合向量检索和BM25检索）"""
        try:
            # 获取稠密检索结果
            dense_results = self.do_search(query, top_k * 2, score_threshold)
            
            # 获取稀疏检索结果
            sparse_results = self.do_bm25_search(query, top_k * 2, None)  # BM25不使用score_threshold

            if not dense_results and not sparse_results:
                return []
            elif not dense_results:
                return sparse_results[:top_k]
            elif not sparse_results:
                return dense_results[:top_k]

            # RRF融合
            hybrid_results = reciprocal_rank_fusion(
                [dense_results, sparse_results],
                k=rrf_k,
                top_k=top_k,
                normalize_scores=True
            )

            # 应用分数阈值（针对融合后的分数）
            if score_threshold is not None and hybrid_results:
                # 对于RRF分数，我们使用相对阈值
                max_rrf_score = max(score for _, score in hybrid_results) if hybrid_results else 0
                if max_rrf_score > 0:
                    relative_threshold = max_rrf_score * 0.1  # 相对阈值
                    hybrid_results = [(doc, score) for doc, score in hybrid_results
                                    if score >= relative_threshold]

            return hybrid_results

        except Exception as e:
            print(f"混合检索出错: {e}")
            # 降级为向量搜索
            return self.do_search(query, top_k, score_threshold)

    def do_rag_fusion_search(self,
                            query: str,
                            top_k: int,
                            score_threshold: float = SCORE_THRESHOLD,
                            query_count: int = 3,
                            model_name: str = None,
                            dense_weight: float = 0.7,
                            sparse_weight: float = 0.3,
                            rrf_k: int = 60,
                            **kwargs
                            ) -> List[Tuple[Document, float]]:
        """
        FAISS优化的RAG-fusion检索实现
        """
        if not RAG_FUSION_AVAILABLE or not self.supports_rag_fusion():
            print("RAG-fusion不支持，降级为混合检索")
            return self.do_hybrid_search(query, top_k, score_threshold,
                                       dense_weight, sparse_weight, rrf_k)
        
        start_time = time.time()
        success = False
        
        try:
            config = self.get_rag_fusion_config()
            
            # 缓存键
            cache_key = f"{query}_{query_count}_{model_name}_{top_k}_{dense_weight}_{sparse_weight}"
            
            # 检查缓存
            cache_config = config.get("cache", {})
            if (cache_config.get("enable", True) and cache_key in self._query_cache):
                cache_entry = self._query_cache[cache_key]
                cache_age = time.time() - cache_entry['timestamp']
                cache_expire = cache_config.get("cache_ttl", 3600)
                
                if cache_age < cache_expire:
                    self._rag_fusion_stats["cache_hits"] += 1
                    print(f"RAG-fusion使用缓存结果 (age: {cache_age:.1f}s)")
                    return cache_entry['results']
            
            # 生成融合查询
            if 'generate_fusion_queries' not in globals():
                print("generate_fusion_queries函数不可用，降级为混合检索")
                return self.do_hybrid_search(query, top_k, score_threshold,
                                           dense_weight, sparse_weight, rrf_k)
            
            # 使用配置中的参数
            query_gen_config = config.get("query_generation", {})
            actual_query_count = min(query_count, query_gen_config.get("max_queries", 5))
            
            queries = generate_fusion_queries(
                original_query=query,
                num_queries=actual_query_count,
                model_name=model_name or RAG_FUSION_LLM_MODEL,
                use_cache=cache_config.get("enable", True),
                timeout=query_gen_config.get("timeout", 30)
            )
            
            if not queries or len(queries) < 2:
                print("RAG-fusion查询生成失败，降级为混合检索")
                return self.do_hybrid_search(query, top_k, score_threshold,
                                           dense_weight, sparse_weight, rrf_k)
            
            print(f"RAG-fusion生成了 {len(queries)} 个查询")
            
            # 并行执行多个查询检索
            all_results = []
            fusion_config = config.get("fusion", {})
            per_query_top_k = fusion_config.get("top_k_per_query", max(top_k, 5))
            
            for i, fusion_query in enumerate(queries):
                try:
                    # 为不同的查询使用不同的策略
                    if i == 0:  # 原查询使用混合检索
                        results = self.do_hybrid_search(
                            fusion_query, per_query_top_k, score_threshold,
                            dense_weight, sparse_weight, rrf_k
                        )
                    else:  # 生成的查询可以使用不同策略
                        search_strategy = kwargs.get("fusion_search_strategy", "hybrid")
                        if search_strategy == "vector":
                            results = self.do_search(fusion_query, per_query_top_k, score_threshold)
                        elif search_strategy == "bm25":
                            results = self.do_bm25_search(fusion_query, per_query_top_k, None)
                        else:  # hybrid (default)
                            # 为生成的查询微调权重
                            adj_dense_weight = dense_weight * (0.9 if i % 2 == 1 else 1.1)
                            adj_sparse_weight = 1.0 - adj_dense_weight
                            results = self.do_hybrid_search(
                                fusion_query, per_query_top_k, score_threshold,
                                adj_dense_weight, adj_sparse_weight, rrf_k
                            )
                    
                    if results:
                        all_results.append(results)
                        
                except Exception as e:
                    print(f"执行查询 '{fusion_query}' 时出错: {e}")
                    continue
            
            if not all_results:
                print("所有RAG-fusion查询都失败，降级为普通检索")
                return self.do_search(query, top_k, score_threshold)
            
            # 使用RRF融合所有结果
            final_rrf_k = fusion_config.get("rrf_k", rrf_k)
            final_top_k = fusion_config.get("final_top_k", top_k)
            
            fused_results = reciprocal_rank_fusion(
                all_results, 
                k=final_rrf_k, 
                top_k=final_top_k,
                normalize_scores=fusion_config.get("normalization", True)
            )
            
            # 可选的重排序
            if kwargs.get("enable_rerank", config.get("rerank", {}).get("enable", False)):
                fused_results = self._rerank_results(query, fused_results, 
                                                   kwargs.get("rerank_top_k", top_k))
            
            success = True
            
            # 缓存结果
            if cache_config.get("enable", True):
                self._clean_query_cache()
                self._query_cache[cache_key] = {
                    'results': fused_results,
                    'timestamp': time.time()
                }
            
            execution_time = time.time() - start_time
            
            # 记录详细日志
            if config.get("logging", {}).get("enabled", False):
                print(f"FAISS RAG-fusion完成: "
                      f"查询数={len(queries)}, 结果组={len(all_results)}, "
                      f"最终文档={len(fused_results)}, 耗时={execution_time:.2f}s")
            
            return fused_results
            
        except Exception as e:
            print(f"FAISS RAG-fusion检索出错: {e}")
            return self.do_hybrid_search(query, top_k, score_threshold,
                                       dense_weight, sparse_weight, rrf_k)
        finally:
            # 更新统计信息
            execution_time = time.time() - start_time
            self._update_rag_fusion_stats(execution_time, success)

    def _rerank_results(self, query: str, results: List[Tuple[Document, float]], 
                       top_k: int) -> List[Tuple[Document, float]]:
        """
        简单的重排序实现
        基于查询与文档的词汇重叠度和语义相关性
        """
        try:
            if not results:
                return results
                
            query_tokens = set(self._text_processor.tokenize(query))
            if not query_tokens:
                return results[:top_k]
            
            reranked_results = []
            for doc, score in results:
                try:
                    doc_tokens = set(self._text_processor.tokenize(doc.page_content))
                    
                    # 计算词汇重叠度
                    overlap = len(query_tokens.intersection(doc_tokens))
                    total = len(query_tokens.union(doc_tokens))
                    overlap_score = overlap / total if total > 0 else 0
                    
                    # 计算长度惩罚（避免过短或过长的文档）
                    doc_length = len(doc.page_content.split())
                    length_penalty = 1.0
                    if doc_length < 50:
                        length_penalty = 0.8
                    elif doc_length > 1000:
                        length_penalty = 0.9
                    
                    # 结合原始分数、重叠度分数和长度惩罚
                    final_score = (score * 0.6 + 
                                 overlap_score * 0.3 + 
                                 (overlap / len(query_tokens) if query_tokens else 0) * 0.1) * length_penalty
                    
                    reranked_results.append((doc, final_score))
                    
                except Exception as e:
                    print(f"重排序单个文档时出错: {e}")
                    reranked_results.append((doc, score))  # 保持原始分数
            
            # 按最终分数重新排序
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            return reranked_results[:top_k]
            
        except Exception as e:
            print(f"重排序失败: {e}")
            return results[:top_k]

    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:
        '''添加文档到知识库'''
        try:
            data = self._docs_to_embeddings(docs)
            
            with self.load_vector_store().acquire() as vs:
                ids = vs.add_embeddings(text_embeddings=zip(data["texts"], data["embeddings"]),
                                        metadatas=data["metadatas"],
                                        ids=kwargs.get("ids"))
                if not kwargs.get("not_refresh_vs_cache"):
                    vs.save_local(self.vs_path)

            # 重新构建BM25索引
            if not kwargs.get("not_refresh_bm25_cache", False):
                self._rebuild_bm25_index()
            

            self._query_cache.clear()

            doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
            torch_gc()
            return doc_infos
            
        except Exception as e:
            print(f"添加文档失败: {e}")
            return []

    def do_delete_doc(self,
                      kb_file: KnowledgeFile,
                      **kwargs):
        '''从知识库删除文档'''
        try:
            with self.load_vector_store().acquire() as vs:
                if not vs.docstore or not vs.docstore._dict:
                    return []
                    
                ids = [k for k, v in vs.docstore._dict.items() if
                       v and v.metadata.get("source", "").lower() == kb_file.filename.lower()]
                
                if len(ids) > 0:
                    vs.delete(ids)
                    print(f"删除了 {len(ids)} 个文档块")
                
                if not kwargs.get("not_refresh_vs_cache"):
                    vs.save_local(self.vs_path)

            # 重新构建BM25索引
            if not kwargs.get("not_refresh_bm25_cache", False):
                self._rebuild_bm25_index()
            
            # 清空RAG-fusion缓存
            self._query_cache.clear()

            return ids
            
        except Exception as e:
            print(f"删除文档失败: {e}")
            return []

    def do_clear_vs(self):
        '''清除向量存储'''
        try:
            with kb_faiss_pool.atomic:
                kb_faiss_pool.pop((self.kb_name, self.vector_name))

            self._clear_bm25_index()
            self._query_cache.clear()  # 清空RAG-fusion缓存
            
            if os.path.exists(self.vs_path):
                shutil.rmtree(self.vs_path)
            
            os.makedirs(self.vs_path, exist_ok=True)
            print(f"向量存储已清空: {self.vs_path}")
            
        except Exception as e:
            print(f"清空向量存储失败: {e}")

    def exist_doc(self, file_name: str):
        '''检查文档是否存在'''
        if super().exist_doc(file_name):
            return "in_db"
        
        content_path = os.path.join(self.kb_path, "content")
        if os.path.isfile(os.path.join(content_path, file_name)):
            return "in_folder"
        else:
            return False

    def supports_rag_fusion(self) -> bool:
        """FAISS支持RAG-fusion"""
        return RAG_FUSION_AVAILABLE and 'generate_fusion_queries' in globals()

    def get_stats(self) -> Dict:
        """获取服务统计信息"""
        try:
            with self.load_vector_store().acquire() as vs:
                doc_count = len(vs.docstore._dict) if vs.docstore and vs.docstore._dict else 0
        except:
            doc_count = 0
        
        stats = {
            "vs_type": self.vs_type(),
            "kb_name": self.kb_name,
            "embed_model": self.embed_model,
            "document_count": doc_count,
            "bm25_corpus_size": len(self._corpus_texts) if self._corpus_texts else 0,
            "bm25_available": self._bm25_retriever is not None,
            "rag_fusion_available": self.supports_rag_fusion(),
            "rag_fusion_cache_size": len(self._query_cache),
            "search_modes_supported": [mode for mode in SearchMode.get_all_modes() 
                                     if mode != SearchMode.RAG_FUSION or self.supports_rag_fusion()]
        }
        
        # 添加RAG-fusion统计
        if self.supports_rag_fusion():
            stats["rag_fusion_stats"] = self._rag_fusion_stats.copy()
        
        return stats

    def get_rag_fusion_config(self) -> Dict:
        """获取RAG-fusion配置（继承自基类但可以覆盖）"""
        base_config = super().get_rag_fusion_config()
        
        # FAISS特定的优化配置
        faiss_config = {
            "fusion": {
                "top_k_per_query": max(base_config.get("fusion", {}).get("final_top_k", 10), 5),
                "enable_rerank": True,
            },
            "performance": {
                "parallel_queries": False,  # FAISS通常在单线程下表现更好
                "cache_embeddings": True,
            }
        }
        
        # 合并配置
        def merge_configs(base, override):
            result = base.copy()
            for key, value in override.items():
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    result[key] = merge_configs(result[key], value)
                else:
                    result[key] = value
            return result
        
        return merge_configs(base_config, faiss_config)


if __name__ == '__main__':
    # 测试代码
    try:
        service = FaissKBService("test")
        print("FAISS服务初始化完成")
        print(f"支持的功能: {service.get_stats()}")
        
        # 测试RAG-fusion
        if service.supports_rag_fusion():
            print("✅ 支持RAG-fusion功能")
        else:
            print("❌ RAG-fusion功能不可用")
            
    except Exception as e:
        print(f"FAISS服务初始化失败: {e}")
