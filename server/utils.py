# -*- coding: utf-8 -*-

# pydantic库，一个数据验证和设置管理的库，常用于数据的校验以及配置管理。
import pydantic
# BaseModel是用于创建数据模型的基类，通常用于类型注解和验证。
from pydantic import BaseModel
from typing import List
# FastAPI是一个用于构建API的现代、快速（高性能）的Web框架。
from fastapi import FastAPI
# Path是用于文件系统路径操作的类
from pathlib import Path
import asyncio
# 从configs模块中导入多个配置项
from configs import (LLM_MODELS, LLM_DEVICE, EMBEDDING_DEVICE,
                     MODEL_PATH, MODEL_ROOT_PATH, ONLINE_LLM_MODEL, logger, log_verbose,
                     FSCHAT_MODEL_WORKERS, HTTPX_DEFAULT_TIMEOUT)
                     
# 安全导入RAG-fusion相关配置
try:
    from configs import (ENABLE_RAG_FUSION, RAG_FUSION_CONFIG, RAG_FUSION_LLM_MODEL,
                         RAG_FUSION_SUPPORTED_MODELS, RAG_FUSION_MODEL_SELECTION)
except ImportError:
    # 如果导入失败，设置默认值
    ENABLE_RAG_FUSION = False
    RAG_FUSION_CONFIG = {}
    RAG_FUSION_LLM_MODEL = "Qwen1.5-7B-Chat"
    RAG_FUSION_SUPPORTED_MODELS = []
    RAG_FUSION_MODEL_SELECTION = {}

# 导入更多必需的配置
try:
    from configs import (
        VECTOR_SEARCH_TOP_K,
        SCORE_THRESHOLD,
        TEMPERATURE,
        SEARCH_ENGINE_TOP_K,
    )
except ImportError:
    VECTOR_SEARCH_TOP_K = 7
    SCORE_THRESHOLD = 1.0
    TEMPERATURE = 0.7
    SEARCH_ENGINE_TOP_K = 7

import os
# ThreadPoolExecutor是一个线程池执行器，用于并行执行多个任务。
# as_completed是一个函数，用于迭代完成的future对象。
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
# httpx是一个第三方HTTP客户端库，支持异步请求。
import httpx
import contextlib
import json
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Callable,
    Generator,
    Dict,
    Any,
    Awaitable,
    Union,
    Tuple
)
import logging
import torch
import time
import re
import hashlib
from functools import lru_cache

from server.minx_chat_openai import MinxChatOpenAI


async def wrap_done(fn: Awaitable, # fn是一个Awaitable对象，即一个可以被await的对象，如异步函数的调用结果。
                    event: asyncio.Event # event是一个asyncio.Event对象，用于在异步操作中进行同步。
                    ):
    """用于执行一个awaitable对象（即可以用await语句等待的对象），
    并在完成或发生异常时通过一个事件（asyncio.Event）来发出信号。"""
    try:
        await fn # 等待fn执行完成
    except Exception as e:
        # 这个方法会自动记录异常的堆栈跟踪信息
        logging.exception(e)
        msg = f"Caught exception: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
    finally:
        # 调用event的set方法来设置事件的状态为已发生（signaled）。
        # 这通常用于通知其他等待这个事件的异步任务可以继续执行。
        event.set()


def get_ChatOpenAI(
        model_name: str, # 模型名称
        temperature: float, # 控制生成文本的随机性，较高的值会产生更多变化的结果。
        max_tokens: int = None, # 生成回复时的最大令tokens（即单词数），默认为None。
        streaming: bool = True, # 是否以流式方式处理聊天请求
        callbacks: List[Callable] = [], # 回调函数列表，用于处理聊天过程中的特定事件，默认为空列表。
        verbose: bool = True, # 是否启用详细模式，以提供更多的日志输出，默认为True。
        **kwargs: Any, # 接受任意数量的关键字参数，用于提供额外的配置选项。
) -> ChatOpenAI:
    '''根据给定的参数创建并配置一个ChatOpenAI对象，用于与OpenAI聊天模型的交互。'''
    # 传入模型名称，获取模型的配置信息。
    config = get_model_worker_config(model_name)
    if model_name == "openai-api":
        # 从配置中获取实际的模型名称
        model_name = config.get("model_name")
    # 使用特定的编码模型实现
    ChatOpenAI._get_encoding_model = MinxChatOpenAI.get_encoding_model
    # 创建ChatOpenAI实例
    model = ChatOpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=config.get("openai_proxy"),
        **kwargs
    )
    return model


def get_OpenAI(
        model_name: str,
        temperature: float,
        max_tokens: int = None,
        streaming: bool = True,
        echo: bool = True, # 用于控制是否回显输入的文本
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> OpenAI:
    # 获取指定模型的配置信息
    config = get_model_worker_config(model_name)
    if model_name == "openai-api":
        # 从配置中获取实际的模型名称
        model_name = config.get("model_name")
    # 创建OpenAI实例
    model = OpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=config.get("openai_proxy"),
        echo=echo,
        **kwargs
    )
    return model


# ================= RAG-fusion工具函数 =================

# 简单的内存缓存
_rag_fusion_query_cache = {}

def _clean_cache():
    """清理过期的缓存条目"""
    current_time = time.time()
    cache_expire_time = RAG_FUSION_CONFIG.get("cache", {}).get("cache_expire_time", 3600)
    
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
    """
    计算两个查询之间的简单相似度（基于词汇重叠）
    返回值在0-1之间，1表示完全相同
    """
    # 简单的基于词汇的相似度计算
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
    # 清理响应文本
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
    
    queries = []
    # 添加原始查询
    queries.append(original_query)
    
    for line in lines:
        # 移除数字编号、破折号等前缀
        clean_line = re.sub(r'^[\d\-\.\)\(]*\s*', '', line).strip()
        
        # 移除引号
        clean_line = clean_line.strip('"\'')
        
        # 过滤过短或过长的查询
        if 5 <= len(clean_line) <= 100 and clean_line not in queries:
            queries.append(clean_line)
    
    return queries


def get_rag_fusion_llm_client(model_name: str = None) -> Union[ChatOpenAI, OpenAI]:
    """
    获取用于RAG-fusion查询生成的LLM客户端
    
    Args:
        model_name: 指定的模型名称，如果为None则使用配置的默认模型
        
    Returns:
        LLM客户端实例
    """
    model_name = model_name or RAG_FUSION_LLM_MODEL
    
    # 检查模型是否支持RAG-fusion
    if model_name not in RAG_FUSION_SUPPORTED_MODELS:
        logger.warning(f"模型 {model_name} 不在RAG-fusion支持列表中，使用默认模型")
        model_name = RAG_FUSION_SUPPORTED_MODELS[0] if RAG_FUSION_SUPPORTED_MODELS else "Qwen1.5-7B-Chat"
    
    # 获取RAG-fusion特定配置
    try:
        from configs.model_config import get_rag_fusion_model_config
        rag_config = get_rag_fusion_model_config(model_name)
    except ImportError:
        rag_config = {"temperature": 0.7, "max_tokens": 200}
    
    temperature = rag_config.get("temperature", 0.7)
    max_tokens = rag_config.get("max_tokens", 200)
    
    try:
        # 优先使用ChatOpenAI（更适合对话）
        return get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=False,  # RAG-fusion不需要流式输出
            verbose=False,    # 减少日志输出
        )
    except Exception as e:
        logger.warning(f"创建ChatOpenAI客户端失败，尝试使用OpenAI: {e}")
        try:
            return get_OpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=False,
                echo=False,
                verbose=False,
            )
        except Exception as e2:
            logger.error(f"创建LLM客户端失败: {e2}")
            raise e2


async def generate_fusion_queries_async(
    original_query: str,
    num_queries: int = 3,
    model_name: str = None,
    use_cache: bool = True,
    timeout: int = 30
) -> List[str]:
    """
    异步生成RAG-fusion查询
    
    Args:
        original_query: 原始查询
        num_queries: 生成的查询总数（包括原查询）
        model_name: 使用的模型名称
        use_cache: 是否使用缓存
        timeout: 超时时间
        
    Returns:
        查询列表
    """
    if not ENABLE_RAG_FUSION:
        return [original_query]
    
    # 检查缓存
    if use_cache:
        _clean_cache()
        cache_key = _get_cache_key(original_query, {"query_count": num_queries, "llm_model": model_name})
        if cache_key in _rag_fusion_query_cache:
            timestamp, cached_queries = _rag_fusion_query_cache[cache_key]
            if RAG_FUSION_CONFIG.get("logging", {}).get("log_generated_queries", True):
                logger.info(f"使用缓存的RAG-fusion查询: {cached_queries}")
            return cached_queries
    
    try:
        # 获取LLM客户端
        llm_client = get_rag_fusion_llm_client(model_name)
        
        # 构建提示
        prompt_template = RAG_FUSION_CONFIG.get("query_generation", {}).get("prompt_template", "")
        if not prompt_template:
            prompt_template = """Based on the original query, generate {num_queries} different but related search queries to improve information retrieval.

Original query: {original_query}

Requirements:
1. Generate {num_queries} queries (including the original one)
2. Each query should approach the topic from a different angle
3. Queries should be concise and focused
4. Use synonyms and related terms when appropriate
5. Return only the queries, one per line

Generated queries:"""
        
        prompt = prompt_template.format(
            original_query=original_query,
            num_queries=num_queries - 1  # 减1因为原查询会被自动添加
        )
        
        # 异步调用LLM
        start_time = time.time()
        if hasattr(llm_client, 'agenerate'):
            response = await asyncio.wait_for(llm_client.agenerate([prompt]), timeout=timeout)
            response_text = response.generations[0][0].text
        else:
            # 如果不支持异步，在线程池中运行
            response_text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: llm_client.generate([prompt]).generations[0][0].text
            )
        
        generation_time = time.time() - start_time
        
        # 解析生成的查询
        queries = _parse_generated_queries(response_text, original_query)
        
        # 过滤相似查询
        filter_similar = RAG_FUSION_CONFIG.get("query_generation", {}).get("filter_similar", True)
        if filter_similar:
            similarity_threshold = RAG_FUSION_CONFIG.get("query_generation", {}).get("similarity_threshold", 0.9)
            queries = _filter_similar_queries(queries, similarity_threshold)
        
        # 限制查询数量
        queries = queries[:num_queries]
        
        # 确保至少有原查询
        if original_query not in queries:
            queries = [original_query] + queries[1:num_queries]
        
        # 缓存结果
        if use_cache:
            max_cache_size = RAG_FUSION_CONFIG.get("cache", {}).get("max_cache_size", 1000)
            if len(_rag_fusion_query_cache) < max_cache_size:
                _rag_fusion_query_cache[cache_key] = (time.time(), queries)
        
        # 记录日志
        if RAG_FUSION_CONFIG.get("logging", {}).get("log_generated_queries", True):
            logger.info(f"RAG-fusion查询生成成功 (耗时: {generation_time:.2f}s): {queries}")
        
        return queries
        
    except asyncio.TimeoutError:
        logger.warning(f"RAG-fusion查询生成超时，返回原查询")
        return [original_query]
    except Exception as e:
        logger.error(f"RAG-fusion查询生成失败: {e}")
        return [original_query]


def generate_fusion_queries(
    original_query: str,
    num_queries: int = 3,
    model_name: str = None,
    use_cache: bool = True,
    timeout: int = 30
) -> List[str]:
    """
    同步版本的RAG-fusion查询生成
    
    Args:
        original_query: 原始查询
        num_queries: 生成的查询总数
        model_name: 使用的模型名称
        use_cache: 是否使用缓存
        timeout: 超时时间
        
    Returns:
        查询列表
    """
    if not ENABLE_RAG_FUSION:
        return [original_query]
    
    # 使用run_async运行异步版本
    return run_async(generate_fusion_queries_async(
        original_query=original_query,
        num_queries=num_queries,
        model_name=model_name,
        use_cache=use_cache,
        timeout=timeout
    ))


def validate_rag_fusion_config() -> bool:
    """
    验证RAG-fusion配置的有效性
    
    Returns:
        bool: 配置是否有效
    """
    if not ENABLE_RAG_FUSION:
        return True
    
    try:
        # 检查基本配置
        assert RAG_FUSION_CONFIG is not None, "RAG_FUSION_CONFIG未正确配置"
        assert RAG_FUSION_LLM_MODEL in RAG_FUSION_SUPPORTED_MODELS, f"默认模型{RAG_FUSION_LLM_MODEL}不在支持列表中"
        
        # 检查模型可用性
        try:
            test_client = get_rag_fusion_llm_client()
            logger.info(f"RAG-fusion模型 {RAG_FUSION_LLM_MODEL} 验证成功")
        except Exception as e:
            logger.warning(f"RAG-fusion模型验证失败: {e}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"RAG-fusion配置验证失败: {e}")
        return False


def get_rag_fusion_stats() -> Dict[str, Any]:
    """
    获取RAG-fusion统计信息
    
    Returns:
        统计信息字典
    """
    return {
        "enabled": ENABLE_RAG_FUSION,
        "cache_size": len(_rag_fusion_query_cache),
        "supported_models": RAG_FUSION_SUPPORTED_MODELS,
        "current_model": RAG_FUSION_LLM_MODEL,
        "config_valid": validate_rag_fusion_config(),
        "cache_stats": {
            "entries": len(_rag_fusion_query_cache),
            "max_size": RAG_FUSION_CONFIG.get("cache", {}).get("max_cache_size", 1000)
        }
    }


def clear_rag_fusion_cache():
    """清空RAG-fusion缓存"""
    global _rag_fusion_query_cache
    _rag_fusion_query_cache.clear()
    logger.info("RAG-fusion缓存已清空")


# ================= 原有工具函数继续 =================

class BaseResponse(BaseModel):
    # API的状态码
    code: int = pydantic.Field(200, description="API status code")
    # 状态信息
    msg: str = pydantic.Field("success", description="API status message")
    # 返回的数据，Any，意味着它可以是任何类型的数据
    data: Any = pydantic.Field(None, description="API data")
    # 内嵌的类，用于为BaseResponse类提供配置信息
    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListResponse(BaseResponse):

    data: List[str] = pydantic.Field(..., description="List of names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


class ChatMessage(BaseModel):
    # 表示提出的问题文本，通过...表示该字段是必填的
    question: str = pydantic.Field(..., description="Question text")
    # 表示对question的回答文本
    response: str = pydantic.Field(..., description="Response text")
    # 记录了在当前问题之前的交互历史，每个子列表包含一组问题和对应的回答。
    history: List[List[str]] = pydantic.Field(..., description="History text")
    # 记录生成响应时参考的文档和这些文档的得分或其他相关信息。
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )
    # 配置类，用于提供模型的配置信息。
    class Config:
        schema_extra = {
            "example": {
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n" +
                            "2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n" +
                            "3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n" +
                            "4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n" +
                            "5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n" +
                            "6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，" +
                        "由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t" +
                    "( 一)  从业单位  (组织)  按'自愿参保'原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }


# ================= ApiRequest类 - 支持RAG-fusion =================

def set_httpx_config():
    """设置httpx配置"""
    pass  # 简化版，实际实现可以更复杂

def api_address() -> str:
    """获取API地址"""
    from configs.server_config import API_SERVER
    host = API_SERVER.get("host", "127.0.0.1")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = API_SERVER.get("port", 7861)
    return f"http://{host}:{port}"

def get_httpx_client(base_url: str = None, use_async: bool = False, timeout: float = HTTPX_DEFAULT_TIMEOUT):
    """获取httpx客户端"""
    if use_async:
        return httpx.AsyncClient(base_url=base_url, timeout=timeout)
    else:
        return httpx.Client(base_url=base_url, timeout=timeout)


class ApiRequest:
    '''
    api.py调用的封装（同步模式），简化api调用方式，支持RAG-fusion
    '''

    def __init__(
            self,
            base_url: str = None,
            timeout: float = HTTPX_DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url or api_address()
        self.timeout = timeout
        self._use_async = False
        self._client = None

    @property
    def client(self):
        if self._client is None or self._client.is_closed:
            self._client = get_httpx_client(base_url=self.base_url,
                                            use_async=self._use_async,
                                            timeout=self.timeout)
        return self._client

    def get(self, url: str, params=None, **kwargs):
        """GET请求"""
        return self.client.get(url, params=params, **kwargs)
    
    def post(self, url: str, json=None, **kwargs):
        """POST请求"""
        return self.client.post(url, json=json, **kwargs)

    def _get_response_value(self, response, as_json: bool = True, value_func: Callable = None):
        """获取响应值"""
        try:
            if response.status_code == 200:
                if as_json:
                    data = response.json()
                    if value_func:
                        return value_func(data)
                    return data
                else:
                    return response.text
            else:
                logger.error(f"API请求失败: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            logger.error(f"处理API响应失败: {e}")
            return {}

    def _httpx_stream2generator(self, response, as_json: bool = True):
        """将httpx stream响应转换为生成器"""
        try:
            with response as r:
                for chunk in r.iter_text():
                    if chunk.strip():
                        if as_json:
                            try:
                                yield json.loads(chunk)
                            except json.JSONDecodeError:
                                yield {"text": chunk}
                        else:
                            yield chunk
        except Exception as e:
            logger.error(f"流处理错误: {e}")
            yield {"error": str(e)}

    def search_kb_docs(self,
                       query: str,
                       knowledge_base_name: str,
                       top_k: int = VECTOR_SEARCH_TOP_K,
                       score_threshold: float = SCORE_THRESHOLD,
                       file_name: str = "",
                       metadata: dict = {},
                       # RAG-fusion 和混合检索参数
                       search_mode: str = None,
                       dense_weight: float = None,
                       sparse_weight: float = None,
                       rrf_k: int = None,
                       enable_rag_fusion: bool = None,
                       fusion_search_strategy: str = None,
                       fusion_query_count: int = None,
                       fusion_model_name: str = None,
                       fusion_timeout: int = None,
                       **kwargs) -> List[Dict]:
        """
        从知识库检索文档片段（支持RAG-fusion）
        """
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "file_name": file_name,
            "metadata": metadata,
        }
        
        # 只添加非None的可选参数
        optional_params = {
            "search_mode": search_mode,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "rrf_k": rrf_k,
            "enable_rag_fusion": enable_rag_fusion,
            "fusion_search_strategy": fusion_search_strategy,
            "fusion_query_count": fusion_query_count,
            "fusion_model_name": fusion_model_name,
            "fusion_timeout": fusion_timeout,
        }
        
        # 只添加有值的参数
        for key, value in optional_params.items():
            if value is not None:
                data[key] = value
        
        # 添加其他kwargs参数
        for key, value in kwargs.items():
            if value is not None and key not in data:
                data[key] = value

        response = self.post("/knowledge_base/search_docs", json=data)
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", []))

    def knowledge_base_chat(self,
                            query: str,
                            knowledge_base_name: str,
                            top_k: int = VECTOR_SEARCH_TOP_K,
                            score_threshold: float = SCORE_THRESHOLD,
                            history: List[Dict] = [],
                            stream: bool = True,
                            model: str = None,
                            temperature: float = TEMPERATURE,
                            max_tokens: int = None,
                            prompt_name: str = "default",
                            # RAG-fusion 和混合检索参数  
                            search_mode: str = None,
                            dense_weight: float = None,
                            sparse_weight: float = None,
                            rrf_k: int = None,
                            enable_rag_fusion: bool = None,
                            fusion_search_strategy: str = None,
                            fusion_query_count: int = None,
                            fusion_model_name: str = None,
                            fusion_timeout: int = None,
                            **kwargs):
        """
        对应api.py/chat/knowledge_base_chat接口（支持RAG-fusion）
        """
        if model is None:
            model = LLM_MODELS[0] if LLM_MODELS else "Qwen1.5-7B-Chat"
            
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }
        
        # 只添加非None的可选参数
        optional_params = {
            "search_mode": search_mode,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "rrf_k": rrf_k,
            "enable_rag_fusion": enable_rag_fusion,
            "fusion_search_strategy": fusion_search_strategy,
            "fusion_query_count": fusion_query_count,
            "fusion_model_name": fusion_model_name,
            "fusion_timeout": fusion_timeout,
        }
        
        # 只添加有值的参数
        for key, value in optional_params.items():
            if value is not None:
                data[key] = value
        
        # 添加其他kwargs参数
        for key, value in kwargs.items():
            if value is not None and key not in data:
                data[key] = value

        response = self.post("/chat/knowledge_base_chat", json=data, stream=True)
        return self._httpx_stream2generator(response, as_json=True)

    def list_knowledge_bases(self):
        """列出所有知识库"""
        response = self.get("/knowledge_base/list_knowledge_bases")
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", []))

    def chat(self,
             query: str,
             conversation_id: str = None,
             history_len: int = -1,
             history: List[Dict] = [],
             stream: bool = True,
             model: str = None,
             temperature: float = TEMPERATURE,
             max_tokens: int = None,
             prompt_name: str = "default",
             **kwargs):
        """普通对话接口"""
        if model is None:
            model = LLM_MODELS[0] if LLM_MODELS else "Qwen1.5-7B-Chat"
            
        data = {
            "query": query,
            "conversation_id": conversation_id,
            "history_len": history_len,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        response = self.post("/chat/chat", json=data, stream=True, **kwargs)
        return self._httpx_stream2generator(response, as_json=True)


def torch_gc():
    '''释放由PyTorch在使用GPU或Apple的Metal Performance Shaders (MPS)时占用的内存'''
    try:
        import torch
        if torch.cuda.is_available():
            # with torch.cuda.device(DEVICE):
            # 清空CUDA的缓存，释放未使用的内存。
            torch.cuda.empty_cache()
            # 调用 ipc_collect 函数收集和释放跨进程通信(IPC)资源
            torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
            # 检查是否可以使用MPS（Metal Performance Shaders），这是Apple为其设备提供的一个加速计算框架。
            try:
                from torch.mps import empty_cache
                # 调用MPS的 empty_cache 函数来释放内存
                empty_cache()
            except Exception as e:
                msg = ("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，" +
                       "以支持及时清理 torch 产生的内存占用。")
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
    except Exception:
        ...


def run_async(cor):
    '''
    在同步环境中运行异步代码.

    cor: 一个协程（coroutine），协程是Python中用于异步编程的一个构建块。
    '''
    try:
        # 获取当前线程的事件循环，事件循环是异步编程中的核心概念，用于调度和执行异步任务。
        loop = asyncio.get_event_loop()
    except:
        # 创建一个新的事件循环并将其赋值给 loop 变量
        loop = asyncio.new_event_loop()

    # 使用获取或创建的事件循环 loop 来运行传入的协程 cor。
    # run_until_complete 方法将运行事件循环，直到参数 cor 指定的协程完成。
    # 完成后，返回协程的结果。
    return loop.run_until_complete(cor)


def iter_over_async(ait, loop=None):
    '''
    目的是将异步生成器封装成同步生成器，从而允许在同步环境中使用异步生成器产生的值。

    ait: 异步迭代器
    loop: 一个事件循环对象，默认为 None。
    '''
    # 用异步迭代器的 __aiter__ 方法，确保 ait 是一个异步迭代器对象。
    ait = ait.__aiter__()

    async def get_next():
        '''获取 ait 的下一个元素'''
        try:
            # 使用 await 关键字等待异步迭代器的 __anext__ 方法完成，
            # 获取下一个元素并将其赋值给 obj。
            obj = await ait.__anext__()
            #  如果成功获取到元素，返回一个元组，
            #  第一个元素为 False，表示没有到达迭代结束，
            #  第二个元素是获取到的对象 obj。
            return False, obj
        # 如果在尝试获取下一个元素时引发了 StopAsyncIteration 异常，
        # 表示异步迭代器已经没有更多元素可以迭代。
        except StopAsyncIteration:
            # 在捕获到 StopAsyncIteration 异常时，返回一个元组，
            # 第一个元素为 True，表示迭代已结束，第二个元素为 None。
            return True, None

    if loop is None:
        try:
            # 获取当前事件循环，如果失败（比如没有运行的事件循环），
            # 则通过 asyncio.new_event_loop() 创建一个新的事件循环。
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()

    # 开始一个无限循环，用于迭代异步生成器。
    while True:
        # 在事件循环中运行 get_next 异步函数，等待其完成，并获取返回的元组值。
        done, obj = loop.run_until_complete(get_next())
        # 检查 done 是否为 True，如果是，表示异步迭代器已经没有更多元素可以迭代，因此跳出循环。
        if done:
            break
        #  如果 done 为 False，则 yield 返回的对象 obj，
        #  允许同步环境中的代码逐个处理异步生成器产生的元素。
        yield obj


def MakeFastAPIOffline(
        app: FastAPI, # 需要被打补丁的FastAPI应用实例
        static_dir=Path(__file__).parent / "static", # 静态文件目录
        static_url="/static-offline-docs", # 静态文件URL
        docs_url: Optional[str] = "/docs", # 设置Swagger UI文档的URL
        redoc_url: Optional[str] = "/redoc", # 设置ReDoc文档的URL
) -> None:
    """为FastAPI应用打上一个"补丁"，使得其文档页面不再依赖于CDN（内容分发网络），
    而是可以在没有网络的环境下也能访问。这是通过提供静态文件服务和自定义文档路由来实现的。"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    # StaticFiles用于服务静态文件
    from fastapi.staticfiles import StaticFiles
    # HTMLResponse用于生成HTML响应
    from starlette.responses import HTMLResponse
    # 从app对象中获取OpenAPI的URL
    openapi_url = app.openapi_url
    # OAuth2的重定向URL
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        '''
        从FastAPI应用中移除指定URL的路由。这是为了去除默认的文档路由，以便添加自定义的静态文档路由。
        '''
        index = None
        # 通过遍历app.routes列表来查找与给定url匹配的路由
        for i, r in enumerate(app.routes):
            # app.routes是一个包含了应用中所有路由的列表。
            # 一旦找到匹配的路由，就记录其索引到index变量，并跳出循环。
            if r.path.lower() == url.lower():
                index = i
                break
        # 如果index是一个整数（这意味着找到了匹配的路由），
        # 那么使用app.routes.pop(index)来移除该路由。
        if isinstance(index, int):
            app.routes.pop(index)

    # 使用app.mount方法来挂载静态文件目录，使得应用可以服务于那些静态文件。
    app.mount(
        static_url, # 静态文件服务的URL路径
        StaticFiles(directory=Path(static_dir).as_posix()), # 实例，指定了静态文件目录的位置。
        name="static-offline-docs", # 为这个静态文件挂载点指定一个名称
    )


    ## 设置Swagger UI文档页面

    # 如果docs_url（Swagger UI文档的URL）不为空，
    # 首先移除该URL以及OAuth2重定向URL的默认路由，然后设置自定义的路由。
    if docs_url is not None:
        # 移除指定URL的路由
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        # 使用@app.get装饰器创建一个新的路由，当访问docs_url时，执行custom_swagger_ui_html异步函数。
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            '''这个异步函数接收一个Request对象，并返回一个HTMLResponse对象，包含自定义的Swagger UI文档页面。'''
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            # 返回一个由get_swagger_ui_html函数生成的HTMLResponse，
            # 这个HTMLResponse包含了自定义的Swagger UI文档页面。
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        # 如果有OAuth2认证流程，还需要处理OAuth2重定向。
        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        # 返回由get_swagger_ui_oauth2_redirect_html生成的OAuth2重定向页面的HTMLResponse。
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()


    ## 设置ReDoc文档页面

    if redoc_url is not None:
        # 如果为非空，说明开发者希望提供ReDoc文档页面的访问。
        # 移除原有的ReDoc路由
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            # 首先计算根路径root和图标favicon的URL，确保它们指向正确的静态文件位置。
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            # 调用get_redoc_html函数，生成ReDoc页面的HTML响应。
            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False, # 避免从Google字体服务加载任何外部字体
                redoc_favicon_url=favicon,
            )



# 从model_config中获取模型信息

def list_embed_models() -> List[str]:
    '''
    get names of configured embedding models
    '''
    return list(MODEL_PATH["embed_model"])


def list_config_llm_models() -> Dict[str, Dict]:
    '''
    get configured llm models with different types.
    return {config_type: {model_name: config}, ...}
    提供一个统一的方式来获取不同类型（本地、在线、工作节点）的模型配置信息
    '''
    workers = FSCHAT_MODEL_WORKERS.copy()
    workers.pop("default", None)

    return {
        "local": MODEL_PATH["llm_model"].copy(), # 本地配置
        "online": ONLINE_LLM_MODEL.copy(), # 在线配置
        "worker": workers, # 工作节点配置
    }


def get_model_path(model_name: str, type: str = None) -> Optional[str]:
    """根据模型的名称（model_name）和类型（type）获取模型的路径
    model_name : 模型名称
    type : 模型类型
    """
    if type in MODEL_PATH:
        paths = MODEL_PATH[type]
    else:
        paths = {}
        # 将MODEL_PATH中的所有路径合并，不考虑它们的类型。
        for v in MODEL_PATH.values():
            paths.update(v)

    # 取model_name对应的路径
    if path_str := paths.get(model_name):  # 以 "chatglm-6b": "THUDM/chatglm-6b-new" 为例，以下都是支持的路径
        # 将其赋值给path_str，然后将path_str转换为Path对象。
        path = Path(path_str)
        # 检查path是否指向一个目录（is_dir()）
        if path.is_dir():  # 任意绝对路径
            return str(path)

        # 在一个名为MODEL_ROOT_PATH的根路径下查找模型
        root_path = Path(MODEL_ROOT_PATH)
        if root_path.is_dir():
            path = root_path / model_name
            if path.is_dir():  # use key, {MODEL_ROOT_PATH}/chatglm-6b
                return str(path)
            path = root_path / path_str
            if path.is_dir():  # use value, {MODEL_ROOT_PATH}/THUDM/chatglm-6b-new
                return str(path)
            path = root_path / path_str.split("/")[-1]
            if path.is_dir():  # use value split by "/", {MODEL_ROOT_PATH}/chatglm-6b-new
                return str(path)
        return path_str  # THUDM/chatglm06b


# 从server_config中获取服务信息

def get_model_worker_config(model_name: str = None) -> dict:
    '''
    加载模型工作进程（worker）的配置项，它根据传入的模型名称（model_name），从多个配置源中合并配置信息，并最终返回一个配置字典。
    优先级:FSCHAT_MODEL_WORKERS[model_name] > ONLINE_LLM_MODEL[model_name] > FSCHAT_MODEL_WORKERS["default"]
    '''
    from configs.model_config import ONLINE_LLM_MODEL, MODEL_PATH
    from configs.server_config import FSCHAT_MODEL_WORKERS
    from server import model_workers
    # 1. 初始化config变量
    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()
    # 2. 更新config变量
    # 首先尝试使用ONLINE_LLM_MODEL[model_name]的配置更新config
    config.update(ONLINE_LLM_MODEL.get(model_name, {}).copy())
    config.update(FSCHAT_MODEL_WORKERS.get(model_name, {}).copy())
    # 3. 处理在线模型配置
    # 判断model_name是否在ONLINE_LLM_MODEL字典中
    if model_name in ONLINE_LLM_MODEL:
        # 如果在，表明这是一个在线模型，需要将config中的online_api键设为True。
        config["online_api"] = True
        if provider := config.get("provider"):
            try:
                # 从model_workers中获取一个与provider同名的属性，并将这个属性赋值给config中的worker_class。
                config["worker_class"] = getattr(model_workers, provider)
            except Exception as e:
                msg = f"在线模型 '{model_name}' 的provider没有正确配置"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
    # 4. 处理本地模型配置
    if model_name in MODEL_PATH["llm_model"]:
        # 如果在，表明这是一个本地模型。接着，通过调用get_model_path函数获取模型路径，
        # 并更新config中的model_path。
        path = get_model_path(model_name)
        config["model_path"] = path
        if path and os.path.isdir(path):
            # 如果该路径存在且是一个目录，则在config中设置model_path_exists为True。
            config["model_path_exists"] = True
        config["device"] = llm_device(config.get("device"))
    # 返回config变量，即加载完成的模型配置信息。
    return config


def get_all_model_worker_configs() -> dict:
    """获取所有模型工作进程的配置信息"""
    # 存储每个模型名称对应的配置信息
    result = {}
    # 获取所有配置中的模型名称，并转换为一个集合model_names。
    model_names = set(FSCHAT_MODEL_WORKERS.keys())
    # 遍历model_names中的每个模型名称
    for name in model_names:
        if name != "default":
            # 取该模型的配置信息，并将结果存储在result字典中，键为模型名称。
            result[name] = get_model_worker_config(name)
    return result


def fschat_controller_address() -> str:
    '''获取服务控制器（FSChat Controller）的网络地址'''
    from configs.server_config import FSCHAT_CONTROLLER

    host = FSCHAT_CONTROLLER["host"]
    # 这样的替换通常用于将服务绑定到本地主机，便于本地访问。
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_CONTROLLER["port"]
    return f"http://{host}:{port}"


def fschat_model_worker_address(model_name: str = None) -> str:
    """获取特定模型工作进程的服务地址"""
    if model_name is None:
        model_name = LLM_MODELS[0] if LLM_MODELS else "Qwen1.5-7B-Chat"
        
    if model := get_model_worker_config(model_name):
        host = model["host"]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = model["port"]
        return f"http://{host}:{port}"
    return ""


def fschat_openai_api_address() -> str:
    '''获取OpenAI API的服务地址'''
    from configs.server_config import FSCHAT_OPENAI_API

    host = FSCHAT_OPENAI_API["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_OPENAI_API["port"]
    return f"http://{host}:{port}/v1"


def webui_address() -> str:
    '''webui address'''
    from configs.server_config import WEBUI_SERVER

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]
    return f"http://{host}:{port}"


def get_prompt_template(type: str, name: str) -> Optional[str]:
    '''
    从配置中加载特定类型和名称的提示模板（prompt template）。
    它主要用于不同场景下的对话或查询，通过加载预先定义的模板来标准化输入和输出格式。
    type: "llm_chat","agent_chat","knowledge_base_chat","search_engine_chat"的其中一种，如果有新功能，应该进行加入。
    '''

    # 首先，从configs包中导入prompt_config模块。
    from configs import prompt_config
    import importlib
    # 然后使用importlib.reload方法重新加载prompt_config模块。
    # 这个步骤确保了即使在运行时修改了配置内容，也能加载最新的配置，而不需要重启应用程序。
    importlib.reload(prompt_config)
    # 加载模板
    return prompt_config.PROMPT_TEMPLATES[type].get(name)


def set_httpx_config(
        timeout: float = HTTPX_DEFAULT_TIMEOUT, # 设置httpx的默认超时时间
        proxy: Union[str, Dict] = None, # 可选，用于设置代理
):
    ''' 配置httpx库的默认超时时间和代理设置，主要用于网络请求时的配置。

    设置httpx默认timeout。httpx默认timeout是5秒，在请求LLM回答时不够用。
    将本项目相关服务加入无代理列表，避免fastchat的服务器请求错误。(windows下无效)
    对于chatgpt等在线API，如要使用代理需要手动配置。搜索引擎的代理如何处置还需考虑。
    '''

    import httpx
    import os
    # 为连接（connect）、读取（read）和写入（write）操作设置相同的超时时间
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # 函数根据proxy参数的类型（字符串或字典）来设置代理
    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    # 设置好的代理会添加到环境变量中，这样就可以在进程范围内生效。
    for k, v in proxies.items():
        os.environ[k] = v

    # 为了确保某些特定地址不通过代理访问，函数将localhost和127.0.0.1以及本项目相关的服务地址添加到无代理列表（NO_PROXY环境变量）中。
    no_proxy = [x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()]
    no_proxy += [
        # do not use proxy for locahost
        "http://127.0.0.1",
        "http://localhost",
    ]
    # 不使用代理访问特定服务
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        # 对于每一个地址，首先将其拆分为部分，并只取前两部分（http(s)://和主机名），
        # 然后将它们重新组合成不包含端口和路径的基础地址（host）。
        host = ":".join(x.split(":")[:2])

        # 如果这个基础地址host不在之前定义的无代理列表no_proxy中，则将其添加到这个列表中。
        # 这个过程确保了特定的服务地址在进行网络请求时不会通过配置的代理。
        if host not in no_proxy:
            no_proxy.append(host)

    # 设置NO_PROXY环境变量，让Python的网络请求库（如requests、httpx和urllib）在向这些地址发起请求时不使用代理。
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    def _get_proxies():
        return proxies

    # 重写urllib.request.getproxies函数
    import urllib.request
    urllib.request.getproxies = _get_proxies


def detect_device() -> Literal["cuda", "mps", "cpu"]:
    '''自动检测系统中可用的设备，并返回最优的设备类型。'''
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"


def llm_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    '''确定用于运行LLM的具体设备'''
    device = device or LLM_DEVICE
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device


def embedding_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    '''确定运行嵌入（embedding）相关操作的最佳设备'''
    device = device or EMBEDDING_DEVICE
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device


def run_in_thread_pool(
        func: Callable, # func是需要在线程池中运行的函数
        params: List[Dict] = [], # 一个字典列表，每个字典包含一组传递给func的关键字参数。
) -> Generator:
    '''
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    '''
    tasks = []
    # 创建一个线程池环境。这个线程池会自动管理线程的创建、执行和销毁。
    with ThreadPoolExecutor() as pool:
        # 遍历params列表中的每个参数字典
        for kwargs in params:
            # 使用pool.submit(func, **kwargs)提交任务到线程池
            thread = pool.submit(func, **kwargs)
            # 每个提交的任务（thread）都被添加到任务列表tasks中
            tasks.append(thread)

        # 使用as_completed(tasks)遍历所有已完成的任务
        # as_completed是一个函数，它返回一个迭代器，该迭代器会在每个任务完成时产生任务对象。
        for obj in as_completed(tasks):
            # 对于每个完成的任务，使用obj.result()获取任务的结果，并通过yield语句将结果逐一返回。
            yield obj.result()


def get_httpx_client(
        use_async: bool = False, # 用于指定返回的httpx客户端是否为异步客户端。默认值为False，即默认返回同步客户端。
        proxies: Union[str, Dict] = None, #  代理设置，可以是字符串或字典。用于配置访问网络资源时使用的代理。
        timeout: float = HTTPX_DEFAULT_TIMEOUT, # 请求超时时间
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    '''
    用来获取httpx客户端实例的一个函数，支持同步和异步客户端，并且可以处理代理设置和超时时间。

    返回值： 根据use_async的值，返回httpx.Client（同步客户端）或httpx.AsyncClient（异步客户端）。
    '''

    # 包括了不使用代理访问localhost和127.0.0.1的设置。
    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # 获取不应使用代理访问的特定地址，并更新default_proxies字典，以排除这些地址使用代理。
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        default_proxies.update({host: None})

    # 从环境变量中获取http_proxy、https_proxy和all_proxy的设置，并更新default_proxies字典。
    default_proxies.update({
        "http://": (os.environ.get("http_proxy")
                    if os.environ.get("http_proxy") and len(os.environ.get("http_proxy").strip())
                    else None),
        "https://": (os.environ.get("https_proxy")
                     if os.environ.get("https_proxy") and len(os.environ.get("https_proxy").strip())
                     else None),
        "all://": (os.environ.get("all_proxy")
                   if os.environ.get("all_proxy") and len(os.environ.get("all_proxy").strip())
                   else None),
    })

    # 同时，它处理了no_proxy环境变量，以排除某些主机使用代理。
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # Origin code
            default_proxies.update({'all://' + host: None})  # PR 1838 fix, if not add 'all://', httpx will raise error

    # 合并用户提供的代理
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # 构造httpx客户端实例，代码更新了传递给httpx客户端的参数，包括超时时间和代理设置，
    # 并根据use_async参数的值构造并返回相应的客户端实例。
    kwargs.update(timeout=timeout, proxies=default_proxies)

    if log_verbose:
        logger.info(f'{get_httpx_client.__class__.__name__}:kwargs: {kwargs}')

    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)


def get_server_configs() -> Dict:
    '''
    多个配置文件中收集配置项，并将它们整合为一个字典返回，这个字典中包含了多种配置，可以供前端使用。
    '''
    try:
        from configs.kb_config import (
            DEFAULT_KNOWLEDGE_BASE,
            DEFAULT_SEARCH_ENGINE,
            DEFAULT_VS_TYPE,
            CHUNK_SIZE,
            OVERLAP_SIZE,
            SCORE_THRESHOLD,
            VECTOR_SEARCH_TOP_K,
            SEARCH_ENGINE_TOP_K,
            ZH_TITLE_ENHANCE,
            text_splitter_dict,
            TEXT_SPLITTER_NAME,
        )
        # 安全导入RAG-fusion配置
        try:
            from configs.kb_config import (ENABLE_RAG_FUSION, RAG_FUSION_CONFIG)
        except ImportError:
            ENABLE_RAG_FUSION = False
            RAG_FUSION_CONFIG = {}
    except ImportError as e:
        logger.warning(f"导入kb_config失败: {e}")
        # 设置默认值
        DEFAULT_KNOWLEDGE_BASE = "default"
        DEFAULT_SEARCH_ENGINE = "duckduckgo"
        DEFAULT_VS_TYPE = "faiss"
        CHUNK_SIZE = 250
        OVERLAP_SIZE = 50
        SCORE_THRESHOLD = 1.0
        VECTOR_SEARCH_TOP_K = 7
        SEARCH_ENGINE_TOP_K = 7
        ZH_TITLE_ENHANCE = False
        text_splitter_dict = {}
        TEXT_SPLITTER_NAME = "SpacyTextSplitter"
        ENABLE_RAG_FUSION = False
        RAG_FUSION_CONFIG = {}

    try:
        from configs.model_config import (
            LLM_MODELS,
            HISTORY_LEN,
            TEMPERATURE,
        )
        # 安全导入RAG-fusion模型配置
        try:
            from configs.model_config import RAG_FUSION_SUPPORTED_MODELS
        except ImportError:
            RAG_FUSION_SUPPORTED_MODELS = []
    except ImportError as e:
        logger.warning(f"导入model_config失败: {e}")
        # 设置默认值
        LLM_MODELS = ["chatglm3-6b"]
        HISTORY_LEN = 3
        TEMPERATURE = 0.7
        RAG_FUSION_SUPPORTED_MODELS = []

    try:
        from configs.prompt_config import PROMPT_TEMPLATES
    except ImportError as e:
        logger.warning(f"导入prompt_config失败: {e}")
        PROMPT_TEMPLATES = {}

    # 在_custom字典中，定义了一些特定的配置项，这些配置项可能是动态获取的
    _custom = {
        "controller_address": fschat_controller_address(),
        "openai_api_address": fschat_openai_api_address(),
        "api_address": api_address(),
        # 新增RAG-fusion状态
        "rag_fusion_stats": get_rag_fusion_stats(),
    }

    # 通过字典解包（**操作符）和列表推导式创建了一个新字典，
    # 该字典合并了函数内部定义的所有本地变量（排除了以_开头的变量，例如_custom自身）和_custom字典中的项。

    # locals().items()返回一个包含函数局部变量的字典。
    # 通过检查键（即变量名）的首字符不是_，可以排除掉私有或临时变量，只保留需要的配置项。
    return {**{k: v for k, v in locals().items() if k[0] != "_"}, **_custom}


def list_online_embed_models() -> List[str]:
    '''列出所有在线可用的嵌入模型的名称'''
    from server import model_workers
    # 用于存储支持嵌入的在线模型名称
    ret = []
    # 获取在线模型的配置项，并遍历这些项。
    for k, v in list_config_llm_models()["online"].items():
        # 对于每个模型，首先尝试获取其提供者，这是一个指示模型工作类名称的字符串。
        if provider := v.get("provider"):
            # 使用getattr函数尝试从model_workers模块中获取对应的工作类。
            worker_class = getattr(model_workers, provider, None)
            # 如果该类存在并且具有嵌入能力（通过调用can_embedding()方法检查），
            # 则将模型名称添加到返回列表中。
            if worker_class is not None and worker_class.can_embedding():
                ret.append(k)
    # 返回包含支持嵌入的在线模型名称的列表
    return ret


def load_local_embeddings(model: str = None, device: str = embedding_device()):
    '''
    从缓存中加载embeddings，可以避免多线程时竞争加载。
    '''
    from server.knowledge_base.kb_cache.base import embeddings_pool
    from configs import EMBEDDING_MODEL
    # 确定加载的模型
    model = model or EMBEDDING_MODEL
    # 加载并返回嵌入
    return embeddings_pool.load_embeddings(model=model, device=device)


def get_temp_dir(id: str = None) -> Tuple[str, str]:
    '''
    创建一个临时目录，返回（路径，文件夹名称）
    id : 可以用于指定临时目录的名称

    返回一个包含两个字符串的元组：目录的完整路径和目录的名称。
    '''
    from configs.basic_config import BASE_TEMP_DIR
    # tempfile模块提供了生成临时文件和目录的功能
    import tempfile

    if id is not None:  # 如果指定的临时目录已存在，直接返回
        path = os.path.join(BASE_TEMP_DIR, id)
        if os.path.isdir(path):
            return path, id

    # 创建一个新的临时目录，mkdtemp函数确保每次调用都会创建一个唯一的目录。
    path = tempfile.mkdtemp(dir=BASE_TEMP_DIR)
    # 返回新创建的临时目录的路径和名称
    return path, os.path.basename(path)


# ================= 启动时初始化RAG-fusion =================

def _initialize_rag_fusion():
    """在模块加载时初始化RAG-fusion"""
    if ENABLE_RAG_FUSION:
        try:
            # 验证配置
            if validate_rag_fusion_config():
                logger.info("RAG-fusion功能初始化成功")
            else:
                logger.warning("RAG-fusion配置验证失败，功能可能无法正常工作")
        except Exception as e:
            logger.error(f"RAG-fusion初始化失败: {e}")

# 在模块导入时执行初始化
if __name__ != "__main__":
    _initialize_rag_fusion()
