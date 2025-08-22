# 该文件封装了对api.py的请求，可以被不同的webui使用
# 通过ApiRequest和AsyncApiRequest支持同步/异步调用
# 增强支持RAG-fusion功能

from typing import *
from pathlib import Path
# 此处导入的配置为发起请求（如WEBUI）机器上的配置，主要用于为前端设置默认值。分布式部署时可以与服务器上的不同
from configs import (
    EMBEDDING_MODEL,
    DEFAULT_VS_TYPE,
    LLM_MODELS,
    TEMPERATURE,
    SCORE_THRESHOLD,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    VECTOR_SEARCH_TOP_K,
    SEARCH_ENGINE_TOP_K,
    HTTPX_DEFAULT_TIMEOUT,
    logger, log_verbose,
    TEXT_SPLITTER_NAME,
    # 添加混合检索相关配置导入
    DEFAULT_SEARCH_MODE, DEFAULT_DENSE_WEIGHT,
    DEFAULT_SPARSE_WEIGHT, DEFAULT_RRF_K,
)

# 添加RAG-fusion相关配置导入
try:
    from configs import (
        ENABLE_RAG_FUSION,
        RAG_FUSION_CONFIG,
        RAG_FUSION_QUERY_COUNT,
        RAG_FUSION_LLM_MODEL,
        RAG_FUSION_SUPPORTED_MODELS,
    )
    RAG_FUSION_AVAILABLE = ENABLE_RAG_FUSION
except ImportError:
    RAG_FUSION_AVAILABLE = False
    RAG_FUSION_CONFIG = {}
    RAG_FUSION_QUERY_COUNT = 3
    RAG_FUSION_LLM_MODEL = "Qwen1.5-7B-Chat"
    RAG_FUSION_SUPPORTED_MODELS = []

import httpx
import contextlib
import json
import os
from io import BytesIO
from server.utils import set_httpx_config, api_address, get_httpx_client

from pprint import pprint
from langchain_core._api import deprecated

set_httpx_config()


class ApiRequest:
    '''
    api.py调用的封装（同步模式）,简化api调用方式
    增强支持分片策略选择功能和RAG-fusion功能
    '''

    def __init__(
            self,
            base_url: str = api_address(),
            timeout: float = HTTPX_DEFAULT_TIMEOUT,
    ):
        self.base_url = base_url
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

    def get(
            self,
            url: str,
            params: Union[Dict, List[Tuple], bytes] = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("GET", url, params=params, **kwargs)
                else:
                    return self.client.get(url, params=params, **kwargs)
            except Exception as e:
                msg = f"error when get {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    def post(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                # print(kwargs)
                if stream:
                    return self.client.stream("POST", url, data=data, json=json, **kwargs)
                else:
                    return self.client.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when post {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    def delete(
            self,
            url: str,
            data: Dict = None,
            json: Dict = None,
            retry: int = 3,
            stream: bool = False,
            **kwargs: Any
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("DELETE", url, data=data, json=json, **kwargs)
                else:
                    return self.client.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when delete {url}: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                retry -= 1

    def _httpx_stream2generator(
            self,
            response: contextlib._GeneratorContextManager,
            as_json: bool = False,
    ):
        '''
        将httpx.stream返回的GeneratorContextManager转化为普通生成器
        '''

        async def ret_async(response, as_json):
            try:
                async with response as r:
                    async for chunk in r.aiter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk)
                                yield data
                            except Exception as e:
                                msg = f"API returned JSON error: '{chunk}'. Error:{e}."
                                logger.error(f'{e.__class__.__name__}: {msg}',
                                             exc_info=e if log_verbose else None)
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"Unable to connect to API server, please ensure 'api.py' is running properly. ({e})"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API communication timeout, please ensure FastChat and API service are started (see Wiki '5. Start API Service or Web UI'). ({e})"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API communication error: {e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                yield {"code": 500, "msg": msg}

        def ret_sync(response, as_json):
            try:
                with response as r:
                    for chunk in r.iter_text(None):
                        if not chunk:  # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                if chunk.startswith("data: "):
                                    data = json.loads(chunk[6:-2])
                                elif chunk.startswith(":"):  # skip sse comment line
                                    continue
                                else:
                                    data = json.loads(chunk)
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： '{chunk}'。错误信息是：{e}。"
                                logger.error(f'{e.__class__.__name__}: {msg}',
                                             exc_info=e if log_verbose else None)
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 'api.py' 已正常启动。({e})"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                logger.error(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                yield {"code": 500, "msg": msg}

        if self._use_async:
            return ret_async(response, as_json)
        else:
            return ret_sync(response, as_json)

    def _get_response_value(
            self,
            response: httpx.Response,
            as_json: bool = False,
            value_func: Callable = None,
    ):
        '''
        转换同步或异步请求返回的响应
        `as_json`: 返回json
        `value_func`: 用户可以自定义返回值，该函数接受response或json
        '''

        def to_json(r):
            try:
                return r.json()
            except Exception as e:
                msg = "API未能返回正确的JSON。" + str(e)
                if log_verbose:
                    logger.error(f'{e.__class__.__name__}: {msg}',
                                 exc_info=e if log_verbose else None)
                return {"code": 500, "msg": msg, "data": None}

        if value_func is None:
            value_func = (lambda r: r)

        async def ret_async(response):
            if as_json:
                return value_func(to_json(await response))
            else:
                return value_func(await response)

        if self._use_async:
            return ret_async(response)
        else:
            if as_json:
                return value_func(to_json(response))
            else:
                return value_func(response)

    # ================= 新增：分片策略相关API =================
    
    def get_available_text_splitters(self) -> Dict[str, str]:
        """
        获取所有可用的文本分片器及其描述
        """
        try:
            response = self.post("/knowledge_base/get_available_splitters")
            return self._get_response_value(response, as_json=True, 
                                          value_func=lambda r: r.get("data", {}))
        except Exception as e:
            logger.warning(f"获取分片器列表失败，使用默认列表: {e}")
            # 返回默认的分片器列表
            return {
                "RecursiveCharacterTextSplitter": "RecursiveCharacterTextSplitter(Generic, recommended for English)",
                "EnglishSentenceSplitter": "英文句子分片器（精确句子边界）",
                "EnglishParagraphSplitter": "英文段落分片器（保持段落结构）",
                "SemanticChunkSplitter": "语义分块器（智能语义分组）",
                "SlidingWindowSplitter": "滑动窗口分片器（最大化覆盖率）",
            }
    
    def validate_text_splitter_config(self, splitter_name: str, config: Dict) -> Dict:
        """
        验证分片器配置参数
        """
        try:
            data = {
                "splitter_name": splitter_name,
                "config": config
            }
            response = self.post("/knowledge_base/validate_splitter_config", json=data)
            return self._get_response_value(response, as_json=True)
        except Exception as e:
            logger.warning(f"验证分片器配置失败: {e}")
            return {"code": 200, "msg": "配置验证跳过", "data": config}

    # ================= 新增：RAG-fusion相关API =================
    
    def get_system_info(self) -> Dict:
        """
        获取系统级别的RAG-fusion信息
        """
        try:
            response = self.post("/knowledge_base/get_system_info")
            return self._get_response_value(response, as_json=True)
        except Exception as e:
            logger.warning(f"获取系统信息失败: {e}")
            return {
                "code": 200,
                "msg": "系统信息获取失败",
                "data": {
                    "rag_fusion_available": RAG_FUSION_AVAILABLE,
                    "supported_models": RAG_FUSION_SUPPORTED_MODELS,
                    "default_model": RAG_FUSION_LLM_MODEL
                }
            }
    
    def validate_rag_fusion_setup(self) -> Dict:
        """
        验证RAG-fusion配置是否正确
        """
        try:
            response = self.post("/knowledge_base/validate_rag_fusion_setup")
            return self._get_response_value(response, as_json=True)
        except Exception as e:
            logger.warning(f"验证RAG-fusion配置失败: {e}")
            return {
                "code": 200,
                "msg": "配置验证失败",
                "data": {"valid": False, "error": str(e)}
            }
    
    def get_rag_fusion_models(self) -> List[str]:
        """
        获取可用的RAG-fusion模型列表
        """
        try:
            response = self.post("/knowledge_base/get_rag_fusion_models")
            return self._get_response_value(response, as_json=True, 
                                          value_func=lambda r: r.get("data", {}).get("models", []))
        except Exception as e:
            logger.warning(f"获取RAG-fusion模型列表失败: {e}")
            return RAG_FUSION_SUPPORTED_MODELS
    
    def get_kb_capabilities(self, knowledge_base_name: str) -> Dict:
        """
        获取知识库支持的功能能力
        """
        try:
            data = {"knowledge_base_name": knowledge_base_name}
            response = self.post("/knowledge_base/get_kb_capabilities", json=data)
            return self._get_response_value(response, as_json=True)
        except Exception as e:
            logger.warning(f"获取知识库能力信息失败: {e}")
            return {
                "code": 200,
                "msg": "能力信息获取失败",
                "data": {
                    "vector_search": {"supported": True},
                    "rag_fusion": {"supported": RAG_FUSION_AVAILABLE}
                }
            }
    
    def get_search_modes(self, knowledge_base_name: str) -> Dict:
        """
        获取知识库支持的搜索模式
        """
        try:
            data = {"knowledge_base_name": knowledge_base_name}
            response = self.post("/knowledge_base/get_search_modes", json=data)
            return self._get_response_value(response, as_json=True)
        except Exception as e:
            logger.warning(f"获取搜索模式失败: {e}")
            return {
                "code": 200,
                "msg": "搜索模式获取失败",
                "data": {
                    "supported_modes": {"vector": {"supported": True}},
                    "total_supported": 1
                }
            }
    
    def rag_fusion_search(
            self,
            query: str,
            knowledge_base_name: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            query_count: int = RAG_FUSION_QUERY_COUNT,
            llm_model: str = RAG_FUSION_LLM_MODEL,
            timeout: int = 30,
            dense_weight: float = DEFAULT_DENSE_WEIGHT,
            sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
            rrf_k: int = DEFAULT_RRF_K,
            enable_cache: bool = True,
            enable_rerank: bool = False,
            fusion_strategy: str = "hybrid",
    ) -> Dict:
        """
        专门的RAG-fusion检索API
        """
        try:
            data = {
                "query": query,
                "knowledge_base_name": knowledge_base_name,
                "top_k": top_k,
                "score_threshold": score_threshold,
                "query_count": query_count,
                "llm_model": llm_model,
                "timeout": timeout,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "rrf_k": rrf_k,
                "enable_cache": enable_cache,
                "enable_rerank": enable_rerank,
                "fusion_strategy": fusion_strategy,
            }
            
            response = self.post("/knowledge_base/rag_fusion_search", json=data)
            return self._get_response_value(response, as_json=True)
            
        except Exception as e:
            logger.error(f"RAG-fusion检索失败: {e}")
            return {
                "code": 500,
                "msg": f"RAG-fusion检索失败: {str(e)}",
                "data": {"results": [], "error": str(e)}
            }
    
    def compare_search_modes(
            self,
            query: str,
            knowledge_base_name: str,
            top_k: int = 5,
            modes: List[str] = None
    ) -> Dict:
        """
        比较不同搜索模式的结果
        """
        try:
            if modes is None:
                modes = ["vector", "hybrid"]
                if RAG_FUSION_AVAILABLE:
                    modes.append("rag_fusion")
            
            data = {
                "query": query,
                "knowledge_base_name": knowledge_base_name,
                "top_k": top_k,
                "modes": modes
            }
            
            response = self.post("/knowledge_base/compare_search_modes", json=data)
            return self._get_response_value(response, as_json=True)
            
        except Exception as e:
            logger.error(f"搜索模式比较失败: {e}")
            return {
                "code": 500,
                "msg": f"搜索模式比较失败: {str(e)}",
                "data": {"comparison_results": {}, "error": str(e)}
            }

    # 服务器信息
    def get_server_configs(self, **kwargs) -> Dict:
        response = self.post("/server/configs", **kwargs)
        return self._get_response_value(response, as_json=True)

    def list_search_engines(self, **kwargs) -> List:
        response = self.post("/server/list_search_engines", **kwargs)
        return self._get_response_value(response, as_json=True, value_func=lambda r: r["data"])

    def get_prompt_template(
            self,
            type: str = "llm_chat",
            name: str = "default",
            **kwargs,
    ) -> str:
        data = {
            "type": type,
            "name": name,
        }
        response = self.post("/server/get_prompt_template", json=data, **kwargs)
        return self._get_response_value(response, value_func=lambda r: r.text)

    # 对话相关操作
    def chat_chat(
            self,
            query: str,
            conversation_id: str = None,
            history_len: int = -1,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            **kwargs,
    ):
        '''
        对应api.py/chat/chat接口
        '''
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

        # print(f"received input message:")
        # pprint(data)

        response = self.post("/chat/chat", json=data, stream=True, **kwargs)
        return self._httpx_stream2generator(response, as_json=True)

    @deprecated(
        since="0.3.0",
        message="自定义Agent问答将于 Langchain-Chatchat 0.3.x重写, 0.2.x中相关功能将废弃",
        removal="0.3.0")
    def agent_chat(
            self,
            query: str,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        '''
        对应api.py/chat/agent_chat 接口
        '''
        data = {
            "query": query,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post("/chat/agent_chat", json=data, stream=True)
        return self._httpx_stream2generator(response, as_json=True)

    def knowledge_base_chat(
            self,
            query: str,
            knowledge_base_name: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            # 添加搜索参数（支持RAG-fusion）
            search_mode: str = DEFAULT_SEARCH_MODE,
            dense_weight: float = DEFAULT_DENSE_WEIGHT,
            sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
            rrf_k: int = DEFAULT_RRF_K,
            # RAG-fusion特有参数
            rag_fusion_query_count: int = None,
            rag_fusion_model: str = None,
            rag_fusion_timeout: int = None,
    ):
        '''
        对应api.py/chat/knowledge_base_chat接口
        增强支持RAG-fusion参数
        '''
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
            # 搜索参数
            "search_mode": search_mode,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "rrf_k": rrf_k,
        }
        
        # 添加RAG-fusion特有参数
        if search_mode == "rag_fusion":
            if rag_fusion_query_count is not None:
                data["rag_fusion_query_count"] = rag_fusion_query_count
            if rag_fusion_model is not None:
                data["rag_fusion_model"] = rag_fusion_model
            if rag_fusion_timeout is not None:
                data["rag_fusion_timeout"] = rag_fusion_timeout

        # print(f"received input message:")
        # pprint(data)

        response = self.post(
            "/chat/knowledge_base_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)

    def upload_temp_docs(
            self,
            files: List[Union[str, Path, bytes]],
            knowledge_id: str = None,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            text_splitter_name: str = None,  # 新增：分片器名称
    ):
        '''
        对应api.py/knowledge_base/upload_tmep_docs接口
        增强支持分片器选择
        '''

        def convert_file(file, filename=None):
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data = {
            "knowledge_id": knowledge_id,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }
        
        # 添加分片器名称（如果指定）
        if text_splitter_name:
            data["text_splitter_name"] = text_splitter_name

        response = self.post(
            "/knowledge_base/upload_temp_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        return self._get_response_value(response, as_json=True)

    def file_chat(
            self,
            query: str,
            knowledge_id: str,
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            # 添加搜索参数（支持RAG-fusion）
            search_mode: str = DEFAULT_SEARCH_MODE,
            dense_weight: float = DEFAULT_DENSE_WEIGHT,
            sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
            rrf_k: int = DEFAULT_RRF_K,
            # RAG-fusion特有参数
            rag_fusion_query_count: int = None,
            rag_fusion_model: str = None,
    ):
        '''
        对应api.py/chat/file_chat接口
        增强支持RAG-fusion参数
        '''
        data = {
            "query": query,
            "knowledge_id": knowledge_id,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
            # 搜索参数
            "search_mode": search_mode,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "rrf_k": rrf_k,
        }
        
        # 添加RAG-fusion特有参数
        if search_mode == "rag_fusion":
            if rag_fusion_query_count is not None:
                data["rag_fusion_query_count"] = rag_fusion_query_count
            if rag_fusion_model is not None:
                data["rag_fusion_model"] = rag_fusion_model

        response = self.post(
            "/chat/file_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)

    @deprecated(
        since="0.3.0",
        message="搜索引擎问答将于 Langchain-Chatchat 0.3.x重写, 0.2.x中相关功能将废弃",
        removal="0.3.0"
    )
    def search_engine_chat(
            self,
            query: str,
            search_engine_name: str,
            top_k: int = SEARCH_ENGINE_TOP_K,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = LLM_MODELS[0],
            temperature: float = TEMPERATURE,
            max_tokens: int = None,
            prompt_name: str = "default",
            split_result: bool = False,
    ):
        '''
        对应api.py/chat/search_engine_chat接口
        '''
        data = {
            "query": query,
            "search_engine_name": search_engine_name,
            "top_k": top_k,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
            "split_result": split_result,
        }

        # print(f"received input message:")
        # pprint(data)

        response = self.post(
            "/chat/search_engine_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)

    # 知识库相关操作

    def list_knowledge_bases(
            self,
    ):
        '''
        对应api.py/knowledge_base/list_knowledge_bases接口
        '''
        response = self.get("/knowledge_base/list_knowledge_bases")
        return self._get_response_value(response,
                                        as_json=True,
                                        value_func=lambda r: r.get("data", []))

    def create_knowledge_base(
            self,
            knowledge_base_name: str,
            vector_store_type: str = DEFAULT_VS_TYPE,
            embed_model: str = EMBEDDING_MODEL,
            enable_rag_fusion: bool = None,  # 新增：RAG-fusion选项
    ):
        '''
        对应api.py/knowledge_base/create_knowledge_base接口
        增强支持RAG-fusion配置
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "vector_store_type": vector_store_type,
            "embed_model": embed_model,
        }
        
        # 添加RAG-fusion配置
        if enable_rag_fusion is not None:
            data["enable_rag_fusion"] = enable_rag_fusion

        response = self.post(
            "/knowledge_base/create_knowledge_base",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def delete_knowledge_base(
            self,
            knowledge_base_name: str,
    ):
        '''
        对应api.py/knowledge_base/delete_knowledge_base接口
        '''
        response = self.post(
            "/knowledge_base/delete_knowledge_base",
            json=f"{knowledge_base_name}",
        )
        return self._get_response_value(response, as_json=True)

    def list_kb_docs(
            self,
            knowledge_base_name: str,
    ):
        '''
        对应api.py/knowledge_base/list_files接口
        '''
        response = self.get(
            "/knowledge_base/list_files",
            params={"knowledge_base_name": knowledge_base_name}
        )
        return self._get_response_value(response,
                                        as_json=True,
                                        value_func=lambda r: r.get("data", []))

    def search_kb_docs(
            self,
            knowledge_base_name: str,
            query: str = "",
            top_k: int = VECTOR_SEARCH_TOP_K,
            score_threshold: float = SCORE_THRESHOLD,
            file_name: str = "",
            metadata: dict = {},
            # 搜索模式参数（支持RAG-fusion）
            search_mode: str = DEFAULT_SEARCH_MODE,
            dense_weight: float = DEFAULT_DENSE_WEIGHT,
            sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
            rrf_k: int = DEFAULT_RRF_K,
            # RAG-fusion特有参数
            rag_fusion_query_count: int = None,
            rag_fusion_model: str = None,
            rag_fusion_timeout: int = None,
            enable_rerank: bool = False,
            return_metadata: bool = True,
    ) -> List:
        '''
        对应api.py/knowledge_base/search_docs接口
        增强支持RAG-fusion功能
        '''
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "file_name": file_name,
            "metadata": metadata,
            # 搜索参数
            "search_mode": search_mode,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "rrf_k": rrf_k,
            "enable_rerank": enable_rerank,
            "return_metadata": return_metadata,
        }
        
        # 添加RAG-fusion特有参数
        if search_mode == "rag_fusion":
            if rag_fusion_query_count is not None:
                data["rag_fusion_query_count"] = rag_fusion_query_count
            if rag_fusion_model is not None:
                data["rag_fusion_model"] = rag_fusion_model
            if rag_fusion_timeout is not None:
                data["rag_fusion_timeout"] = rag_fusion_timeout

        response = self.post(
            "/knowledge_base/search_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_docs_by_id(
            self,
            knowledge_base_name: str,
            docs: Dict[str, Dict],
    ) -> bool:
        '''
        对应api.py/knowledge_base/update_docs_by_id接口
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "docs": docs,
        }
        response = self.post(
            "/knowledge_base/update_docs_by_id",
            json=data
        )
        return self._get_response_value(response)

    def upload_kb_docs(
            self,
            files: List[Union[str, Path, bytes]],
            knowledge_base_name: str,
            override: bool = False,
            to_vector_store: bool = True,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            docs: Dict = {},
            not_refresh_vs_cache: bool = False,
            text_splitter_name: str = None,  # 新增：分片器名称
            **kwargs  # 新增：额外的分片器参数
    ):
        '''
        对应api.py/knowledge_base/upload_docs接口
        增强支持分片器选择
        '''

        def convert_file(file, filename=None):
            if isinstance(file, bytes):  # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"):  # a file io like object
                filename = filename or file.name
            else:  # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data = {
            "knowledge_base_name": knowledge_base_name,
            "override": override,
            "to_vector_store": to_vector_store,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }
        
        # 添加分片器相关参数
        if text_splitter_name and text_splitter_name != TEXT_SPLITTER_NAME:
            data["text_splitter_name"] = text_splitter_name
            logger.info(f"使用指定分片器: {text_splitter_name}")
        
        # 添加额外的分片器参数
        for key, value in kwargs.items():
            if key.startswith(('splitter_', 'text_splitter_')):
                data[key] = value

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)
        response = self.post(
            "/knowledge_base/upload_docs",
            data=data,
            files=[("files", (filename, file)) for filename, file in files],
        )
        return self._get_response_value(response, as_json=True)

    def delete_kb_docs(
            self,
            knowledge_base_name: str,
            file_names: List[str],
            delete_content: bool = False,
            not_refresh_vs_cache: bool = False,
    ):
        '''
        对应api.py/knowledge_base/delete_docs接口
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "delete_content": delete_content,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        response = self.post(
            "/knowledge_base/delete_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_kb_info(self, knowledge_base_name, kb_info):
        '''
        对应api.py/knowledge_base/update_info接口
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "kb_info": kb_info,
        }

        response = self.post(
            "/knowledge_base/update_info",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_kb_docs(
            self,
            knowledge_base_name: str,
            file_names: List[str],
            override_custom_docs: bool = False,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            docs: Dict = {},
            not_refresh_vs_cache: bool = False,
            text_splitter_name: str = None,  # 新增：分片器名称
            **kwargs  # 新增：额外的分片器参数
    ):
        '''
        对应api.py/knowledge_base/update_docs接口
        增强支持分片器选择
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "override_custom_docs": override_custom_docs,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }
        
        # 添加分片器相关参数
        if text_splitter_name and text_splitter_name != TEXT_SPLITTER_NAME:
            data["text_splitter_name"] = text_splitter_name
            logger.info(f"更新文档使用分片器: {text_splitter_name}")
        
        # 添加额外的分片器参数
        for key, value in kwargs.items():
            if key.startswith(('splitter_', 'text_splitter_')):
                data[key] = value

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)

        response = self.post(
            "/knowledge_base/update_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def recreate_vector_store(
            self,
            knowledge_base_name: str,
            allow_empty_kb: bool = True,
            vs_type: str = DEFAULT_VS_TYPE,
            embed_model: str = EMBEDDING_MODEL,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=OVERLAP_SIZE,
            zh_title_enhance=ZH_TITLE_ENHANCE,
            text_splitter_name: str = None,  # 新增：分片器名称
            **kwargs  # 新增：额外的分片器参数
    ):
        '''
        对应api.py/knowledge_base/recreate_vector_store接口
        增强支持分片器选择
        '''
        data = {
            "knowledge_base_name": knowledge_base_name,
            "allow_empty_kb": allow_empty_kb,
            "vs_type": vs_type,
            "embed_model": embed_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
        }
        
        # 添加分片器相关参数
        if text_splitter_name and text_splitter_name != TEXT_SPLITTER_NAME:
            data["text_splitter_name"] = text_splitter_name
            logger.info(f"重建向量库使用分片器: {text_splitter_name}")
        
        # 添加额外的分片器参数
        for key, value in kwargs.items():
            if key.startswith(('splitter_', 'text_splitter_')):
                data[key] = value

        response = self.post(
            "/knowledge_base/recreate_vector_store",
            json=data,
            stream=True,
            timeout=None,
        )
        return self._httpx_stream2generator(response, as_json=True)

    # LLM模型相关操作
    def list_running_models(
            self,
            controller_address: str = None,
    ):
        '''
        获取Fastchat中正运行的模型列表
        '''
        data = {
            "controller_address": controller_address,
        }

        if log_verbose:
            logger.info(f'{self.__class__.__name__}:data: {data}')

        response = self.post(
            "/llm_model/list_running_models",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", []))

    def get_default_llm_model(self, local_first: bool = True) -> Tuple[str, bool]:
        '''
        从服务器上获取当前运行的LLM模型。
        当 local_first=True 时，优先返回运行中的本地模型，否则优先按LLM_MODELS配置顺序返回。
        返回类型为（model_name, is_local_model）
        '''

        def ret_sync():
            running_models = self.list_running_models()
            if not running_models:
                return "", False

            model = ""
            for m in LLM_MODELS:
                if m not in running_models:
                    continue
                is_local = not running_models[m].get("online_api")
                if local_first and not is_local:
                    continue
                else:
                    model = m
                    break

            if not model:  # LLM_MODELS中配置的模型都不在running_models里
                model = list(running_models)[0]
            is_local = not running_models[model].get("online_api")
            return model, is_local

        async def ret_async():
            running_models = await self.list_running_models()
            if not running_models:
                return "", False

            model = ""
            for m in LLM_MODELS:
                if m not in running_models:
                    continue
                is_local = not running_models[m].get("online_api")
                if local_first and not is_local:
                    continue
                else:
                    model = m
                    break

            if not model:  # LLM_MODELS中配置的模型都不在running_models里
                model = list(running_models)[0]
            is_local = not running_models[model].get("online_api")
            return model, is_local

        if self._use_async:
            return ret_async()
        else:
            return ret_sync()

    def list_config_models(
            self,
            types: List[str] = ["local", "online"],
    ) -> Dict[str, Dict]:
        '''
        获取服务器configs中配置的模型列表，返回形式为{"type": {model_name: config}, ...}。
        '''
        data = {
            "types": types,
        }
        response = self.post(
            "/llm_model/list_config_models",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", {}))

    def get_model_config(
            self,
            model_name: str = None,
    ) -> Dict:
        '''
        获取服务器上模型配置
        '''
        data = {
            "model_name": model_name,
        }
        response = self.post(
            "/llm_model/get_model_config",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", {}))

    def list_search_engines(self) -> List[str]:
        '''
        获取服务器支持的搜索引擎
        '''
        response = self.post(
            "/server/list_search_engines",
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r: r.get("data", {}))

    def stop_llm_model(
            self,
            model_name: str,
            controller_address: str = None,
    ):
        '''
        停止某个LLM模型。
        注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
        '''
        data = {
            "model_name": model_name,
            "controller_address": controller_address,
        }

        response = self.post(
            "/llm_model/stop",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def change_llm_model(
            self,
            model_name: str,
            new_model_name: str,
            controller_address: str = None,
    ):
        '''
        向fastchat controller请求切换LLM模型。
        '''
        if not model_name or not new_model_name:
            return {
                "code": 500,
                "msg": f"未指定模型名称"
            }

        def ret_sync():
            running_models = self.list_running_models()
            if new_model_name == model_name or new_model_name in running_models:
                return {
                    "code": 200,
                    "msg": "无需切换"
                }

            if model_name not in running_models:
                return {
                    "code": 500,
                    "msg": f"指定的模型'{model_name}'没有运行。当前运行模型：{running_models}"
                }

            config_models = self.list_config_models()
            if new_model_name not in config_models.get("local", {}):
                return {
                    "code": 500,
                    "msg": f"要切换的模型'{new_model_name}'在configs中没有配置。"
                }

            data = {
                "model_name": model_name,
                "new_model_name": new_model_name,
                "controller_address": controller_address,
            }

            response = self.post(
                "/llm_model/change",
                json=data,
            )
            return self._get_response_value(response, as_json=True)

        async def ret_async():
            running_models = await self.list_running_models()
            if new_model_name == model_name or new_model_name in running_models:
                return {
                    "code": 200,
                    "msg": "无需切换"
                }

            if model_name not in running_models:
                return {
                    "code": 500,
                    "msg": f"指定的模型'{model_name}'没有运行。当前运行模型：{running_models}"
                }

            config_models = await self.list_config_models()
            if new_model_name not in config_models.get("local", {}):
                return {
                    "code": 500,
                    "msg": f"要切换的模型'{new_model_name}'在configs中没有配置。"
                }

            data = {
                "model_name": model_name,
                "new_model_name": new_model_name,
                "controller_address": controller_address,
            }

            response = self.post(
                "/llm_model/change",
                json=data,
            )
            return self._get_response_value(response, as_json=True)

        if self._use_async:
            return ret_async()
        else:
            return ret_sync()

    def embed_texts(
            self,
            texts: List[str],
            embed_model: str = EMBEDDING_MODEL,
            to_query: bool = False,
    ) -> List[List[float]]:
        '''
        对文本进行向量化，可选模型包括本地 embed_models 和支持 embeddings 的在线模型
        '''
        data = {
            "texts": texts,
            "embed_model": embed_model,
            "to_query": to_query,
        }
        resp = self.post(
            "/other/embed_texts",
            json=data,
        )
        return self._get_response_value(resp, as_json=True, value_func=lambda r: r.get("data"))

    def chat_feedback(
            self,
            message_id: str,
            score: int,
            reason: str = "",
    ) -> int:
        '''
        反馈对话评价
        '''
        data = {
            "message_id": message_id,
            "score": score,
            "reason": reason,
        }
        resp = self.post("/chat/feedback", json=data)
        return self._get_response_value(resp)


class AsyncApiRequest(ApiRequest):
    def __init__(self, base_url: str = api_address(), timeout: float = HTTPX_DEFAULT_TIMEOUT):
        super().__init__(base_url, timeout)
        self._use_async = True


# ================= 增强：分片策略相关工具函数 =================

def get_splitter_display_name(splitter_name: str) -> str:
    """
    获取分片器的友好显示名称
    """
    display_names = {
        "RecursiveCharacterTextSplitter": "Recursive Character Splitter",
        "ChineseRecursiveTextSplitter": "Chinese Recursive Splitter",
        "ChineseTextSplitter": "Chinese Text Splitter",
        "SpacyTextSplitter": "Spacy Text Splitter",
        "MarkdownHeaderTextSplitter": "Markdown Header Splitter",
        "EnglishSentenceSplitter": "English Sentence Splitter",
        "EnglishParagraphSplitter": "English Paragraph Splitter",
        "SemanticChunkSplitter": "Semantic Chunk Splitter",
        "SlidingWindowSplitter": "Sliding Window Splitter",
    }
    return display_names.get(splitter_name, splitter_name)


def get_splitter_description(splitter_name: str) -> str:
    """
    获取分片器的详细描述
    """
    descriptions = {
        "RecursiveCharacterTextSplitter": "General recursive character splitter, suitable for most English documents, splits by character count recursively",
        "ChineseRecursiveTextSplitter": "Chinese-optimized recursive splitter, suitable for Chinese document processing",
        "ChineseTextSplitter": "Specialized splitter for Chinese text",
        "SpacyTextSplitter": "Intelligent splitter based on Spacy library, considers linguistic features",
        "MarkdownHeaderTextSplitter": "Specialized splitter for Markdown documents, maintains heading hierarchy",
        "EnglishSentenceSplitter": "Precise splitting based on English sentence boundaries, maintains sentence integrity and grammatical structure",
        "EnglishParagraphSplitter": "Splitting based on English paragraph structure, maintains paragraph logical integrity",
        "SemanticChunkSplitter": "Intelligent splitting based on semantic similarity, ensures semantically related content in the same chunk",
        "SlidingWindowSplitter": "Sliding window splitting strategy, maximizes retrieval coverage through overlapping windows",
    }
    return descriptions.get(splitter_name, "Unknown splitter")


def get_splitter_recommended_params(splitter_name: str) -> Dict[str, Any]:
    """
    获取分片器的推荐参数
    """
    recommendations = {
        "RecursiveCharacterTextSplitter": {
            "chunk_size": 250,
            "chunk_overlap": 50,
            "suitable_for": ["English documents", "Technical documents", "General text"]
        },
        "EnglishSentenceSplitter": {
            "chunk_size": 200,
            "chunk_overlap": 40,
            "suitable_for": ["Short text", "Dialog data", "Precise analysis"]
        },
        "EnglishParagraphSplitter": {
            "chunk_size": 400,
            "chunk_overlap": 80,
            "suitable_for": ["Long documents", "Academic papers", "Technical manuals"]
        },
        "SemanticChunkSplitter": {
            "chunk_size": 300,
            "chunk_overlap": 75,
            "suitable_for": ["High-quality retrieval", "Complex documents", "Academic research"]
        },
        "SlidingWindowSplitter": {
            "chunk_size": 250,
            "chunk_overlap": 125,
            "suitable_for": ["High recall", "Important documents", "Detailed analysis"]
        },
    }
    return recommendations.get(splitter_name, {
        "chunk_size": 250,
        "chunk_overlap": 50,
        "suitable_for": ["General documents"]
    })


def validate_splitter_params(splitter_name: str, chunk_size: int, chunk_overlap: int) -> Dict[str, str]:
    """
    验证分片器参数的合理性
    """
    warnings = {}
    
    # 基础验证
    if chunk_overlap >= chunk_size:
        warnings["overlap_too_large"] = "Overlap size should not be greater than or equal to chunk size"
    
    if chunk_size < 50:
        warnings["chunk_too_small"] = "Chunk size too small may lead to incomplete information"
    
    if chunk_size > 1000:
        warnings["chunk_too_large"] = "Chunk size too large may affect retrieval accuracy"
    
    # 特定分片器的建议
    recommendations = get_splitter_recommended_params(splitter_name)
    recommended_size = recommendations.get("chunk_size", 250)
    recommended_overlap = recommendations.get("chunk_overlap", 50)
    
    if abs(chunk_size - recommended_size) > recommended_size * 0.5:
        warnings["size_deviation"] = f"Recommended chunk size is approximately {recommended_size} characters"
    
    if abs(chunk_overlap - recommended_overlap) > recommended_overlap * 0.5:
        warnings["overlap_deviation"] = f"Recommended overlap is approximately {recommended_overlap} characters"
    
    return warnings


# ================= 新增：RAG-fusion相关工具函数 =================

def get_search_mode_display_name(mode: str) -> str:
    """
    获取搜索模式的友好显示名称
    """
    display_names = {
        "vector": "🧠 Vector Search",
        "bm25": "🔤 Keyword Search",
        "hybrid": "🔀 Hybrid Search",
        "rag_fusion": "🚀 RAG-fusion Search",
        "adaptive": "🎯 Adaptive Search"
    }
    return display_names.get(mode, mode)


def get_search_mode_description(mode: str) -> str:
    """
    获取搜索模式的详细描述
    """
    descriptions = {
        "vector": "Semantic search based on vector similarity, suitable for understanding query intent and semantic relevance",
        "bm25": "Keyword search based on BM25 algorithm, suitable for exact vocabulary matching and term lookup",
        "hybrid": "Hybrid strategy combining vector search and BM25 search, balancing semantic understanding and keyword matching",
        "rag_fusion": "Advanced retrieval strategy that generates multiple related queries and fuses results, providing the most comprehensive search results",
        "adaptive": "Intelligent mode that automatically selects the best retrieval strategy based on query characteristics"
    }
    return descriptions.get(mode, "Unknown search mode")


def validate_rag_fusion_params(params: Dict[str, Any]) -> Dict[str, str]:
    """
    验证RAG-fusion参数的合理性
    """
    warnings = {}
    
    query_count = params.get("query_count", 3)
    timeout = params.get("timeout", 30)
    model = params.get("model", "")
    
    # 查询数量验证
    if query_count < 2:
        warnings["query_count_too_small"] = "Query count should be at least 2 (including original query)"
    elif query_count > 10:
        warnings["query_count_too_large"] = "Too many queries may affect performance, recommend no more than 10"
    
    # 超时验证
    if timeout < 10:
        warnings["timeout_too_short"] = "Timeout too short may cause query generation failure"
    elif timeout > 120:
        warnings["timeout_too_long"] = "Timeout too long may affect user experience"
    
    # 模型验证
    if model and RAG_FUSION_SUPPORTED_MODELS:
        if model not in RAG_FUSION_SUPPORTED_MODELS:
            warnings["model_not_supported"] = f"Model '{model}' may not be supported"
    
    return warnings


def estimate_rag_fusion_cost(query_count: int, avg_query_length: int = 20) -> Dict[str, Any]:
    """
    估算RAG-fusion的性能成本
    """
    # 基础成本估算（相对值）
    base_cost = 1.0  # 普通检索的成本
    query_generation_cost = query_count * 0.3  # 查询生成成本
    retrieval_cost = query_count * base_cost  # 多次检索成本
    fusion_cost = 0.2  # 结果融合成本
    
    total_cost = query_generation_cost + retrieval_cost + fusion_cost
    
    # 时间估算（秒）
    estimated_time = 2 + (query_count - 1) * 1.5  # 基础时间 + 额外查询时间
    
    return {
        "relative_cost": round(total_cost, 2),
        "estimated_time": round(estimated_time, 1),
        "query_generation_cost": round(query_generation_cost, 2),
        "retrieval_cost": round(retrieval_cost, 2),
        "fusion_cost": round(fusion_cost, 2),
        "cost_breakdown": {
            "query_generation": f"{query_generation_cost/total_cost*100:.1f}%",
            "retrieval": f"{retrieval_cost/total_cost*100:.1f}%",
            "fusion": f"{fusion_cost/total_cost*100:.1f}%"
        }
    }


def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def check_success_msg(data: Union[str, dict, list], key: str = "msg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if (isinstance(data, dict)
            and key in data
            and "code" in data
            and data["code"] == 200):
        return data[key]
    return ""


if __name__ == "__main__":
    api = ApiRequest()
    aapi = AsyncApiRequest()

    # 测试新的分片器功能
    print("测试分片器功能:")
    
    try:
        splitters = api.get_available_text_splitters()
        print(f"可用分片器: {list(splitters.keys())}")
        
        for name in ["EnglishSentenceSplitter", "SemanticChunkSplitter"]:
            if name in splitters:
                print(f"\n{name}:")
                print(f"  显示名: {get_splitter_display_name(name)}")
                print(f"  描述: {get_splitter_description(name)}")
                params = get_splitter_recommended_params(name)
                print(f"  推荐参数: chunk_size={params['chunk_size']}, chunk_overlap={params['chunk_overlap']}")
                print(f"  适用场景: {', '.join(params['suitable_for'])}")
    except Exception as e:
        print(f"分片器测试失败: {e}")

    # 测试RAG-fusion功能
    print("\n测试RAG-fusion功能:")
    
    try:
        if RAG_FUSION_AVAILABLE:
            system_info = api.get_system_info()
            print(f"系统信息: {system_info.get('msg', 'N/A')}")
            
            validation = api.validate_rag_fusion_setup()
            print(f"配置验证: {validation.get('msg', 'N/A')}")
            
            models = api.get_rag_fusion_models()
            print(f"可用模型: {models[:3]}..." if len(models) > 3 else f"可用模型: {models}")
            
            # 测试成本估算
            cost = estimate_rag_fusion_cost(3)
            print(f"成本估算: 相对成本={cost['relative_cost']}, 预估时间={cost['estimated_time']}s")
        else:
            print("RAG-fusion功能未启用")
            
    except Exception as e:
        print(f"RAG-fusion测试失败: {e}")

    # 原有测试代码保持不变
    # print(api.list_knowledge_bases())
