import os
import urllib
import time
from fastapi import File, Form, Body, Query, UploadFile
from configs import (DEFAULT_VS_TYPE, EMBEDDING_MODEL,
                     VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE,
                     logger, log_verbose, )
from server.utils import BaseResponse, ListResponse, run_in_thread_pool
from server.knowledge_base.utils import (validate_kb_name, list_files_from_folder, get_file_path,
                                         files2docs_in_thread, KnowledgeFile)
from fastapi.responses import FileResponse
from sse_starlette import EventSourceResponse
from pydantic import Json
import json
import urllib.parse
from server.knowledge_base.kb_service.base import KBServiceFactory, SearchMode
from server.db.repository.knowledge_file_repository import get_file_detail
from langchain.docstore.document import Document
from server.knowledge_base.model.kb_document_model import DocumentWithVSId
from typing import List, Dict, Optional
import urllib.parse

# 导入RAG-fusion相关配置和工具
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


def search_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
        score_threshold: float = Body(SCORE_THRESHOLD,
                                      description="知识库匹配相关度阈值，取值范围在0-1之间，"
                                                  "SCORE越小，相关度越高，"
                                                  "取到1相当于不筛选，建议设置在0.5左右",
                                      ge=0, le=2),
        file_name: str = Body("", description="文件名称，支持 sql 通配符"),
        metadata: dict = Body({}, description="根据 metadata 进行过滤，仅支持一级键"),
        
        # 混合检索和RAG-fusion参数
        search_mode: str = Body(SearchMode.VECTOR_ONLY, 
                               description="检索模式: 'vector'(向量检索), 'bm25'(关键词检索), 'hybrid'(混合检索), 'rag_fusion'(RAG-fusion), 'adaptive'(自适应)",
                               examples=["vector"]),
        dense_weight: float = Body(0.7, description="稠密检索权重", ge=0.0, le=1.0),
        sparse_weight: float = Body(0.3, description="稀疏检索权重", ge=0.0, le=1.0),
        rrf_k: int = Body(60, description="RRF算法参数", ge=1),
        
        # RAG-fusion特有参数
        enable_rag_fusion: Optional[bool] = Body(None, description="是否启用RAG-fusion（当search_mode为其他值时可强制启用）"),
        fusion_search_strategy: Optional[str] = Body(None, description="RAG-fusion搜索策略: 'parallel'(并行), 'sequential'(顺序)", examples=["parallel"]),
        fusion_query_count: Optional[int] = Body(None, description="RAG-fusion生成的查询数量（包括原查询）", ge=2, le=10),
        fusion_model_name: Optional[str] = Body(None, description="RAG-fusion使用的LLM模型", examples=["Qwen1.5-7B-Chat"]),
        fusion_timeout: Optional[int] = Body(None, description="RAG-fusion查询生成超时时间（秒）", ge=5, le=120),

) -> List[DocumentWithVSId]:
    '''在指定的知识库中搜索文档，支持多种检索模式包括RAG-fusion'''
    
    if not validate_kb_name(knowledge_base_name):
        return []
    
    # 获取对应的知识库服务实例
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    
    if kb is None:
        return data
    
    if query:
        # 验证和规范化搜索模式
        valid_modes = [SearchMode.VECTOR_ONLY, SearchMode.BM25_ONLY, SearchMode.HYBRID]
        
        # 检查是否支持RAG-fusion
        if RAG_FUSION_AVAILABLE and hasattr(kb, 'supports_rag_fusion') and kb.supports_rag_fusion():
            valid_modes.extend([SearchMode.RAG_FUSION, SearchMode.ADAPTIVE])
        
        # 处理强制启用RAG-fusion的情况
        if enable_rag_fusion and RAG_FUSION_AVAILABLE:
            search_mode = SearchMode.RAG_FUSION
        
        if search_mode not in valid_modes:
            # 如果模式不受支持，记录警告并使用默认模式
            logger.warning(f"检索模式 '{search_mode}' 不受支持，使用向量检索")
            search_mode = SearchMode.VECTOR_ONLY
        
        # 特殊处理：如果选择了RAG-fusion但不可用，降级为混合检索
        if (search_mode == SearchMode.RAG_FUSION and 
            (not RAG_FUSION_AVAILABLE or not hasattr(kb, 'supports_rag_fusion') or not kb.supports_rag_fusion())):
            logger.warning("RAG-fusion不可用，降级为混合检索")
            search_mode = SearchMode.HYBRID
        
        # 验证权重参数（用于混合检索和RAG-fusion）
        if search_mode in [SearchMode.HYBRID, SearchMode.RAG_FUSION]:
            total_weight = dense_weight + sparse_weight
            if abs(total_weight - 1.0) > 0.01:
                # 重新归一化权重
                dense_weight = dense_weight / total_weight
                sparse_weight = sparse_weight / total_weight
                logger.info(f"权重已归一化: dense={dense_weight:.3f}, sparse={sparse_weight:.3f}")
        
        # 准备搜索参数
        search_params = {
            "query": query,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "search_mode": search_mode,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "rrf_k": rrf_k
        }
        
        # 为RAG-fusion添加特有参数（只有当它们不是None时才添加）
        if search_mode == SearchMode.RAG_FUSION:
            if fusion_query_count is not None:
                search_params["rag_fusion_query_count"] = fusion_query_count
            if fusion_model_name is not None:
                # 验证模型是否支持
                if RAG_FUSION_SUPPORTED_MODELS and fusion_model_name not in RAG_FUSION_SUPPORTED_MODELS:
                    logger.warning(f"模型 '{fusion_model_name}' 可能不受支持，但仍将尝试使用")
                search_params["rag_fusion_model"] = fusion_model_name
            if fusion_timeout is not None:
                search_params["rag_fusion_timeout"] = fusion_timeout
        
        try:
            # 执行搜索
            docs = kb.search_docs(**search_params)
            
            # 构造返回数据，添加搜索模式信息
            data = []
            for doc, score in docs:
                doc_data = DocumentWithVSId(**doc.dict(), score=score, id=doc.metadata.get("id"))
                # 在元数据中记录使用的搜索模式
                if not hasattr(doc_data, 'metadata') or doc_data.metadata is None:
                    doc_data.metadata = {}
                doc_data.metadata["search_mode_used"] = search_mode
                data.append(doc_data)
            
            # 记录搜索日志
            logger.info(f"搜索完成: 模式={search_mode}, 查询='{query}', 结果数={len(data)}")
            
        except Exception as e:
            logger.error(f"搜索出错: {e}")
            # 如果RAG-fusion失败，尝试降级
            if search_mode == SearchMode.RAG_FUSION:
                logger.info("RAG-fusion搜索失败，尝试混合检索")
                try:
                    fallback_params = search_params.copy()
                    fallback_params["search_mode"] = SearchMode.HYBRID
                    # 移除RAG-fusion特有参数
                    for key in ["rag_fusion_query_count", "rag_fusion_model", "rag_fusion_timeout"]:
                        fallback_params.pop(key, None)
                    
                    docs = kb.search_docs(**fallback_params)
                    data = [DocumentWithVSId(**doc.dict(), score=score, id=doc.metadata.get("id")) 
                           for doc, score in docs]
                    logger.info(f"降级搜索成功，返回 {len(data)} 个结果")
                except Exception as e2:
                    logger.error(f"降级搜索也失败: {e2}")
                    data = []
            
    elif file_name or metadata:
        # 文档列表功能保持不变
        data = kb.list_docs(file_name=file_name, metadata=metadata)
        for d in data:
            if "vector" in d.metadata:
                del d.metadata["vector"]
    
    return data


def rag_fusion_search(
        query: str = Body(..., description="用户输入", examples=["如何使用机器学习？"]),
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="最终返回的文档数量"),
        score_threshold: float = Body(SCORE_THRESHOLD, description="相关度阈值", ge=0, le=1),
        
        # RAG-fusion核心参数
        query_count: int = Body(RAG_FUSION_QUERY_COUNT, description="生成的查询总数（包括原查询）", ge=2, le=10),
        llm_model: str = Body(RAG_FUSION_LLM_MODEL, description="用于查询生成的LLM模型"),
        timeout: int = Body(30, description="查询生成超时时间（秒）", ge=5, le=120),
        
        # 检索策略参数
        per_query_top_k: Optional[int] = Body(None, description="每个查询的检索数量（默认为top_k*2）", ge=1, le=50),
        dense_weight: float = Body(0.7, description="稠密检索权重", ge=0.0, le=1.0),
        sparse_weight: float = Body(0.3, description="稀疏检索权重", ge=0.0, le=1.0),
        rrf_k: int = Body(60, description="RRF融合算法参数", ge=1),
        
        # 高级选项
        enable_cache: bool = Body(True, description="是否启用查询缓存"),
        enable_rerank: bool = Body(False, description="是否启用结果重排序"),
        
) -> BaseResponse:
    """专门的RAG-fusion检索API"""
    
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    
    if not RAG_FUSION_AVAILABLE:
        return BaseResponse(code=400, msg="RAG-fusion功能未启用")
    
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    
    # 检查知识库是否支持RAG-fusion
    if not (hasattr(kb, 'supports_rag_fusion') and kb.supports_rag_fusion()):
        return BaseResponse(code=400, msg=f"知识库 {knowledge_base_name} 不支持RAG-fusion功能")
    
    try:
        # 准备RAG-fusion参数
        rag_params = {
            "query": query,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "search_mode": SearchMode.RAG_FUSION,
            "rag_fusion_query_count": query_count,
            "rag_fusion_model": llm_model,
            "rag_fusion_timeout": timeout,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "rrf_k": rrf_k
        }
        
        # 执行RAG-fusion检索
        start_time = time.time()
        docs = kb.search_docs(**rag_params)
        execution_time = time.time() - start_time
        
        # 处理结果
        results = []
        for doc, score in docs:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "id": doc.metadata.get("id"),
            }
            results.append(result)
        
        # 构造响应数据
        response_data = {
            "results": results,
            "query_info": {
                "original_query": query,
                "query_count": query_count,
                "llm_model": llm_model,
                "execution_time": execution_time,
            },
            "search_params": {
                "top_k": top_k,
                "score_threshold": score_threshold,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "rrf_k": rrf_k,
            }
        }
        
        return BaseResponse(code=200, msg="RAG-fusion检索成功", data=response_data)
        
    except Exception as e:
        logger.error(f"RAG-fusion检索失败: {e}")
        return BaseResponse(code=500, msg=f"RAG-fusion检索失败: {str(e)}")


def update_info(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        kb_info: str = Body(..., description="知识库介绍", examples=["这是一个知识库"]),
):
    """更新知识库介绍信息"""
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    
    kb.update_info(kb_info)
    return BaseResponse(code=200, msg=f"知识库介绍修改完成", data={"kb_info": kb_info})


def update_docs_by_id(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        docs: Dict[str, Document] = Body(..., description="要更新的文档内容，形如：{id: Document, ...}")
) -> BaseResponse:
    '''根据文档ID更新指定知识库中的文档内容'''

    # 根据知识库名称获取对应的知识库服务实例
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        # 如果知识库不存在，则返回一个状态码为500的BaseResponse对象，提示知识库不存在。
        return BaseResponse(code=500, msg=f"指定的知识库 {knowledge_base_name} 不存在")

    # 如果知识库存在，则尝试更新文档。如果更新成功，返回提示"文档更新成功"的BaseResponse对象；
    if kb.update_doc_by_ids(docs=docs):
        return BaseResponse(msg=f"文档更新成功")
    else:
        # 如果更新失败，返回提示"文档更新失败"的BaseResponse对象。
        return BaseResponse(msg=f"文档更新失败")


def list_files(
        knowledge_base_name: str  # 知识库名称
) -> ListResponse:
    '''列出指定知识库中的所有文件名'''
    if not validate_kb_name(knowledge_base_name):
        return ListResponse(code=403, msg="Don't attack me", data=[])

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return ListResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
    else:
        all_doc_names = kb.list_files()
        return ListResponse(data=all_doc_names)


def _save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          override: bool):
    """
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    """

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        '''
        保存单个文件。
        '''
        try:
            filename = file.filename
            file_path = get_file_path(knowledge_base_name, filename)
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read()  # 读取上传文件的内容
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                # 文件名相同且文件大小相同，返回错误信息
                return dict(code=404, msg=f"文件 {filename} 已存在", data=data)

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"保存文件 {filename} 时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return dict(code=500, msg=msg, data=data)

    params = [{"file": file, "knowledge_base_name": knowledge_base_name, "override": override} for file in files]
    for result in run_in_thread_pool(func=save_file, params=params):
        yield result


def upload_docs(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
        override: bool = Form(False, description="覆盖已有文件"),
        to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
        chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        docs: Json = Form({}, description="自定义的docs，需要转为json字符串"),
        not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """
    API接口：上传文件，并/或向量化
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    file_names = list(docs.keys())

    # 先将上传的文件保存到磁盘
    for result in _save_files_in_thread(files, knowledge_base_name, override):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    # 对保存的文件进行向量化
    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True)
        if hasattr(result, 'data') and result.data and "failed_files" in result.data:
            failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return BaseResponse(code=200, msg="文件上传与向量化完成", data={"failed_files": failed_files})


def delete_docs(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        file_names: List[str] = Body(..., description="删除的文件名称", examples=[["file_name.md", "test.txt"]]),
        delete_content: bool = Body(False, description="是否删除文件内容")
) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    for file_name in file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"未找到文件 {file_name}"

        try:
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb.delete_doc(kb_file, delete_content)
        except Exception as e:
            msg = f"删除文件 {file_name} 出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    return BaseResponse(code=200, msg=f"文件删除完成", data={"failed_files": failed_files})


def update_docs(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        file_names: List[str] = Body(..., description="文件名称，支持多文件", examples=[["file_name1", "text.txt"]]),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        override_custom_docs: bool = Body(False, description="是否覆盖之前自定义的docs"),
        docs: Json = Body({}, description="自定义的docs，需要转为json字符串",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """更新知识库文档"""
    
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    
    failed_files = {}
    kb_files = []

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        # 获取文件详情
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs:
            try:
                kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name))
            except Exception as e:
                msg = f"加载文档 {file_name} 时出错：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                failed_files[file_name] = msg

    # 从文件生成docs，并进行向量化
    for status, result in files2docs_in_thread(kb_files,
                                               chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap,
                                               zh_title_enhance=zh_title_enhance):
        if status:
            kb_name, file_name, new_docs = result
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb_file.splited_docs = new_docs
            kb.update_doc(kb_file, not_refresh_vs_cache=True)
        else:
            kb_name, file_name, error = result
            failed_files[file_name] = error

    # 将自定义的docs进行向量化
    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
            kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"为 {file_name} 添加自定义docs时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg
    
    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"更新文档完成", data={"failed_files": failed_files})


def update_kb_docs(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        file_names: List[str] = Body(..., description="文件名称，支持 sql 通配符，如果为空则返回全部文件", examples=[["file_name1", "text.txt"]]),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        override_custom_docs: bool = Body(False, description="是否覆盖之前自定义的docs"),
        docs: dict = Body({}, description="自定义的docs，需要转为json字符串"),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """
    更新知识库文档（兼容性接口）
    """
    # 调用新的update_docs函数
    return update_docs(
        knowledge_base_name=knowledge_base_name,
        file_names=file_names,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        zh_title_enhance=zh_title_enhance,
        override_custom_docs=override_custom_docs,
        docs=docs,
        not_refresh_vs_cache=not_refresh_vs_cache
    )


def download_doc(
        knowledge_base_name: str = Query(..., description="知识库名称", examples=["samples"]),
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
        preview: bool = Query(False, description="是否预览")
):
    """
    下载知识库文档
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    if preview:
        content_disposition_type = "inline"
    else:
        content_disposition_type = None

    try:
        kb_file = KnowledgeFile(filename=file_name,
                                knowledge_base_name=knowledge_base_name)

        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type='application/octet-stream',
                headers={"content-disposition-type": content_disposition_type} if preview else None)
    except Exception as e:
        msg = f"{kb_file.filename} 读取文件失败，错误信息是：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"{kb_file.filename} 读取文件失败")


def recreate_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        chunk_size: int = Body(CHUNK_SIZE),
        chunk_overlap: int = Body(OVERLAP_SIZE),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE),
):
    """
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
    set allow_empty_kb to True make it applied on empty knowledge base which it not in the info.db or having no files.
    """

    def output():
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 '{knowledge_base_name}'"}
        else:
            if kb.exists():
                kb.clear_vs()
            kb.create_kb()
            files = list_files_from_folder(knowledge_base_name)
            kb_files = [(file, KnowledgeFile(filename=file, knowledge_base_name=knowledge_base_name)) for file in files]
            i = 0
            for file, kb_file in kb_files:
                try:
                    kb.add_doc(kb_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                               zh_title_enhance=zh_title_enhance)
                    i += 1
                    yield {"code": 200, "msg": f"({i}/{len(files)}): {file}"}
                except Exception as e:
                    msg = f"添加文件'{file}'到知识库'{knowledge_base_name}'时出错：{e}。已跳过。"
                    logger.error(f'{e.__class__.__name__}: {msg}',
                                 exc_info=e if log_verbose else None)
                    yield {"code": 500, "msg": msg}
            yield {"code": 200, "msg": f"知识库'{knowledge_base_name}'重建完毕，共导入{i}个文件。"}

    return EventSourceResponse(output())


def recreate_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        chunk_size: int = Body(CHUNK_SIZE),
        chunk_overlap: int = Body(OVERLAP_SIZE),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE),
):
    """
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
    set allow_empty_kb to True make it applied on empty knowledge base which it not in the info.db or having no files.
    """

    def output():
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 '{knowledge_base_name}'"}
        else:
            if kb.exists():
                kb.clear_vs()
            kb.create_kb()
            files = list_files_from_folder(knowledge_base_name)
            kb_files = [(file, KnowledgeFile(filename=file, knowledge_base_name=knowledge_base_name)) for file in files]
            i = 0
            for file, kb_file in kb_files:
                try:
                    kb.add_doc(kb_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                               zh_title_enhance=zh_title_enhance)
                    i += 1
                    yield {"code": 200, "msg": f"({i}/{len(files)}): {file}"}
                except Exception as e:
                    msg = f"添加文件'{file}'到知识库'{knowledge_base_name}'时出错：{e}。已跳过。"
                    logger.error(f'{e.__class__.__name__}: {msg}',
                                 exc_info=e if log_verbose else None)
                    yield {"code": 500, "msg": msg}
            yield {"code": 200, "msg": f"知识库'{knowledge_base_name}'重建完毕，共导入{i}个文件。"}

    return EventSourceResponse(output())
