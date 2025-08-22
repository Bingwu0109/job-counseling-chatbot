import urllib
from typing import Dict, Any, List
from server.utils import BaseResponse, ListResponse
from server.knowledge_base.utils import validate_kb_name
from server.knowledge_base.kb_service.base import KBServiceFactory, SearchMode
from server.db.repository.knowledge_base_repository import list_kbs_from_db
from configs import EMBEDDING_MODEL, logger, log_verbose
from fastapi import Body

# 导入RAG-fusion相关配置（安全导入）
try:
    from configs import (
        ENABLE_RAG_FUSION,
        RAG_FUSION_CONFIG,
        RAG_FUSION_QUERY_COUNT,
        RAG_FUSION_LLM_MODEL,
        RAG_FUSION_SUPPORTED_MODELS,
        kbs_config,
    )
    RAG_FUSION_AVAILABLE = ENABLE_RAG_FUSION
except ImportError:
    RAG_FUSION_AVAILABLE = False
    RAG_FUSION_CONFIG = {}
    RAG_FUSION_QUERY_COUNT = 3
    RAG_FUSION_LLM_MODEL = "Qwen1.5-7B-Chat"
    RAG_FUSION_SUPPORTED_MODELS = []
    kbs_config = {}


def list_kbs():
    """列出所有知识库（最简修复版本）"""
    try:
        # 从数据库获取知识库列表
        kb_list_raw = list_kbs_from_db()
        
        # 确保返回字符串列表
        kb_names = []
        
        if kb_list_raw:
            for kb_item in kb_list_raw:
                try:
                    if isinstance(kb_item, str):
                        # 直接是字符串
                        if kb_item.strip():
                            kb_names.append(kb_item.strip())
                    elif isinstance(kb_item, dict):
                        # 从字典中提取kb_name
                        kb_name = kb_item.get("kb_name")
                        if kb_name and isinstance(kb_name, str) and kb_name.strip():
                            kb_names.append(kb_name.strip())
                except Exception as e:
                    logger.debug(f"处理知识库项目失败: {e}")
                    continue
        
        # 去重并排序
        kb_names = sorted(list(set(kb_names)))
        
        return ListResponse(
            data=kb_names,
            msg=f"成功获取 {len(kb_names)} 个知识库"
        )
        
    except Exception as e:
        logger.error(f"列出知识库时出错: {e}")
        # 确保出错时也返回正确的格式
        return ListResponse(
            code=500, 
            msg=f"列出知识库失败: {e}", 
            data=[]  # 空的字符串列表
        )


def create_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
              vector_store_type: str = Body("faiss"),
              embed_model: str = Body(EMBEDDING_MODEL),
              ) -> BaseResponse:
    """创建知识库"""
    
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")
    
    # 检查是否已存在同名知识库
    try:
        kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
        if kb is not None:
            return BaseResponse(code=404, msg=f"已存在同名知识库 {knowledge_base_name}")
    except:
        pass
    
    # 创建知识库
    try:
        kb = KBServiceFactory.get_service(knowledge_base_name, vector_store_type, embed_model)
        kb.create_kb()
        
        return BaseResponse(
            code=200, 
            msg=f"已新增知识库 {knowledge_base_name}",
            data={
                "kb_name": knowledge_base_name,
                "vs_type": vector_store_type,
                "embed_model": embed_model
            }
        )
        
    except Exception as e:
        msg = f"创建知识库出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)


def delete_kb(knowledge_base_name: str = Body(..., examples=["samples"])) -> BaseResponse:
    """删除知识库"""
    
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    try:
        status = kb.clear_vs()
        status = kb.drop_kb()
        
        if status:
            return BaseResponse(
                code=200, 
                msg=f"成功删除知识库 {knowledge_base_name}"
            )
            
        return BaseResponse(code=500, msg=f"删除知识库失败 {knowledge_base_name}")
        
    except Exception as e:
        msg = f"删除知识库时出现意外： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)


def get_kb_info(knowledge_base_name: str = Body(..., examples=["samples"])) -> BaseResponse:
    """获取知识库详细信息"""
    
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    
    try:
        # 获取基本信息
        kb_info = {
            "kb_name": kb.kb_name,
            "kb_info": getattr(kb, 'kb_info', ''),
            "vs_type": kb.vs_type() if hasattr(kb, 'vs_type') else 'unknown',
            "embed_model": getattr(kb, 'embed_model', 'unknown'),
        }
        
        # 获取文件数量
        try:
            kb_info["file_count"] = kb.count_files() if hasattr(kb, 'count_files') else 0
        except:
            kb_info["file_count"] = 0
        
        # 获取文档数量
        try:
            if hasattr(kb, 'get_stats'):
                stats = kb.get_stats()
                kb_info["document_count"] = stats.get("document_count", 0)
            else:
                kb_info["document_count"] = 0
        except:
            kb_info["document_count"] = 0
        
        # 检查能力
        capabilities = {
            "vector_search": True,
            "bm25_search": hasattr(kb, 'do_bm25_search'),
            "hybrid_search": hasattr(kb, 'do_hybrid_search'),
        }
        
        # 检查RAG-fusion支持
        if RAG_FUSION_AVAILABLE:
            try:
                capabilities["rag_fusion"] = (
                    hasattr(kb, 'supports_rag_fusion') and
                    callable(getattr(kb, 'supports_rag_fusion', None)) and
                    kb.supports_rag_fusion()
                )
            except:
                capabilities["rag_fusion"] = False
        else:
            capabilities["rag_fusion"] = False
        
        kb_info["capabilities"] = capabilities
        
        return BaseResponse(code=200, msg="获取知识库信息成功", data=kb_info)
        
    except Exception as e:
        msg = f"获取知识库信息出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)


def get_system_info() -> BaseResponse:
    """获取系统信息"""
    try:
        system_info = {
            "rag_fusion_available": RAG_FUSION_AVAILABLE,
            "supported_models": RAG_FUSION_SUPPORTED_MODELS if RAG_FUSION_AVAILABLE else [],
            "default_model": RAG_FUSION_LLM_MODEL if RAG_FUSION_AVAILABLE else "未配置",
            "vs_types": list(kbs_config.keys()) if kbs_config else ["faiss"]
        }
        
        return BaseResponse(
            code=200,
            msg="获取系统信息成功",
            data=system_info
        )
    except Exception as e:
        return BaseResponse(
            code=500,
            msg=f"获取系统信息失败: {e}",
            data={}
        )
