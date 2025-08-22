from .basic_config import *
from .model_config import *
from .kb_config import *
from .server_config import *
from .prompt_config import *

# ================= RAG-fusion配置导入验证 =================

# 验证RAG-fusion关键配置是否正确导入
try:
    # 验证基础配置
    assert ENABLE_RAG_FUSION is not None, "RAG_FUSION基础配置导入失败"
    assert RAG_FUSION_CONFIG is not None, "RAG_FUSION详细配置导入失败"
    
    # 验证模型配置
    assert RAG_FUSION_SUPPORTED_MODELS is not None, "RAG_FUSION支持模型列表导入失败"
    assert RAG_FUSION_MODEL_SELECTION is not None, "RAG_FUSION模型选择策略导入失败"
    
    # 验证向量库支持
    for kb_type, config in kbs_config.items():
        if "supports_rag_fusion" not in config:
            config["supports_rag_fusion"] = False  # 为未明确配置的向量库添加默认值
    
    print("✅ RAG-fusion配置导入验证成功")
    
except (NameError, AssertionError) as e:
    print(f"❌ RAG-fusion配置导入验证失败: {e}")
    # 设置默认值以确保系统正常运行
    if 'ENABLE_RAG_FUSION' not in globals():
        ENABLE_RAG_FUSION = False
    if 'RAG_FUSION_CONFIG' not in globals():
        RAG_FUSION_CONFIG = {"enable": False}

# ================= 配置兼容性检查 =================

def validate_rag_fusion_config():
    """验证RAG-fusion配置的完整性和合理性"""
    
    issues = []
    
    # 检查基础配置
    if ENABLE_RAG_FUSION:
        if RAG_FUSION_LLM_MODEL not in RAG_FUSION_SUPPORTED_MODELS:
            issues.append(f"RAG-fusion默认模型 {RAG_FUSION_LLM_MODEL} 不在支持列表中")
        
        if RAG_FUSION_QUERY_COUNT < 2 or RAG_FUSION_QUERY_COUNT > 10:
            issues.append(f"RAG-fusion查询数量 {RAG_FUSION_QUERY_COUNT} 不在合理范围内(2-10)")
        
        if RAG_FUSION_TEMPERATURE < 0 or RAG_FUSION_TEMPERATURE > 2:
            issues.append(f"RAG-fusion温度 {RAG_FUSION_TEMPERATURE} 不在合理范围内(0-2)")
    
    # 检查向量库支持
    supported_vs = [vs for vs, config in kbs_config.items() 
                   if config.get("supports_rag_fusion", False)]
    
    if ENABLE_RAG_FUSION and DEFAULT_VS_TYPE not in [vs for vs in supported_vs]:
        issues.append(f"当前向量库 {DEFAULT_VS_TYPE} 不支持RAG-fusion功能")
    
    # 输出检查结果
    if issues:
        print("⚠️  RAG-fusion配置问题:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("✅ RAG-fusion配置验证通过")
    
    return len(issues) == 0

# 执行配置验证
if __name__ != "__main__":  # 避免在直接运行时执行
    validate_rag_fusion_config()

# ================= 版本信息 =================

VERSION = "v0.2.10"

# RAG-fusion功能版本标识
RAG_FUSION_VERSION = "v1.0.0"
FEATURE_FLAGS = {
    "rag_fusion": ENABLE_RAG_FUSION,
    "hybrid_search": True,  # 继承原有混合检索功能
    "adaptive_search": True,  # 自适应检索功能
    "multi_splitter": True,   # 多种分片策略
}

print(f"Langchain-Chatchat {VERSION} 已加载")
if ENABLE_RAG_FUSION:
    print(f"🚀 RAG-fusion功能已启用 (版本: {RAG_FUSION_VERSION})")
    print(f"   - 支持的向量库: {[vs for vs, config in kbs_config.items() if config.get('supports_rag_fusion', False)]}")
    print(f"   - 默认查询扩展数量: {RAG_FUSION_QUERY_COUNT}")
    print(f"   - 默认LLM模型: {RAG_FUSION_LLM_MODEL}")
else:
    print("RAG-fusion功能当前未启用")

# ================= 便捷导入 =================

# 将关键配置提升到包级别，方便其他模块导入
__all__ = [
    # 基础配置
    "VERSION", "log_verbose", "logger",
    
    # 知识库配置
    "DEFAULT_KNOWLEDGE_BASE", "DEFAULT_VS_TYPE", "VECTOR_SEARCH_TOP_K", 
    "SCORE_THRESHOLD", "CHUNK_SIZE", "OVERLAP_SIZE",
    
    # 混合检索配置
    "DEFAULT_SEARCH_MODE", "DEFAULT_DENSE_WEIGHT", "DEFAULT_SPARSE_WEIGHT", 
    "DEFAULT_RRF_K",
    
    # RAG-fusion配置
    "ENABLE_RAG_FUSION", "RAG_FUSION_CONFIG", "RAG_FUSION_QUERY_COUNT",
    "RAG_FUSION_LLM_MODEL", "RAG_FUSION_SUPPORTED_MODELS",
    
    # 模型配置
    "LLM_MODELS", "EMBEDDING_MODEL", "TEMPERATURE", "MAX_TOKENS",
    
    # 工具配置
    "kbs_config", "text_splitter_dict", "TEXT_SPLITTER_NAME",
    
    # 验证函数
    "validate_rag_fusion_config",
]