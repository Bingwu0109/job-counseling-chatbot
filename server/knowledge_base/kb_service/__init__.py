# server/knowledge_base/kb_service/__init__.py

"""
知识库服务模块初始化文件
支持多种向量存储类型和检索模式，包括RAG-fusion功能
"""

from .base import (
    KBService,
    KBServiceFactory, 
    SupportedVSType,
    SearchMode,
    EmbeddingsFunAdapter,
    get_kb_details,
    get_kb_file_details,
    reciprocal_rank_fusion,
    score_threshold_process,
    normalize,
    normalize_v2,
)

# 导入具体的服务实现
from .faiss_kb_service import FaissKBService
from .default_kb_service import DefaultKBService

# 尝试导入其他服务（如果可用）
try:
    from .es_kb_service import ESKBService
except ImportError:
    ESKBService = None

try:
    from .milvus_kb_service import MilvusKBService
except ImportError:
    MilvusKBService = None

try:
    from .pg_kb_service import PGKBService
except ImportError:
    PGKBService = None

try:
    from .chromadb_kb_service import ChromaKBService
except ImportError:
    ChromaKBService = None

try:
    from .zilliz_kb_service import ZillizKBService
except ImportError:
    ZillizKBService = None

# 导入RAG-fusion相关功能
try:
    from configs import (
        ENABLE_RAG_FUSION, 
        RAG_FUSION_CONFIG,
        kbs_config
    )
    RAG_FUSION_AVAILABLE = ENABLE_RAG_FUSION
except ImportError:
    RAG_FUSION_AVAILABLE = False

# 版本信息
__version__ = "0.2.10"

# 支持的功能特性
SUPPORTED_FEATURES = {
    "vector_search": True,
    "bm25_search": True, 
    "hybrid_search": True,
    "rag_fusion": RAG_FUSION_AVAILABLE,
    "adaptive_search": True,
}

# 向量存储类型映射
VS_TYPE_MAPPING = {
    SupportedVSType.FAISS: FaissKBService,
    SupportedVSType.DEFAULT: DefaultKBService,
}

# 动态添加可用的服务
if ESKBService is not None:
    VS_TYPE_MAPPING[SupportedVSType.ES] = ESKBService

if MilvusKBService is not None:
    VS_TYPE_MAPPING[SupportedVSType.MILVUS] = MilvusKBService

if PGKBService is not None:
    VS_TYPE_MAPPING[SupportedVSType.PG] = PGKBService

if ChromaKBService is not None:
    VS_TYPE_MAPPING[SupportedVSType.CHROMADB] = ChromaKBService

if ZillizKBService is not None:
    VS_TYPE_MAPPING[SupportedVSType.ZILLIZ] = ZillizKBService

def get_supported_vs_types() -> List[str]:
    """获取支持的向量存储类型列表"""
    return list(VS_TYPE_MAPPING.keys())

def get_rag_fusion_supported_vs_types() -> List[str]:
    """获取支持RAG-fusion的向量存储类型列表"""
    if not RAG_FUSION_AVAILABLE:
        return []
    
    supported_types = []
    for vs_type in VS_TYPE_MAPPING.keys():
        config = kbs_config.get(vs_type, {})
        if config.get("supports_rag_fusion", False):
            supported_types.append(vs_type)
    
    return supported_types

def is_rag_fusion_supported(vs_type: str) -> bool:
    """检查指定的向量存储类型是否支持RAG-fusion"""
    return vs_type in get_rag_fusion_supported_vs_types()

# 功能检查函数
def check_service_capabilities(vs_type: str) -> Dict[str, bool]:
    """检查指定向量存储服务的功能支持情况"""
    config = kbs_config.get(vs_type, {})
    
    return {
        "vector_search": True,  # 所有服务都支持向量检索
        "bm25_search": vs_type in [SupportedVSType.FAISS, SupportedVSType.ES],
        "hybrid_search": config.get("supports_hybrid_search", False),
        "rag_fusion": config.get("supports_rag_fusion", False) and RAG_FUSION_AVAILABLE,
    }

# 导出的主要接口
__all__ = [
    # 核心类
    "KBService",
    "KBServiceFactory", 
    "SupportedVSType",
    "SearchMode",
    "EmbeddingsFunAdapter",
    
    # 具体服务实现
    "FaissKBService",
    "DefaultKBService",
    "ESKBService",
    "MilvusKBService", 
    "PGKBService",
    "ChromaKBService",
    "ZillizKBService",
    
    # 工具函数
    "get_kb_details",
    "get_kb_file_details", 
    "reciprocal_rank_fusion",
    "score_threshold_process",
    "normalize",
    "normalize_v2",
    
    # 功能检查函数
    "get_supported_vs_types",
    "get_rag_fusion_supported_vs_types",
    "is_rag_fusion_supported",
    "check_service_capabilities",
    
    # 常量
    "SUPPORTED_FEATURES",
    "RAG_FUSION_AVAILABLE",
    "VS_TYPE_MAPPING",
]

# 模块初始化时的日志输出
if __name__ != "__main__":
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"知识库服务模块初始化完成 (版本: {__version__})")
    logger.info(f"支持的向量存储类型: {list(VS_TYPE_MAPPING.keys())}")
    
    if RAG_FUSION_AVAILABLE:
        rag_fusion_types = get_rag_fusion_supported_vs_types()
        logger.info(f"支持RAG-fusion的类型: {rag_fusion_types}")
    else:
        logger.info("RAG-fusion功能未启用")
