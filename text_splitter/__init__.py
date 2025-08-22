# 原有的中文分片器
from .chinese_text_splitter import ChineseTextSplitter
from .ali_text_splitter import AliTextSplitter
from .zh_title_enhance import zh_title_enhance
from .chinese_recursive_text_splitter import ChineseRecursiveTextSplitter

# 新增的英文优化分片器
from .english_sentence_splitter import EnglishSentenceSplitter
from .english_paragraph_splitter import EnglishParagraphSplitter
from .semantic_chunk_splitter import SemanticChunkSplitter
from .sliding_window_splitter import SlidingWindowSplitter

# 导出所有分片器
__all__ = [
    # 原有分片器
    'ChineseTextSplitter',
    'AliTextSplitter',
    'zh_title_enhance',
    'ChineseRecursiveTextSplitter',

    # 新增英文分片器
    'EnglishSentenceSplitter',
    'EnglishParagraphSplitter',
    'SemanticChunkSplitter',
    'SlidingWindowSplitter',
]

# 分片器映射字典，用于动态获取分片器类
SPLITTER_MAPPING = {
    'ChineseTextSplitter': ChineseTextSplitter,
    'AliTextSplitter': AliTextSplitter,
    'ChineseRecursiveTextSplitter': ChineseRecursiveTextSplitter,
    'EnglishSentenceSplitter': EnglishSentenceSplitter,
    'EnglishParagraphSplitter': EnglishParagraphSplitter,
    'SemanticChunkSplitter': SemanticChunkSplitter,
    'SlidingWindowSplitter': SlidingWindowSplitter,
}


def get_text_splitter(splitter_name: str, **kwargs):
    """
    根据分片器名称获取分片器实例

    Args:
        splitter_name: 分片器名称
        **kwargs: 分片器初始化参数

    Returns:
        分片器实例
    """
    if splitter_name in SPLITTER_MAPPING:
        return SPLITTER_MAPPING[splitter_name](**kwargs)
    else:
        raise ValueError(f"未找到分片器: {splitter_name}")