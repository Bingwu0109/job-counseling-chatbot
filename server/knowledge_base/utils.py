import os
import hashlib
import time
import re
from configs import (
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    logger,
    log_verbose,
    text_splitter_dict,
    LLM_MODELS,
    TEXT_SPLITTER_NAME,
    # 新增：导入新分片器的配置
    ENGLISH_SENTENCE_SPLITTER_CONFIG,
    ENGLISH_PARAGRAPH_SPLITTER_CONFIG,
    SEMANTIC_CHUNK_SPLITTER_CONFIG,
    SLIDING_WINDOW_SPLITTER_CONFIG,
    TEXT_SPLITTER_SELECTION_CONFIG,
)
import importlib
from text_splitter import zh_title_enhance as func_zh_title_enhance
import langchain.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from pathlib import Path
from server.utils import run_in_thread_pool, get_model_worker_config
import json
from typing import List, Union, Dict, Tuple, Generator, Any
import chardet

# ================= RAG-fusion相关配置导入 =================
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


def validate_kb_name(knowledge_base_id: str) -> bool:
    '''验证给定的知识库ID是否有效'''
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path(knowledge_base_name: str):
    '''根据给定的知识库名称构建并返回知识库的文件系统路径'''
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    '''构建并返回给定知识库中文档存储的路径'''
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str, vector_name: str):
    '''获取知识库中向量存储的路径'''
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store", vector_name)


def get_file_path(knowledge_base_name: str, doc_name: str):
    '''构建并返回知识库中某个文档的完整路径'''
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def list_kbs_from_folder():
    '''列出存储在特定根目录下的所有知识库文件夹'''
    return [f for f in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, f))]


def list_files_from_folder(kb_name: str):
    '''列出给定知识库中所有文件的路径，排除了一些特定的临时文件和目录。'''
    # 获取知识库中文档存储的根路径
    doc_path = get_doc_path(kb_name)
    result = []

    def is_skiped_path(path: str):
        '''用来判断给定的路径是否应该被忽略。
           如果文件名以temp, tmp, ., 或~$开头，该路径被认为是应该被跳过的。'''
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False

    def process_entry(entry):
        '''用于处理每一个文件系统条目。它会递归地处理目录，并收集文件路径。
           如果遇到符号链接，它会跟踪链接到的目标并处理目标位置的条目。'''
        if is_skiped_path(entry.path):
            return
        # 如果entry是符号链接，使用os.scandir递归处理链接指向的目录。
        if entry.is_symlink():
            target_path = os.path.realpath(entry.path)
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    process_entry(target_entry)
        elif entry.is_file():
            # 如果entry是文件，并且不应被跳过（使用is_skiped_path判断），
            # 则将其相对于doc_path的POSIX风格路径添加到result列表中。
            file_path = (Path(os.path.relpath(entry.path, doc_path)).as_posix())  # 路径统一为 posix 格式
            result.append(file_path)
        elif entry.is_dir():
            # 如果entry是目录，递归处理该目录中的所有条目。
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    process_entry(sub_entry)
    # 遍历知识库文档根目录下的所有条目，并对每个条目调用process_entry函数进行处理。
    with os.scandir(doc_path) as it:
        for entry in it:
            process_entry(entry)
    # 返回知识库中所有文件的路径列表，这些路径都是相对于知识库文档目录的，并且以POSIX风格表示。
    return result

# 用于映射文件扩展名到特定的加载器，并列出支持的文件扩展名。
LOADER_DICT = {"UnstructuredHTMLLoader": ['.html', '.htm'],
               "MHTMLLoader": ['.mhtml'],
               "UnstructuredMarkdownLoader": ['.md'],
               "JSONLoader": [".json"],
               "JSONLinesLoader": [".jsonl"],
               "CSVLoader": [".csv"],
               # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
               "RapidOCRPDFLoader": [".pdf"],
               "RapidOCRDocLoader": ['.docx', '.doc'],
               "RapidOCRPPTLoader": ['.ppt', '.pptx', ],
               "RapidOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
               "UnstructuredFileLoader": ['.eml', '.msg', '.rst',
                                          '.rtf', '.txt', '.xml',
                                          '.epub', '.odt','.tsv'],
               "UnstructuredEmailLoader": ['.eml', '.msg'],
               "UnstructuredEPubLoader": ['.epub'],
               "UnstructuredExcelLoader": ['.xlsx', '.xls', '.xlsd'],
               "NotebookLoader": ['.ipynb'],
               "UnstructuredODTLoader": ['.odt'],
               "PythonLoader": ['.py'],
               "UnstructuredRSTLoader": ['.rst'],
               "UnstructuredRTFLoader": ['.rtf'],
               "SRTLoader": ['.srt'],
               "TomlLoader": ['.toml'],
               "UnstructuredTSVLoader": ['.tsv'],
               "UnstructuredWordDocumentLoader": ['.docx', '.doc'],
               "UnstructuredXMLLoader": ['.xml'],
               "UnstructuredPowerPointLoader": ['.ppt', '.pptx'],
               "EverNoteLoader": ['.enex'],
               }

# 这个列表包含了所有支持的文件扩展名，每种文件扩展名都只会列出一次，即使它可能被多个加载器支持。
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


# patch json.dumps to disable ensure_ascii
def _new_json_dumps(obj, **kwargs):
    '''修改json.dumps函数的默认行为，可以确保JSON序列化时能够正确处理并保留非ASCII字符，如中文等。'''
    kwargs["ensure_ascii"] = False
    # 调用原来的 json.dumps 函数
    # 强制 json.dumps 输出非ASCII字符，而不是将它们转换为ASCII编码的转义序列。
    return _origin_json_dumps(obj, **kwargs)

# 检查当前 json.dumps 是否已经被替换为 _new_json_dumps，如果没有，则进行替换。
if json.dumps is not _new_json_dumps:
    _origin_json_dumps = json.dumps
    json.dumps = _new_json_dumps


class JSONLinesLoader(langchain.document_loaders.JSONLoader):
    '''
    行式 Json 加载器，要求文件扩展名为 .jsonl
    行式JSON文件是指每一行都是一个完整的JSON对象的文本文件。
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True

# 使用这里新定义的处理行式JSON文件的类
langchain.document_loaders.JSONLinesLoader = JSONLinesLoader


def get_LoaderClass(file_extension):
    '''file_extension : 文件扩展名'''
    # 循环遍历LOADER_DICT字典
    for LoaderClass, extensions in LOADER_DICT.items():
        # 检查传入的文件扩展名是否存在于当前迭代到的加载器支持的扩展名列表中
        if file_extension in extensions:
            # 如果存在，函数立即返回这个加载器类
            return LoaderClass


def get_loader(loader_name: str,  # 如果存在，函数立即返回这个加载器类
               file_path: str,  # 文件路径
               loader_kwargs: Dict = None  # 传递给加载器的关键字参数字典
               ):
    '''
    根据loader_name和文件路径或内容返回文档加载器。
    '''
    loader_kwargs = loader_kwargs or {}
    try:
        # 导入与 loader_name 匹配的加载器类
        if loader_name in ["RapidOCRPDFLoader", "RapidOCRLoader", "FilteredCSVLoader",
                           "RapidOCRDocLoader", "RapidOCRPPTLoader"]:
            # 根据loader_name的值，选择导入哪个模块。
            # 如果loader_name属于一组特定的加载器名称，则导入document_loaders模块；
            document_loaders_module = importlib.import_module('document_loaders')
        else:
            # 导入 langchain.document_loaders。
            document_loaders_module = importlib.import_module('langchain.document_loaders')
        # 从导入的模块中获取与 loader_name 对应的加载器类
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        # 如果在这一过程中发生任何异常，捕获异常并记录错误信息。
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        # 导入 langchain.document_loaders 模块，并从中获取 UnstructuredFileLoader 作为默认加载器类。
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")
    # 根据加载器名称，设置 loader_kwargs 中的默认值。
    if loader_name == "UnstructuredFileLoader":
        # 如果加载器是 UnstructuredFileLoader，则设置 autodetect_encoding 为 True。
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif loader_name == "CSVLoader":
        # 如果是 CSVLoader 并且没有指定编码，则尝试自动检测文件编码。
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, 'rb') as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]
    # 对于 JSONLoader 和 JSONLinesLoader，设置 jq_schema 默认值为 "."，text_content 默认值为 False。
    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    # 使用获取到的加载器类 DocumentLoader 和 loader_kwargs，创建加载器实例，传入文件路径和其他参数。
    loader = DocumentLoader(file_path, **loader_kwargs)
    # 返回这个加载器实例
    return loader


def auto_select_text_splitter(
    file_path: str = None,
    file_extension: str = None,
    content_length: int = None,
    default_splitter: str = TEXT_SPLITTER_NAME
) -> str:
    """
    根据文件类型和内容长度自动选择合适的分片器
    
    Args:
        file_path: 文件路径
        file_extension: 文件扩展名
        content_length: 内容长度
        default_splitter: 默认分片器
    
    Returns:
        推荐的分片器名称
    """
    try:
        # 如果没有启用自动选择，直接返回默认分片器
        if not TEXT_SPLITTER_SELECTION_CONFIG.get("auto_selection", False):
            return default_splitter
        
        # 根据文件类型选择
        if file_extension and file_extension in TEXT_SPLITTER_SELECTION_CONFIG.get("document_type_mapping", {}):
            return TEXT_SPLITTER_SELECTION_CONFIG["document_type_mapping"][file_extension]
        
        # 根据内容长度选择
        if content_length is not None:
            length_mapping = TEXT_SPLITTER_SELECTION_CONFIG.get("content_length_mapping", {})
            if content_length < 1000:
                return length_mapping.get("short", "EnglishSentenceSplitter")
            elif content_length < 10000:
                return length_mapping.get("medium", "EnglishParagraphSplitter")
            else:
                return length_mapping.get("long", "SemanticChunkSplitter")
        
        return default_splitter
    except Exception as e:
        logger.warning(f"自动选择分片器失败，使用默认分片器: {e}")
        return default_splitter


def make_text_splitter(
        splitter_name: str = TEXT_SPLITTER_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        llm_model: str = LLM_MODELS[0],
        file_path: str = None,
        **kwargs
):

    splitter_name = splitter_name or "SpacyTextSplitter"
    

    if TEXT_SPLITTER_SELECTION_CONFIG.get("auto_selection", False) and file_path:
        file_extension = os.path.splitext(file_path)[-1].lower().lstrip('.')
        try:
            #
            if os.path.exists(file_path):
                content_length = os.path.getsize(file_path)
                splitter_name = auto_select_text_splitter(
                    file_path=file_path,
                    file_extension=file_extension,
                    content_length=content_length,
                    default_splitter=splitter_name
                )
                logger.info(f"自动选择分片器: {splitter_name} (文件: {file_path})")
        except Exception as e:
            logger.warning(f"自动选择分片器失败，使用指定分片器: {e}")
    
    try:
        # ================= 处理新增的英文优化分片器 =================
        if splitter_name in ["EnglishSentenceSplitter", "EnglishParagraphSplitter", 
                           "SemanticChunkSplitter", "SlidingWindowSplitter"]:
            try:
                # 导入自定义分片器模块
                text_splitter_module = importlib.import_module('text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)
                
                # 获取分片器的配置
                splitter_config = text_splitter_dict.get(splitter_name, {}).get("config", {})
                
                # 合并配置参数
                splitter_kwargs = {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                }
                
                # 添加特定分片器的配置
                if splitter_name == "EnglishSentenceSplitter":
                    splitter_kwargs.update(ENGLISH_SENTENCE_SPLITTER_CONFIG)
                elif splitter_name == "EnglishParagraphSplitter":
                    splitter_kwargs.update(ENGLISH_PARAGRAPH_SPLITTER_CONFIG)
                elif splitter_name == "SemanticChunkSplitter":
                    splitter_kwargs.update(SEMANTIC_CHUNK_SPLITTER_CONFIG)
                elif splitter_name == "SlidingWindowSplitter":
                    splitter_kwargs.update(SLIDING_WINDOW_SPLITTER_CONFIG)
                
                # 添加来自配置文件的参数
                splitter_kwargs.update(splitter_config)
                # 添加传入的额外参数
                splitter_kwargs.update(kwargs)
                
                logger.info(f"创建英文优化分片器: {splitter_name}")
                text_splitter = TextSplitter(**splitter_kwargs)
                return text_splitter
                
            except Exception as e:
                logger.error(f"创建英文优化分片器 {splitter_name} 失败: {e}")
                # 降级到默认分片器
                logger.info("降级使用 RecursiveCharacterTextSplitter")
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
                return TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # ================= 处理原有分片器 =================
        elif splitter_name == "MarkdownHeaderTextSplitter":
            # 获取必要的头部分割信息，并创建一个MarkdownHeaderTextSplitter实例。
            headers_to_split_on = text_splitter_dict[splitter_name]['headers_to_split_on']
            text_splitter = langchain.text_splitter.MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on)
        else:
            # 优先使用用户自定义的text_splitter
            try:
                text_splitter_module = importlib.import_module('text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:
                # 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)

            # 根据分割器字典text_splitter_dict中的信息决定如何创建TextSplitter实例。
            # 这包括检查分割器的来源是tiktoken还是huggingface，并据此从相应的库中加载分割器。
            if text_splitter_dict.get(splitter_name, {}).get("source") == "tiktoken":
                # 从tiktoken加载
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
            elif text_splitter_dict.get(splitter_name, {}).get("source") == "huggingface":
                # 从huggingface加载
                if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "":
                    config = get_model_worker_config(llm_model)
                    text_splitter_dict[splitter_name]["tokenizer_name_or_path"] = \
                        config.get("model_path")

                if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "gpt2":
                    from transformers import GPT2TokenizerFast
                    from langchain.text_splitter import CharacterTextSplitter
                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:  ## 字符长度加载
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True)
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                # 如果既不是tiktoken也不是huggingface来源，尝试使用TextSplitter的标准构造函数创建实例。
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        **kwargs
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        **kwargs
                    )
    except Exception as e:
        logger.error(f"创建分片器 {splitter_name} 失败: {e}")
        # 最终的降级方案：使用RecursiveCharacterTextSplitter
        text_splitter_module = importlib.import_module('langchain.text_splitter')
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
    # If you use SpacyTextSplitter you can use GPU to do split likes Issue #1287
    # text_splitter._tokenizer.max_length = 37016792
    # text_splitter._tokenizer.prefer_gpu()
    return text_splitter


def get_available_text_splitters() -> Dict[str, str]:
    """
    获取所有可用的文本分片器及其描述
    
    Returns:
        字典，键为分片器名称，值为描述
    """
    splitters = {
        # 原有分片器
        "RecursiveCharacterTextSplitter": "递归字符分片器（通用，推荐用于英文）",
        "ChineseRecursiveTextSplitter": "中文递归分片器",
        "ChineseTextSplitter": "中文文本分片器", 
        "SpacyTextSplitter": "Spacy文本分片器",
        "MarkdownHeaderTextSplitter": "Markdown标题分片器",
        
        # 新增的英文优化分片器
        "EnglishSentenceSplitter": "英文句子分片器（精确句子边界）",
        "EnglishParagraphSplitter": "英文段落分片器（保持段落结构）",
        "SemanticChunkSplitter": "语义分块器（智能语义分组）",
        "SlidingWindowSplitter": "滑动窗口分片器（最大化覆盖率）",
    }
    
    # 只返回在配置中定义的分片器
    available_splitters = {}
    for name, description in splitters.items():
        if name in text_splitter_dict or name in [
            "EnglishSentenceSplitter", "EnglishParagraphSplitter", 
            "SemanticChunkSplitter", "SlidingWindowSplitter"
        ]:
            available_splitters[name] = description
    
    return available_splitters


# ================= RAG-fusion相关工具函数 =================

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


def validate_rag_fusion_config() -> bool:
    """验证RAG-fusion配置的有效性"""
    if not RAG_FUSION_AVAILABLE:
        return True
    
    try:
        assert RAG_FUSION_CONFIG is not None, "RAG_FUSION_CONFIG未正确配置"
        assert RAG_FUSION_LLM_MODEL in RAG_FUSION_SUPPORTED_MODELS, f"默认模型{RAG_FUSION_LLM_MODEL}不在支持列表中"
        return True
    except Exception as e:
        logger.error(f"RAG-fusion配置验证失败: {e}")
        return False


def get_rag_fusion_stats() -> Dict[str, Any]:
    """获取RAG-fusion统计信息"""
    return {
        "enabled": RAG_FUSION_AVAILABLE,
        "cache_size": len(_rag_fusion_query_cache),
        "supported_models": RAG_FUSION_SUPPORTED_MODELS if RAG_FUSION_AVAILABLE else [],
        "current_model": RAG_FUSION_LLM_MODEL if RAG_FUSION_AVAILABLE else None,
        "config_valid": validate_rag_fusion_config(),
        "cache_stats": {
            "entries": len(_rag_fusion_query_cache),
            "max_size": RAG_FUSION_CONFIG.get("cache", {}).get("max_cache_size", 1000) if RAG_FUSION_CONFIG else 1000
        }
    }


def clear_rag_fusion_cache():
    """清空RAG-fusion缓存"""
    global _rag_fusion_query_cache
    _rag_fusion_query_cache.clear()
    logger.info("RAG-fusion缓存已清空")


def is_rag_fusion_supported_file(file_extension: str) -> bool:
    """
    检查文件类型是否适合RAG-fusion
    某些文件类型可能不适合进行多查询检索
    """
    # 不适合RAG-fusion的文件类型
    unsuitable_exts = {'.csv', '.json', '.jsonl', '.xml'}
    return file_extension.lower() not in unsuitable_exts


def get_optimal_splitter_for_rag_fusion(file_extension: str, content_length: int = None) -> str:
    """
    为RAG-fusion推荐最佳的分片器
    RAG-fusion通常需要更语义化的分片策略
    """
    if not RAG_FUSION_AVAILABLE:
        return TEXT_SPLITTER_NAME
    
    # 针对RAG-fusion的分片器推荐
    if file_extension in ['.md', '.markdown']:
        return "MarkdownHeaderTextSplitter"
    elif file_extension in ['.txt', '.doc', '.docx']:
        if content_length and content_length > 10000:
            return "SemanticChunkSplitter"  # 大文档用语义分块
        else:
            return "EnglishParagraphSplitter"  # 中小文档用段落分片
    elif file_extension in ['.html', '.htm']:
        return "EnglishParagraphSplitter"
    else:
        return "RecursiveCharacterTextSplitter"  # 默认选择


class KnowledgeFile:
    def __init__(
            self,
            filename: str,  # 文件名
            knowledge_base_name: str,  # 知识库名称
            loader_kwargs: Dict = {},  # 加载器的关键字参数，默认为空字典
    ):
        '''
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        '''
        self.kb_name = knowledge_base_name  # 知识库名称
        self.filename = str(Path(filename).as_posix())  # Path().as_posix()方法将路径转换为POSIX风格，这在跨平台编程时非常有用。
        self.ext = os.path.splitext(filename)[-1].lower()  # 获取文件的扩展名
        if self.ext not in SUPPORTED_EXTS:  # 检查文件扩展名是否在预定义的支持列表SUPPORTED_EXTS中
            raise ValueError(f"暂未支持的文件格式 {self.filename}")
        self.loader_kwargs = loader_kwargs  # 加载器的关键字参数字典
        self.filepath = get_file_path(knowledge_base_name, filename)  # 文件路径
        self.docs = None  # 文件加载后的文档内容
        self.splited_docs = None  # 分割后的文档
        self.document_loader_name = get_LoaderClass(self.ext)  # 获取对应的文档加载器类名
        self.text_splitter_name = TEXT_SPLITTER_NAME  # 文本分割器名称
        # RAG-fusion相关属性
        self._rag_fusion_suitable = None

    def file2docs(self, refresh: bool = False):
        '''将文件内容加载为文档列表'''
        # 判断是否需要加载文档
        if self.docs is None or refresh:
            # 记录当前正在使用的文档加载器和目标文件路径
            logger.info(f"{self.document_loader_name} used for {self.filepath}")
            # 创建一个加载器
            loader = get_loader(loader_name=self.document_loader_name,
                                file_path=self.filepath,
                                loader_kwargs=self.loader_kwargs)
            # 使用加载器的load方法从文件中加载文档
            self.docs = loader.load()
        return self.docs

    def docs2texts(
            self,
            docs: List[Document] = None,  # 文档列表
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,  # 是否刷新文档列表
            chunk_size: int = CHUNK_SIZE,  # 块大小
            chunk_overlap: int = OVERLAP_SIZE,  # 重叠大小
            text_splitter: TextSplitter = None,  # 文本分割器实例
            text_splitter_name: str = None,  # 指定的文本分割器名称
    ):
        '''从文档到文本块的转换，同时提供了文本分割和可选的增强处理功能。'''
        # 允许直接传入文档列表或从文件中加载
        docs = docs or self.file2docs(refresh=refresh)
        # 如果没有文档可处理，立即返回空列表。
        if not docs:
            return []
            
        # 使用指定的分片器名称或默认分片器名称
        splitter_name = text_splitter_name or self.text_splitter_name
        
        if self.ext not in [".csv"]:
            if text_splitter is None:
                # 创建一个文本分割器，传入文件路径用于自动选择
                text_splitter = make_text_splitter(
                    splitter_name=splitter_name, 
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    file_path=self.filepath  # 新增：传入文件路径
                )
            # 动态决定如何创建分割器
            if splitter_name == "MarkdownHeaderTextSplitter":
                # 只对第一个文档的页面内容进行分割
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                # 对整个文档列表进行分割
                docs = text_splitter.split_documents(docs)
        # 再次检查分割后的文档列表是否为空，如果是，则返回空列表。
        if not docs:
            return []

        print(f"文档切分示例（使用{splitter_name}）：{docs[0]}")
        
        # 如果启用了中文标题增强（zh_title_enhance为True），则对文档标题进行增强处理。
        # 注意：对于英文文档，通常应该禁用中文标题增强
        if zh_title_enhance and splitter_name not in [
            "EnglishSentenceSplitter", "EnglishParagraphSplitter", 
            "SemanticChunkSplitter", "SlidingWindowSplitter"
        ]:
            docs = func_zh_title_enhance(docs)
        
        self.splited_docs = docs
        # 返回处理后的文档列表
        return self.splited_docs

    def file2text(
            self,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
            text_splitter_name: str = None,  # 新增：允许指定分片器名称
    ):

        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            # 使用docs2texts方法处理文档，将其分割为文本块
            self.splited_docs = self.docs2texts(
                docs=docs,
                zh_title_enhance=zh_title_enhance,
                refresh=refresh,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
                text_splitter_name=text_splitter_name
            )
        return self.splited_docs

    def file_exist(self):
        '''检查关联文件是否存在于文件系统中'''
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        '''获取文件的最后修改时间'''
        return os.path.getmtime(self.filepath)

    def get_size(self):
        '''获取文件的大小'''
        return os.path.getsize(self.filepath)

    # ================= RAG-fusion相关方法 =================

    def is_rag_fusion_suitable(self) -> bool:
        """检查文件是否适合RAG-fusion"""
        if self._rag_fusion_suitable is None:
            self._rag_fusion_suitable = (
                RAG_FUSION_AVAILABLE and 
                is_rag_fusion_supported_file(self.ext)
            )
        return self._rag_fusion_suitable
    
    def get_optimal_rag_fusion_splitter(self) -> str:
        """获取适合RAG-fusion的最佳分片器"""
        content_length = self.get_size() if self.file_exist() else None
        return get_optimal_splitter_for_rag_fusion(self.ext, content_length)
    
    def file2text_for_rag_fusion(self, **kwargs):
        """针对RAG-fusion优化的文本分片"""
        if self.is_rag_fusion_suitable():
            # 使用推荐的分片器
            optimal_splitter = self.get_optimal_rag_fusion_splitter()
            kwargs['text_splitter_name'] = optimal_splitter
            
            # 针对RAG-fusion调整参数
            if 'chunk_size' not in kwargs:
                kwargs['chunk_size'] = min(CHUNK_SIZE * 1.5, 400)  # RAG-fusion适合稍大的chunk
            
            logger.info(f"RAG-fusion模式: 使用{optimal_splitter}分片器处理{self.filename}")
        
        return self.file2text(**kwargs)


def create_knowledge_file(*args, **kwargs):
    """创建KnowledgeFile的工厂函数，支持RAG-fusion增强"""
    return KnowledgeFile(*args, **kwargs)


def files2docs_in_thread(
        files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],  # 文件列表
        chunk_size: int = CHUNK_SIZE,  # 块大小
        chunk_overlap: int = OVERLAP_SIZE,  # 块重叠
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,  # 是否增强中文标题
        text_splitter_name: str = None,  # 新增：允许指定分片器名称
) -> Generator:
    '''
    一个设计用于在多线程环境中批量将磁盘文件转换成langchain Document的生成器函数。
    增强支持指定分片器名称

    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    '''

    def file2docs(*, file: KnowledgeFile, **kwargs) -> Tuple[bool, Tuple[str, str, List[Document]]]:
        '''用于处理单个文件的转换'''
        try:
            # 调用file.file2text(**kwargs)方法将文件转换为文档
            # 返回True和一个元组，包含知识库名称、文件名和转换得到的文档列表。
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))
        except Exception as e:
            msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            # 返回一个元组，其中包含处理状态、知识库名称、文件名和文档列表或错误消息。
            return False, (file.kb_name, file.filename, msg)

    # 存储将要传递给file2docs函数的参数
    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            # 如果元素是元组并且长度至少为2，提取文件名和知识库名，并创建一个KnowledgeFile对象。
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                # 如果元素是字典，同样提取文件名和知识库名，
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                # 并将剩余的内容作为额外参数（kwargs）更新到字典中，然后创建KnowledgeFile对象。
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            # 将必要的参数（包括file对象和其他方法参数）添加到kwargs字典中，然后将这个字典添加到kwargs_list列表。
            kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            if text_splitter_name:  # 新增：传递分片器名称
                kwargs["text_splitter_name"] = text_splitter_name
            kwargs_list.append(kwargs)
        except Exception as e:
            # 如果在处理过程中发生异常，会直接通过生成器返回错误信息。
            yield False, (kb_name, filename, str(e))

    # 使用run_in_thread_pool函数并行执行file2docs函数，传入kwargs_list作为参数。
    # 遍历run_in_thread_pool函数的结果，并通过生成器返回。
    for result in run_in_thread_pool(func=file2docs, params=kwargs_list):
        yield result


if __name__ == "__main__":
    from pprint import pprint
    
    # 测试新的分片器
    print("测试可用的分片器:")
    splitters = get_available_text_splitters()
    for name, desc in splitters.items():
        print(f"  {name}: {desc}")
    
    print("\n测试分片器创建:")
    # 测试创建不同的分片器
    test_splitters = ["RecursiveCharacterTextSplitter", "EnglishSentenceSplitter", 
                     "EnglishParagraphSplitter", "SemanticChunkSplitter"]
    
    for splitter_name in test_splitters:
        try:
            splitter = make_text_splitter(splitter_name=splitter_name)
            print(f"  ✓ {splitter_name}: {type(splitter).__name__}")
        except Exception as e:
            print(f"  ✗ {splitter_name}: {e}")

    # 测试RAG-fusion相关功能
    print("\n测试RAG-fusion工具函数:")
    print(f"RAG-fusion可用: {RAG_FUSION_AVAILABLE}")
    print(f"统计信息: {get_rag_fusion_stats()}")
    
    if RAG_FUSION_AVAILABLE:
        # 测试查询相似度计算
        q1 = "What is machine learning?"
        q2 = "How does machine learning work?"
        similarity = _calculate_query_similarity(q1, q2)
        print(f"查询相似度: {similarity:.3f}")
        
        # 测试文件适用性检查
        test_files = ['.txt', '.pdf', '.csv', '.json', '.md']
        for ext in test_files:
            suitable = is_rag_fusion_supported_file(ext)
            splitter = get_optimal_splitter_for_rag_fusion(ext)
            print(f"{ext}: 适合RAG-fusion={suitable}, 推荐分片器={splitter}")

    # 原有的测试代码
    print("\n测试KnowledgeFile:")
    try:
        kb_file = KnowledgeFile(
            filename="test.txt",
            knowledge_base_name="samples")
        print(f"KnowledgeFile创建成功: {kb_file.filename}")
        print(f"是否适合RAG-fusion: {kb_file.is_rag_fusion_suitable()}")
        print(f"推荐的RAG-fusion分片器: {kb_file.get_optimal_rag_fusion_splitter()}")
    except Exception as e:
        print(f"KnowledgeFile测试失败: {e}")
