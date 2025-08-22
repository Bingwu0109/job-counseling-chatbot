import os

# 默认使用的知识库
DEFAULT_KNOWLEDGE_BASE = "samples"

# 默认向量库/全文检索引擎类型。可选：faiss, milvus(离线) & zilliz(在线), pgvector, chromadb 全文检索引擎es
DEFAULT_VS_TYPE = "es" # "faiss"

# 缓存向量库数量（针对FAISS）
CACHED_VS_NUM = 1

# 缓存临时向量库数量（针对FAISS），用于文件对话
CACHED_MEMO_VS_NUM = 10


CHUNK_SIZE = 250

OVERLAP_SIZE = 50

VECTOR_SEARCH_TOP_K = 3

# 知识库匹配的距离阈值，一般取值范围在0-1之间，SCORE越小，距离越小从而相关度越高。
# 但有用户报告遇到过匹配分值超过1的情况，为了兼容性默认设为1，在WEBUI中调整范围为0-2
SCORE_THRESHOLD = 1.0

# ================= 混合检索配置（英文版） =================

# 默认检索模式：'vector'(向量检索), 'bm25'(关键词检索), 'hybrid'(混合检索)
DEFAULT_SEARCH_MODE = "hybrid"

# 混合检索权重配置
DEFAULT_DENSE_WEIGHT = 0.7   # 稠密检索(向量检索)权重
DEFAULT_SPARSE_WEIGHT = 0.3  # 稀疏检索(BM25检索)权重

# RRF (Reciprocal Rank Fusion) 算法参数
DEFAULT_RRF_K = 60

# ================= RAG-fusion配置 =================

# 是否启用RAG-fusion
ENABLE_RAG_FUSION = True  # 改为 True

# RAG-fusion生成查询数量（包括原查询）
RAG_FUSION_QUERY_COUNT = 3

# 用于RAG-fusion查询生成的LLM模型（从model_config.py中的LLM_MODELS选择）
RAG_FUSION_LLM_MODEL = "Qwen1.5-7B-Chat"  # 默认使用第一个本地模型

# RAG-fusion查询生成的temperature
RAG_FUSION_TEMPERATURE = 0.7

# RAG-fusion最大token数
RAG_FUSION_MAX_TOKENS = 200

# RAG-fusion超时时间（秒）
RAG_FUSION_TIMEOUT = 30

# RAG-fusion查询生成提示模板
RAG_FUSION_QUERY_PROMPT = """Based on the original query, generate {num_queries} different but related search queries to improve information retrieval.

Original query: {original_query}

Requirements:
1. Generate {num_queries} queries (including the original one)
2. Each query should approach the topic from a different angle
3. Queries should be concise and focused
4. Use synonyms and related terms when appropriate
5. Return only the queries, one per line

Generated queries:"""

# RAG-fusion结果融合配置
RAG_FUSION_CONFIG = {
    # 基本参数
    "enable": True,  # 改为 True
    "query_count": RAG_FUSION_QUERY_COUNT,
    "llm_model": RAG_FUSION_LLM_MODEL,
    "temperature": RAG_FUSION_TEMPERATURE,
    "max_tokens": RAG_FUSION_MAX_TOKENS,
    "timeout": RAG_FUSION_TIMEOUT,
    
    # 查询生成配置
    "query_generation": {
        "prompt_template": RAG_FUSION_QUERY_PROMPT,
        "min_query_length": 5,          # 生成查询的最小长度
        "max_query_length": 100,        # 生成查询的最大长度
        "filter_similar": True,         # 是否过滤过于相似的查询
        "similarity_threshold": 0.9,    # 查询相似度阈值
        "retry_attempts": 3,            # 失败重试次数
    },
    
    # 检索配置
    "retrieval": {
        "per_query_top_k": 5,           # 每个查询检索的文档数
        "search_mode": "hybrid",        # 检索模式：vector, bm25, hybrid
        "enable_rerank": False,         # 是否启用重排序
        "rerank_top_k": 10,            # 重排序前的文档数
    },
    
    # 结果融合配置  
    "fusion": {
        "algorithm": "rrf",             # 融合算法：rrf, weighted_sum, max
        "rrf_k": 60,                   # RRF算法参数
        "final_top_k": VECTOR_SEARCH_TOP_K,  # 最终返回的文档数
        "score_normalization": True,    # 是否进行分数归一化
        "diversity_boost": 0.1,        # 多样性增强权重
    },
    
    # 缓存配置
    "cache": {
        "enable_query_cache": True,     # 是否缓存生成的查询
        "cache_expire_time": 3600,      # 缓存过期时间（秒）
        "max_cache_size": 1000,        # 最大缓存条目数
    },
    
    # 日志和调试
    "logging": {
        "log_generated_queries": True,  # 是否记录生成的查询
        "log_retrieval_results": False, # 是否记录每个查询的检索结果
        "log_fusion_process": True,     # 是否记录融合过程
    }
}

# BM25检索配置（英文优化版）
BM25_CONFIG = {
    # BM25算法参数（针对英文优化）
    "k1": 1.2,        # 控制词频饱和度的参数
    "b": 0.75,        # 控制文档长度归一化的参数
    
    # 分词配置（英文）
    "tokenizer": "english",         # 分词器类型：english（使用空格和标点分词）
    "enable_custom_dict": False,    # 英文不需要自定义词典
    "enable_stopwords": True,       # 启用英文停用词过滤
    "lowercase": True,              # 转换为小写
    "remove_punctuation": True,     # 移除标点符号
    
    # 英文停用词列表
    "stopwords": {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
        'was', 'will', 'with', 'would', 'could', 'should', 'may', 'might', 'can',
        'shall', 'must', 'ought', 'need', 'dare', 'used', 'am', 'i', 'you', 'we',
        'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our',
        'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'this', 'these',
        'that', 'those', 'there', 'here', 'when', 'where', 'why', 'what', 'which',
        'who', 'whom', 'whose', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'now'
    },
    
    # 索引配置
    "enable_index_cache": True,     # 是否启用索引缓存
    "index_cache_expire": 3600,     # 索引缓存过期时间(秒)
    "auto_rebuild_threshold": 0.3,  # 自动重建索引的阈值(文档变化比例)
    
    # 查询优化（英文）
    "enable_query_expansion": False,  # 是否启用查询扩展
    "min_token_length": 2,           # 最小token长度（英文单词通常较长）
    "max_token_length": 30,          # 最大token长度
    "enable_stemming": True,         # 启用词干提取
    "enable_lemmatization": False,   # 启用词形还原（可选，需要额外库）
}

# 自适应检索配置（英文版）
ADAPTIVE_SEARCH_CONFIG = {
    # 关键词指示符（倾向于BM25检索）
    "keyword_indicators": [
        "how", "what", "when", "where", "why", "who", "which", "define", "definition",
        "explain", "explanation", "describe", "description", "steps", "step", "method",
        "methods", "way", "ways", "process", "procedure", "tutorial", "guide", "instruction",
        "instructions", "setup", "install", "installation", "configure", "configuration",
        "use", "usage", "operate", "operation", "implement", "implementation"
    ],
    
    # 语义指示符（倾向于向量检索）
    "semantic_indicators": [
        "similar", "similarity", "like", "alike", "related", "relation", "relationship",
        "compare", "comparison", "contrast", "difference", "differences", "distinguish",
        "connection", "associate", "association", "correlate", "correlation", "analyze",
        "analysis", "principle", "principles", "concept", "concepts", "theory", "theories",
        "characteristic", "characteristics", "feature", "features", "property", "properties",
        "nature", "essence", "fundamental", "basic", "advanced", "complex"
    ],
    
    # 查询长度阈值
    "short_query_threshold": 15,     # 短查询阈值（单词数，英文通常较短）
    "long_query_threshold": 40,      # 长查询阈值（单词数）
    
    # 自动选择权重
    "bm25_threshold": 2,            # BM25关键词匹配阈值
    "vector_threshold": 1,          # 向量语义匹配阈值
}

# 检索结果融合配置
RESULT_FUSION_CONFIG = {
    "enable_score_normalization": True,    # 是否启用分数归一化
    "enable_diversity_boost": False,       # 是否启用结果多样性增强
    "diversity_lambda": 0.1,               # 多样性权重
    "min_score_threshold": 0.01,           # 最小分数阈值
}

# ================= 新增分片策略配置 =================

# 英文句子分片配置
ENGLISH_SENTENCE_SPLITTER_CONFIG = {
    "language": "english",
    "keep_separator": True,                 # 保留分隔符
    "min_sentence_length": 15,             # 最小句子长度
    "max_sentence_length": 1000,           # 最大句子长度
    "sentence_endings": ['.', '!', '?'],   # 句子结束符
    "preserve_quotes": True,               # 保持引用完整性
    "handle_abbreviations": True,          # 处理缩写
}

# 英文段落分片配置
ENGLISH_PARAGRAPH_SPLITTER_CONFIG = {
    "paragraph_separator": "\n\n",         # 段落分隔符
    "preserve_structure": True,            # 保持结构
    "min_paragraph_length": 50,           # 最小段落长度
    "max_paragraph_length": 2000,         # 最大段落长度
    "merge_short_paragraphs": True,       # 合并短段落
    "split_long_paragraphs": True,        # 拆分长段落
    "keep_headings": True,                # 保持标题
}

# 语义分块配置
SEMANTIC_CHUNK_SPLITTER_CONFIG = {
    "embed_model": "bge-large-en-v1.5",   # 使用您的英文embedding模型
    "similarity_threshold": 0.75,         # 语义相似度阈值
    "min_chunk_size": 100,                # 最小chunk大小
    "max_chunk_size": 800,                # 最大chunk大小
    "window_size": 3,                     # 滑动窗口大小
    "stride": 1,                          # 窗口步长
    "boundary_strategy": "sentence",       # 边界策略：sentence, paragraph, both
    "merge_threshold": 0.85,              # 合并阈值
    "split_threshold": 0.6,               # 分割阈值
}

# 滑动窗口分片配置
SLIDING_WINDOW_SPLITTER_CONFIG = {
    "window_size": 200,                   # 窗口大小（字符数）
    "step_size": 100,                     # 步长（重叠部分）
    "min_window_size": 50,                # 最小窗口大小
    "boundary_strategy": "sentence",       # 边界策略：sentence, word, character
    "preserve_sentences": True,           # 保持句子完整性
    "overlap_strategy": "symmetric",       # 重叠策略：symmetric, forward, backward
    "dynamic_sizing": True,               # 动态调整窗口大小
}

# ================= 原有配置继续 =================

# 默认搜索引擎。可选：bing, duckduckgo, metaphor
DEFAULT_SEARCH_ENGINE = "duckduckgo"

# 搜索引擎匹配结题数量
SEARCH_ENGINE_TOP_K = 3


# Bing 搜索必备变量
# 使用 Bing 搜索需要使用 Bing Subscription Key,需要在azure port中申请试用bing search
# 具体申请方式请见
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/create-bing-search-service-resource
# 使用python创建bing api 搜索实例详见:
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
BING_SEARCH_URL = "https://api.bing.microsoft.com/v7.0/search"
# 注意不是bing Webmaster Tools的api key，

# 此外，如果是在服务器上，报Failed到establish a new connection: [Errno 110] Connection timed out
# 是因为服务器加了防火墙，需要联系管理员加白名单，如果公司的服务器的话，就别想了GG
BING_SUBSCRIPTION_KEY = ""

# metaphor搜索需要KEY
METAPHOR_API_KEY = ""

# 心知天气 API KEY，用于天气Agent。申请：https://www.seniverse.com/
SENIVERSE_API_KEY = ""

# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
# 注意：英文知识库建议设为False
ZH_TITLE_ENHANCE = False

# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)

# 每个知识库的初始化介绍，用于在初始化知识库时显示和Agent调用，没写则没有介绍，不会被Agent调用。
KB_INFO = {
    "knowledge_base_name": "knowledge base description",
    "samples": "Q&A about project issues and solutions",
}


# 通常情况下不需要更改以下内容

# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
if not os.path.exists(KB_ROOT_PATH):
    os.mkdir(KB_ROOT_PATH)
# 数据库默认存储路径。
# 如果使用sqlite，可以直接修改DB_ROOT_PATH；如果使用其它数据库，请直接修改SQLALCHEMY_DATABASE_URI。
DB_ROOT_PATH = os.path.join(KB_ROOT_PATH, "info.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_ROOT_PATH}"

# 可选向量库类型及对应配置
kbs_config = {
    "faiss": {
        # 添加混合检索支持标识
        "supports_hybrid_search": True,
        "supports_rag_fusion": True,    # 添加RAG-fusion支持标识
    },
    "milvus": {
        "host": "127.0.0.1",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
        "supports_hybrid_search": False,  # Milvus暂不支持我们的混合检索实现
        "supports_rag_fusion": False,    # Milvus暂不支持RAG-fusion
    },
    "zilliz": {
        "host": "in01-a7ce524e41e3935.ali-cn-hangzhou.vectordb.zilliz.com.cn",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": True,
        "supports_hybrid_search": False,  # Zilliz暂不支持我们的混合检索实现
        "supports_rag_fusion": False,    # Zilliz暂不支持RAG-fusion
        },
    "pg": {
        "connection_uri": "postgresql://postgres:postgres@127.0.0.1:5432/langchain_chatchat",
        "supports_hybrid_search": False,  # PG暂不支持我们的混合检索实现
        "supports_rag_fusion": False,    # PG暂不支持RAG-fusion
    },

    "es": {
        "host": "0.0.0.0",
        "port": "9200",
        "index_name": "index01",
        "user": "elastic",
        "password": "yuxFGk2i*_PS=_qcPTHv",
        "dims_length": 1024,  # 注意：bge-large-en-v1.5的向量维度
        "supports_hybrid_search": True,   # ES支持混合检索
        "supports_rag_fusion": True,     # ES支持RAG-fusion
    },
    "milvus_kwargs":{
        "search_params":{"metric_type": "L2"}, #在此处增加search_params
        "index_params":{"metric_type": "L2","index_type": "HNSW"} # 在此处增加index_params
    },
    "chromadb": {
        "supports_hybrid_search": False,  # ChromaDB暂不支持我们的混合检索实现
        "supports_rag_fusion": False,    # ChromaDB暂不支持RAG-fusion
    }
}

# TextSplitter配置项，如果你不明白其中的含义，就不要修改。
text_splitter_dict = {
    # ================= 原有分片器 =================
    "ChineseRecursiveTextSplitter": {
        "source": "huggingface",   # 选择tiktoken则使用openai的方法
        "tokenizer_name_or_path": "",
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on":
            [
                ("#", "head1"),
                ("##", "head2"),
                ("###", "head3"),
                ("####", "head4"),
            ]
    },
    
    # ================= 新增的4种英文优化分片器 =================
    "EnglishSentenceSplitter": {
        "source": "nltk",                     # 使用NLTK进行句子分割
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": OVERLAP_SIZE,
        "config": ENGLISH_SENTENCE_SPLITTER_CONFIG,
    },
    
    "EnglishParagraphSplitter": {
        "source": "regex",                    # 使用正则表达式识别段落
        "chunk_size": CHUNK_SIZE * 2,         # 段落通常比句子长
        "chunk_overlap": OVERLAP_SIZE * 2,
        "config": ENGLISH_PARAGRAPH_SPLITTER_CONFIG,
    },
    
    "SemanticChunkSplitter": {
        "source": "embedding",               # 基于embedding的语义分块
        "chunk_size": CHUNK_SIZE + 50,       # 语义分块可以稍大一些
        "chunk_overlap": OVERLAP_SIZE + 25,
        "config": SEMANTIC_CHUNK_SPLITTER_CONFIG,
    },
    
    "SlidingWindowSplitter": {
        "source": "window",                  # 滑动窗口策略
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": OVERLAP_SIZE,
        "config": SLIDING_WINDOW_SPLITTER_CONFIG,
    },
}

# TEXT_SPLITTER 名称 - 英文推荐使用RecursiveCharacterTextSplitter
# 可以改为新的分片器进行测试：EnglishSentenceSplitter, EnglishParagraphSplitter, SemanticChunkSplitter, SlidingWindowSplitter
TEXT_SPLITTER_NAME = "RecursiveCharacterTextSplitter"

# 分片器选择策略配置
TEXT_SPLITTER_SELECTION_CONFIG = {
    "auto_selection": False,              # 是否启用自动选择分片器
    "document_type_mapping": {            # 根据文档类型选择分片器
        "pdf": "EnglishParagraphSplitter",
        "txt": "EnglishSentenceSplitter", 
        "md": "MarkdownHeaderTextSplitter",
        "docx": "EnglishParagraphSplitter",
        "html": "EnglishParagraphSplitter",
    },
    "content_length_mapping": {           # 根据内容长度选择分片器
        "short": "EnglishSentenceSplitter",    # < 1000 字符
        "medium": "EnglishParagraphSplitter",  # 1000-10000 字符
        "long": "SemanticChunkSplitter",       # > 10000 字符
    },
    "quality_priority": [                 # 质量优先级排序
        "SemanticChunkSplitter",          # 最佳语义理解
        "EnglishParagraphSplitter",       # 保持逻辑结构
        "EnglishSentenceSplitter",        # 句子级精确度
        "SlidingWindowSplitter",          # 最大覆盖率
        "RecursiveCharacterTextSplitter", # 通用选择
    ],
}

# Embedding模型定制词语的词表文件
EMBEDDING_KEYWORD_FILE = "embedding_keywords.txt"
