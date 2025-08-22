import os

# 可以指定一个绝对路径，统一存放所有的Embedding和LLM模型。
# 每个模型可以是一个单独的目录，也可以是某个目录下的二级子目录。
# 如果模型目录名称和 MODEL_PATH 中的 key 或 value 相同，程序会自动检测加载，无需修改 MODEL_PATH 中的路径。
MODEL_ROOT_PATH = "/root/autodl-tmp"

# 选用的 Embedding 名称
EMBEDDING_MODEL = "bge-large-en-v1.5"

# Embedding 模型运行设备。设为 "auto" 会自动检测(会有警告)，也可手动设定为 "cuda","mps","cpu","xpu" 其中之一。
EMBEDDING_DEVICE = "auto"

# The selected reranker model
RERANKER_MODEL = "bge-reranker-large"
# Whether reranker model is enabled or not
USE_RERANKER = True
RERANKER_MAX_LENGTH = 1024

# 如果需要在 EMBEDDING_MODEL 中增加自定义的关键字时配置
EMBEDDING_KEYWORD_FILE = "keywords.txt"
EMBEDDING_MODEL_OUTPUT_PATH = "output"

# 要运行的 LLM 名称，可以包括本地模型和在线模型。列表中本地模型将在启动项目时全部加载。
# 列表中第一个模型将作为 API 和 WEBUI 的默认模型。
# 在这里，我们使用目前主流的两个离线模型，其中，chatglm3-6b 为默认加载模型。
# 如果你的显存不足，可使用 Qwen-1_8B-Chat, 该模型 FP16 仅需 3.8G显存。

LLM_MODELS = ["Qwen1.5-7B-Chat", "zhipu-api", "openai-api"]
Agent_MODEL = None

# LLM 模型运行设备。设为"auto"会自动检测(会有警告)，也可手动设定为 "cuda","mps","cpu","xpu" 其中之一。
LLM_DEVICE = "cuda" # "auto"

HISTORY_LEN = 3

MAX_TOKENS = 2048

TEMPERATURE = 0.7

# ================= RAG-fusion配置 =================

# 是否启用RAG-fusion功能
ENABLE_RAG_FUSION = True

# RAG-fusion生成的查询数量（建议3-5个）
RAG_FUSION_QUERY_COUNT = 3

# RAG-fusion默认使用的LLM模型（优先使用本地快速模型）
RAG_FUSION_LLM_MODEL = "Qwen1.5-7B-Chat"

# RAG-fusion详细配置
RAG_FUSION_CONFIG = {
    # 查询生成配置
    "query_generation": {
        "max_queries": 5,           # 最多生成查询数量
        "min_queries": 2,           # 最少生成查询数量
        "max_tokens": 200,          # 生成时最大token数
        "temperature": 0.7,         # 生成温度
        "timeout": 30,              # 超时时间（秒）
        "retry_attempts": 2,        # 重试次数
        "query_diversity_threshold": 0.3,  # 查询多样性阈值
    },
    
    # 缓存配置
    "cache": {
        "enable": True,
        "max_cache_size": 1000,     # 最大缓存条目数
        "cache_ttl": 3600,          # 缓存生存时间（秒）
        "cleanup_interval": 300,     # 缓存清理间隔（秒）
    },
    
    # 融合配置
    "fusion": {
        "method": "rrf",            # 融合方法：rrf(Reciprocal Rank Fusion), weighted_sum
        "rrf_k": 60,               # RRF参数k值
        "normalization": True,      # 是否对分数进行标准化
        "top_k_per_query": 10,     # 每个查询返回的top-k结果
        "final_top_k": 20,         # 最终融合后返回的top-k结果
    },
    
    # 自适应配置
    "adaptive": {
        "enable": True,
        "query_length_threshold": 50,      # 查询长度阈值
        "complexity_keywords": [           # 复杂查询关键词
            "compare", "analyze", "relationship", "difference", 
            "why", "how", "explain", "详细", "分析", "比较", "关系"
        ],
        "simple_query_fallback": "vector", # 简单查询的回退模式
    },
    
    # 性能配置
    "performance": {
        "parallel_queries": True,   # 是否并行执行查询
        "max_concurrent": 3,        # 最大并发数
        "batch_size": 16,          # 批处理大小
    }
}

# RAG-fusion支持的模型列表（优先使用本地模型以降低延迟）
RAG_FUSION_SUPPORTED_MODELS = [
    # 本地模型（推荐用于RAG-fusion查询生成）
    "Qwen1.5-7B-Chat",
    "Qwen1.5-1.8B-Chat", 
    "Qwen1.5-4B-Chat",
    "chatglm3-6b",
    "chatglm2-6b",
    "baichuan2-7b-chat",
    "internlm2-chat-7b",
    "Llama-2-7b-chat-hf",
    
    # 在线模型（备选）
    "zhipu-api",
    "openai-api",
    "qwen-api",
    "minimax-api",
    "qianfan-api",
]

# RAG-fusion模型选择策略
RAG_FUSION_MODEL_SELECTION = {
    "prefer_local": True,           # 优先使用本地模型
    "fallback_to_online": True,     # 本地模型不可用时是否回退到在线模型
    "max_retry_attempts": 3,        # 最大重试次数
    "timeout_threshold": 30,        # 超时阈值（秒）
    
    # 模型性能配置（根据模型大小和性能调整）
    "model_performance": {
        "Qwen1.5-7B-Chat": {"speed": "fast", "quality": "high", "priority": 1},
        "Qwen1.5-4B-Chat": {"speed": "very_fast", "quality": "medium", "priority": 2},
        "Qwen1.5-1.8B-Chat": {"speed": "very_fast", "quality": "medium", "priority": 3},
        "chatglm3-6b": {"speed": "fast", "quality": "high", "priority": 4},
        "chatglm2-6b": {"speed": "fast", "quality": "medium", "priority": 5},
        "zhipu-api": {"speed": "medium", "quality": "very_high", "priority": 6},
        "openai-api": {"speed": "medium", "quality": "very_high", "priority": 7},
    }
}

# ================= 文本分片器配置 =================

# 新增分片器的详细配置
ENGLISH_SENTENCE_SPLITTER_CONFIG = {
    "chunk_size": 200,
    "chunk_overlap": 20,
    "separators": [". ", "! ", "? ", "\n\n", "\n", " "],
    "keep_separator": True,
    "length_function": len,
}

ENGLISH_PARAGRAPH_SPLITTER_CONFIG = {
    "chunk_size": 400,
    "chunk_overlap": 50,
    "separators": ["\n\n", "\n", ". ", "! ", "? ", " "],
    "keep_separator": True,
    "length_function": len,
}

SEMANTIC_CHUNK_SPLITTER_CONFIG = {
    "chunk_size": 300,
    "similarity_threshold": 0.7,
    "min_chunk_size": 50,
    "max_chunk_size": 500,
    "model_name": EMBEDDING_MODEL,
}

SLIDING_WINDOW_SPLITTER_CONFIG = {
    "window_size": 250,
    "step_size": 125,  # 50% overlap
    "min_window_size": 100,
}

# 分片器选择策略配置
TEXT_SPLITTER_SELECTION_CONFIG = {
    "file_type_mapping": {
        ".pdf": "RecursiveCharacterTextSplitter",
        ".md": "MarkdownHeaderTextSplitter", 
        ".markdown": "MarkdownHeaderTextSplitter",
        ".txt": "EnglishParagraphSplitter",
        ".docx": "EnglishParagraphSplitter",
        ".doc": "EnglishParagraphSplitter",
        ".html": "HTMLHeaderTextSplitter",
        ".htm": "HTMLHeaderTextSplitter",
    },
    
    "content_based_selection": {
        "enable": True,
        "large_document_threshold": 10000,     # 大文档阈值（字符数）
        "structured_keywords": ["##", "###", "---", "<h1>", "<h2>"],
        "code_keywords": ["```", "def ", "class ", "function", "import "],
    },
    
    "rag_fusion_optimization": {
        "preferred_splitters": [
            "SemanticChunkSplitter",
            "EnglishParagraphSplitter", 
            "RecursiveCharacterTextSplitter"
        ],
        "chunk_size_multiplier": 1.2,  # RAG-fusion适合稍大的chunk
    }
}

ONLINE_LLM_MODEL = {
    "openai-api": {
        "model_name": "gpt-4",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "",
        "openai_proxy": "",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "model_name": "gpt-3.5-turbo",  # RAG-fusion可以使用更便宜的模型
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },

    # 智谱AI API,具体注册及api key获取请前往 http://open.bigmodel.cn
    "zhipu-api": {
        "api_key": "",
        "version": "glm-4",
        "provider": "ChatGLMWorker",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "version": "glm-3-turbo",       # 使用更快的模型版本
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },

    # 具体注册及api key获取请前往 https://api.minimax.chat/
    "minimax-api": {
        "group_id": "",
        "api_key": "",
        "is_pro": False,
        "provider": "MiniMaxWorker",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },

    # 具体注册及api key获取请前往 https://xinghuo.xfyun.cn/
    "xinghuo-api": {
        "APPID": "",
        "APISecret": "",
        "api_key": "",
        "version": "v3.5", # 你使用的讯飞星火大模型版本，可选包括 "v3.5","v3.0", "v2.0", "v1.5"
        "provider": "XingHuoWorker",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "version": "v2.0",              # 使用更便宜的版本
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },

    # 百度千帆 API，申请方式请参考 https://cloud.baidu.com/doc/WENXINWORKSHOP/s/4lilb2lpf
    "qianfan-api": {
        "version": "ERNIE-Bot",  # 注意大小写。当前支持 "ERNIE-Bot" 或 "ERNIE-Bot-turbo"， 更多的见官方文档。
        "version_url": "",  # 也可以不填写version，直接填写在千帆申请模型发布的API地址
        "api_key": "",
        "secret_key": "",
        "provider": "QianFanWorker",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "version": "ERNIE-Bot-turbo",   # 使用更快的turbo版本
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },

    # 火山方舟 API，文档参考 https://www.volcengine.com/docs/82379
    "fangzhou-api": {
        "version": "chatglm-6b-model",
        "version_url": "",
        "api_key": "",
        "secret_key": "",
        "provider": "FangZhouWorker",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },

    # 阿里云通义千问 API，文档参考 https://help.aliyun.com/zh/dashscope/developer-reference/api-details
    "qwen-api": {
        "version": "qwen-max",
        "api_key": "",
        "provider": "QwenWorker",
        "embed_model": "text-embedding-v1",  # embedding 模型名称
        # RAG-fusion特殊配置
        "rag_fusion": {
            "version": "qwen-turbo",        # 使用更快的turbo版本
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },

    # 百川 API，申请方式请参考 https://www.baichuan-ai.com/home#api-enter
    "baichuan-api": {
        "version": "Baichuan2-53B",
        "api_key": "",
        "secret_key": "",
        "provider": "BaiChuanWorker",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "version": "Baichuan2-13B",     # 使用更小的模型版本
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },

    # Azure API
    "azure-api": {
        "deployment_name": "",  # 部署容器的名字
        "resource_name": "",  # https://{resource_name}.openai.azure.com/openai/ 填写resource_name的部分，其他部分不要填写
        "api_version": "",  # API的版本，不是模型版本
        "api_key": "",
        "provider": "AzureWorker",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "deployment_name": "",         # 可以指定专门用于RAG-fusion的部署
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },

    # 昆仑万维天工 API https://model-platform.tiangong.cn/
    "tiangong-api": {
        "version": "SkyChat-MegaVerse",
        "api_key": "",
        "secret_key": "",
        "provider": "TianGongWorker",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    },
    # Gemini API https://makersuite.google.com/app/apikey
    "gemini-api": {
        "api_key": "",
        "provider": "GeminiWorker",
        # RAG-fusion特殊配置
        "rag_fusion": {
            "model_name": "gemini-pro",
            "max_tokens": 200,
            "temperature": 0.7,
            "timeout": 30,
        }
    }

}

# 在以下字典中修改属性值，以指定本地embedding模型存储位置。支持3种设置方法：
# 1、将对应的值修改为模型绝对路径
# 2、不修改此处的值（以 text2vec 为例）：
#       2.1 如果{MODEL_ROOT_PATH}下存在如下任一子目录：
#           - text2vec
#           - GanymedeNil/text2vec-large-chinese
#           - text2vec-large-chinese
#       2.2 如果以上本地路径不存在，则使用huggingface模型

MODEL_PATH = {
    "embed_model": {
        "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
        "ernie-base": "nghuyong/ernie-3.0-base-zh",
        "text2vec-base": "shibing624/text2vec-base-chinese",
        "text2vec": "GanymedeNil/text2vec-large-chinese",
        "text2vec-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
        "text2vec-sentence": "shibing624/text2vec-base-chinese-sentence",
        "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
        "text2vec-bge-large-chinese": "shibing624/text2vec-bge-large-chinese",
        "m3e-small": "moka-ai/m3e-small",
        "m3e-base": "moka-ai/m3e-base",
        "m3e-large": "moka-ai/m3e-large",

        "bge-small-zh": "BAAI/bge-small-zh",
        "bge-base-zh": "BAAI/bge-base-zh",
        "bge-large-zh": "BAAI/bge-large-zh",
        "bge-large-zh-noinstruct": "BAAI/bge-large-zh-noinstruct",
        "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
        "bge-large-en-v1.5": "/root/autodl-tmp/bge-large-en-v1.5", # "BAAI/bge-large-zh-v1.5",

        "bge-m3": "BAAI/bge-m3",

        "piccolo-base-zh": "sensenova/piccolo-base-zh",
        "piccolo-large-zh": "sensenova/piccolo-large-zh",
        "nlp_gte_sentence-embedding_chinese-large": "damo/nlp_gte_sentence-embedding_chinese-large",
        "text-embedding-ada-002": "your OPENAI_API_KEY",
    },

    "llm_model": {
        "chatglm2-6b": "THUDM/chatglm2-6b",
        "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
        "chatglm3-6b": "THUDM/chatglm3-6b",
        "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",

        "Orion-14B-Chat": "OrionStarAI/Orion-14B-Chat",
        "Orion-14B-Chat-Plugin": "OrionStarAI/Orion-14B-Chat-Plugin",
        "Orion-14B-LongChat": "OrionStarAI/Orion-14B-LongChat",

        "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
        "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
        "Llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",

        "Qwen-1_8B-Chat": "Qwen/Qwen-1_8B-Chat",
        "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
        "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",
        "Qwen-72B-Chat": "Qwen/Qwen-72B-Chat",

        # Qwen1.5 模型 VLLM可能出现问题
        "Qwen1.5-0.5B-Chat": "Qwen/Qwen1.5-0.5B-Chat",
        "Qwen1.5-1.8B-Chat": "Qwen/Qwen1.5-1.8B-Chat",
        "Qwen1.5-4B-Chat": "Qwen/Qwen1.5-4B-Chat",
        "Qwen1.5-7B-Chat": "/root/autodl-tmp/Qwen1.5-7B-Chat",
        "Qwen1.5-14B-Chat": "Qwen/Qwen1.5-14B-Chat",
        "Qwen1.5-72B-Chat": "Qwen/Qwen1.5-72B-Chat",

        "baichuan-7b-chat": "baichuan-inc/Baichuan-7B-Chat",
        "baichuan-13b-chat": "baichuan-inc/Baichuan-13B-Chat",
        "baichuan2-7b-chat": "baichuan-inc/Baichuan2-7B-Chat",
        "baichuan2-13b-chat": "baichuan-inc/Baichuan2-13B-Chat",

        "internlm-7b": "internlm/internlm-7b",
        "internlm-chat-7b": "internlm/internlm-chat-7b",
        "internlm2-chat-7b": "internlm/internlm2-chat-7b",
        "internlm2-chat-20b": "internlm/internlm2-chat-20b",

        "BlueLM-7B-Chat": "vivo-ai/BlueLM-7B-Chat",
        "BlueLM-7B-Chat-32k": "vivo-ai/BlueLM-7B-Chat-32k",

        "Yi-34B-Chat": "https://huggingface.co/01-ai/Yi-34B-Chat",

        "agentlm-7b": "THUDM/agentlm-7b",
        "agentlm-13b": "THUDM/agentlm-13b",
        "agentlm-70b": "THUDM/agentlm-70b",

        "falcon-7b": "tiiuae/falcon-7b",
        "falcon-40b": "tiiuae/falcon-40b",
        "falcon-rw-7b": "tiiuae/falcon-rw-7b",

        "aquila-7b": "BAAI/Aquila-7B",
        "aquilachat-7b": "BAAI/AquilaChat-7B",
        "open_llama_13b": "openlm-research/open_llama_13b",
        "vicuna-13b-v1.5": "lmsys/vicuna-13b-v1.5",
        "koala": "young-geng/koala",
        "mpt-7b": "mosaicml/mpt-7b",
        "mpt-7b-storywriter": "mosaicml/mpt-7b-storywriter",
        "mpt-30b": "mosaicml/mpt-30b",
        "opt-66b": "facebook/opt-66b",
        "opt-iml-max-30b": "facebook/opt-iml-max-30b",
        "gpt2": "gpt2",
        "gpt2-xl": "gpt2-xl",
        "gpt-j-6b": "EleutherAI/gpt-j-6b",
        "gpt4all-j": "nomic-ai/gpt4all-j",
        "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
        "pythia-12b": "EleutherAI/pythia-12b",
        "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
        "dolly-v2-12b": "databricks/dolly-v2-12b",
        "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
    },

    "reranker": {
        "bge-reranker-large": "/root/autodl-tmp/bge-reranker-large", # "BAAI/bge-reranker-large",
        "bge-reranker-base": "BAAI/bge-reranker-base",
    }
}

# 通常情况下不需要更改以下内容

# nltk 模型存储路径
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

# 使用VLLM可能导致模型推理能力下降，无法完成Agent任务
VLLM_MODEL_DICT = {
    "chatglm2-6b": "THUDM/chatglm2-6b",
    "chatglm2-6b-32k": "THUDM/chatglm2-6b-32k",
    "chatglm3-6b": "THUDM/chatglm3-6b",
    "chatglm3-6b-32k": "THUDM/chatglm3-6b-32k",

    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf": "meta-llama/Llama-2-70b-chat-hf",

    "Qwen-1_8B-Chat": "Qwen/Qwen-1_8B-Chat",
    "Qwen-7B-Chat": "Qwen/Qwen-7B-Chat",
    "Qwen-14B-Chat": "Qwen/Qwen-14B-Chat",
    "Qwen-72B-Chat": "Qwen/Qwen-72B-Chat",

    "baichuan-7b-chat": "baichuan-inc/Baichuan-7B-Chat",
    "baichuan-13b-chat": "baichuan-inc/Baichuan-13B-Chat",
    "baichuan2-7b-chat": "baichuan-inc/Baichuan-7B-Chat",
    "baichuan2-13b-chat": "baichuan-inc/Baichuan-13B-Chat",

    "BlueLM-7B-Chat": "vivo-ai/BlueLM-7B-Chat",
    "BlueLM-7B-Chat-32k": "vivo-ai/BlueLM-7B-Chat-32k",

    "internlm-7b": "internlm/internlm-7b",
    "internlm-chat-7b": "internlm/internlm-chat-7b",
    "internlm2-chat-7b": "internlm/Models/internlm2-chat-7b",
    "internlm2-chat-20b": "internlm/Models/internlm2-chat-20b",

    "aquila-7b": "BAAI/Aquila-7B",
    "aquilachat-7b": "BAAI/AquilaChat-7B",

    "falcon-7b": "tiiuae/falcon-7b",
    "falcon-40b": "tiiuae/falcon-40b",
    "falcon-rw-7b": "tiiuae/falcon-rw-7b",
    "gpt2": "gpt2",
    "gpt2-xl": "gpt2-xl",
    "gpt-j-6b": "EleutherAI/gpt-j-6b",
    "gpt4all-j": "nomic-ai/gpt4all-j",
    "gpt-neox-20b": "EleutherAI/gpt-neox-20b",
    "pythia-12b": "EleutherAI/pythia-12b",
    "oasst-sft-4-pythia-12b-epoch-3.5": "OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
    "dolly-v2-12b": "databricks/dolly-v2-12b",
    "stablelm-tuned-alpha-7b": "stabilityai/stablelm-tuned-alpha-7b",
    "open_llama_13b": "openlm-research/open_llama_13b",
    "vicuna-13b-v1.3": "lmsys/vicuna-13b-v1.3",
    "koala": "young-geng/koala",
    "mpt-7b": "mosaicml/mpt-7b",
    "mpt-7b-storywriter": "mosaicml/mpt-7b-storywriter",
    "mpt-30b": "mosaicml/mpt-30b",
    "opt-66b": "facebook/opt-66b",
    "opt-iml-max-30b": "facebook/opt-iml-max-30b",

}

# 支持Agent的模型列表（RAG-fusion不一定需要支持Agent的模型）
SUPPORT_AGENT_MODEL = [
    "openai-api",  # GPT4 模型
    "qwen-api",  # Qwen Max模型
    "zhipu-api",  # 智谱AI GLM4模型
    "Qwen",  # 所有Qwen系列本地模型
    "chatglm3-6b",
    "internlm2-chat-20b",
    "Orion-14B-Chat-Plugin",
]

# ================= RAG-fusion模型验证函数 =================

def is_model_supported_for_rag_fusion(model_name: str) -> bool:
    """
    检查模型是否支持RAG-fusion功能
    
    Args:
        model_name: 模型名称
        
    Returns:
        bool: 是否支持RAG-fusion
    """
    return model_name in RAG_FUSION_SUPPORTED_MODELS

def get_optimal_rag_fusion_model() -> str:
    """
    获取RAG-fusion的最优模型
    
    Returns:
        str: 推荐的模型名称
    """
    # 优先返回本地快速模型
    for model in RAG_FUSION_SUPPORTED_MODELS:
        if model in LLM_MODELS:  # 确保模型在当前配置的模型列表中
            return model
    
    # 如果没有本地模型，返回第一个支持的模型
    return RAG_FUSION_SUPPORTED_MODELS[0] if RAG_FUSION_SUPPORTED_MODELS else "Qwen1.5-7B-Chat"

def get_rag_fusion_model_config(model_name: str) -> dict:
    """
    获取指定模型的RAG-fusion配置
    
    Args:
        model_name: 模型名称
        
    Returns:
        dict: 模型的RAG-fusion配置
    """
    if model_name in ONLINE_LLM_MODEL and "rag_fusion" in ONLINE_LLM_MODEL[model_name]:
        return ONLINE_LLM_MODEL[model_name]["rag_fusion"]
    
    # 本地模型的默认配置
    return {
        "max_tokens": 200,
        "temperature": 0.7,
        "timeout": 30,
    }

def validate_rag_fusion_config() -> bool:
    """
    验证RAG-fusion配置的完整性和有效性
    
    Returns:
        bool: 配置是否有效
    """
    if not ENABLE_RAG_FUSION:
        return True
        
    try:
        # 检查必要配置项
        assert RAG_FUSION_CONFIG is not None, "RAG_FUSION_CONFIG不能为None"
        assert RAG_FUSION_LLM_MODEL in RAG_FUSION_SUPPORTED_MODELS, f"默认模型{RAG_FUSION_LLM_MODEL}不在支持列表中"
        assert 1 <= RAG_FUSION_QUERY_COUNT <= 10, "查询数量应在1-10之间"
        
        # 检查配置结构
        required_keys = ["query_generation", "cache", "fusion", "performance"]
        for key in required_keys:
            assert key in RAG_FUSION_CONFIG, f"缺少必要配置项: {key}"
        
        return True
        
    except Exception as e:
        print(f"RAG-fusion配置验证失败: {e}")
        return False

# 初始化时验证配置
if ENABLE_RAG_FUSION and not validate_rag_fusion_config():
    print("警告: RAG-fusion配置验证失败，将禁用RAG-fusion功能")
    ENABLE_RAG_FUSION = False
