import nltk
import sys
import os

# 这行代码用于修改Python的模块搜索路径（sys.path）
# __file__是当前脚本的路径，os.path.dirname(__file__)得到当前脚本所在的目录，
# 再次使用os.path.dirname()则得到父目录的路径。
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from configs import VERSION
from configs.model_config import NLTK_DATA_PATH
# 用于配置服务器是否允许跨域请求
from configs.server_config import OPEN_CROSS_DOMAIN
import argparse
# 导入uvicorn，一个ASGI服务器，用于运行FastAPI应用。
import uvicorn
# Body用于定义接收请求体中数据的参数
from fastapi import Body
# 用于处理跨源资源共享（CORS），允许或限制跨站点的请求。
from fastapi.middleware.cors import CORSMiddleware
# 用于生成重定向响应的工具
from starlette.responses import RedirectResponse
from server.chat.chat import chat
from server.chat.search_engine_chat import search_engine_chat
from server.chat.completion import completion
from server.chat.feedback import chat_feedback
from server.embeddings_api import embed_texts_endpoint
from server.llm_api import (list_running_models, list_config_models,
                            change_llm_model, stop_llm_model,
                            get_model_config, list_search_engines)
from server.utils import (BaseResponse, ListResponse, FastAPI, MakeFastAPIOffline,
                          get_server_configs, get_prompt_template)
from typing import List, Literal
# 设置了nltk数据的搜索路径
nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path


async def document():
    # 函数返回一个重定向响应（RedirectResponse），客户端访问这个端点时会被重定向到/docs路径。
    # FastAPI默认提供/docs路径用于自动生成的API文档，所以这个函数可能是用于将根路径或某个特定路径重定向到API文档。
    return RedirectResponse(url="/docs")


def create_app(run_mode: str = None):
    ''''''
    # 创建了一个FastAPI应用实例，设置了应用的标题和版本号，其中VERSION是从配置文件导入的变量。
    app = FastAPI(
        title="Langchain-Chatchat API Server",
        version=VERSION
    )
    MakeFastAPIOffline(app)
    # 如果是，就给应用添加CORS中间件（CORSMiddleware），配置为允许所有来源（allow_origins=["*"]）、
    # 允许凭证（allow_credentials=True）、允许所有方法（allow_methods=["*"]）和
    # 允许所有头部（allow_headers=["*"]），实现跨域资源共享。
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app, run_mode=run_mode)
    return app


def mount_app_routes(app: FastAPI, run_mode: str = None):
    '''用于配置和挂载路由，接收一个FastAPI应用实例作为第一个参数，可选的运行模式字符串作为第二个参数。'''
    # 返回swagger文档的响应
    app.get("/",
            response_model=BaseResponse,
            summary="swagger 文档")(document)

    # Tag: Chat
    app.post("/chat/chat",
             tags=["Chat"],
             summary="与llm模型对话(通过LLMChain)",
             )(chat)

    app.post("/chat/search_engine_chat",
             tags=["Chat"],
             summary="与搜索引擎对话",
             )(search_engine_chat)

    app.post("/chat/feedback",
             tags=["Chat"],
             summary="返回llm模型对话评分",
             )(chat_feedback)

    # 挂载：知识库相关接口
    mount_knowledge_routes(app)
    # 挂载：摘要相关接口
    mount_filename_summary_routes(app)

    # LLM模型相关接口
    app.post("/llm_model/list_running_models",
             tags=["LLM Model Management"],
             summary="列出当前已加载的模型",
             )(list_running_models)

    app.post("/llm_model/list_config_models",
             tags=["LLM Model Management"],
             summary="列出configs已配置的模型",
             )(list_config_models)

    app.post("/llm_model/get_model_config",
             tags=["LLM Model Management"],
             summary="获取模型配置（合并后）",
             )(get_model_config)

    app.post("/llm_model/stop",
             tags=["LLM Model Management"],
             summary="停止指定的LLM模型（Model Worker)",
             )(stop_llm_model)

    app.post("/llm_model/change",
             tags=["LLM Model Management"],
             summary="切换指定的LLM模型（Model Worker)",
             )(change_llm_model)

    # 服务器相关接口
    app.post("/server/configs",
             tags=["Server State"],
             summary="获取服务器原始配置信息",
             )(get_server_configs)

    app.post("/server/list_search_engines",
             tags=["Server State"],
             summary="获取服务器支持的搜索引擎",
             )(list_search_engines)


    # 获取服务器配置的prompt模板，通过type和name两个参数来指定所需的模板类型和名称。
    @app.post("/server/get_prompt_template",
             tags=["Server State"],
             summary="获取服务区配置的 prompt 模板")
    def get_server_prompt_template(
        type: Literal["llm_chat", "knowledge_base_chat", "search_engine_chat", "agent_chat"]=Body("llm_chat", description="模板类型，可选值：llm_chat，knowledge_base_chat，search_engine_chat，agent_chat"),
        name: str = Body("default", description="模板名称"),
    ) -> str:
        return get_prompt_template(type=type, name=name)

    # 其它接口
    app.post("/other/completion",
             tags=["Other"],
             summary="要求llm模型补全(通过LLMChain)",
             )(completion)

    app.post("/other/embed_texts",
            tags=["Other"],
            summary="将文本向量化，支持本地模型和在线模型",
            )(embed_texts_endpoint)


def mount_knowledge_routes(app: FastAPI):
    '''用于在FastAPI应用中挂载与知识库管理相关的路由'''
    from server.chat.knowledge_base_chat import knowledge_base_chat
    from server.chat.file_chat import upload_temp_docs, file_chat
    from server.chat.agent_chat import agent_chat
    from server.knowledge_base.kb_api import list_kbs, create_kb, delete_kb
    from server.knowledge_base.kb_doc_api import (list_files, upload_docs, delete_docs,
                                                update_docs, download_doc, recreate_vector_store,
                                                search_docs, DocumentWithVSId, update_info,
                                                update_docs_by_id,)

    app.post("/chat/knowledge_base_chat",
             tags=["Chat"],
             summary="与知识库对话")(knowledge_base_chat)

    app.post("/chat/file_chat",
             tags=["Knowledge Base Management"],
             summary="文件对话"
             )(file_chat)

    app.post("/chat/agent_chat",
             tags=["Chat"],
             summary="与agent对话")(agent_chat)

    # Tag: Knowledge Base Management
    app.get("/knowledge_base/list_knowledge_bases",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="获取知识库列表")(list_kbs)

    app.post("/knowledge_base/create_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="创建知识库"
             )(create_kb)

    app.post("/knowledge_base/delete_knowledge_base",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="删除知识库"
             )(delete_kb)

    app.get("/knowledge_base/list_files",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="获取知识库内的文件列表"
            )(list_files)

    app.post("/knowledge_base/search_docs",
             tags=["Knowledge Base Management"],
             response_model=List[DocumentWithVSId],
             summary="搜索知识库"
             )(search_docs)

    app.post("/knowledge_base/update_docs_by_id",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="直接更新知识库文档"
             )(update_docs_by_id)


    app.post("/knowledge_base/upload_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="上传文件到知识库，并/或进行向量化"
             )(upload_docs)

    app.post("/knowledge_base/delete_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="删除知识库内指定文件"
             )(delete_docs)

    app.post("/knowledge_base/update_info",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="更新知识库介绍"
             )(update_info)
    app.post("/knowledge_base/update_docs",
             tags=["Knowledge Base Management"],
             response_model=BaseResponse,
             summary="更新现有文件到知识库"
             )(update_docs)

    app.get("/knowledge_base/download_doc",
            tags=["Knowledge Base Management"],
            summary="下载对应的知识文件")(download_doc)

    app.post("/knowledge_base/recreate_vector_store",
             tags=["Knowledge Base Management"],
             summary="根据content中文档重建向量库，流式输出处理进度。"
             )(recreate_vector_store)

    app.post("/knowledge_base/upload_temp_docs",
             tags=["Knowledge Base Management"],
             summary="上传文件到临时目录，用于文件对话。"
             )(upload_temp_docs)


def mount_filename_summary_routes(app: FastAPI):
    '''配置和挂载与文件摘要相关的路由'''
    from server.knowledge_base.kb_summary_api import (summary_file_to_vector_store, recreate_summary_vector_store,
                                                      summary_doc_ids_to_vector_store)

    app.post("/knowledge_base/kb_summary_api/summary_file_to_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="单个知识库根据文件名称摘要"
             )(summary_file_to_vector_store)
    app.post("/knowledge_base/kb_summary_api/summary_doc_ids_to_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="单个知识库根据doc_ids摘要",
             response_model=BaseResponse,
             )(summary_doc_ids_to_vector_store)
    app.post("/knowledge_base/kb_summary_api/recreate_summary_vector_store",
             tags=["Knowledge kb_summary_api Management"],
             summary="重建单个知识库文件摘要"
             )(recreate_summary_vector_store)

'''
ssl_keyfile：该文件包含了用于SSL加密的私钥。在建立一个HTTPS连接时，服务器需要这个私钥来完成与客户端的安全握手过程。

ssl_certfile：该文件包含了由证书颁发机构（CA）签发的SSL证书。这个证书包含了公钥（与私钥配对使用）和服务器的身份信息。
              客户端使用这个证书中的公钥来加密与服务器的通信，确保信息安全，并且可以验证服务器的身份，防止中间人攻击。
              
使用SSL证书和密钥文件可以使Web服务支持HTTPS协议，而HTTPS协议是HTTP的安全版本，
它在客户端和服务器之间的通信过程中提供数据加密、完整性校验和身份验证。
'''

def run_api(host, port, **kwargs):
    # 检查是否传入了SSL密钥文件和证书文件的路径
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        # uvicorn.run是Uvicorn库的函数，用于启动ASGI应用。
        # 在这种情况下，启动的服务将支持HTTPS连接。
        # Uvicorn是一个轻量级、高性能的ASGI服务器，适用于异步Web应用。
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        # 如果没有提供SSL密钥文件和证书文件，则简单地启动一个不使用SSL的HTTP服务。
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='langchain-ChatGLM',
                                     description='About langchain-ChatGLM, local knowledge based ChatGLM with langchain'
                                                 ' ｜ 基于本地知识库的 ChatGLM 问答')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args() # 解析命令行输入的参数
    args_dict = vars(args) # 将args（一个命名空间对象）转换为字典。这样做是为了方便访问命令行参数的值。

    app = create_app()

    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
