from langchain.docstore.document import Document
from configs import EMBEDDING_MODEL, logger
from server.model_workers.base import ApiEmbeddingsParams
from server.utils import BaseResponse, get_model_worker_config, list_embed_models, list_online_embed_models
from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from typing import Dict, List
# 获取当前可用的在线嵌入模型列表
online_embed_models = list_online_embed_models()


def embed_texts(
        texts: List[str], # 文本列表
        embed_model: str = EMBEDDING_MODEL, # 嵌入模型
        to_query: bool = False, # 是否用于查询
) -> BaseResponse:
    '''
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    '''
    try:
        # 使用本地Embeddings模型
        if embed_model in list_embed_models():
            from server.utils import load_local_embeddings

            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=embeddings.embed_documents(texts))

        # 使用在线API
        if embed_model in list_online_embed_models():
            config = get_model_worker_config(embed_model)
            worker_class = config.get("worker_class")
            embed_model = config.get("embed_model")
            worker = worker_class()
            if worker_class.can_embedding():
                params = ApiEmbeddingsParams(texts=texts, to_query=to_query, embed_model=embed_model)
                resp = worker.do_embeddings(params)
                return BaseResponse(**resp)

        return BaseResponse(code=500, msg=f"指定的模型 {embed_model} 不支持 Embeddings 功能。")
    except Exception as e:
        # 如果指定的模型既不在本地列表也不在在线列表中，返回一个错误信息。
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")


async def aembed_texts(
    texts: List[str],
    embed_model: str = EMBEDDING_MODEL,
    to_query: bool = False,
) -> BaseResponse:
    ''' 在异步环境中执行，不会阻塞事件循环。
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    '''
    try:
        if embed_model in list_embed_models(): # 使用本地Embeddings模型
            from server.utils import load_local_embeddings

            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=await embeddings.aembed_documents(texts))

        if embed_model in list_online_embed_models(): # 使用在线API
            # run_in_threadpool是FastAPI提供的一个工具，它允许将同步代码在异步函数中运行，而不会阻塞事件循环。
            return await run_in_threadpool(embed_texts,
                                           texts=texts,
                                           embed_model=embed_model,
                                           to_query=to_query)
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")


def embed_texts_endpoint(
        texts: List[str] = Body(..., description="要嵌入的文本列表", examples=[["hello", "world"]]),
        embed_model: str = Body(EMBEDDING_MODEL,
                                description=f"使用的嵌入模型，除了本地部署的Embedding模型，也支持在线API({online_embed_models})提供的嵌入服务。"),
        to_query: bool = Body(False, description="向量是否用于查询。有些模型如Minimax对存储/查询的向量进行了区分优化。"),
) -> BaseResponse:
    ''' 这个端点的主要作用是提供一个HTTP接口，使得客户端可以通过发送HTTP请求来进行文本向量化。'''
    return embed_texts(texts=texts, embed_model=embed_model, to_query=to_query)


def embed_documents(
        docs: List[Document],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> Dict:
    """ 用于处理一系列文档对象的向量化，其中文档对象是之前导入的Document类的实例。

    将 List[Document] 向量化，转化为 VectorStore.add_embeddings 可以接受的参数
    """
    # 从每个Document对象中提取文本内容（page_content）和元数据（metadata）。
    texts = [x.page_content for x in docs]
    metadatas = [x.metadata for x in docs]
    # 将这些文本向量化
    embeddings = embed_texts(texts=texts, embed_model=embed_model, to_query=to_query).data
    if embeddings is not None:
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
