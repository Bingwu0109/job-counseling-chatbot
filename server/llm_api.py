from fastapi import Body
from configs import logger, log_verbose, LLM_MODELS, HTTPX_DEFAULT_TIMEOUT
from server.utils import (BaseResponse, fschat_controller_address, list_config_llm_models,
                          get_httpx_client, get_model_worker_config)
from typing import List


def list_running_models(
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="该参数未使用，占位用"),
) -> BaseResponse:
    '''
    从fastchat controller获取已加载模型列表及其配置项
    '''
    try:
        # 获取模型列表和配置的操作
        controller_address = controller_address or fschat_controller_address()
        # 获取一个HTTP客户端（基于httpx库）
        with get_httpx_client() as client:
            # 获取正在运行的模型列表
            r = client.post(controller_address + "/list_models")
            # 模型的列表
            models = r.json()["models"]
            # 获取模型的配置，然后构建一个字典data，其键为模型名称，值为模型配置的data属性。
            data = {m: get_model_config(m).data for m in models}
            return BaseResponse(data=data)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get available models from controller: {controller_address}。错误信息是： {e}")


def list_config_models(
    types: List[str] = Body(["local", "online"], description="模型配置项类别，如local, online, worker"),
    placeholder: str = Body(None, description="占位用，无实际效果")
) -> BaseResponse:
    '''
    从本地获取configs中配置的模型列表
    '''
    data = {}
    # 返回的是一个字典，其键是模型的类型（如local或online），值是对应类型的模型名称列表。
    for type, models in list_config_llm_models().items():
        if type in types:
            data[type] = {m: get_model_config(m).data for m in models}
    return BaseResponse(data=data)


def get_model_config(
    model_name: str = Body(description="配置中LLM模型的名称"),
    placeholder: str = Body(None, description="占位用，无实际效果")
) -> BaseResponse:
    '''
    获取某个特定LLM模型的配置项（合并后的）
    '''
    config = {}
    # 删除ONLINE_MODEL配置中的敏感信息
    for k, v in get_model_worker_config(model_name=model_name).items():
        # 删除包含敏感信息的配置项
        if not (k == "worker_class"
            or "key" in k.lower()
            or "secret" in k.lower()
            or k.lower().endswith("id")):
            config[k] = v

    return BaseResponse(data=config)


def stop_llm_model(
    model_name: str = Body(..., description="要停止的LLM模型名称", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()])
) -> BaseResponse:
    '''
    向fastchat controller请求停止某个LLM模型。
    注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        # 获取一个httpx客户端实例
        with get_httpx_client() as client:
            # 发送一个POST请求到controller_address指定的地址上，路径为/release_worker。
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name},
            )
            # 如果请求成功，函数将返回请求响应的JSON数据。
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"failed to stop LLM model {model_name} from controller: {controller_address}。错误信息是： {e}")


def change_llm_model(
    model_name: str = Body(..., description="当前运行模型", examples=[LLM_MODELS[0]]),
    new_model_name: str = Body(..., description="要切换的新模型", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()])
):
    '''
    向fastchat controller请求切换LLM模型。
    '''
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            # 发送一个POST请求到controller_address指定的地址，请求路径为/release_worker。
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
            )
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"failed to switch LLM model from controller: {controller_address}。错误信息是： {e}")


def list_search_engines() -> BaseResponse:
    '''列出服务器上配置的搜索引擎列表'''
    from server.chat.search_engine_chat import SEARCH_ENGINES

    return BaseResponse(data=list(SEARCH_ENGINES))
