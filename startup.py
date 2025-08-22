# 用于编写单线程并发代码
import asyncio
# 提供了一个本地和远程并发执行的能力，支持子进程、通信和共享数据等。
import multiprocessing as mp
import os
# 用于产生新进程，连接到它们的输入/输出/错误管道，并获取它们的返回码。
import subprocess
import sys
from multiprocessing import Process
from datetime import datetime
from pprint import pprint
# 一个装饰器，用于标记旧的API函数或方法已不推荐使用。
from langchain_core._api import deprecated


try:
    # numexpr是一个快速数值表达式求值器，它可以利用多核心进行优化。
    import numexpr
    # 检测系统中的CPU核心数，并将这个数量设置为环境变量NUMEXPR_MAX_THREADS
    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass



sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs import (
    LOG_PATH,
    log_verbose,
    logger,
    LLM_MODELS,
    EMBEDDING_MODEL,
    TEXT_SPLITTER_NAME,
    FSCHAT_CONTROLLER,
    FSCHAT_OPENAI_API,
    FSCHAT_MODEL_WORKERS,
    API_SERVER,
    WEBUI_SERVER,
    HTTPX_DEFAULT_TIMEOUT,
)
from server.utils import (fschat_controller_address, fschat_model_worker_address,
                          fschat_openai_api_address, get_httpx_client, get_model_worker_config,
                          MakeFastAPIOffline, FastAPI, llm_device, embedding_device)
from server.knowledge_base.migrate import create_tables
import argparse
from typing import List, Dict
from configs import VERSION


@deprecated(
    since="0.3.0", # 自从哪个版本开始过时
    message="模型启动功能将于 Langchain-Chatchat 0.3.x重写,支持更多模式和加速启动，0.2.x中相关功能将废弃",
    removal="0.3.0") # 何时将完全移除
def create_controller_app(
        dispatch_method: str, # 分派方法
        log_level: str = "INFO", # 日志级别
) -> FastAPI:
    '''这个函数的目的是创建和配置一个FastAPI应用，包括设置日志级别、创建和注册控制器实例'''
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH  # 日志文件的存放路径
    # 导入app（一个FastAPI应用实例）、Controller（一个控制器类）和logger（一个日志记录器）
    from fastchat.serve.controller import app, Controller, logger
    # 设置了日志记录器的日志级别
    logger.setLevel(log_level)
    # Controller是负责处理请求和分发任务的核心类
    controller = Controller(dispatch_method)
    # 何从该模块导入controller的代码都将获取到这个新创建的实例
    sys.modules["fastchat.serve.controller"].controller = controller
    # 调用了一个名为MakeFastAPIOffline的函数，将app（FastAPI应用实例）设置为离线模式，
    # 即在没有网络连接的情况下也能运行或进行某些特定的配置。
    MakeFastAPIOffline(app)
    # 设置了FastAPI应用的标题
    app.title = "FastChat Controller"
    # FastAPI应用就可以通过这个属性访问控制器实例
    app._controller = controller
    return app


def create_model_worker_app(log_level: str = "INFO", **kwargs) -> FastAPI:
    """
    创建并配置一个模型工作进程应用，主要用于处理各种模型的服务请求。

    kwargs包含的字段如下：
    host:  # 主机地址
    port: # 端口
    model_names:[`model_name`] # 模型名称
    controller_address: # 控制器地址
    worker_address:

    对于Langchain支持的模型：
        langchain_model:True
        不会使用fschat

    对于online_api:
        online_api:True
        worker_class: `provider`

    对于离线模型：
        model_path: `model_name_or_path`,huggingface的repo-id或本地路径
        device:`LLM_DEVICE`
    """
    import fastchat.constants
    # 日志目录的路径
    fastchat.constants.LOGDIR = LOG_PATH
    import argparse
    # parser : 解析命令行参数
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    # 遍历传给函数的所有关键字参数
    for k, v in kwargs.items():
        # 将每个关键字参数设置为 args 对象的属性
        setattr(args, k, v)
    if worker_class := kwargs.get("langchain_model"):  # Langchian支持的模型不用做操作
        from fastchat.serve.base_model_worker import app
        worker = ""
    elif worker_class := kwargs.get("worker_class"):
        # 在线模型API
        from fastchat.serve.base_model_worker import app
        # 创建了一个新的worker实例
        worker = worker_class(model_names=args.model_names, # 模型名称
                              controller_addr=args.controller_address, # 控制器地址
                              worker_addr=args.worker_address) # 工作进程地址
        # sys.modules["fastchat.serve.base_model_worker"].worker = worker
        # 根据需要调整日志的详细程度
        sys.modules["fastchat.serve.base_model_worker"].logger.setLevel(log_level)
    else:
        # 本地模型
        from configs.model_config import VLLM_MODEL_DICT
        if kwargs["model_names"][0] in VLLM_MODEL_DICT and args.infer_turbo == "vllm":
            import fastchat.serve.vllm_worker
            from fastchat.serve.vllm_worker import VLLMWorker, app, worker_id
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs

            args.tokenizer = args.model_path
            args.tokenizer_mode = 'auto'
            args.trust_remote_code = True
            args.download_dir = None
            args.load_format = 'auto'
            args.dtype = 'auto'
            args.seed = 0
            args.worker_use_ray = False
            args.pipeline_parallel_size = 1
            args.tensor_parallel_size = 1
            args.block_size = 16
            args.swap_space = 4  # GiB
            args.gpu_memory_utilization = 0.90
            args.max_num_batched_tokens = None  # 一个批次中的最大令牌（tokens）数量，这个取决于你的显卡和大模型设置，设置太大显存会不够
            args.max_num_seqs = 256
            args.disable_log_stats = False
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.no_register = False
            args.num_gpus = 1  # vllm worker的切分是tensor并行，这里填写显卡的数量
            args.engine_use_ray = False
            args.disable_log_requests = False

            # 0.2.1 vllm后要加的参数, 但是这里不需要
            args.max_model_len = None
            args.revision = None
            args.quantization = None
            args.max_log_len = None
            args.tokenizer_revision = None

            # 0.2.2 vllm需要新加的参数
            args.max_paddings = 256

            if args.model_path:
                args.model = args.model_path
            if args.num_gpus > 1:
                args.tensor_parallel_size = args.num_gpus

            for k, v in kwargs.items():
                setattr(args, k, v)

            # 创建了一个用于处理模型推理的异步引擎
            engine_args = AsyncEngineArgs.from_cli_args(args)
            engine = AsyncLLMEngine.from_engine_args(engine_args)

            worker = VLLMWorker(
                controller_addr=args.controller_address,
                worker_addr=args.worker_address,
                worker_id=worker_id,
                model_path=args.model_path,
                model_names=args.model_names,
                limit_worker_concurrency=args.limit_worker_concurrency,
                no_register=args.no_register,
                llm_engine=engine,
                conv_template=args.conv_template,
            )

            sys.modules["fastchat.serve.vllm_worker"].engine = engine
            sys.modules["fastchat.serve.vllm_worker"].worker = worker
            sys.modules["fastchat.serve.vllm_worker"].logger.setLevel(log_level)

        else:
            from fastchat.serve.model_worker import app, GptqConfig, AWQConfig, ModelWorker, worker_id
            # 处理GPU设置
            args.gpus = "0"  # GPU的编号,如果有多个GPU，可以设置为"0,1,2,3"
            args.max_gpu_memory = "22GiB"
            args.num_gpus = 1  # model worker的切分是model并行，这里填写显卡的数量
            # 量化配置
            args.load_8bit = False
            args.cpu_offloading = None
            args.gptq_ckpt = None
            args.gptq_wbits = 16
            args.gptq_groupsize = -1
            args.gptq_act_order = False
            args.awq_ckpt = None
            args.awq_wbits = 16
            args.awq_groupsize = -1
            args.model_names = [""]
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.stream_interval = 2
            args.no_register = False
            args.embed_in_truncate = False
            for k, v in kwargs.items():
                setattr(args, k, v)
            if args.gpus:
                if args.num_gpus is None:
                    args.num_gpus = len(args.gpus.split(','))
                if len(args.gpus.split(",")) < args.num_gpus:
                    raise ValueError(
                        f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                    )
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            gptq_config = GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            )
            awq_config = AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            )
            # 创建ModelWorker实例
            worker = ModelWorker(
                controller_addr=args.controller_address,
                worker_addr=args.worker_address,
                worker_id=worker_id,
                model_path=args.model_path,
                model_names=args.model_names,
                limit_worker_concurrency=args.limit_worker_concurrency,
                no_register=args.no_register,
                device=args.device,
                num_gpus=args.num_gpus,
                max_gpu_memory=args.max_gpu_memory,
                load_8bit=args.load_8bit,
                cpu_offloading=args.cpu_offloading,
                gptq_config=gptq_config,
                awq_config=awq_config,
                stream_interval=args.stream_interval,
                conv_template=args.conv_template,
                embed_in_truncate=args.embed_in_truncate,
            )
            # 设置模块级变量
            sys.modules["fastchat.serve.model_worker"].args = args
            sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
            # sys.modules["fastchat.serve.model_worker"].worker = worker
            sys.modules["fastchat.serve.model_worker"].logger.setLevel(log_level)

    MakeFastAPIOffline(app)
    app.title = f"FastChat LLM Server ({args.model_names[0]})"
    app._worker = worker
    return app


def create_openai_api_app(
        controller_address: str,
        api_keys: List = [],
        log_level: str = "INFO",
) -> FastAPI:
    '''负责创建和配置一个用于提供OpenAI API服务的FastAPI应用'''
    import fastchat.constants
    # 设置日志目录的路径
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings
    from fastchat.utils import build_logger
    # 创建一个日志器（logger），并设置其名称为openai_api。日志文件命名为openai_api.log。
    logger = build_logger("openai_api", "openai_api.log")
    # 设置日志级别
    logger.setLevel(log_level)
    # 通过app.add_middleware方法向FastAPI应用添加CORSMiddleware中间件，配置允许的跨域请求。这包括：
    # 允许携带证书的请求（allow_credentials=True）。
    # 允许来自任何源的请求（allow_origins=["*"]）。
    # 允许任何HTTP方法（allow_methods=["*"]）。
    # 允许任何HTTP头（allow_headers=["*"]）。
    # 这样的配置使得API能够被不同的客户端安全地访问。
    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sys.modules["fastchat.serve.openai_api_server"].logger = logger
    app_settings.controller_address = controller_address
    app_settings.api_keys = api_keys
    # 配置FastAPI应用
    MakeFastAPIOffline(app)
    app.title = "FastChat OpeanAI API Server"
    return app


def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    '''
    app是一个FastAPI的应用实例，started_event是一个多进程（multiprocessing）库中的Event实例，用于跨进程的事件通知。
    '''
    # 注册一个在应用启动时自动执行的异步函数on_startup
    @app.on_event("startup")
    async def on_startup():
        # 是否存在一个跨进程的通知需求
        if started_event is not None:
            # 如果不为None，意味着提供了一个多进程事件对象，函数将调用started_event.set()方法。
            # 这个方法的调用将事件状态设置为“已发生”（set），这可以用来通知其他进程应用已经启动完成。
            started_event.set()


def run_controller(log_level: str = "INFO", started_event: mp.Event = None):
    '''
       主要目的是启动和运行一个控制器应用，该应用是基于FastAPI构建的异步Web应用。
       这个函数通过结合使用ASGI服务器、异步HTTP客户端库以及自定义的应用创建和事件设置逻辑，
       实现了一个灵活且高效的控制器服务启动流程。
    '''
    import uvicorn
    # httpx库，这是一个异步HTTP客户端库，用于发送HTTP请求。
    import httpx
    # Body，通常用于FastAPI应用中定义请求体的数据模型。
    from fastapi import Body
    import time
    import sys
    # 配置 httpx 客户端，可能包括超时设置、代理配置等。
    from server.utils import set_httpx_config
    set_httpx_config()
    # 创建控制器应用的FastAPI实例
    app = create_controller_app(
        dispatch_method=FSCHAT_CONTROLLER.get("dispatch_method"),
        log_level=log_level,
    )
    # 为创建的控制器应用设置启动事件，如果提供了started_event（一个多进程 Event 实例），
    # 这将允许控制器应用在启动时通知其他进程。
    _set_app_event(app, started_event)

    # add interface to release and load model worker
    # 装饰器，用于将下面定义的函数注册为处理POST请求的接口，路径为/release_worker。
    @app.post("/release_worker")
    def release_worker(
            # 这个参数需要提供一个字符串，指定要释放的模型名称。Body(...)中的...表示这个参数是必需的。
            model_name: str = Body(..., description="要释放模型的名称", samples=["chatglm-6b"]),
            # worker_address: str = Body(None, description="要释放模型的地址，与名称二选一", samples=[FSCHAT_CONTROLLER_address()]),
            # 允许调用者指定一个新模型的名称，该模型将在释放当前模型后被加载。
            new_model_name: str = Body(None, description="释放后加载该模型"),
            # 指示是否保留原有模型。如果为True，则即使请求中指定了新模型，原模型也不会被释放。
            keep_origin: bool = Body(False, description="不释放原模型，加载新模型")
    ) -> Dict:
        # 获取当前可用的模型列表
        available_models = app._controller.list_models()

        # 如果请求中指定的新模型已经在可用模型列表中，则记录一条信息日志，
        # 并返回一个状态码为500的错误响应，指出新模型已经存在。
        if new_model_name in available_models:
            msg = f"要切换的LLM模型 {new_model_name} 已经存在"
            logger.info(msg)
            return {"code": 500, "msg": msg}

        # 根据是否提供了new_model_name，分别记录不同的日志信息，表明是切换模型还是仅仅停止当前模型。
        if new_model_name:
            logger.info(f"开始切换LLM模型：从 {model_name} 到 {new_model_name}")
        else:
            logger.info(f"即将停止LLM模型： {model_name}")

        # 如果请求中指定的要释放的模型不在可用模型列表中，记录一条错误日志，
        # 并返回一个状态码为500的错误响应，指出请求中的模型不可用。
        if model_name not in available_models:
            msg = f"the model {model_name} is not available"
            logger.error(msg)
            return {"code": 500, "msg": msg}
        # 获取指定模型名称对应的模型工作者地址
        worker_address = app._controller.get_worker_address(model_name)

        # 如果无法获取模型工作者的地址（即worker_address为空），则记录一条错误日志，
        # 并返回一个状态码为500的错误响应，说明找不到指定模型名称对应的模型工作者地址。
        if not worker_address:
            msg = f"can not find model_worker address for {model_name}"
            logger.error(msg)
            return {"code": 500, "msg": msg}

        # 创建一个HTTP客户端实例，用于发送网络请求
        with get_httpx_client() as client:
            # 使用创建的HTTP客户端向模型工作者发送一个POST请求
            r = client.post(worker_address + "/release",
                            json={"new_model_name": new_model_name, "keep_origin": keep_origin})
            # 使用创建的HTTP客户端向模型工作者发送一个POST请求
            if r.status_code != 200:
                msg = f"failed to release model: {model_name}"
                logger.error(msg)
                return {"code": 500, "msg": msg}

        #  如果new_model_name不为空，说明请求中指定了释放当前模型后要加载的新模型。
        if new_model_name:
            # 设置一个计时器
            timer = HTTPX_DEFAULT_TIMEOUT  # wait for new model_worker register
            # 在超时时间内循环
            while timer > 0:
                # 获取当前可用的模型列表
                models = app._controller.list_models()
                # 如果新模型名称出现在模型列表中，使用break跳出循环，表示新模型已成功注册。
                if new_model_name in models:
                    break
                time.sleep(1)
                timer -= 1

            # 循环结束后，通过检查timer的值判断是否在超时时间内成功等待到新模型的注册。
            # 如果timer大于0，说明在超时之前新模型已注册。
            if timer > 0:
                msg = f"sucess change model from {model_name} to {new_model_name}"
                logger.info(msg)
                return {"code": 200, "msg": msg}
            else:
                msg = f"failed change model from {model_name} to {new_model_name}"
                logger.error(msg)
                return {"code": 500, "msg": msg}
        else:
            msg = f"sucess to release model: {model_name}"
            logger.info(msg)
            return {"code": 200, "msg": msg}

    # 获取主机地址
    host = FSCHAT_CONTROLLER["host"]
    # 端口号
    port = FSCHAT_CONTROLLER["port"]

    if log_level == "ERROR":
        # 如果是，那么将sys.stdout和sys.stderr重定向回它们的原始系统定义。
        # 目的是在日志级别为"ERROR"时，确保所有的标准输出和错误输出都能正确地显示在控制台上，
        # 而不是被重定向或者过滤掉。
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    # 使用uvicorn.run函数来启动FastAPI应用
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def run_model_worker(
        model_name: str = LLM_MODELS[0], # 模型名称
        controller_address: str = "", # 控制器的地址
        log_level: str = "INFO", # 日志级别
        q: mp.Queue = None, # 一个multiprocessing.Queue对象，默认为None，用于进程间的消息传递。
        started_event: mp.Event = None, # 一个multiprocessing.Event对象，默认为None，用于通知其他进程工作器已经启动。
):
    '''启动一个基于FastAPI和uvicorn的模型工作器（model worker）'''
    import uvicorn
    from fastapi import Body
    import sys
    from server.utils import set_httpx_config
    # 配置HTTP客户端
    set_httpx_config()
    # 获取模型工作器的配置
    kwargs = get_model_worker_config(model_name)
    # 主机地址
    host = kwargs.pop("host")
    # 端口号
    port = kwargs.pop("port")
    kwargs["model_names"] = [model_name]
    kwargs["controller_address"] = controller_address or fschat_controller_address()
    kwargs["worker_address"] = fschat_model_worker_address(model_name)
    model_path = kwargs.get("model_path", "")
    kwargs["model_path"] = model_path
    # 创建一个FastAPI应用实例
    app = create_model_worker_app(log_level=log_level, **kwargs)
    # 在FastAPI应用实例上设置某个事件，标记应用已启动的事件。
    _set_app_event(app, started_event)
    if log_level == "ERROR":
        # 将sys.stdout和sys.stderr设置回它们的原始系统定义
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # add interface to release and load model
    @app.post("/release")
    def release_model(
            new_model_name: str = Body(None, description="释放后加载该模型"),
            keep_origin: bool = Body(False, description="不释放原模型，加载新模型")
    ) -> Dict:
        # 如果keep_origin为True，并且new_model_name被指定，
        # 那么将向q（一个多进程队列）发送一个包含当前模型名称、字符串"start"和新模型名称的列表。
        if keep_origin:
            if new_model_name:
                q.put([model_name, "start", new_model_name])
        else:
            # 如果指定了new_model_name，则向q发送一个包含当前模型名称、字符串"replace"和新模型名称的列表，表示替换当前模型为新模型。
            if new_model_name:
                q.put([model_name, "replace", new_model_name])
            else:
                # 如果没有指定new_model_name，则向q发送一个包含当前模型名称和字符串"stop"的列表，表示停止当前模型。
                q.put([model_name, "stop", None])
        return {"code": 200, "msg": "done"}
    # 使用uvicorn启动了之前创建的FastAPI应用app
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def run_openai_api(log_level: str = "INFO", started_event: mp.Event = None):
    '''启动一个模拟OpenAI API的服务，该服务基于FastAPI框架和uvicorn服务器。'''
    import uvicorn
    import sys
    from server.utils import set_httpx_config
    # 设置httpx客户端的配置，这可能包括超时时间、代理设置等，以优化网络请求的性能。
    set_httpx_config()
    # 获取控制器的地址，这个地址用于与模型控制器进行通信，以实现对模型的动态管理。
    controller_addr = fschat_controller_address()
    # 调用函数创建一个模拟OpenAI API的FastAPI应用实例
    app = create_openai_api_app(controller_addr, log_level=log_level)
    # 在应用启动时设置一个事件，这通常用于在服务启动后进行一些初始化操作或通知其他进程。
    _set_app_event(app, started_event)
    # 获取服务应该监听的主机地址和端口号
    host = FSCHAT_OPENAI_API["host"]
    port = FSCHAT_OPENAI_API["port"]
    if log_level == "ERROR":
        # 如果日志级别设置为"ERROR"，则将sys.stdout和sys.stderr设置回它们的原始系统定义，
        # 这样做的目的是在仅关注错误信息的日志级别下，确保标准输出和错误能够被正确地显示。
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    # 启动服务，监听指定的主机地址和端口，等待处理请求。
    uvicorn.run(app, host=host, port=port)


def run_api_server(started_event: mp.Event = None, run_mode: str = None):
    '''主要作用是启动一个API服务器，这个服务器基于FastAPI框架和uvicorn服务器。'''
    from server.api import create_app
    # uvicorn用于运行ASGI应用的服务器，支持异步Python网络框架。
    import uvicorn
    from server.utils import set_httpx_config
    # 配置httpx客户端，这可能涉及设置请求超时时间、最大并发数等，以优化HTTP请求的处理。
    set_httpx_config()
    # 创建一个FastAPI应用实例
    app = create_app(run_mode=run_mode)
    # 在应用启动时设置一个事件，这可以用于通知其他进程API服务器已经启动或进行一些初始化操作。
    _set_app_event(app, started_event)
    # 网络地址和端口号
    host = API_SERVER["host"]
    port = API_SERVER["port"]
    # 启动FastAPI应用
    uvicorn.run(app, host=host, port=port)


def run_webui(started_event: mp.Event = None, run_mode: str = None):
    '''启动一个Web用户界面（UI），这个UI基于Streamlit框架构建，用于提供一个交互式的界面给用户。'''
    from server.utils import set_httpx_config
    # 配置HTTP客户端，包括设置超时时间、代理等配置，以优化与后端服务的网络请求。
    set_httpx_config()

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]
    # 构造一个命令行参数列表cmd，包含了运行Streamlit应用所需的所有命令行参数，如服务器的地址、端口号、主题颜色等。
    cmd = ["streamlit", "run", "webui.py",
           "--server.address", host,
           "--server.port", str(port),
           "--theme.base", "light",
           "--theme.primaryColor", "#165dff",
           "--theme.secondaryBackgroundColor", "#f5f5f5",
           "--theme.textColor", "#000000",
           ]
    # 在"lite"模式下以不同的方式运行
    if run_mode == "lite":
        cmd += [
            "--",
            "lite",
        ]
    # 启动一个子进程来运行Streamlit应用
    p = subprocess.Popen(cmd)
    # 调用started_event.set()标记启动事件。如果started_event被提供，
    # 这表明UI已经启动，可以通知其他等待该事件的进程继续运行。
    started_event.set()
    # 等待子进程结束，这意味着run_webui函数将阻塞，直到Streamlit应用被关闭。
    p.wait()


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # 程序会运行fastchat的controller/openai_api/model_worker服务器，并运行api.py和webui.py。
    parser.add_argument(
        "-a",
        "--all-webui",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py and webui.py",
        dest="all_webui",
    )
    # 只运行api.py，不包括webui.py
    parser.add_argument(
        "--all-api",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py",
        dest="all_api",
    )
    # 运行fastchat的controller/openai_api/model_worker服务器
    parser.add_argument(
        "--llm-api",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers",
        dest="llm_api",
    )
    # 运行fastchat的controller和openai_api服务器
    parser.add_argument(
        "-o",
        "--openai-api",
        action="store_true",
        help="run fastchat's controller/openai_api servers",
        dest="openai_api",
    )
    # 运行指定模型名的model_worker服务器。如果需要使用非默认的LLM模型，可以指定--model-name。
    parser.add_argument(
        "-m",
        "--model-worker",
        action="store_true",
        help="run fastchat's model_worker server with specified model name. "
             "specify --model-name if not using default LLM_MODELS",
        dest="model_worker",
    )
    # 指定model_worker使用的模型名称。如果需要启动多个model_worker，可以空格分隔模型名称。
    parser.add_argument(
        "-n",
        "--model-name",
        type=str,
        nargs="+",
        default=LLM_MODELS,
        help="specify model name for model worker. "
             "add addition names with space seperated to start multiple model workers.",
        dest="model_name",
    )
    # 指定worker注册到的控制器地址。默认是FSCHAT_CONTROLLER。
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        help="specify controller address the worker is registered to. default is FSCHAT_CONTROLLER",
        dest="controller_address",
    )
    # 运行api.py服务器
    parser.add_argument(
        "--api",
        action="store_true",
        help="run api.py server",
        dest="api",
    )
    # 运行在线模型API，例如zhipuai。
    parser.add_argument(
        "-p",
        "--api-worker",
        action="store_true",
        help="run online model api such as zhipuai",
        dest="api_worker",
    )
    # 运行webui.py服务器
    parser.add_argument(
        "-w",
        "--webui",
        action="store_true",
        help="run webui.py server",
        dest="webui",
    )
    # 减少fastchat服务的日志输出
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="减少fastchat服务log信息",
        dest="quiet",
    )
    # 以Lite模式运行，仅支持在线API的LLM对话和搜索引擎对话。
    parser.add_argument(
        "-i",
        "--lite",
        action="store_true",
        help="以Lite模式运行：仅支持在线API的LLM对话、搜索引擎对话",
        dest="lite",
    )
    args = parser.parse_args()
    return args, parser


def dump_server_info(after_start=False, args=None):
    '''在启动服务器时打印出服务器配置和环境信息'''

    # 获取操作系统的信息
    import platform
    import langchain
    import fastchat
    from server.utils import api_address, webui_address

    print("\n")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print(f"操作系统：{platform.platform()}.")
    print(f"python版本：{sys.version}")
    print(f"项目版本：{VERSION}")
    print(f"langchain版本：{langchain.__version__}. fastchat版本：{fastchat.__version__}")
    print("\n")

    models = LLM_MODELS
    if args and args.model_name:
        models = args.model_name

    print(f"当前使用的分词器：{TEXT_SPLITTER_NAME}")
    print(f"当前启动的LLM模型：{models} @ {llm_device()}")

    for model in models:
        pprint(get_model_worker_config(model))
    print(f"当前Embbedings模型： {EMBEDDING_MODEL} @ {embedding_device()}")

    # 打印服务端运行信息
    if after_start:
        print("\n")
        print(f"服务端运行信息：")
        if args.openai_api:
            print(f"    OpenAI API Server: {fschat_openai_api_address()}")
        if args.api:
            print(f"    Chatchat  API  Server: {api_address()}")
        if args.webui:
            print(f"    Chatchat WEBUI Server: {webui_address()}")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print("\n")


async def start_main_server():
    import time
    import signal

    # >>> 第1步：信号处理设置

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            # 闭包函数，抛出异常
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # 第1步：信号处理设置
    # This will be inherited by the child process if it is forked (not spawned)
    # 注册一个信号处理器，用于处理SIGINT信号（通常是由Ctrl+C触发的中断信号）
    signal.signal(signal.SIGINT, handler("SIGINT"))
    # 注册了一个SIGTERM信号的处理器（SIGTERM通常用于请求程序终止
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    # >>> 第2步：多进程启动模式设置

    # 设置了多进程的启动方式为“spawn”，默认情况下，在Unix/Linux上使用的是"fork"，
    # 而"spawn"方式则会启动一个全新的Python解释器进程。
    mp.set_start_method("spawn")


    # >>> 第3步：多进程管理器初始化

    # 创建了一个多进程管理器
    manager = mp.Manager()
    run_mode = None
    # 使用之前创建的多进程管理器manager来创建一个多进程安全的队列。
    # manager.Queue()方法返回一个跨进程的队列实例，这个队列可以被多个进程共享，用于进程间的通信。
    queue = manager.Queue()
    args, parser = parse_args()


    # >>> 第4步：解析命令行参数

    if args.all_webui:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = True

    elif args.all_api:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = False

    elif args.llm_api:
        args.openai_api = True
        args.model_worker = True
        args.api_worker = True
        args.api = False
        args.webui = False

    if args.lite:
        args.model_worker = False
        run_mode = "lite"

    dump_server_info(args=args)

    if len(sys.argv) > 1:
        logger.info(f"正在启动服务：")
        logger.info(f"如需查看 llm_api 日志，请前往 {LOG_PATH}")


    # >>> 第5步：进程字典和事件对象的创建

    processes = {"online_api": {}, "model_worker": {}}

    def process_count():
        '''计算当前运行的进程数量'''
        return len(processes) + len(processes["online_api"]) + len(processes["model_worker"]) - 2

    # 设置日志级别
    if args.quiet or not log_verbose:
        log_level = "ERROR"
    else:
        log_level = "INFO"
    # 创建了一个跨进程的事件对象，Event对象用于跨进程间的同步，
    # 可以设置（set）和清除（clear）状态，并检查事件是否已被设置（is_set）。
    controller_started = manager.Event()


    # >>> 第6步：根据参数配置启动进程

    if args.openai_api:
        # 创建了一个进程用于运行控制器
        process = Process(
            target=run_controller, # 进程启动时执行的函数
            name=f"controller", # 进程名称
            # log_level控制日志输出级别，started_event是一个事件对象。
            kwargs=dict(log_level=log_level, started_event=controller_started),
            daemon=True, # 将这个进程设置为守护进程，守护进程是随主进程退出而退出的。
        )
        processes["controller"] = process

        # 新的进程用于运行OpenAI API相关的服务
        process = Process(
            target=run_openai_api,
            name=f"openai_api",
            daemon=True,
        )
        processes["openai_api"] = process

    model_worker_started = []

    # 根据提供的参数启动模型工作器（model worker）进程
    if args.model_worker:
        # 遍历用户指定的模型名称列表
        for model_name in args.model_name:
            # 为每个模型名获取配置信息
            config = get_model_worker_config(model_name)
            # 判断是否需要启动一个本地模型工作器进程，而不是连接到一个在线API。
            if not config.get("online_api"):
                # 创建一个新的事件对象，允许在不同进程间共享状态。
                e = manager.Event()
                # 将事件对象添加到一个列表中
                model_worker_started.append(e)
                # 创建一个新的进程对象
                process = Process(
                    target=run_model_worker, # 指定进程启动时调用的函数
                    name=f"model_worker - {model_name}", # 为进程指定一个名称
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True, # 设置进程为守护进程
                )
                # 将新创建的进程对象保存在一个嵌套字典中
                processes["model_worker"][model_name] = process

    # 是否指定了需要启动API工作器
    if args.api_worker:
        # 遍历用户指定的模型名称列表
        for model_name in args.model_name:
            # 获取每个模型名称对应的配置信息
            config = get_model_worker_config(model_name)
            if (config.get("online_api")
                    and config.get("worker_class")
                    and model_name in FSCHAT_MODEL_WORKERS):
                # 创建一个新的事件对象
                e = manager.Event()
                # 将新创建的事件对象添加到列表中
                model_worker_started.append(e)
                # 创建Process对象以启动新的进程
                process = Process(
                    target=run_model_worker, # 指定启动进程时要调用的函数
                    name=f"api_worker - {model_name}", # 为进程指定一个名称
                    kwargs=dict(model_name=model_name, # 向run_model_worker函数传递的参数
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True, # 设置进程为守护进程
                )
                # 将新创建的进程对象存储在一个嵌套字典中，用于管理和追踪所有启动的进程。
                processes["online_api"][model_name] = process

    # 使用multiprocessing.Manager()创建的manager对象来初始化一个新的事件对象api_started
    api_started = manager.Event()

    if args.api:
        # 创建Process对象来启动API服务器的进程
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(started_event=api_started, run_mode=run_mode),
            daemon=True,
        )
        # 将创建的进程对象存储在processes字典中，使用"api"作为键
        processes["api"] = process

    # 使用multiprocessing.Manager()创建的manager对象来初始化另一个新的事件对象webui_started。
    webui_started = manager.Event()

    if args.webui:
        process = Process(
            target=run_webui, # 指定启动进程时要调用的函数
            name=f"WEBUI Server",
            kwargs=dict(started_event=webui_started, run_mode=run_mode),
            daemon=True, # 设置进程为守护进程
        )
        processes["webui"] = process # 将新创建的进程对象保存在processes字典中，使用"webui"作为键。

    # 检查是否有任何进程需要启动
    if process_count() == 0:
        # 如果返回值为0，意味着没有配置任何进程。
        parser.print_help()
    else:
        # 如果有一个或多个进程需要启动，进入到启动进程的逻辑。
        try:
            # 获取控制器进程
            if p := processes.get("controller"):
                # 如果存在，则启动该进程，并更新其名称以包括进程ID，
                # 然后等待controller_started事件，这意味着控制器启动完成。
                p.start()
                p.name = f"{p.name} ({p.pid})"
                controller_started.wait()  # 等待controller启动完成

            # 获取并启动openai_api进程，如果配置了该进程。
            if p := processes.get("openai_api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"

            # 遍历processes字典中的"model_worker"键对应的所有进程，并启动每个进程。
            # 这些进程负责运行模型工作器。
            for n, p in processes.get("model_worker", {}).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            # 遍历并启动"online_api"相关的所有进程。这些进程通过在线API与模型交互。
            for n, p in processes.get("online_api", []).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            # 对于所有已启动的模型工作器进程，使用e.wait()在model_worker_started列表中等待每个进程的启动事件。
            # 这确保在继续之前，所有模型工作器都已成功启动。
            for e in model_worker_started:
                e.wait()

            # 尝试获取并启动API服务器进程，如果配置了该进程。
            if p := processes.get("api"):
                # 启动后，等待api_started事件，表示API服务器已启动完成。
                p.start()
                p.name = f"{p.name} ({p.pid})"
                api_started.wait()

            # 尝试获取并启动WebUI服务器进程，如果配置了该进程。
            if p := processes.get("webui"):
                # 启动后，等待webui_started事件，表示WebUI服务器已启动完成。
                p.start()
                p.name = f"{p.name} ({p.pid})"
                webui_started.wait()

            # 在所有相关进程成功启动后，调用dump_server_info函数来记录或显示服务器的信息。
            dump_server_info(after_start=True, args=args)


            # >>> 第7步：进程间通信与管理

            # 用于持续监听队列中的命令
            while True:
                # 从queue中获取一个命令，这里的queue是一个多进程队列（multiprocessing.Queue），可以跨不同进程安全地传递消息。
                cmd = queue.get()
                # 创建一个新的事件对象，用于跨进程通信。
                e = manager.Event()
                # 检查获取的命令是否为列表类型
                if isinstance(cmd, list):
                    # 解包命令列表，包含三个元素的列表，分别是原模型名称、命令类型（如"start"）和新模型名称。
                    model_name, cmd, new_model_name = cmd
                    # 检查命令是否为"start"，即是否需要启动一个新的模型工作器进程。
                    if cmd == "start":
                        logger.info(f"准备启动新模型进程：{new_model_name}")
                        # 创建并启动一个新的Process对象
                        process = Process(
                            target=run_model_worker, # 指定进程启动时调用的函数，负责启动模型工作器服务的逻辑。
                            name=f"model_worker - {new_model_name}", # 为进程指定一个名称
                            # 向run_model_worker函数传递的关键字参数，包括新模型名称、控制器地址、日志级别、消息队列和启动事件。
                            kwargs=dict(model_name=new_model_name,
                                        controller_address=args.controller_address,
                                        log_level=log_level,
                                        q=queue,
                                        started_event=e),
                            daemon=True, # 设置进程为守护进程，确保主进程结束时，子进程也会被自动终止。
                        )
                        # 启动进程
                        process.start()
                        # 更新进程名称，加入进程ID，以便于后续管理和识别。
                        process.name = f"{process.name} ({process.pid})"
                        # 将新启动的进程对象保存在processes字典中，以便于管理和跟踪。
                        processes["model_worker"][new_model_name] = process
                        # 等待新启动的模型工作器进程的启动事件。这个事件在进程成功启动并准备就绪时被设置。
                        e.wait()
                        logger.info(f"成功启动新模型进程：{new_model_name}")
                    elif cmd == "stop":
                        # 检查之前从队列中获取的命令是否为"stop"，即是否需要停止某个模型工作器进程。
                        # 从processes字典中获取与model_name相对应的模型工作器进程。
                        if process := processes["model_worker"].get(model_name):
                            # 在尝试终止进程之前，先暂停执行1秒。
                            time.sleep(1)
                            # 调用进程的terminate方法来请求终止进程。
                            process.terminate()
                            # 等待进程实际终止，join方法会阻塞当前执行的线程，直到被终止的进程实际退出。
                            process.join()
                            logger.info(f"停止模型进程：{model_name}")
                        else:
                            logger.error(f"未找到模型进程：{model_name}")
                    elif cmd == "replace":
                        # 检查命令是否为"replace"，即需要替换一个模型工作器进程。
                        # 从存储所有模型工作器进程的字典processes["model_worker"]中弹出（即移除并返回）与model_name对应的进程。
                        if process := processes["model_worker"].pop(model_name, None):
                            logger.info(f"停止模型进程：{model_name}")
                            # 记录当前时间，用于后续计算停止旧进程并启动新进程所需的总时间。
                            start_time = datetime.now()
                            time.sleep(1)
                            # 请求终止旧的模型工作器进程，并等待其实际终止，释放相关资源。
                            process.terminate()
                            process.join()
                            # 创建并启动一个新的Process对象用于新模型工作器
                            process = Process(
                                target=run_model_worker, # 函数负责启动和管理模型工作器的逻辑。
                                name=f"model_worker - {new_model_name}", # 为进程指定名称
                                kwargs=dict(model_name=new_model_name,
                                            controller_address=args.controller_address,
                                            log_level=log_level,
                                            q=queue,
                                            started_event=e),
                                daemon=True,
                            )
                            # 启动新的模型工作器进程
                            process.start()
                            # 更新进程名称，加入进程ID。
                            process.name = f"{process.name} ({process.pid})"
                            # 将新的模型工作器进程添加到processes["model_worker"]字典中，使用新模型名称作为键。
                            processes["model_worker"][new_model_name] = process
                            # 等待新模型工作器进程的启动事件，确保新进程已经准备就绪。
                            e.wait()
                            # 计算从停止旧进程到新进程准备就绪的总用时。
                            timing = datetime.now() - start_time
                            logger.info(f"成功启动新模型进程：{new_model_name}。用时：{timing}。")
                        else:
                            # 表示未能找到要替换的模型工作器进程
                            logger.error(f"未找到模型进程：{model_name}")

            # for process in processes.get("model_worker", {}).values():
            #     process.join()
            # for process in processes.get("online_api", {}).values():
            #     process.join()

            # for name, process in processes.items():
            #     if name not in ["model_worker", "online_api"]:
            #         if isinstance(p, dict):
            #             for work_process in p.values():
            #                 work_process.join()
            #         else:
            #             process.join()
        except Exception as e:
            logger.error(e)
            logger.warning("Caught KeyboardInterrupt! Setting stop event...")
        finally:

            # >>> 第8步：异常处理与清理

            # 不管是否发生异常，finally块的代码都将执行。
            # 遍历processes字典中的所有值
            for p in processes.values():
                logger.warning("Sending SIGKILL to %s", p)
                # 判断对于每个包含进程字典的项，遍历并调用每个进程的kill()方法来强制终止。
                if isinstance(p, dict):
                    for process in p.values():
                        # kill()方法会向进程发送SIGKILL信号，这是强制终止进程的最后手段。
                        process.kill()
                else:
                    # 如果p不是字典类型，则直接调用p.kill()来终止进程。
                    p.kill()
            # 遍历所有进程
            for p in processes.values():
                # 记录每个进程的状态信息
                logger.info("Process status: %s", p)


if __name__ == "__main__":
    # 调用一个函数来创建数据库表格
    create_tables()
    # 检查Python的版本信息，判断当前Python的版本是否低于3.10
    if sys.version_info < (3, 10):
        # 获取当前线程的事件循环，在Python 3.10之前，这是获取或创建事件循环的标准方式。
        loop = asyncio.get_event_loop()
    else:
        try:
            # 获取当前正在运行的事件循环
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 创建一个新的事件循环
            loop = asyncio.new_event_loop()
        # 设置当前线程的事件循环为新创建的事件循环
        asyncio.set_event_loop(loop)
    # 使用事件循环运行start_main_server协程，直到该协程运行完成。
    # start_main_server是启动主服务器的异步函数，它会启动一个web服务器，监听来自客户端的连接请求等。
    loop.run_until_complete(start_main_server())

# 服务启动后接口调用示例：
# import openai
# openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://localhost:8888/v1"

# model = "chatglm3-6b"

# # create a chat completion
# completion = openai.ChatCompletion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)
