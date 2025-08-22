from fastchat.conversation import Conversation
from configs import LOG_PATH, TEMPERATURE
import fastchat.constants
fastchat.constants.LOGDIR = LOG_PATH
from fastchat.serve.base_model_worker import BaseModelWorker
# 导入uuid模块，通常用于生成唯一的标识符，可能在创建会话或用户标识时使用。
import uuid
import json
import sys
# pydantic用于数据验证和设置管理，BaseModel是创建数据模型的基类，
# root_validator用于在整个模型上执行自定义验证。
from pydantic import BaseModel, root_validator
import fastchat
# Python的异步I/O框架，用于编写并发代码
import asyncio
from server.utils import get_model_worker_config
from typing import Dict, List, Optional

# 设置了模块级别的__all__变量，这个列表包含了模块导出的公共名称。
# 当从这个模块中使用from module import *时，只有这个列表中的名称会被导入。
__all__ = ["ApiModelWorker", "ApiChatParams", "ApiCompletionParams", "ApiEmbeddingsParams"]


class ApiConfigParams(BaseModel):
    '''
    在线API配置参数，未提供的值会自动从model_config.ONLINE_LLM_MODEL中读取
    '''
    api_base_url: Optional[str] = None
    api_proxy: Optional[str] = None
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    group_id: Optional[str] = None # for minimax
    is_pro: bool = False # for minimax

    APPID: Optional[str] = None # for xinghuo
    APISecret: Optional[str] = None # for xinghuo
    is_v2: bool = False # for xinghuo

    worker_name: Optional[str] = None

    class Config:
        extra = "allow"

    # pydantic提供的一个装饰器，用于在模型实例化之前或之后执行验证。
    # 这里使用pre=True参数，意味着这个验证器会在模型的字段赋值之前运行。
    @root_validator(pre=True)
    # cls是指向类本身的引用，v是一个字典，包含了尝试创建或更新模型实例时提供的所有字段。
    def validate_config(cls, v: Dict) -> Dict:
        # 根据worker_name从提供的字典v中获取配置。
        # 如果worker_name对应的配置存在，则将其赋值给config变量。
        if config := get_model_worker_config(v.get("worker_name")):
            # 遍历类的所有字段
            for n in cls.__fields__:
                # 如果某个字段在config中有定义，则将这个值更新到v字典中，
                # 这样在模型实例化时，这些字段就会使用从配置中获取到的值。
                if n in config:
                    v[n] = config[n]
        return v

    def load_config(self, worker_name: str):
        '''用于根据提供的worker_name加载和更新模型的配置'''
        self.worker_name = worker_name
        # 根据worker_name获取对应的配置
        if config := get_model_worker_config(worker_name):
            # 遍历模型的所有字段
            for n in self.__fields__:
                # 如果模型的字段在获取到的配置中存在，则使用setattr函数更新实例的相应字段。
                if n in config:
                    setattr(self, n, config[n])
        return self


class ApiModelParams(ApiConfigParams):
    '''
    模型配置参数
    '''
    version: Optional[str] = None # 模型的版本
    version_url: Optional[str] = None # 模型版本对应的URL地址
    api_version: Optional[str] = None #  API版本
    deployment_name: Optional[str] = None # 部署名称
    resource_name: Optional[str] = None # 资源名称

    temperature: float = TEMPERATURE
    max_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0


class ApiChatParams(ApiModelParams):
    '''
    chat请求参数
    '''
    # 列表中的每个元素都是一个字典，字典的键和值都是字符串类型。这个属性用于存储聊天会话中的消息。
    # 每个字典代表一条消息，其中可能包含消息内容、发送者标识、时间戳等信息。
    messages: List[Dict[str, str]]
    # 系统消息
    system_message: Optional[str] = None # for minimax
    # 角色
    role_meta: Dict = {} # for minimax


class ApiCompletionParams(ApiModelParams):
    '''用于处理完成（completion）请求的参数，只有一个prompt字段。'''
    prompt: str


class ApiEmbeddingsParams(ApiConfigParams):
    '''于处理嵌入（embeddings）请求的参数，如文本列表、嵌入模型和查询标志。'''
    texts: List[str]
    embed_model: Optional[str] = None
    to_query: bool = False # for minimax


class ApiModelWorker(BaseModelWorker):
    '''定义了如何初始化模型工作者、处理各种API请求（如chat、completion、embeddings）、生成响应等。'''

    # 初始值为None，表示默认情况下不支持嵌入模型。
    DEFAULT_EMBED_MODEL: str = None # None means not support embedding

    def __init__(
        self,
        model_names: List[str], # 模型名称列表
        controller_addr: str = None, # 控制器地址
        worker_addr: str = None, # 工作单元地址
        context_len: int = 2048,  # 上下文长度
        no_register: bool = False, # 是否在控制器中注册这个工作单元
        **kwargs,
    ):
        print("### ApiModelWorker ###")
        # 设置默认值
        # 用uuid.uuid4().hex[:8]生成的，提供了一个默认的、简短的、随机的工作单元标识符。
        kwargs.setdefault("worker_id", uuid.uuid4().hex[:8])
        kwargs.setdefault("model_path", "")
        # 可以同时处理的任务数量，以防止资源过度使用。
        kwargs.setdefault("limit_worker_concurrency", 5)
        super().__init__(model_names=model_names,
                        controller_addr=controller_addr,
                        worker_addr=worker_addr,
                        **kwargs)
        import fastchat.serve.base_model_worker
        import sys
        self.logger = fastchat.serve.base_model_worker.logger
        # 恢复被fastchat覆盖的标准输出
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        # 创建一个新的事件循环，并将其设置为当前线程的事件循环。
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        # 上下文长度值
        self.context_len = context_len
        # 初始化一个asyncio.Semaphore对象，并使用limit_worker_concurrency作为并发限制，
        # 用于控制同时处理的任务数量。
        self.semaphore = asyncio.Semaphore(self.limit_worker_concurrency)
        # 工作单元的版本
        self.version = None

        if not no_register and self.controller_addr:
            # 初始化心跳机制，这可能用于定期向控制器报告工作单元的状态。
            self.init_heart_beat()


    def count_token(self, params):
        '''计算其长度（即字符数）'''
        prompt = params["prompt"]
        return {"count": len(str(prompt)), "error_code": 0}

    def generate_stream_gate(self, params: Dict):
        '''生成任务'''
        # 追踪generate_stream_gate方法被调用的次数
        self.call_ct += 1

        try:
            prompt = params["prompt"]
            # 判断prompt是否符合聊天条件
            if self._is_chat(prompt):
                # 如果是，将prompt转换为消息列表（通过prompt_to_messages），
                # 然后对这些消息进行验证（通过validate_messages）。
                messages = self.prompt_to_messages(prompt)
                messages = self.validate_messages(messages)
            else: # 使用chat模仿续写功能，不支持历史消息
                # 如果不是聊天条件，构造一个单消息列表，模仿续写功能，消息内容提示用户从prompt指定的文本开始续写。
                messages = [{"role": self.user_role, "content": f"please continue writing from here: {prompt}"}]
            # 创建一个ApiChatParams实例
            p = ApiChatParams(
                messages=messages,
                temperature=params.get("temperature"),
                top_p=params.get("top_p"),
                max_tokens=params.get("max_new_tokens"),
                version=self.version,
            )
            # 调用do_chat方法处理聊天请求，并逐个将结果通过_jsonify方法转换为JSON格式后返回（使用yield实现生成器）。
            for resp in self.do_chat(p):
                yield self._jsonify(resp)
        except Exception as e:
            # 如果在执行过程中发生异常，方法将捕获异常并返回一个包含错误码（500）和错误信息的JSON对象。
            yield self._jsonify({"error_code": 500, "text": f"{self.model_names[0]}请求API时发生错误：{e}"})

    def generate_gate(self, params):
        '''调用generate_stream_gate方法并处理其返回的每个结果'''
        try:
            # generate_stream_gate是一个生成器，可以逐个产生处理结果。
            for x in self.generate_stream_gate(params):
                ...
            # 对最后一个结果x进行处理：假设x是一个字节字符串（由于使用了decode()方法），
            # 先去除最后一个字节（[:-1]），然后解码并解析成JSON对象。
            return json.loads(x[:-1].decode())
        except Exception as e:
            return {"error_code": 500, "text": str(e)}


    def do_chat(self, params: ApiChatParams) -> Dict:
        '''
        执行Chat的方法，默认使用模块里面的chat函数。
        要求返回形式：{"error_code": int, "text": str}
        '''
        return {"error_code": 500, "text": f"{self.model_names[0]}未实现chat功能"}

    # def do_completion(self, p: ApiCompletionParams) -> Dict:
    #     '''
    #     执行Completion的方法，默认使用模块里面的completion函数。
    #     要求返回形式：{"error_code": int, "text": str}
    #     '''
    #     return {"error_code": 500, "text": f"{self.model_names[0]}未实现completion功能"}

    def do_embeddings(self, params: ApiEmbeddingsParams) -> Dict:
        '''
        执行Embeddings的方法，默认使用模块里面的embed_documents函数。
        要求返回形式：{"code": int, "data": List[List[float]], "msg": str}
        '''
        return {"code": 500, "msg": f"{self.model_names[0]}未实现embeddings功能"}

    def get_embeddings(self, params):
        # fastchat对LLM做Embeddings限制很大，似乎只能使用openai的。
        # 在前端通过OpenAIEmbeddings发起的请求直接出错，无法请求过来。
        print("get_embedding")
        print(params)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        raise NotImplementedError

    def validate_messages(self, messages: List[Dict]) -> List[Dict]:
        '''
        有些API对mesages有特殊格式，可以重写该函数替换默认的messages。
        之所以跟prompt_to_messages分开，是因为他们应用场景不同、参数不同
        '''
        return messages


    # help methods
    @property
    def user_role(self):
        return self.conv.roles[0]

    @property
    def ai_role(self):
        return self.conv.roles[1]

    def _jsonify(self, data: Dict) -> str:
        '''
        将chat函数返回的结果按照fastchat openai-api-server的格式返回
        '''
        return json.dumps(data, ensure_ascii=False).encode() + b"\0"

    def _is_chat(self, prompt: str) -> bool:
        '''
        检查prompt是否由chat messages拼接而来
        TODO: 存在误判的可能，也许从fastchat直接传入原始messages是更好的做法
        '''
        key = f"{self.conv.sep}{self.user_role}:"
        return key in prompt

    def prompt_to_messages(self, prompt: str) -> List[Dict]:
        '''
        将prompt字符串拆分成messages.

        将一段对话(prompt)拆分成多条消息(messages)，每条消息包含角色(role)和内容(content)
        '''
        result = []
        # 用户角色
        user_role = self.user_role
        # AI角色
        ai_role = self.ai_role
        # 用户消息的开始
        user_start = user_role + ":"
        # AI消息的开始
        ai_start = ai_role + ":"
        # 遍历由prompt字符串经过self.conv.sep分割后得到的列表
        for msg in prompt.split(self.conv.sep)[1:-1]:
            # 首先判断当前消息是否以用户角色开始
            if msg.startswith(user_start):
                # 如果是，那么去掉消息开始的角色标识和冒号，然后去除两端的空白字符，得到消息内容。
                if content := msg[len(user_start):].strip():
                    # 如果内容非空，将这条消息以字典的形式添加到result列表中。
                    # 字典中包含两个键值对，"role"键对应的值是用户角色，"content"键对应的值是消息内容。
                    result.append({"role": user_role, "content": content})
            elif msg.startswith(ai_start):
                # 如果消息以AI角色开始，处理逻辑与处理用户消息类似，只不过角色是AI角色。
                if content := msg[len(ai_start):].strip():
                    result.append({"role": ai_role, "content": content})
            else:
                # 如果消息既不是以用户角色开始，也不是以AI角色开始，
                # 那么会抛出一个RuntimeError异常，提示存在未知角色的消息。
                raise RuntimeError(f"unknown role in msg: {msg}")
        return result

    @classmethod
    def can_embedding(cls):
        return cls.DEFAULT_EMBED_MODEL is not None
