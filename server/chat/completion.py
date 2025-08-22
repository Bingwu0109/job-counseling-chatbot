from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_OpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, Optional
import asyncio
from langchain.prompts import PromptTemplate

from server.utils import get_prompt_template


async def completion(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
                     stream: bool = Body(False, description="流式输出"),
                     echo: bool = Body(False, description="除了输出之外，还回显输入"),
                     model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                     temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                     max_tokens: Optional[int] = Body(1024, description="限制LLM生成Token数量，默认None代表模型最大值"),
                     # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
                     prompt_name: str = Body("default",
                                             description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                     ):

    #todo 因ApiModelWorker 默认是按chat处理的，会对params["prompt"] 解析为messages，因此ApiModelWorker 使用时需要有相应处理
    async def completion_iterator(query: str,
                                  model_name: str = LLM_MODELS[0],
                                  prompt_name: str = prompt_name,
                                  echo: bool = echo,
                                  ) -> AsyncIterable[str]:
        '''异步地生成文本，使用了一个配置好的LLM来根据用户的输入生成相应的文本。'''
        nonlocal max_tokens
        # 用于异步处理回调
        callback = AsyncIteratorCallbackHandler()
        # 如果max_tokens是一个非正整数，则将其设为None
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None
        # 初始化聊天模型
        model = get_OpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
            echo=echo
        )
        # 获取指定的提示模板
        prompt_template = get_prompt_template("completion", prompt_name)
        # 创建一个prompt对象
        prompt = PromptTemplate.from_template(prompt_template)
        # 实例化一个LLMChain对象，这个对象将用于生成文本。
        chain = LLMChain(prompt=prompt, llm=model)

        # 启动一个异步任务task，这个任务调用LLMChain对象的acall方法生成文本，并在完成时调用回调。
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        if stream:
            # 如果stream为True，则通过async for循环异步迭代callback对象，
            # 使用服务器发送事件（Server-Sent Events, SSE）以流的形式实时返回每个生成的文本片段。
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield token
        else:
            # 如果stream为False，则同样异步迭代callback对象，但将所有生成的文本片段拼接后，作为一个整体一次性返回。
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer
        # 确保上述异步任务正常完成
        await task

    # 根据客户端请求的需要，可以以事件流的方式向客户端推送生成的文本数据，这对于需要实时更新数据的前端应用非常有用。
    return EventSourceResponse(completion_iterator(query=query,
                                                 model_name=model_name,
                                                 prompt_name=prompt_name),
                             )
