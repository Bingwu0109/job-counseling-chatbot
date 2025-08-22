from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from configs import LLM_MODELS, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
import json
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union
from server.chat.utils import History
from langchain.prompts import PromptTemplate
from server.utils import get_prompt_template
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory
from server.db.repository import add_message_to_db
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               conversation_id: str = Body("", description="对话框ID"),
               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                             {"role": "assistant", "content": "虎头虎脑"}]]
                                                         ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    # 返回类型: AsyncIterab、；≥ le[str]（异步可迭代的字符串）
    async def chat_iterator() -> AsyncIterable[str]:
        # 使用nonlocal关键字，表明这些变量在外层函数中定义，允许在chat_iterator内部修改它们。
        nonlocal history, max_tokens
        # 用于异步处理回调。
        # 在异步操作中注册回调函数，这些回调函数可以在数据生成时被调用，并将数据传递给一个等待中的异步迭代器。
        callback = AsyncIteratorCallbackHandler()
        # 存储所有回调处理器
        callbacks = [callback]
        # 上下文记忆处理
        memory = None

        # 将聊天请求保存到数据库，并获取一个消息ID。
        message_id = add_message_to_db(chat_type="llm_chat", query=query, conversation_id=conversation_id)
        # 处理对话相关的回调操作
        conversation_callback = ConversationCallbackHandler(conversation_id=conversation_id, message_id=message_id,
                                                            chat_type="llm_chat",
                                                            query=query)
        callbacks.append(conversation_callback)
        # 如果max_tokens是一个非正整数，则将其设为None
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None
        # 初始化聊天模型
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

        # 优先使用前端传入的历史消息
        if history:
            # 如果有历史消息（history）提供，首先将这些历史消息转换成History对象。
            history = [History.from_data(h) for h in history]
            # 获取与prompt_name相对应的提示模板
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            # 将历史消息和用户的当前消息（通过提示模板转换得到）组合成一个新的聊天提示（chat_prompt），
            # 这个聊天提示将作为语言模型的输入。
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
        # 从数据库获取历史消息
        elif conversation_id and history_len > 0:
            # 使用一个特定的带历史的提示模板
            prompt = get_prompt_template("llm_chat", "with_history")
            chat_prompt = PromptTemplate.from_template(prompt)
            # 实例化一个记忆体，它通过对话ID和历史消息数量参数，
            # 从数据库中获取历史消息，并将这些消息作为模型的上下文。
            memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                llm=model,
                                                message_limit=history_len)
        else:
            # 没有历史消息时的默认处理，使用默认的提示模板创建聊天提示
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        # 使用LLMChain类创建一个链式调用对象，将准备好的聊天提示、语言模型实例以及可能的记忆体作为参数传入。
        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        # 创建一个异步任务，这个任务调用chain.acall({"input": query})来生成聊天回应，
        # wrap_done函数用于在任务完成时调用callback.done，以确保所有回调都得到处理。
        task = asyncio.create_task(wrap_done(
            chain.acall({"input": query}),
            callback.done),
        )

        # 流式输出逻辑，这种方式适用于需要实时显示聊天回复的场景，可以逐个token地向用户展示回复的生成过程。
        if stream:
            # 通过callback.aiter()异步迭代器逐个获取token，
            # 每获取到一个token就通过json.dumps将其序列化成JSON格式并yield返回。
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"text": token, "message_id": message_id},
                    ensure_ascii=False)
        else:
            # 非流式输出逻辑，代码会累积所有生成的tokens，直到所有tokens都生成完毕，然后将累积的答案作为一个完整的文本返回。
            answer = ""
            # 使用callback.aiter()来异步迭代获取token，但是会将所有tokens拼接成一个完整的答案，
            # 最后通过json.dumps序列化并一次性返回。
            async for token in callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "message_id": message_id},
                ensure_ascii=False)
        # 确保了前面启动的异步生成聊天回应的任务能够执行完成
        await task

    # 实现了将聊天生成的过程作为一个持续的流发送给客户端，客户端在接收到这种类型的响应后，
    # 可以持续接收到服务器推送的数据，直到连接被关闭。
    return EventSourceResponse(chat_iterator())
