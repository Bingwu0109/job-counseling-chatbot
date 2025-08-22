import logging
from typing import Any, List, Dict

from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import get_buffer_string, BaseMessage, HumanMessage, AIMessage
from langchain.schema.language_model import BaseLanguageModel
from server.db.repository.message_repository import filter_message
from server.db.models.message_model import MessageModel


class ConversationBufferDBMemory(BaseChatMemory):
    conversation_id: str # 会话ID
    human_prefix: str = "Human"
    ai_prefix: str = "Assistant"
    llm: BaseLanguageModel
    memory_key: str = "history"
    max_token_limit: int = 2000 # 用于控制存储在内存中的消息数量和长度
    message_limit: int = 10
    '''负责获取和处理数据库中的聊天记录以构建适合输入到语言模型的消息缓冲区'''

    @property
    def buffer(self) -> List[BaseMessage]:
        """String buffer of memory."""
        # fetch limited messages desc, and return reversed
        # 根据当前实例的会话ID和消息限制从数据库中检索聊天记录，这些记录默认是按时间降序排列的，即最新的消息排在最前面。
        messages = filter_message(conversation_id=self.conversation_id, limit=self.message_limit)
        # 返回的记录按时间倒序，转为正序，这样做是为了让聊天记录的顺序与实际对话的顺序一致。
        messages = list(reversed(messages))
        chat_messages: List[BaseMessage] = []
        # 对于检索到的每条消息，都会根据消息的类型（查询或响应）创建HumanMessage或AIMessage对象，
        # 并将这些对象添加到chat_messages列表中。
        for message in messages:
            chat_messages.append(HumanMessage(content=message["query"]))
            chat_messages.append(AIMessage(content=message["response"]))
        # 在数据库中没有检索到任何消息
        if not chat_messages:
            return []

        # 计算当前消息缓冲区的令牌数量
        curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))
        # 如果令牌数量超过了设置的最大限制，则进入一个循环，
        # 从chat_messages列表中移除最早的消息，直到总令牌数不再超过限制为止。
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit and chat_messages:
                pruned_memory.append(chat_messages.pop(0))
                curr_buffer_length = self.llm.get_num_tokens(get_buffer_string(chat_messages))
        # 返回处理后的chat_messages列表，该列表现在包含了优化后的聊天记录，
        # 既不超过令牌限制，也保留了足够的历史信息供模型生成响应。
        return chat_messages

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """根据当前的内存状态（即聊天记录缓冲区）来构造一个输出字典，该字典可以直接用于语言模型的输入。"""
        # 获取当前的聊天记录缓冲区
        buffer: Any = self.buffer
        if self.return_messages:
            # 直接返回原始的聊天记录列表（buffer）
            final_buffer: Any = buffer
        else:
            # 将这个列表转换成一个字符串，转换成字符串的过程中会使用到human_prefix和ai_prefix来区分人类用户和AI的消息。
            final_buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: final_buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Nothing should be saved or changed"""
        pass

    def clear(self) -> None:
        """Nothing to clear, got a memory like a vault."""
        pass