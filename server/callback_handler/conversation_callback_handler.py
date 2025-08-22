from typing import Any, Dict, List
# BaseCallbackHandler 作为基类提供回调处理的基本框架
from langchain.callbacks.base import BaseCallbackHandler
# LLMResult 用于表示语言模型的结果
from langchain.schema import LLMResult
# update_message 函数用于更新数据库中的信息
from server.db.repository import update_message


class ConversationCallbackHandler(BaseCallbackHandler):
    # 在处理回调时是否应该抛出错误
    raise_error: bool = True

    def __init__(self, conversation_id: str, message_id: str, chat_type: str, query: str):
        self.conversation_id = conversation_id # 会话ID
        self.message_id = message_id # 消息ID
        self.chat_type = chat_type # 聊天类型
        self.query = query # 查询内容
        self.start_at = None # 记录开始处理的时间点

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False.
        论是否设置为详细模式，都总是进行详细回调。
        """
        return True

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # 如果想存更多信息，则prompts 也需要持久化
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        # 先从响应中提取文本结果
        answer = response.generations[0][0].text
        # 更新数据库中相应消息的内容
        update_message(self.message_id, answer)
