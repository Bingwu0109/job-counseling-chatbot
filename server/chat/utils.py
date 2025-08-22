from pydantic import BaseModel, Field
from langchain.prompts.chat import ChatMessagePromptTemplate
from configs import logger, log_verbose
from typing import List, Tuple, Dict, Union


class History(BaseModel):
    """
    该类用于表示对话历史的数据结构，并且基于pydantic库实现数据验证和管理。

    可从dict生成，如
    h = History(**{"role":"user","content":"你好"})
    也可转换为tuple，如
    h.to_msy_tuple = ("human", "你好")
    """

    # 分别表示对话中的角色（用户或助手）和内容，Field(...)表示这两个字段是必填的。
    role: str = Field(...)
    content: str = Field(...)

    def to_msg_tuple(self):
        '''用于将History实例的数据转换为元组格式'''
        return "ai" if self.role=="assistant" else "human", self.content

    def to_msg_template(self, is_raw=True) -> ChatMessagePromptTemplate:
        '''用于将History实例的数据转换为聊天消息模板'''
        role_maps = {
            "ai": "assistant",
            "human": "user",
        }
        role = role_maps.get(self.role, self.role)
        if is_raw: # 当前默认历史消息都是没有input_variable的文本。
            content = "{% raw %}" + self.content + "{% endraw %}"
        else:
            content = self.content

        return ChatMessagePromptTemplate.from_template(
            content,
            "jinja2",
            role=role,
        )

    @classmethod
    def from_data(cls, h: Union[List, Tuple, Dict]) -> "History":
        '''用于从不同类型的数据（列表、元组或字典）创建History实例。
           这提供了灵活的数据输入方式，便于从不同的数据源构造History对象。'''
        if isinstance(h, (list,tuple)) and len(h) >= 2:
            h = cls(role=h[0], content=h[1])
        elif isinstance(h, dict):
            h = cls(**h)

        return h
