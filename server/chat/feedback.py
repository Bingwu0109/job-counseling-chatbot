from fastapi import Body
from configs import logger, log_verbose
from server.utils import BaseResponse
from server.db.repository import feedback_message_to_db

def chat_feedback(message_id: str = Body("", max_length=32, description="聊天记录id"),
            score: int = Body(0, max=100, description="用户评分，满分100，越大表示评价越高"),
            reason: str = Body("", description="用户评分理由，比如不符合事实等")
            ):
    try:
        # 将用户反馈的信息保存到数据库
        feedback_message_to_db(message_id, score, reason)
    except Exception as e:
        msg = f"反馈聊天记录出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)
    # 函数最后返回一个BaseResponse对象，状态码为200（表示成功），并在消息中包含已反馈聊天记录的ID。
    return BaseResponse(code=200, msg=f"已反馈聊天记录 {message_id}")
