# wraps函数用于在定义装饰器时保留原始函数的名称、文档字符串等属性。
from functools import wraps
# contextmanager装饰器用于将一个生成器函数转换成一个上下文管理器
from contextlib import contextmanager
# SessionLocal用于创建和管理数据库会话的工厂函数
from server.db.base import SessionLocal
# Session表示一个数据库会话，用于执行数据库操作。
from sqlalchemy.orm import Session


@contextmanager
def session_scope() -> Session:
    """上下文管理器用于自动获取 Session, 避免错误"""
    # 返回一个Session 对象，这是SQLAlchemy ORM用于数据库操作的会话。
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def with_session(f):
    '''函数装饰器，用于自动处理函数中的数据库会话。'''
    @wraps(f)
    def wrapper(*args, **kwargs):
        # 自动管理一个数据库会话
        with session_scope() as session:
            try:
                # 将这个会话作为第一个参数传递给被装饰的函数f，执行并处理事务提交和回滚。
                result = f(session, *args, **kwargs)
                session.commit()
                return result
            except:
                session.rollback()
                raise
    # 返回wrapper函数，替换原始函数f。
    return wrapper


def get_db() -> SessionLocal:
    '''一个生成器，用于获取和管理一个数据库会话。'''
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db0() -> SessionLocal:
    '''get_db0不使用yield语句，因此不会自动关闭会话，这意味着调用者需要手动管理会话的关闭。'''
    db = SessionLocal()
    return db
