# SQLAlchemy是一个流行的SQL工具包和对象关系映射（ORM）工具，
# 这段代码主要用于配置和创建数据库会话。

# create_engine是SQLAlchemy用于连接数据库的函数，它返回一个数据库引擎对象，该对象提供了数据库连接的源。
from sqlalchemy import create_engine
# declarative_base函数用于生成一个基类，该基类用于声明模型（数据库表的映射类）。
# DeclarativeMeta是一个类型注解，用于指示Base的类型。
from sqlalchemy.ext.declarative import declarative_base, DeclarativeMeta
# sessionmaker是一个工厂函数，用于构造新的Session对象，Session用于管理数据库中的事务和记录。
from sqlalchemy.orm import sessionmaker
# 这个变量包含了数据库的连接信息，例如数据库类型、用户名、密码、主机名和数据库名。
from configs import SQLALCHEMY_DATABASE_URI
import json

# 创建了一个数据库引擎
engine = create_engine(
    SQLALCHEMY_DATABASE_URI, # 数据库的连接字符串
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False), # 用于自定义如何将对象序列化为JSON格式
)
# 使用sessionmaker函数创建了一个Session类的工厂
# autocommit=False表示禁用自动提交事务（这意味着你需要手动调用commit()方法来提交事务）
# autoflush=False表示禁用自动刷新，bind=engine 将此Session绑定到之前创建的engine，以便知道如何连接数据库。
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# 使用declarative_base()函数创建了一个基础类Base，所有的模型（数据库表的映射）都应该继承这个基础类。
Base: DeclarativeMeta = declarative_base()
