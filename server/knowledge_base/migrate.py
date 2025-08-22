from configs import (
    EMBEDDING_MODEL, DEFAULT_VS_TYPE, ZH_TITLE_ENHANCE,
    CHUNK_SIZE, OVERLAP_SIZE,
    logger, log_verbose
)
from server.knowledge_base.utils import (
    get_file_path, list_kbs_from_folder,
    list_files_from_folder, files2docs_in_thread,
    KnowledgeFile
)
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.models.conversation_model import ConversationModel
from server.db.models.message_model import MessageModel
from server.db.repository.knowledge_file_repository import add_file_to_db # ensure Models are imported
from server.db.repository.knowledge_metadata_repository import add_summary_to_db

from server.db.base import Base, engine
from server.db.session import session_scope
import os
from dateutil.parser import parse
from typing import Literal, List


def create_tables():
    '''创建表格
    Base: 这通常是一个由 SQLAlchemy 的 declarative base 返回的基类，用于所有模型类的声明基础。
    metadata: 是 SQLAlchemy 中的一个属性，用于存储定义的表格结构（如列、数据类型等）。
    create_all: 这是一个方法，用于根据 metadata 中定义的表结构在数据库中创建表格。
    bind=engine: bind 参数用于指定连接到哪个数据库。engine 是 SQLAlchemy 用于连接数据库的引擎。
    '''
    Base.metadata.create_all(bind=engine)


def reset_tables():
    '''重置表格
    drop_all: 这个方法用于删除 metadata 中定义的所有表格，bind=engine 表示连接到指定的数据库。
    '''
    Base.metadata.drop_all(bind=engine)
    # 重新创建表格
    create_tables()


def import_from_db(
        sqlite_path: str = None, # 用于指定要导入数据的 SQLite 数据库文件路径
        # csv_path: str = None,
) -> bool:
    """
    在知识库与向量库无变化的情况下，从备份数据库中导入数据到 info.db。
    适用于版本升级时，info.db 结构变化，但无需重新向量化的情况。
    请确保两边数据库表名一致，需要导入的字段名一致
    当前仅支持 sqlite
    """
    # 导入 sqlite3 模块来操作 SQLite 数据库
    import sqlite3 as sql
    from pprint import pprint
    # 获取所有通过 SQLAlchemy 基类 Base 定义的模型
    models = list(Base.registry.mappers)

    try:
        # 使用 sqlite3 连接到指定的 SQLite 数据库文件
        con = sql.connect(sqlite_path)
        # 设置 row_factory 为 sql.Row 以使行可以像字典一样被访问
        con.row_factory = sql.Row
        cur = con.cursor()
        # 查询所有表名，并存储到 tables 列表中
        tables = [x["name"] for x in cur.execute("select name from sqlite_master where type='table'").fetchall()]
        # 遍历每个模型，检查对应的表是否存在于 SQLite 数据库中
        for model in models:
            # 如果表存在，则遍历该表中的每行数据。
            table = model.local_table.fullname
            if table not in tables:
                continue
            print(f"processing table: {table}")
            with session_scope() as session:
                # 对于每行数据，过滤出模型中定义的列，并处理特殊字段（如 create_time）。
                for row in cur.execute(f"select * from {table}").fetchall():
                    data = {k: row[k] for k in row.keys() if k in model.columns}
                    if "create_time" in data:
                        data["create_time"] = parse(data["create_time"])
                    pprint(data)
                    # 处理后的数据添加到会话中
                    session.add(model.class_(**data))
        con.close()
        return True
    except Exception as e:
        print(f"无法读取备份数据库：{sqlite_path}。错误信息：{e}")
        return False


def file_to_kbfile(kb_name: str, files: List[str]) -> List[KnowledgeFile]:
    '''将文件名列表转换为 KnowledgeFile 对象列表，并返回这个列表。'''
    # 用来存放成功创建的 KnowledgeFile 对象
    kb_files = []
    # 遍历传入的文件名列表
    for file in files:
        try:
            # 创建一个 KnowledgeFile 对象，并将其添加到列表 kb_files 中。
            kb_file = KnowledgeFile(filename=file, knowledge_base_name=kb_name)
            kb_files.append(kb_file)
        except Exception as e:
            msg = f"{e}，已跳过"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
    # 返回填充了 KnowledgeFile 对象的列表 kb_files
    return kb_files


def folder2db(
        kb_names: List[str], # 知识库的名称列
        # 操作模式，可以是 recreate_vs（重建向量搜索库）、update_in_db（更新数据库中的向量库）、或 increment（增量更新向量库）。
        mode: Literal["recreate_vs", "update_in_db", "increment"],
        vs_type: Literal["faiss", "milvus", "pg", "chromadb"] = DEFAULT_VS_TYPE, # 向量搜索库的类型
        embed_model: str = EMBEDDING_MODEL, # 用于生成文档向量的嵌入模型
        chunk_size: int = CHUNK_SIZE, # 在处理文档时，一次处理的块大小
        chunk_overlap: int = OVERLAP_SIZE, # 处理文档块时的重叠大小
        zh_title_enhance: bool = ZH_TITLE_ENHANCE, # 是否增强中文标题
):
    """
    用于将文件夹中的文档转换并存储到数据库中，并且根据不同的模式更新或创建向量搜索库。
    """
    def files2vs(kb_name: str, kb_files: List[KnowledgeFile]):
        # 使用 files2docs_in_thread 函数在多线程中将文件转换为文档，返回转换成功与否的标志和结果。
        for success, result in files2docs_in_thread(kb_files,
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    zh_title_enhance=zh_title_enhance):
            if success:
                # 如果转换成功，将转换得到的文档添加到向量搜索库中。
                _, filename, docs = result
                print(f"正在将 {kb_name}/{filename} 添加到向量库，共包含{len(docs)}条文档")
                kb_file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
                kb_file.splited_docs = docs
                kb.add_doc(kb_file=kb_file, not_refresh_vs_cache=True)
            else:
                # 如果转换失败，打印错误结果。
                print(result)
    # 获取知识库名称列表
    kb_names = kb_names or list_kbs_from_folder()
    # 遍历知识库名称
    for kb_name in kb_names:
        # 获取对应的知识库服务对象
        kb = KBServiceFactory.get_service(kb_name, vs_type, embed_model)
        # 如果知识库不存在，则调用 create_kb 方法创建知识库。
        if not kb.exists():
            kb.create_kb()

        # 1 - 重建向量搜索库
        if mode == "recreate_vs":
            # 清除现有的向量搜索库
            kb.clear_vs()
            # 重新创建知识库
            kb.create_kb()
            # 从本地文件夹中获取文件列表，将这些文件转换为 KnowledgeFile 对象列表。
            kb_files = file_to_kbfile(kb_name, list_files_from_folder(kb_name))
            # 调用 files2vs 函数处理这些文件并添加到向量搜索库中。
            files2vs(kb_name, kb_files)
            # 保存更新后的向量搜索库
            kb.save_vector_store()
        # 更新数据库中的向量库
        elif mode == "update_in_db":
            # 从数据库中获取文件列表
            files = kb.list_files()
            # 将这些文件转换为 KnowledgeFile 对象列表
            kb_files = file_to_kbfile(kb_name, files)
            # 更新向量搜索库
            files2vs(kb_name, kb_files)
            # 保存更新后的向量搜索库
            kb.save_vector_store()
        # 增量向量化
        elif mode == "increment":
            # 从数据库和本地文件夹中分别获取文件列表
            db_files = kb.list_files()
            folder_files = list_files_from_folder(kb_name)
            # 计算差集，找出仅存在于本地文件夹中的文件，这些文件尚未被添加到数据库。
            files = list(set(folder_files) - set(db_files))
            # 将这些新增的文件转换为 KnowledgeFile 对象列表
            kb_files = file_to_kbfile(kb_name, files)
            # 调用 files2vs 函数处理这些文件并添加到向量搜索库中
            files2vs(kb_name, kb_files)
            # 保存更新后的向量搜索库
            kb.save_vector_store()
        else:
            # 如果提供了不支持的模式参数，则打印一条错误消息。
            print(f"unsupported migrate mode: {mode}")


def prune_db_docs(kb_names: List[str]):
    """
    确保数据库中只包含那些仍然存在于本地文件夹中的文件对应的文档

    kb_names: 一个包含知识库名称的字符串列表
    """
    # 遍历知识库名称列表
    for kb_name in kb_names:
        # 获取对应的知识库服务对象
        kb = KBServiceFactory.get_service_by_name(kb_name)
        # 如果知识库服务对象存在
        if kb is not None:
            # 获取数据库中记录的文件列表
            files_in_db = kb.list_files()
            # 获取本地文件夹中的文件列表
            files_in_folder = list_files_from_folder(kb_name)
            # 计算两个列表的差集，即数据库中有而文件夹中没有的文件，这些文件对应的文档需要被删除。
            files = list(set(files_in_db) - set(files_in_folder))
            # 使用 file_to_kbfile 函数将待删除的文件名转换为 KnowledgeFile 对象列表
            kb_files = file_to_kbfile(kb_name, files)
            # 遍历 kb_files 列表
            for kb_file in kb_files:
                # 从数据库中删除文档，并打印删除成功的消息。
                kb.delete_doc(kb_file, not_refresh_vs_cache=True)
                print(f"success to delete docs for file: {kb_name}/{kb_file.filename}")
            # 用 save_vector_store 方法保存对向量搜索库的更新
            kb.save_vector_store()


def prune_folder_files(kb_names: List[str]):
    """
    删除那些在数据库中未记录，但存在于本地文件夹中的文件。
    """
    # 遍历知识库名称列表
    for kb_name in kb_names:
        # 获取对应的知识库服务对象
        kb = KBServiceFactory.get_service_by_name(kb_name)
        # 如果知识库服务对象存在
        if kb is not None:
            # 获取数据库中记录的文件列表
            files_in_db = kb.list_files()
            # 获取本地文件夹中的文件列表
            files_in_folder = list_files_from_folder(kb_name)
            # 计算两个列表的差集，即文件夹中有而数据库中没有的文件，这些文件需要被删除。
            files = list(set(files_in_folder) - set(files_in_db))
            # 遍历待删除的文件列表，对每个文件调用 os.remove 方法删除，并打印删除成功的消息。
            for file in files:
                os.remove(get_file_path(kb_name, file))
                print(f"success to delete file: {kb_name}/{file}")
