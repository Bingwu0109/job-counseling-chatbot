import logging
import os
import langchain
import tempfile
import shutil


# 是否显示详细日志
log_verbose = False
langchain.verbose = False

# 通常情况下不需要更改以下内容

# 日志格式

# %(asctime)s会被替换为记录日志的时间，%(filename)s会被替换为产生日志记录的源文件名，
# %(lineno)d会被替换为日志记录发生的代码行号，%(levelname)s会被替换为日志的级别名称（如INFO、WARNING等），
# %(message)s会被替换为具体的日志信息。
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
# 创建了一个日志记录器
logger = logging.getLogger()
# 设置了日志记录器的日志级别为INFO
logger.setLevel(logging.INFO)
# 使用basicConfig函数来为日志系统做基础配置，format=LOG_FORMAT指定了全局的日志格式。
logging.basicConfig(format=LOG_FORMAT)


# 日志存储路径
LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

# 临时文件目录，主要用于文件对话
# 调用tempfile.gettempdir()函数获取系统的临时文件目录的路径
BASE_TEMP_DIR = os.path.join(tempfile.gettempdir(), "chatchat")
try:
    # shutil.rmtree函数用于删除一个目录及其所有内容
    shutil.rmtree(BASE_TEMP_DIR)
except Exception:
    # 如果在尝试删除目录时出现任何异常（比如目录不存在或者权限问题）
    pass
# 创建目录
os.makedirs(BASE_TEMP_DIR, exist_ok=True)
