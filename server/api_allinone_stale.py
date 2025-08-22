"""Usage
调用默认模型：
python server/api_allinone.py

加载多个非默认模型：
python server/api_allinone.py --model-path-address model1@host1@port1 model2@host2@port2 

多卡启动：
python server/api_allinone.py --model-path-address model@host@port --num-gpus 2 --gpus 0,1 --max-gpu-memory 10GiB

"""
import sys
import os
# 将当前文件所在的目录添加到模块搜索路径中
sys.path.append(os.path.dirname(__file__))
# 当前文件的父目录添加到搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from llm_api_stale import launch_all, parser, controller_args, worker_args, server_args
# 创建一个Flask或Starlette等Web应用的实例
from api import create_app
import uvicorn

# 定义了一个列表，列出了与API服务器相关的参数名称
parser.add_argument("--api-host", type=str, default="0.0.0.0")
parser.add_argument("--api-port", type=int, default=7861)
parser.add_argument("--ssl_keyfile", type=str)
parser.add_argument("--ssl_certfile", type=str)

api_args = ["api-host", "api-port", "ssl_keyfile", "ssl_certfile"]


def run_api(host, port, **kwargs):
    # 创建一个Web应用实例
    app = create_app()
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        # 使用uvicorn.run函数启动API服务器
        # 如果提供了SSL证书文件的路径，则服务器会以HTTPS模式运行。
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    print("Luanching api_allinone，it would take a while, please be patient...")
    print("正在启动api_allinone，LLM服务启动约3-10分钟，请耐心等待...")
    # 初始化消息
    # 解析命令行参数
    args = parser.parse_args()
    # 将解析结果存储在变量args中
    args_dict = vars(args)
    # 启动LLM服务
    launch_all(args=args, controller_args=controller_args, worker_args=worker_args, server_args=server_args)
    # 启动API服务器
    run_api(
        host=args.api_host,
        port=args.api_port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )
    print("Luanching api_allinone done.")
    print("api_allinone启动完毕.")
