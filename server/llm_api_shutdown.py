"""
调用示例：
python llm_api_shutdown.py --serve all
可选"all","controller","model_worker","openai_api_server"， all表示停止所有服务
"""
import sys
import os

# 将脚本的上一级目录添加到Python的搜索路径中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# 用于新建进程
import subprocess
import argparse

parser = argparse.ArgumentParser()
# 有四个可选值："all", "controller", "model_worker", "openai_api_server"。默认值是"all"。
parser.add_argument("--serve", choices=["all", "controller", "model_worker", "openai_api_server"], default="all")

args = parser.parse_args()
# 定义了一个基本的shell命令字符串，用于找出并杀掉某些进程。
# 具体来说，它使用ps命令列出所有进程，然后通过grep命令找出包含fastchat.serve的进程，
# 再用grep -v grep排除掉grep进程本身，最后通过awk取出进程ID，并用xargs kill -9命令强制杀掉这些进程。
base_shell = "ps -eo user,pid,cmd|grep fastchat.serve{}|grep -v grep|awk '{{print $2}}'|xargs kill -9"
# 根据命令行参数执行相应的操作
if args.serve == "all":
    shell_script = base_shell.format("")
else:
    serve = f".{args.serve}"
    shell_script = base_shell.format(serve)
# 执行上面生成的shell命令字符串。shell=True表示命令将在shell中运行。
# check=True表示如果执行的命令返回非零值，则会抛出异常。
subprocess.run(shell_script, shell=True, check=True)
# 打印一条消息表明指定的服务已经被停止
print(f"llm api sever --{args.serve} has been shutdown!")
