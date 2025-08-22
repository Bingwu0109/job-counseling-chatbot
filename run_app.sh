#!/bin/bash
run_name=$1
image_name=$2
image_version=$3
ssh_port=$4
jupyter_port=$5
webui_port=$6

docker run --cpus 12 --shm-size 16G --memory 24gb -itd \
    -p ${ssh_port}:22 \
    -p ${jupyter_port}:8888 \
    -p ${webui_port}:8501 \
    -v /etc/localtime:/etc/localtime \
    -v $(pwd)/Langchain-Chatchat:/root/Langchain-Chatchat \
    --name ${run_name} \
    -e JUPYTER_PASSWORD="123456" \
    -e JUPYTER_ROOT="root" \
    -e ROOT_PASS="123456" \
    ${image_name}:${image_version}

echo "Container ${run_name} started."
echo "SSH port: ${ssh_port}, Jupyter port: ${jupyter_port}, Web UI port: ${webui_port}"