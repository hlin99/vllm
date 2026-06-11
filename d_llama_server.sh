# vllm serve /workspace/Llama-3.1-8B-Instruct/ --port 8866 --max_model_len 16384 --no-enable-prefix-caching --no-enable-chunked-prefill --gpu_memory_utilization 0.9 --max-num-seqs 1
#
#!/bin/bash

# 获取传入的参数，如果未传入则使用默认值
# $1 为第一个参数（TP Size），默认值为 1
TP_SIZE=${1:-1}

# $2 为第二个参数（模型路径），默认值为 /workspace/Llama-3.1-8B-Instruct/
MODEL_PATH=${2:-"/workspace/Llama-3.1-8B-Instruct/"}

echo "====================================="
echo "启动 vLLM 服务..."
echo "并行数量 (TP Size): $TP_SIZE"
echo "模型路径 (Model Path): $MODEL_PATH"
echo "====================================="



# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

export VLLM_HOST_IP="10.239.11.165"
unset GLOO_SOCKET_IFNAME

export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/.nixl.mesonpy.libs/plugins:$LD_LIBRARY_PATH
export NIXL_PLUGIN_DIR=/opt/venv/lib/python3.12/site-packages/.nixl.mesonpy.libs/plugins


BLOCK_SIZE=64
VLLM_NIXL_DEVICE_TO_DEVICE=true
NIXL_BUFFER_DEVICE=cpu
VLLM_NIXL_BACKEND=UCX
export UCX_MEMTYPE_CACHE=0
export UCX_NET_DEVICES=mlx5_0:1
export UCX_TLS=rc,sm,self
export UCX_TLS=ib,rc,sm,self,ze_copy

export VLLM_NIXL_SIDE_CHANNEL_HOST=192.168.1.22


vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --port 8866 \
    --max_model_len 16384 \
    --no-enable-chunked-prefill \
    --gpu_memory_utilization 0.9 \
    --max-num-seqs 1 \
    --block-size $BLOCK_SIZE \
    --kv-transfer-config "{\"kv_connector\": \"NixlConnector\", \"kv_role\": \"kv_consumer\", \"kv_buffer_device\": \"${NIXL_BUFFER_DEVICE}\", \"kv_connector_extra_config\": {\"backends\": [\"${VLLM_NIXL_BACKEND}\"]}}"

