#!/bin/bash

# 参数 1: random-input-len，默认值为 16000
INPUT_LEN=${1:-16000}

# 参数 2: model name，默认值为 /workspace/Llama-3.1-8B-Instruct/
MODEL_NAME=${2:-/workspace/Llama-3.1-8B-Instruct/}

echo "▶ 准备发起 vLLM Benchmark 请求..."
echo "  - 模型路径: $MODEL_NAME"
echo "  - Input 长度: $INPUT_LEN"
echo "------------------------------------------------"

vllm bench serve \
    --host 127.0.0.1 \
    --port 8866 \
    --model "$MODEL_NAME" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" \
    --random-output-len 1 \
    --random-range-ratio 0 \
    --num-prompts 64 \
    --burstiness 1000 \
    --request-rate inf \
    --max-concurrency 1 \
    --ignore-eos
