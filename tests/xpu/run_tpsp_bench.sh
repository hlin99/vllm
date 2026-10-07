#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )) || [[ "$1" != "llama" && "$1" != "qwen3-moe" ]]; then
    echo "Usage: $0 {llama|qwen3-moe} /path/to/local/model" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
model_path="$2"
if [[ ! -f "$model_path/config.json" ]]; then
    echo "Model config not found: $model_path/config.json" >&2
    exit 2
fi

export PYTHONPATH="${TPSP_ROOT:-/workspace/TPSP}${PYTHONPATH:+:$PYTHONPATH}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-4,5,6,7}"
export HF_HUB_OFFLINE=1
vllm_bin="${VLLM_BIN:-$repo_root/.venv/bin/vllm}"
port="${TPSP_PORT:-8003}"
result_dir="${TPSP_RESULT_DIR:-/tmp/tpsp-vllm-bench}"
if [[ ! -x "$vllm_bin" ]]; then
    echo "vLLM executable not found: $vllm_bin" >&2
    exit 2
fi

case "$1" in
    llama)
        override='{"LlamaForCausalLM":"vllm.models.llama.xpu.model:TPSPLlamaForCausalLM"}'
        served_name="llama-tpsp"
        max_len=65536
        max_batch=65536
        num_prompts=16
        prefill=(--no-enable-chunked-prefill)
        export ASYNC_TP_OUTPUT_POOL_MB="${ASYNC_TP_OUTPUT_POOL_MB:-2048}"
        memory_utilization=0.85
        ;;
    qwen3-moe)
        override='{"Qwen3MoeForCausalLM":"vllm.models.qwen3_moe.xpu.model:TPSPQwen3MoeForCausalLM"}'
        served_name="qwen3-moe-tpsp"
        max_len=131072
        max_batch=32768
        num_prompts=4
        prefill=(--enable-chunked-prefill)
        export ASYNC_TP_OUTPUT_POOL_MB="${ASYNC_TP_OUTPUT_POOL_MB:-1024}"
        memory_utilization=0.9
        ;;
esac

if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    echo "Port $port already has a running server" >&2
    exit 2
fi

"$vllm_bin" serve "$model_path" \
    --model-class-overrides "$override" \
    --served-model-name "$served_name" \
    --tensor-parallel-size 4 \
    --host 127.0.0.1 --port "$port" \
    --max-model-len "$max_len" --max-num-batched-tokens "$max_batch" \
    "${prefill[@]}" --no-enable-prefix-caching \
    --gpu-memory-utilization "$memory_utilization" \
    --enforce-eager --dtype bfloat16 &
server_pid=$!
cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

ready=0
for _ in {1..120}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        wait "$server_pid"
        exit 1
    fi
    sleep 5
done
if (( ! ready )); then
    echo "vLLM did not become healthy on port $port" >&2
    exit 1
fi

"$vllm_bin" bench serve \
    --backend openai --base-url "http://127.0.0.1:$port" \
    --endpoint /v1/completions --model "$served_name" --tokenizer "$model_path" \
    --dataset-name random --random-input-len 65535 --random-output-len 1 \
    --num-prompts "$num_prompts" --max-concurrency 1 --request-rate inf --seed 0 \
    --save-result --result-dir "$result_dir" \
    --result-filename "$served_name-tp4-64k.json"
