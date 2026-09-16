#bash
#

vllm serve /workspace/Llama-3.1-8B-Instruct/ \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
