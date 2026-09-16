#bash
#

vllm serve /workspace/Llama-3.1-8B-Instruct/ \
  --enable-prefix-caching \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": 137438953472,
      "block_size": 64
    }
  }'
