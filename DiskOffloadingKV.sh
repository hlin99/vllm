#bash
#

vllm serve /workspace/Llama-3.1-8B-Instruct/ \
  --enable-prefix-caching \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec",
      "cpu_bytes_to_use": 34359738368,
      "block_size": 64,
      "eviction_policy": "lru",
      "offload_prompt_only": true,
      "secondary_tiers": [
        {
          "type": "fs",
          "root_dir": "/workspace/kv-cache",
          "n_read_threads": 32,
          "n_write_threads": 16,
          "locality": "LOCAL"
        }
      ]
    }
  }'
