#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -eu
cd "$(dirname "$0")/../../.."

# The default checkpoint contains only six layers; optionally pass another path.
export VLLM_TEST_MODEL="${1:-../DeepSeek-V4-Flash-6layers}"
export VLLM_TEST_HF_OVERRIDES='{"num_hidden_layers":6,"num_nextn_predict_layers":0,"compress_ratios":[0,0,4,128,4,128]}'
export VLLM_BATCH_INVARIANT=1
export VLLM_TEST_TP_SIZE=1
export VLLM_TEST_NUM_FILLERS="${VLLM_TEST_NUM_FILLERS:-15}"
export VLLM_TEST_MAX_TOKENS="${VLLM_TEST_MAX_TOKENS:-512}"

.venv/bin/python -c '
from vllm.platforms import current_platform
assert current_platform.is_cuda() and current_platform.is_device_capability_family(120), "This test requires an SM120 GPU."
'

echo "Comparing every logit bit: alone, prefix hit, batch-first, batch-last, mixed cached."
echo "Target + $VLLM_TEST_NUM_FILLERS filler requests; $VLLM_TEST_MAX_TOKENS tokens each."
echo "Collecting full-vocabulary logits for the target only."
echo "Prefill budgets: 256 and 272 (eager), 257 and 8192 (CUDA graphs)."
echo "Budget 272 forces a single-token prefill tail."
exec .venv/bin/python -u -m pytest -q -s \
    tests/v1/determinism/test_batch_invariance.py::test_raw_logits_bs1_vs_bsN_chunking_and_prefix_cache
