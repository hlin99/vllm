#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# e33d190dd5：公共六层真实模型回归入口；569f3f1de3：支持 light/heavy 两种负载。
set -eu  # 命令失败或引用未设置变量时立即退出，不掩盖测试失败。
cd "$(dirname "$0")/../../.."  # 从脚本目录定位仓库根，统一模型相对路径与解释器路径。

# The default checkpoint contains only six layers; optionally pass another path.
export VLLM_TEST_MODEL="${1:-../DeepSeek-V4-Flash-6layers}"  # 首参覆盖默认六层 checkpoint。
# 用六层架构、禁用 MTP，并覆盖纯滑窗/C4/C128 层，匹配裁剪后的实际权重。
export VLLM_TEST_HF_OVERRIDES='{"num_hidden_layers":6,"num_nextn_predict_layers":0,"compress_ratios":[0,0,4,128,4,128]}'
export VLLM_BATCH_INVARIANT=1  # 开启本系列提交接入的固定归约与稳定选择路径。
export VLLM_TEST_TP_SIZE=1  # 当前 DSV4 确定性实现仅支持单卡。
export VLLM_TEST_NUM_FILLERS="${VLLM_TEST_NUM_FILLERS:-15}"  # 569f3f1de3：默认 heavy 负载。
export VLLM_TEST_MAX_TOKENS="${VLLM_TEST_MAX_TOKENS:-512}"  # light/heavy 包装器可预先覆盖。

# 在加载权重之前检查硬件，避免不支持的平台运行到中途才失败。
.venv/bin/python -c '
from vllm.platforms import current_platform
# 必须同时是 CUDA 平台和 SM120 架构族。
assert current_platform.is_cuda() and current_platform.is_device_capability_family(120), "This test requires an SM120 GPU."
'

# 569f3f1de3：启动前打印场景、负载、收集范围和四种调度预算，方便观察长测进度。
echo "Comparing every logit bit: alone, prefix hit, batch-first, batch-last, mixed cached."
echo "Target + $VLLM_TEST_NUM_FILLERS filler requests; $VLLM_TEST_MAX_TOKENS tokens each."
echo "Collecting full-vocabulary logits for the target only."
echo "Prefill budgets: 256 and 272 (eager), 257 and 8192 (CUDA graphs)."
echo "Budget 272 forces a single-token prefill tail."
# exec 直接传递 pytest 退出码；-u/-s 关闭输出缓冲与捕获，-q 精简 pytest 自身输出。
exec .venv/bin/python -u -m pytest -q -s \
    tests/v1/determinism/test_batch_invariance.py::test_raw_logits_bs1_vs_bsN_chunking_and_prefix_cache
# 上一行只运行本系列新增的逐位 raw-logits 回归，不触发整个 determinism 套件。
