#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 569f3f1de3：轻负载保留相同五场景与四执行配置，仅缩小请求数和生成长度。
set -eu  # 任何错误都使入口失败，不吞掉 pytest 退出状态。
export VLLM_TEST_NUM_FILLERS=3  # 目标之外增加 3 个请求。
export VLLM_TEST_MAX_TOKENS=3  # 每个请求只生成 3 步，降低回归时间。
exec sh "$(dirname "$0")/run_dsv4_e2e.sh" "$@"  # 与 heavy 共用同一真实测试入口。
