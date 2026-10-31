#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 569f3f1de3：重负载入口复用公共脚本，不复制模型与测试配置。
set -eu  # 失败即退出，未定义变量也报错。
export VLLM_TEST_NUM_FILLERS=15  # 目标之外增加 15 个请求，组成最多 16 请求的混批。
export VLLM_TEST_MAX_TOKENS=512  # 每个请求连续生成 512 步，扩大 decode 覆盖范围。
exec sh "$(dirname "$0")/run_dsv4_e2e.sh" "$@"  # 原样转交模型路径参数和退出状态。
