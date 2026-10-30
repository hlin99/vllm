#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -eu
export VLLM_TEST_NUM_FILLERS=15
export VLLM_TEST_MAX_TOKENS=512
exec sh "$(dirname "$0")/run_dsv4_e2e.sh" "$@"
