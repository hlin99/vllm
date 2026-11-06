# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.xpu import XPUFp8BlockScaledMMKernel
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8Static128BlockSym,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_xpu(), reason="Requires XPU")


@pytest.mark.parametrize("tokens", [1, 16])
@pytest.mark.parametrize("k_blocks", [1, 2])
def test_output_block_scales(tokens, k_blocks, default_vllm_config):
    """A single K block must not interpret four block scales as 512 channels."""
    config = FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8Static128BlockSym,
        activation_quant_key=kFp8Dynamic128Sym,
        weight_shape=(512, 128 * k_blocks),
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )
    kernel = XPUFp8BlockScaledMMKernel(config)
    a = torch.ones(tokens, 128 * k_blocks, device="xpu", dtype=torch.float8_e4m3fn)
    b = torch.ones(512, 128 * k_blocks, device="xpu", dtype=torch.float8_e4m3fn)
    a_scale = torch.exp2(
        (torch.arange(tokens * k_blocks, device="xpu", dtype=torch.float32) % 4) - 4
    ).view(tokens, k_blocks)
    b_scale = (
        torch.exp2(torch.arange(4 * k_blocks, device="xpu", dtype=torch.float32))
        .view(4, k_blocks)
        .t()
        .contiguous()
        .t()
    )

    actual = kernel.apply_block_scaled_mm(a, b, a_scale, b_scale)
    expected = (128 * (a_scale @ b_scale.t()).repeat_interleave(128, dim=1)).to(
        torch.bfloat16
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
