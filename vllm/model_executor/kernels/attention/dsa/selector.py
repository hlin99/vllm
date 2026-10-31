# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-independent selection of sparse indexer scoring and Top-K paths."""

from dataclasses import dataclass
from functools import cache
from typing import Literal

import torch


@dataclass(frozen=True)
class IndexerSelectionConfig:
    platform: str
    device_capability: int
    use_batch_invariant: bool = False
    head_dim: int = 128
    query_dtype: torch.dtype = torch.float8_e4m3fn
    weights_dtype: torch.dtype = torch.float32
    quant_block_size: int = 128
    use_fp4_cache: bool = False
    use_pcp: bool = False
    dcp_world_size: int = 1
    candidate_filtering: bool = False
    decode_query_len: int = 1
    deep_gemm_available: bool = True


@dataclass(frozen=True)
class IndexerKernel:
    name: Literal["deep_gemm", "xpu", "triton"]
    platforms: tuple[str, ...]
    supports_batch_invariance: bool = False
    batch_invariant_only: bool = False
    requires_deep_gemm: bool = False
    min_device_capability: int = 0
    supported_head_sizes: tuple[int, ...] | None = None
    supported_query_dtypes: tuple[torch.dtype, ...] | None = None
    supported_weights_dtypes: tuple[torch.dtype, ...] | None = None
    supported_quant_block_sizes: tuple[int, ...] | None = None
    supports_fp4_cache: bool = True
    supports_context_parallelism: bool = True
    supports_candidate_filtering: bool = True
    supports_multi_token_decode: bool = True

    def validate_configuration(self, config: IndexerSelectionConfig) -> list[str]:
        reasons = []
        if config.platform not in self.platforms:
            reasons.append(f"platform {config.platform} not supported")
        if config.device_capability < self.min_device_capability:
            reasons.append("device capability not supported")
        if config.use_batch_invariant and not self.supports_batch_invariance:
            reasons.append("batch invariance not supported")
        if self.batch_invariant_only and not config.use_batch_invariant:
            reasons.append("reserved for batch-invariant indexing")
        if self.requires_deep_gemm and not config.deep_gemm_available:
            reasons.append("DeepGEMM is unavailable")
        if (
            self.supported_head_sizes is not None
            and config.head_dim not in self.supported_head_sizes
        ):
            reasons.append(f"head size {config.head_dim} not supported")
        if (
            self.supported_query_dtypes is not None
            and config.query_dtype not in self.supported_query_dtypes
        ):
            reasons.append(f"query dtype {config.query_dtype} not supported")
        if (
            self.supported_weights_dtypes is not None
            and config.weights_dtype not in self.supported_weights_dtypes
        ):
            reasons.append(f"weights dtype {config.weights_dtype} not supported")
        if (
            self.supported_quant_block_sizes is not None
            and config.quant_block_size not in self.supported_quant_block_sizes
        ):
            reasons.append(
                f"quantization block size {config.quant_block_size} not supported"
            )
        if config.use_fp4_cache and not self.supports_fp4_cache:
            reasons.append("FP4 cache not supported")
        if (
            config.use_pcp or config.dcp_world_size != 1
        ) and not self.supports_context_parallelism:
            reasons.append("context parallelism not supported")
        if config.candidate_filtering and not self.supports_candidate_filtering:
            reasons.append("candidate filtering not supported")
        if config.decode_query_len != 1 and not self.supports_multi_token_decode:
            reasons.append("multi-token decode not supported")
        return reasons


# 保留常规模式的现有路径优先级；BI 是能力筛选条件，不是模型名称判断。
INDEXER_KERNELS = (
    IndexerKernel("deep_gemm", ("cuda",), requires_deep_gemm=True),
    IndexerKernel("xpu", ("xpu",), supports_fp4_cache=False),
    IndexerKernel(
        "triton",
        ("cuda",),
        supports_batch_invariance=True,
        batch_invariant_only=True,
        min_device_capability=9,
        supported_head_sizes=(128,),
        supported_query_dtypes=(torch.float8_e4m3fn,),
        supported_weights_dtypes=(torch.float32, torch.float16, torch.bfloat16),
        supported_quant_block_sizes=(128,),
        supports_fp4_cache=False,
        supports_context_parallelism=False,
        supports_candidate_filtering=False,
        supports_multi_token_decode=False,
    ),
)


@cache
def select_indexer_kernel(config: IndexerSelectionConfig) -> IndexerKernel:
    """Pick the highest-priority implementation satisfying the operator contract."""
    failures = []
    for kernel in INDEXER_KERNELS:
        reasons = kernel.validate_configuration(config)
        if not reasons:
            return kernel
        failures.append(f"{kernel.name}: {', '.join(reasons)}")
    raise NotImplementedError(
        f"No sparse indexer implementation supports {config}. " + "; ".join(failures)
    )


def supports_indexer_batch_invariance(platform: str, device_capability: int) -> bool:
    """Whether a BI indexer implementation exists on the platform.

    Individual operator configurations are validated separately at construction
    and execution; this is not an end-to-end attention capability declaration.
    """
    config = IndexerSelectionConfig(
        platform=platform,
        device_capability=device_capability,
        use_batch_invariant=True,
    )
    return any(
        kernel.supports_batch_invariance and not kernel.validate_configuration(config)
        for kernel in INDEXER_KERNELS
    )
