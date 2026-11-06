# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch

from vllm.config import get_current_vllm_config
from vllm.distributed import get_pcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExperts
from vllm.model_executor.layers.fused_moe.prepare_finalize.passthrough import (
    PassThroughPrepareAndFinalize,
)
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from deep_symm.moe_tp import SymmBuffer
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

logger = init_logger(__name__)


class XPUDeepSymmPrepareAndFinalize(PassThroughPrepareAndFinalize):
    """Local routing with fused PCP communication inside XPU experts."""

    _buffers: WeakKeyDictionary = WeakKeyDictionary()

    def __init__(self, moe: FusedMoEConfig):
        config = get_current_vllm_config()
        parallel = config.parallel_config
        if (
            not current_platform.is_xpu()
            or parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or moe.pcp_size <= 1
            or moe.moe_parallel_config.use_ep
            or moe.moe_parallel_config.is_sequence_parallel
        ):
            raise ValueError(
                "DeepSymm PCP MoE currently requires XPU, PCP>1, TP=DP=1, "
                "without EP or sequence parallelism"
            )
        if (
            config.model_config is None
            or not config.model_config.enforce_eager
            or parallel.enable_dbo
            or moe.is_lora_enabled
            or moe.moe_parallel_config.enable_eplb
            or moe.skip_final_all_reduce
        ):
            raise ValueError(
                "DeepSymm PCP MoE requires eager execution without DBO, "
                "LoRA, EPLB, or deferred cross-rank reduction"
            )
        if moe.in_dtype != torch.bfloat16:
            raise ValueError("DeepSymm PCP MoE requires BF16 activations")
        if moe.num_experts > 512:
            raise ValueError("DeepSymm TP supports at most 512 experts")
        self.moe = moe

    @property
    def handles_pcp(self) -> bool:
        return True

    def post_init_setup(self, fused_experts: FusedMoEExperts):
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExperts

        if not isinstance(fused_experts, XPUExperts):
            raise ValueError("DeepSymm PCP MoE requires the XPU expert backend")
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

        if not hasattr(XpuFusedMoe, "apply_permuted"):
            raise RuntimeError(
                "Update vllm_xpu_kernels: DeepSymm needs XpuFusedMoe.apply_permuted"
            )
        # Import the native module directly: the Python bridge suppresses failures.
        import deep_symm_cpp  # noqa: F401

        fused_experts.pcp_fusion = self
        logger.info_once("Using DeepSymm fused PCP MoE TP dispatch/combine")

    def _buffer(self) -> "SymmBuffer":
        from deep_symm.moe_tp import SymmBuffer

        group = get_pcp_group()
        shape = (
            self.moe.max_num_tokens,
            self.moe.hidden_dim,
            self.moe.experts_per_token,
            self.moe.in_dtype,
        )
        cached = self._buffers.get(group)
        if cached is not None:
            previous_shape, buffer = cached
            if previous_shape != shape:
                raise ValueError(
                    "DeepSymm PCP MoE requires identical workspace shapes across layers"
                )
            return buffer
        if torch.xpu.current_device() != group.rank_in_group:
            raise ValueError("DeepSymm TP requires PCP rank to match local XPU index")
        buffer = SymmBuffer(
            group.device_group,
            num_max_tokens_per_rank=self.moe.max_num_tokens,
            hidden=self.moe.hidden_dim,
            num_topk=self.moe.experts_per_token,
            combine_reduce_type=self.moe.in_dtype,
        )
        self._buffers[group] = (shape, buffer)
        return buffer

    def apply(
        self,
        experts: "XpuFusedMoe",
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> None:
        rows, hidden = hidden_states.shape
        if rows > self.moe.max_num_tokens:
            raise ValueError("DeepSymm PCP MoE token count exceeds reserved capacity")
        buffer = self._buffer()
        expert_rows = rows * self.moe.pcp_size * self.moe.experts_per_token
        remapped = hidden_states.new_empty((expert_rows, hidden))
        remapped, handle = buffer.allgather_local_permute_fusion(
            hidden_states,
            topk_ids,
            topk_weights,
            self.moe.num_experts,
            remapped,
        )
        expert_output = output.new_empty((expert_rows, output.shape[1]))
        experts.apply_permuted(expert_output, remapped, handle.rows_per_expert)
        buffer.unpermute_reducescatter_fusion(expert_output, handle, output)
