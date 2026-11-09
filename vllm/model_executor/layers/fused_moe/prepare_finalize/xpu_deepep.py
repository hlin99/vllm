# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from weakref import WeakKeyDictionary

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class XPUDeepEPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """Native intranode EP dispatch/combine for PCP-local XPU tokens."""

    _buffers: WeakKeyDictionary = WeakKeyDictionary()

    def __init__(self, moe: FusedMoEConfig):
        config = get_current_vllm_config()
        parallel = config.parallel_config
        if (
            not current_platform.is_xpu()
            or parallel.tensor_parallel_size != 1
            or parallel.data_parallel_size != 1
            or parallel.nnodes != 1
            or moe.pcp_size not in (2, 4, 8)
            or not moe.moe_parallel_config.use_ep
            or moe.ep_size != moe.pcp_size
            or moe.moe_parallel_config.is_sequence_parallel
        ):
            raise ValueError(
                "DeepEP XPU PCP MoE requires single-node PCP=EP=2, 4, or 8, "
                "TP=DP=1, without sequence parallelism"
            )
        if (
            config.model_config is None
            or not config.model_config.enforce_eager
            or parallel.enable_dbo
            or moe.is_lora_enabled
            or moe.moe_parallel_config.enable_eplb
            or moe.skip_final_all_reduce
            or parallel.expert_placement_strategy != "linear"
        ):
            raise ValueError(
                "DeepEP XPU PCP MoE requires eager execution and linear expert "
                "placement without DBO, LoRA, EPLB, or deferred reduction"
            )
        if moe.in_dtype != torch.bfloat16 or moe.hidden_dim % 128:
            raise ValueError(
                "DeepEP XPU PCP MoE requires BF16 activations and "
                "a hidden size divisible by 128"
            )
        if moe.num_experts % moe.ep_size:
            raise ValueError("DeepEP XPU requires evenly sharded experts")
        self.moe = moe
        self._handle: tuple | None = None

    @property
    def handles_pcp(self) -> bool:
        return True

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype:
        # XPU hash routing shares int32 input IDs; cast at the native boundary.
        return torch.int32

    def num_dispatchers(self) -> int:
        return self.moe.ep_size

    def output_is_reduced(self) -> bool:
        return True

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts):
        from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExperts

        if not isinstance(fused_experts, XPUExperts):
            raise ValueError("DeepEP XPU PCP MoE requires the XPU expert backend")
        import deep_ep_cpp  # noqa: F401
        from deep_ep_xpu import Buffer  # noqa: F401

        logger.info_once("Using DeepEP XPU PCP MoE EP dispatch/combine")

    def _buffer(self):
        from deep_ep_xpu import Buffer

        group = get_ep_group()
        shape = (self.moe.hidden_dim, self.moe.ep_size)
        if group not in self._buffers:
            hidden_bytes = self.moe.hidden_dim * torch.bfloat16.itemsize
            capacity = max(
                config.get_pcie_buffer_size_hint(hidden_bytes, group.world_size)
                for config in (
                    Buffer.get_dispatch_config(group.world_size),
                    Buffer.get_combine_config(group.world_size),
                )
            )
            buffer = Buffer(group.cpu_group, num_pcie_bytes=capacity)
            if buffer.runtime.get_num_rdma_ranks() != 1:
                raise ValueError("DeepEP XPU PCP MoE requires native intranode EP")
            self._buffers[group] = (shape, buffer)
        cached_shape, buffer = self._buffers[group]
        if cached_shape != shape:
            raise ValueError("DeepEP XPU PCP MoE layers must share a hidden size")
        return buffer

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if apply_router_weight_on_input or not defer_input_quant:
            raise ValueError(
                "DeepEP XPU PCP MoE requires unquantized inputs and "
                "router weights applied on expert outputs"
            )
        if self._handle is not None:
            raise RuntimeError("DeepEP XPU dispatch called before the previous combine")
        buffer = self._buffer()
        topk_ids = topk_ids.to(torch.int64)
        per_rank, per_rdma_rank, per_expert, in_rank, _ = buffer.get_dispatch_layout(
            topk_ids, num_experts
        )
        recv_x, recv_ids, recv_weights, _, self._handle, _ = buffer.dispatch(
            a1,
            num_tokens_per_rank=per_rank,
            num_tokens_per_rdma_rank=per_rdma_rank,
            num_tokens_per_expert=per_expert,
            is_token_in_rank=in_rank,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
        )
        # Native dispatch returns local expert IDs; XPUExperts uses a global map.
        offset = self.moe.moe_parallel_config.ep_rank * self.moe.num_local_experts
        recv_ids = torch.where(recv_ids >= 0, recv_ids + offset, recv_ids)
        return recv_x, None, None, recv_ids, recv_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if self._handle is None:
            raise RuntimeError("DeepEP XPU combine called without a dispatch handle")
        weighted = weight_and_reduce_impl.apply(
            None,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
        )
        # XPUExperts already weights and gathers; combine only adds rank outputs.
        combined, _, _ = self._buffer().combine(weighted, self._handle)
        output.copy_(combined, non_blocking=True)
        self._handle = None
