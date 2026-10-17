# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA implementation of the TPSP backend."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d

from vllm.logger import init_logger
from vllm.model_executor.models.transformers.tpsp_runtime import (
    TPSPBackend,
    TPSPContext,
    TPSPNormType,
    TPSPOpsGroup,
    profile_tpsp,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class CudaTPSPConfig:
    microchunk_tokens_per_rank: int
    sm_count_target: int = 0


@dataclass
class CudaTPSPContext:
    device: torch.device
    tp_size: int
    rank: int
    control_group: dist.ProcessGroup
    max_microchunk_tokens_per_rank: int
    p2p_handle: int = 0
    config: int | CudaTPSPConfig | None = None


class CudaTPSPBackend(TPSPBackend):
    tpsp_max_microchunk_tokens_per_rank = 4_096
    supported_norm_types: frozenset[TPSPNormType] = frozenset(
        {"layer_norm", "rms_norm"}
    )

    def __init__(self, group_name: str, device: torch.device):
        super().__init__(group_name, device)
        self._open_context_ids: set[int] = set()

    def get_default_profile_settings(self) -> tuple[int, int]:
        return 256, 512

    def profile(
        self,
        ops_groups: Sequence[TPSPOpsGroup],
        max_batched_tokens: int,
        *,
        microchunk_tokens: int | None = None,
        threshold_tokens: int | None = None,
        microchunk_tokens_per_rank_min_max: (
            tuple[int, int] | Mapping[str, tuple[int, int]] | None
        ) = None,
        microchunk_step_coarse_per_rank: int = 256,
        microchunk_step_fine_per_rank: int = 64,
    ) -> TPSPContext | None:
        if microchunk_tokens is not None and threshold_tokens is not None:
            return super().profile(
                ops_groups,
                max_batched_tokens,
                microchunk_tokens=microchunk_tokens,
                threshold_tokens=threshold_tokens,
            )

        from vllm.distributed.parallel_state import get_tp_group

        tp_size = get_tp_group().world_size
        if (
            microchunk_tokens_per_rank_min_max is None
            and microchunk_tokens is None
            and type(max_batched_tokens) is int
            and max_batched_tokens > 0
        ):
            microchunk_tokens_per_rank_min_max = self._microchunk_tokens_per_rank_range(
                max_batched_tokens, tp_size
            )
        return profile_tpsp(
            self,
            ops_groups,
            max_batched_tokens,
            microchunk_tokens=microchunk_tokens,
            threshold_tokens=threshold_tokens,
            microchunk_tokens_per_rank_min_max=microchunk_tokens_per_rank_min_max,
            microchunk_step_coarse_per_rank=microchunk_step_coarse_per_rank,
            microchunk_step_fine_per_rank=microchunk_step_fine_per_rank,
        )

    def _microchunk_tokens_per_rank_range(
        self, max_batched_tokens: int, tp_size: int
    ) -> tuple[int, int]:
        max_tokens_per_rank = math.ceil(max_batched_tokens / tp_size)
        cap_tokens_per_rank = self.get_max_microchunk_tokens_per_rank()
        if cap_tokens_per_rank is not None:
            max_tokens_per_rank = min(max_tokens_per_rank, cap_tokens_per_rank)
        return 1, max_tokens_per_rank

    def get_compute_engine_count(self, config: object) -> int | None:
        if isinstance(config, CudaTPSPConfig):
            return config.sm_count_target
        return None

    def get_default_compute_engine_count(self) -> int:
        return 0

    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> CudaTPSPContext | None:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if (
            device != self.device
            or group_name != self.group_name
            or device.type != "cuda"
        ):
            return None
        if (
            dtype != torch.bfloat16
            or tp_size < 2
            or hidden_size <= 0
            or hidden_size % tp_size
            or max_batched_tokens <= 0
        ):
            logger.warning(
                "TPSP unavailable for dtype=%s tp_size=%s hidden_size=%s",
                dtype,
                tp_size,
                hidden_size,
            )
            return None
        if (
            device.index is not None
            and device.index >= torch.accelerator.device_count()
        ):
            logger.warning("CUDA TPSP device index is unavailable: %s", device.index)
            return None
        try:
            group = c10d._resolve_process_group(group_name)
        except (RuntimeError, ValueError) as exc:
            logger.warning("CUDA TPSP process group is unavailable: %s", exc)
            return None
        if dist.get_world_size(group) != tp_size:
            logger.warning("TPSP unavailable: group_name and tp_size disagree")
            return None
        if not all(
            hasattr(torch.ops._C, name)
            for name in (
                "tpsp_fused_matmul_reduce_scatter_norm_all_gather",
                "init_tpsp_p2p",
                "get_tpsp_p2p_handles",
                "map_tpsp_p2p",
                "destroy_tpsp_p2p",
            )
        ):
            logger.warning("CUDA TPSP native operator is not built")
            return None
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        if tp_group.device_group is not group:
            logger.warning("CUDA TPSP group is not the active TP group")
            return None
        control_group = tp_group.cpu_group
        if control_group is None:
            logger.warning("CUDA TPSP CPU control group is unavailable")
            return None
        rank = dist.get_rank(group)
        max_microchunk_tokens_per_rank = math.ceil(max_batched_tokens / tp_size)
        max_microchunk_cap = self.get_max_microchunk_tokens_per_rank()
        if max_microchunk_cap is not None:
            max_microchunk_tokens_per_rank = min(
                max_microchunk_tokens_per_rank, max_microchunk_cap
            )
        context = CudaTPSPContext(
            device,
            tp_size,
            rank,
            control_group,
            max_microchunk_tokens_per_rank,
        )
        context.p2p_handle = torch.ops._C.init_tpsp_p2p(
            device.index if device.index is not None else torch.cuda.current_device(),
            tp_size,
            rank,
            max_microchunk_tokens_per_rank,
            hidden_size,
            max_batched_tokens,
        )
        if not self._all_ranks_ready(bool(context.p2p_handle), control_group):
            if context.p2p_handle:
                torch.ops._C.destroy_tpsp_p2p(context.p2p_handle)
            dist.barrier(group=control_group)
            if rank == 0:
                logger.warning(
                    "CUDA TPSP P2P allocation is unavailable; disabling TPSP"
                )
            return None
        local_handles = torch.ops._C.get_tpsp_p2p_handles(context.p2p_handle)
        gathered_handles = [torch.empty_like(local_handles) for _ in range(tp_size)]
        dist.all_gather(gathered_handles, local_handles, group=control_group)
        peer_handles = torch.cat(gathered_handles)
        mapped = torch.ops._C.map_tpsp_p2p(context.p2p_handle, peer_handles)
        if not self._all_ranks_ready(mapped, control_group):
            torch.ops._C.destroy_tpsp_p2p(context.p2p_handle)
            dist.barrier(group=control_group)
            if rank == 0:
                logger.warning("CUDA TPSP P2P mapping is unavailable; disabling TPSP")
            return None
        if rank == 0:
            logger.info(
                "TPSP enabled with direct P2P for %s tokens per rank",
                max_microchunk_tokens_per_rank,
            )
        self._open_context_ids.add(id(context))
        return context

    @staticmethod
    def _all_ranks_ready(ready: bool, group: dist.ProcessGroup) -> bool:
        supported = torch.tensor(int(ready), device="cpu")
        dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=group)
        return bool(supported.item())

    def _profile_context(self, context: object) -> CudaTPSPContext:
        if context is None:
            raise ValueError("CUDA TPSP requires a projection context")
        if (
            not isinstance(context, CudaTPSPContext)
            or id(context) not in self._open_context_ids
        ):
            raise ValueError("CUDA TPSP context belongs to another backend")
        if not context.p2p_handle:
            raise ValueError("CUDA TPSP requires an available P2P context")
        return context

    def set_config(self, handle: object, config: object) -> None:
        context = self._profile_context(handle)
        context.config = self._resolve_config(context, config)

    def _resolve_config(
        self, context: CudaTPSPContext, config: object
    ) -> CudaTPSPConfig:
        if type(config) is int:
            config = CudaTPSPConfig(config)
        if (
            not isinstance(config, CudaTPSPConfig)
            or type(config.microchunk_tokens_per_rank) is not int
            or config.microchunk_tokens_per_rank <= 0
            or type(config.sm_count_target) is not int
            or config.sm_count_target < 0
        ):
            raise ValueError("Invalid CUDA TPSP tuning configuration")
        return config

    def get_tuning_configs(
        self,
        handle: object,
        microchunk_tokens_per_rank: int,
        group_name: str | None = None,
        *,
        compute_engine_counts: Sequence[int] | None = None,
        compute_engine_step: int = 6,
    ) -> tuple[CudaTPSPConfig, ...]:
        self._profile_context(handle)
        if compute_engine_counts is not None and any(
            target != 0 for target in compute_engine_counts
        ):
            raise ValueError("TPSP uses the default SM count target 0")
        return (CudaTPSPConfig(microchunk_tokens_per_rank, 0),)

    def fused_gemm_rs_norm_ag(
        self,
        projection_context: object,
        x: torch.Tensor,
        projection: torch.nn.Module,
        residual: torch.Tensor,
        norm: torch.nn.Module,
        *,
        config: object | None = None,
        norm_type: TPSPNormType = "rms_norm",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed:
            raise RuntimeError("TPSP backend is closed")
        if norm_type not in self.supported_norm_types:
            raise ValueError(f"CUDA TPSP does not support norm type {norm_type}")
        context = self._profile_context(projection_context)
        hidden_size = norm.weight.numel()
        tokens_per_rank = (x.size(0) + context.tp_size - 1) // context.tp_size
        if residual.shape != (tokens_per_rank, hidden_size):
            raise RuntimeError("TPSP residual shard has an unexpected shape")
        local_residual = residual

        weight = projection.weight
        key = (weight.data_ptr(), weight._version)
        cached = getattr(projection, "_tpsp_transposed_weight", None)
        if cached is None or cached[0] != key:
            cached = (key, weight.T.contiguous())
            projection._tpsp_transposed_weight = cached
        a, b = x.contiguous(), cached[1]
        norm_weight = norm.weight
        eps = norm.eps if norm_type == "layer_norm" else norm.variance_epsilon
        projection_bias = projection.bias
        norm_bias = getattr(norm, "bias", None)
        selected = self._resolve_config(
            context, context.config if config is None else config
        )
        microchunk_tokens_per_rank = selected.microchunk_tokens_per_rank
        p2p_handle = context.p2p_handle
        if (
            min(
                (a.size(0) + context.tp_size - 1) // context.tp_size,
                microchunk_tokens_per_rank,
            )
            > context.max_microchunk_tokens_per_rank
        ):
            raise ValueError("CUDA TPSP microchunk exceeds P2P workspace capacity")
        if eps <= 0 or (norm_bias is not None and norm_type != "layer_norm"):
            raise ValueError(
                "Invalid CUDA TPSP normalization or microchunk configuration"
            )
        if (
            any(
                t.dtype != torch.bfloat16 or t.device != context.device
                for t in (a, b, norm_weight, local_residual)
            )
            or a.ndim != 2
            or b.ndim != 2
            or norm_weight.ndim != 1
            or local_residual.ndim != 2
            or a.shape[1] != b.shape[0]
            or b.shape[1] != norm_weight.numel()
        ):
            raise ValueError("CUDA TPSP requires compatible CUDA BF16 inputs")
        for bias in (projection_bias, norm_bias):
            if bias is not None and (
                bias.dtype != torch.bfloat16
                or bias.device != context.device
                or bias.shape != norm_weight.shape
                or not bias.is_contiguous()
            ):
                raise ValueError("CUDA TPSP bias must match the BF16 norm weight")
        reduced, _, gathered = (
            torch.ops._C.tpsp_fused_matmul_reduce_scatter_norm_all_gather(
                a,
                b,
                norm_weight,
                local_residual,
                projection_bias,
                norm_bias,
                eps,
                1 if norm_type == "layer_norm" else 0,
                microchunk_tokens_per_rank,
                context.tp_size,
                p2p_handle,
                selected.sm_count_target,
            )
        )
        return gathered, reduced

    def close(self, context: CudaTPSPContext | None = None) -> None:
        if context is not None:
            context = self._profile_context(context)
            if context.p2p_handle:
                torch.cuda.synchronize(context.device)
                dist.barrier(group=context.control_group)
                torch.ops._C.destroy_tpsp_p2p(context.p2p_handle)
                dist.barrier(group=context.control_group)
                context.p2p_handle = 0
            self._open_context_ids.remove(id(context))
        else:
            self._closed = True
