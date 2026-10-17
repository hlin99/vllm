# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transformers TPSP backends and runtime contexts."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, get_args

import torch
from torch import nn

_LOG = logging.getLogger(__name__)
_STATIC_PROFILE_THRESHOLD_TOKENS = 512
_STATIC_PROFILE_CHUNK_ROWS = 256

TPSPNormType = Literal[
    "layer_norm",
    "layer_norm_offset_weight",
    "rms_norm",
    "rms_norm_offset_weight",
    "group_rms_norm",
]
# Dense decoder hidden-state norm types; gated and multimodal-only norms are
# outside the current TPSP fusion contract.
TPSP_KNOWN_NORM_TYPES: frozenset[TPSPNormType] = frozenset(get_args(TPSPNormType))
_MODEL_NORM_TYPES: dict[str, TPSPNormType] = {
    "vllm.model_executor.models.nemotron.NemotronLayerNorm1P": (
        "layer_norm_offset_weight"
    ),
    "vllm.model_executor.models.k2_horizon.K2HorizonRMSNorm": "group_rms_norm",
}
# Types without TPSP-specific validation stay on the regular forward path.
_TPSP_VALIDATED_NORM_TYPES: frozenset[TPSPNormType] = frozenset(
    {"layer_norm", "rms_norm", "rms_norm_offset_weight"}
)


class TPSPBackend(ABC):
    tpsp_chunk_granularity: int
    tpsp_max_microchunk_rows: int | None = None
    tpsp_max_microchunk_tokens: int | None = None
    supported_norm_types: frozenset[TPSPNormType] = frozenset()

    def __init__(self, group_name: str, device: torch.device) -> None:
        self.group_name = group_name
        self.device = device
        self._closed = False

    @abstractmethod
    def open(
        self,
        *,
        dtype: torch.dtype,
        tp_size: int,
        hidden_size: int,
        max_batched_tokens: int,
        group_name: str,
        device: torch.device,
    ) -> Any | None: ...

    def profile(
        self,
        ops_groups: Sequence[TPSPOpsGroup],
        max_batched_tokens: int,
        *,
        microchunk_tokens: int | None = None,
        threshold_tokens: int | None = None,
    ) -> TPSPContext | None:
        """Create a static baseline context for named ops groups.

        Args:
            ops_groups: Named projection and normalization pairs to configure.
            max_batched_tokens: Maximum scheduler token count.
            microchunk_tokens: Optional global token-size override.
            threshold_tokens: Optional global token threshold override.

        Returns:
            The baseline TPSP context, or None when unsupported.

        """
        return _build_static_tpsp_context(
            self,
            ops_groups,
            max_batched_tokens,
            microchunk_tokens=microchunk_tokens,
            threshold_tokens=threshold_tokens,
        )

    @abstractmethod
    def set_config(self, handle: object, config: object) -> None: ...

    def get_tuning_configs(
        self,
        handle: object,
        chunk: int,
        group_name: str | None = None,
    ) -> Sequence[object]:
        """Return the backend config for a fixed chunk size."""
        return (chunk,)

    def get_max_microchunk_rows(self, tp_size: int) -> int | None:
        if self.tpsp_max_microchunk_tokens is not None:
            return math.ceil(self.tpsp_max_microchunk_tokens / tp_size)
        return self.tpsp_max_microchunk_rows

    @abstractmethod
    def fused_gemm_rs_norm_ag(
        self,
        projection_context: object,
        x: torch.Tensor,
        projection: nn.Module,
        residual: torch.Tensor,
        norm: nn.Module,
        *,
        config: object | None = None,
        norm_type: TPSPNormType = "rms_norm",
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def close(self, context: Any | None = None) -> None: ...


@dataclass(frozen=True)
class TPSPProfile:
    """Shared TPSP activation decision."""

    enabled: bool
    threshold_tokens: int | None
    max_batched_tokens: int
    reason: str = ""

    def is_active(self, num_tokens: int) -> bool:
        if not self.enabled:
            return False
        if self.threshold_tokens is None:
            raise RuntimeError("TPSP has an invalid enabled profile")
        if not 1 <= num_tokens <= self.max_batched_tokens:
            raise ValueError("current_batched_tokens must be within the profiled range")
        return num_tokens >= self.threshold_tokens


@dataclass(frozen=True)
class TPSPOpsGroup:
    """A consecutive projection and normalization pair."""

    name: str
    projection: nn.Module
    norm: nn.Module


@dataclass(frozen=True)
class TPSPContext:
    """Shared backend and named per-group handles."""

    profile: TPSPProfile
    backend: Any
    handles: dict[str, object]


class UnsupportedTPSPNormError(ValueError):
    """Normalization not supported by the Transformers TPSP path."""


def get_tpsp_norm_type(norm: nn.Module) -> TPSPNormType:
    """Identify known normalization implementations without guessing by attributes."""
    from vllm.model_executor.layers.layernorm import (
        GemmaRMSNorm,
        LayerNorm,
        RMSNorm,
    )
    from vllm.model_executor.models.transformers.fusers.rms_norm import (
        TPAwareGemmaRMSNorm,
        TPAwareRMSNorm,
    )

    norm_cls = type(norm)
    norm_class_name = f"{norm_cls.__module__}.{norm_cls.__name__}"
    if norm_class_name in _MODEL_NORM_TYPES:
        norm_type = _MODEL_NORM_TYPES[norm_class_name]
        if (
            norm_class_name == "vllm.model_executor.models.k2_horizon.K2HorizonRMSNorm"
            and norm.n_groups == 1
        ):
            norm_type = "rms_norm"
    elif norm_cls in (GemmaRMSNorm, TPAwareGemmaRMSNorm):
        norm_type = "rms_norm_offset_weight"
    elif norm_cls in (nn.LayerNorm, LayerNorm):
        norm_type = "layer_norm"
    elif norm_cls in (RMSNorm, TPAwareRMSNorm):
        norm_type = "rms_norm"
        if norm.variance_size_override is not None:
            raise UnsupportedTPSPNormError(
                "TPSP RMSNorm does not support variance_size_override"
            )
    else:
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )

    eps = (
        getattr(norm, "eps", None)
        if norm_type in ("layer_norm", "layer_norm_offset_weight")
        else getattr(norm, "variance_epsilon", None)
    )
    weight = getattr(norm, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 1:
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    bias = getattr(norm, "bias", None)
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.ndim != 1
        or bias.shape != weight.shape
    ):
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    if norm_type not in ("layer_norm", "layer_norm_offset_weight") and bias is not None:
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    if not isinstance(eps, (float, int)):
        raise UnsupportedTPSPNormError(
            f"Unsupported TPSP normalization {type(norm).__name__}"
        )
    if eps <= 0:
        raise ValueError("TPSP norm_eps must be positive")
    return norm_type


def _unsupported_tpsp_norm_types(
    backend: TPSPBackend, ops_groups: Sequence[TPSPOpsGroup]
) -> frozenset[TPSPNormType]:
    unknown_types = backend.supported_norm_types - TPSP_KNOWN_NORM_TYPES
    if unknown_types:
        raise ValueError(
            f"TPSP backend declares unknown norm types: {sorted(unknown_types)}"
        )
    norm_types = {get_tpsp_norm_type(entry.norm) for entry in ops_groups}
    return frozenset(
        (norm_types - backend.supported_norm_types)
        | (norm_types - _TPSP_VALIDATED_NORM_TYPES)
    )


def tpsp_shard_residual(residual: torch.Tensor) -> torch.Tensor:
    """Return the current TP rank's padded shard of a full residual."""
    from vllm.distributed.parallel_state import get_tp_group

    if residual.ndim != 2:
        raise ValueError("TPSP full residual must be a 2D tensor")
    group = get_tp_group()
    rows = math.ceil(residual.size(0) / group.world_size)
    start = group.rank_in_group * rows
    count = min(rows, max(0, residual.size(0) - start))
    if count == rows:
        return residual[start : start + rows].contiguous()
    shard = residual.new_zeros((rows, residual.size(1)))
    shard[:count] = residual[start : start + count]
    return shard


def _build_static_tpsp_context(
    backend: Any,
    ops_groups: Sequence[TPSPOpsGroup],
    max_batched_tokens: int,
    *,
    microchunk_tokens: int | None = None,
    threshold_tokens: int | None = None,
) -> TPSPContext | None:
    """Build the static TPSP context without profiling or scanning."""
    if backend is None:
        raise ValueError("TPSP requires a backend")
    if not ops_groups or any(not entry.name for entry in ops_groups):
        raise ValueError("TPSP requires named ops groups")
    if len({entry.name for entry in ops_groups}) != len(ops_groups):
        raise ValueError("TPSP ops group names must be unique")
    if threshold_tokens is not None and (
        type(threshold_tokens) is not int or threshold_tokens <= 0
    ):
        raise ValueError("TPSP threshold tokens must be positive")
    threshold = (
        _STATIC_PROFILE_THRESHOLD_TOKENS
        if threshold_tokens is None
        else threshold_tokens
    )
    if threshold <= 0:
        raise ValueError("TPSP threshold tokens must be positive")
    if threshold > max_batched_tokens:
        _LOG.warning(
            "TPSP using regular forward: threshold=%d exceeds max_batched_tokens=%d",
            threshold,
            max_batched_tokens,
        )
        return None

    first = ops_groups[0]
    norm_weight = getattr(first.norm, "weight", None)
    if not isinstance(norm_weight, torch.Tensor):
        raise ValueError("TPSP normalizations require a weight tensor")
    hidden_size = norm_weight.numel()
    for entry in ops_groups:
        _projection_input_width(entry.projection)
        weight = getattr(entry.norm, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.numel() != hidden_size:
            raise ValueError("TPSP ops groups require matching norm hidden sizes")
    if unsupported := _unsupported_tpsp_norm_types(backend, ops_groups):
        _LOG.warning(
            "TPSP using regular forward: runtime or backend does not support "
            "norm types %s",
            ", ".join(sorted(unsupported)),
        )
        return None

    from vllm.distributed.parallel_state import get_tp_group

    tp_group = get_tp_group()
    if tp_group.world_size < 2 or hidden_size % tp_group.world_size:
        _LOG.warning("TPSP using regular forward: unsupported TP configuration")
        return None
    chunk_rows = _STATIC_PROFILE_CHUNK_ROWS
    if microchunk_tokens is not None:
        chunk_rows = _microchunk_rows_from_tokens(
            backend, microchunk_tokens, tp_group.world_size
        )

    handles: dict[str, object] = {}
    enabled = False
    try:
        for entry in ops_groups:
            parameter = entry.projection.weight
            handle = backend.open(
                dtype=parameter.dtype,
                tp_size=tp_group.world_size,
                hidden_size=hidden_size,
                max_batched_tokens=max_batched_tokens,
                group_name=tp_group.device_group.group_name,
                device=parameter.device,
            )
            if handle is None:
                _LOG.warning(
                    "TPSP using regular forward: fused backend unavailable for %s",
                    entry.name,
                )
                return None
            handles[entry.name] = handle
            configs = backend.get_tuning_configs(
                handle,
                chunk_rows,
                entry.name,
            )
            if not configs:
                _LOG.warning(
                    "TPSP using regular forward: no baseline config for %s",
                    entry.name,
                )
                return None
            backend.set_config(handle, configs[0])

        enabled = True
        return TPSPContext(
            TPSPProfile(True, threshold, max_batched_tokens),
            backend,
            handles,
        )
    finally:
        if not enabled:
            for handle in handles.values():
                backend.close(handle)
            backend.close()


def _microchunk_rows_from_tokens(
    backend: TPSPBackend, microchunk_tokens: int, tp_size: int
) -> int:
    if type(microchunk_tokens) is not int or microchunk_tokens <= 0:
        raise ValueError("TPSP microchunk tokens must be a positive integer")
    granularity = backend.tpsp_chunk_granularity
    if type(granularity) is not int or granularity <= 0:
        raise ValueError("TPSP chunk granularity must be a positive integer")
    local_rows = math.ceil(microchunk_tokens / tp_size)
    rows = max(granularity, math.ceil(local_rows / granularity) * granularity)
    max_rows = backend.get_max_microchunk_rows(tp_size)
    if max_rows is not None and max_rows >= granularity:
        rows = min(rows, max_rows // granularity * granularity)
    return rows


def _projection_input_width(projection: nn.Module) -> int:
    input_width = getattr(projection, "input_size_per_partition", None)
    if type(input_width) is int and input_width > 0:
        return input_width
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("TPSP projections must have a two-dimensional weight")
    input_width = weight.shape[1]
    if input_width <= 0:
        raise ValueError("TPSP projection input width must be positive")
    return input_width
