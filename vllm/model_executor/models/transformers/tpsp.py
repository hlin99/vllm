# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPSP adapter for Hugging Face Llama decoder layers."""

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.tpsp import TPSPContext, TPSPOpsGroup, tpsp_shard_residual

if TYPE_CHECKING:
    from collections.abc import Iterator

    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class _ForwardState:
    context: TPSPContext | None
    active: bool
    residual: torch.Tensor | None
    expected_projection: nn.Module | None = None
    projection_input: torch.Tensor | None = None
    skip_norm: nn.Module | None = None


_FORWARD_STATE: ContextVar[_ForwardState | None] = ContextVar(
    "transformers_llama_tpsp_state", default=None
)


def _projection_forward(projection: nn.Module, x: torch.Tensor) -> torch.Tensor:
    state = _FORWARD_STATE.get()
    if state is None or not state.active or state.expected_projection is not projection:
        return projection._tpsp_original_forward(x)

    state.projection_input = x
    output_size = getattr(projection, "output_size", None)
    if output_size is None:
        output_size = getattr(projection, "out_features", None)
    if output_size is None:
        raise TypeError(f"Unsupported TPSP projection {type(projection).__name__}")
    return x.new_empty((*x.shape[:-1], output_size))


def _norm_forward(norm: nn.Module, x: torch.Tensor, *args, **kwargs):
    state = _FORWARD_STATE.get()
    if state is not None and state.active and state.skip_norm is norm:
        return x
    return norm._tpsp_original_forward(x, *args, **kwargs)


def _take_projection_input(state: _ForwardState, projection: nn.Module) -> torch.Tensor:
    if state.expected_projection is not projection:
        raise RuntimeError("TPSP projection dispatch is out of order")
    if state.projection_input is None:
        raise RuntimeError(
            f"TPSP did not capture the input to {type(projection).__name__}"
        )
    x = state.projection_input
    state.expected_projection = None
    state.projection_input = None
    return x.reshape(-1, x.shape[-1]).contiguous()


def _bind_layer_forward(layer: nn.Module, next_norm: nn.Module) -> None:
    original_forward = layer.forward
    layer._tpsp_original_forward = original_forward
    layer_signature = inspect.signature(original_forward)
    attention_signature = inspect.signature(layer.self_attn.forward)

    def forward(self, *args, **kwargs):
        state = _FORWARD_STATE.get()
        if state is None or not state.active:
            return self._tpsp_original_forward(*args, **kwargs)

        bound = layer_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        hidden_states = arguments["hidden_states"]
        if arguments.get("output_attentions", False):
            raise ValueError(
                "Transformers Llama TPSP does not support output_attentions"
            )
        if arguments.get("use_cache", False):
            raise ValueError("Transformers Llama TPSP does not support use_cache")
        if any(
            arguments.get(name) is not None
            for name in ("past_key_value", "past_key_values")
        ):
            raise ValueError("Transformers Llama TPSP does not support HF KV caches")

        hidden_size = hidden_states.shape[-1]
        if state.residual is None:
            residual = hidden_states.reshape(-1, hidden_size).contiguous()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            residual = state.residual

        projection = self.self_attn.o_proj
        state.expected_projection = projection
        layer_kwargs_name = next(
            (
                name
                for name, parameter in layer_signature.parameters.items()
                if parameter.kind is inspect.Parameter.VAR_KEYWORD
            ),
            None,
        )
        attention_accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in attention_signature.parameters.values()
        )
        attention_kwargs = {
            name: value
            for name, value in arguments.items()
            if name != "hidden_states"
            and name != layer_kwargs_name
            and name != "output_attentions"
            and name in attention_signature.parameters
        }
        if attention_accepts_kwargs:
            attention_kwargs.update(
                {
                    name: value
                    for name, value in arguments.items()
                    if name != "hidden_states"
                    and name != layer_kwargs_name
                    and name not in attention_kwargs
                    and name != "output_attentions"
                }
            )
            if layer_kwargs_name is not None:
                attention_kwargs.update(arguments[layer_kwargs_name])
        self.self_attn(hidden_states=hidden_states, **attention_kwargs)
        attention_input = _take_projection_input(state, projection)

        if state.residual is None:
            residual = tpsp_shard_residual(residual)
        assert state.context is not None
        hidden_states, residual = state.context.backend.fused_gemm_rs_norm_ag(
            state.context.handles["o_proj"],
            attention_input,
            projection,
            residual,
            self.post_attention_layernorm,
        )

        projection = self.mlp.down_proj
        state.expected_projection = projection
        self.mlp(hidden_states.reshape(1, -1, hidden_size))
        mlp_input = _take_projection_input(state, projection)
        hidden_states, residual = state.context.backend.fused_gemm_rs_norm_ag(
            state.context.handles["down_proj"],
            mlp_input,
            projection,
            residual,
            next_norm,
        )
        state.residual = residual
        return hidden_states.reshape(1, -1, hidden_size)

    layer.forward = MethodType(forward, layer)


class TransformersLlamaTPSP:
    """Install the Llama TPSP path into a Transformers decoder model."""

    def __init__(self, model: nn.Module, vllm_config: "VllmConfig") -> None:
        if getattr(model.config, "model_type", None) != "llama":
            raise ValueError(
                "Transformers TPSP currently supports only Llama architectures"
            )
        if vllm_config.quant_config is not None or vllm_config.lora_config is not None:
            raise ValueError("TPSP does not support quantization or LoRA")

        layers = getattr(model, "layers", None)
        final_norm = getattr(model, "norm", None)
        if not isinstance(layers, nn.ModuleList) or not layers or final_norm is None:
            raise ValueError("Transformers Llama model has an unsupported layer layout")

        self.model = model
        self.model.do_not_compile = True
        self.max_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.context: TPSPContext | None = None
        self.profile_attempted = False

        for layer in layers:
            for projection in (layer.self_attn.o_proj, layer.mlp.down_proj):
                if not hasattr(projection, "weight"):
                    raise ValueError(
                        "Transformers Llama TPSP requires linear projections"
                    )
                projection._tpsp_original_forward = projection.forward
                projection.forward = MethodType(_projection_forward, projection)

        final_norm._tpsp_original_forward = final_norm.forward
        final_norm.forward = MethodType(_norm_forward, final_norm)
        for index, layer in enumerate(layers):
            next_norm = (
                layers[index + 1].input_layernorm
                if index + 1 < len(layers)
                else final_norm
            )
            _bind_layer_forward(layer, next_norm)

    def _maybe_profile(self) -> None:
        if self.profile_attempted:
            return
        self.profile_attempted = True

        from vllm.distributed.parallel_state import get_tp_group
        from vllm.platforms import current_platform

        backend_cls = current_platform.get_tpsp_backend_cls()
        if backend_cls is None:
            logger.warning("Transformers Llama TPSP using regular forward: no backend")
            return

        first_layer = self.model.layers[0]
        backend = backend_cls(
            get_tp_group().device_group.group_name,
            first_layer.self_attn.o_proj.weight.device,
        )
        down_norm = (
            self.model.layers[1].input_layernorm
            if len(self.model.layers) > 1
            else self.model.norm
        )
        self.context = backend.profile(
            [
                TPSPOpsGroup(
                    "o_proj",
                    first_layer.self_attn.o_proj,
                    first_layer.post_attention_layernorm,
                ),
                TPSPOpsGroup(
                    "down_proj",
                    first_layer.mlp.down_proj,
                    down_norm,
                ),
            ],
            self.max_batched_tokens,
        )

    @contextmanager
    def forward_context(self, num_tokens: int) -> "Iterator[None]":
        self._maybe_profile()
        context = self.context
        active = context is not None and context.profile.is_active(num_tokens)
        state = _ForwardState(
            context=context,
            active=active,
            residual=None,
            skip_norm=self.model.norm,
        )
        token = _FORWARD_STATE.set(state)
        try:
            yield
        finally:
            _FORWARD_STATE.reset(token)
