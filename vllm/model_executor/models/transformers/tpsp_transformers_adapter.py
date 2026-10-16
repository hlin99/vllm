# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense TPSP adapter for supported Hugging Face decoder architectures."""

import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.models.transformers.tpsp_runtime import (
    TPSPContext,
    TPSPOpsGroup,
    get_tpsp_norm_type,
    tpsp_shard_residual,
)

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


@dataclass(frozen=True)
class _DecoderLayerOps:
    layer: nn.Module
    attention: nn.Module
    o_proj: nn.Module
    input_norm: nn.Module
    post_attention_norm: nn.Module
    mlp: nn.Module
    down_proj: nn.Module


_FORWARD_STATE: ContextVar[_ForwardState | None] = ContextVar(
    "transformers_decoder_tpsp_state", default=None
)


def _get_norm(layer: nn.Module, names: tuple[str, ...], role: str) -> nn.Module:
    for name in names:
        norm = getattr(layer, name, None)
        if isinstance(norm, nn.Module):
            get_tpsp_norm_type(norm)
            return norm
    raise ValueError(f"Transformers TPSP requires a supported {role} normalization")


def _get_projection_parent(
    layer: nn.Module, projection_name: str, layer_index: int
) -> tuple[nn.Module, nn.Module]:
    matches = [
        (module, getattr(module, projection_name))
        for module in layer.children()
        if isinstance(getattr(module, projection_name, None), nn.Module)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Transformers TPSP layer {layer_index} must have exactly one "
            f"child with {projection_name}"
        )
    parent, projection = matches[0]
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError(
            f"Transformers TPSP requires a linear {projection_name} projection"
        )
    if (
        getattr(projection, "output_size", None) is None
        and getattr(projection, "out_features", None) is None
    ):
        raise ValueError(
            f"Transformers TPSP cannot determine {projection_name} output size"
        )
    return parent, projection


def _decoder_layers(decoder: nn.Module) -> nn.ModuleList:
    for name in ("layers", "h", "blocks"):
        layers = getattr(decoder, name, None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return layers
    raise ValueError("Transformers model has an unsupported decoder layer layout")


def _final_norm(decoder: nn.Module, model: nn.Module) -> nn.Module:
    names = ("norm", "final_norm", "final_layernorm", "final_layer_norm")
    for module in (decoder, model):
        for name in names:
            norm = getattr(module, name, None)
            if isinstance(norm, nn.Module):
                get_tpsp_norm_type(norm)
                return norm
    raise ValueError("Transformers model has no supported final normalization")


def _decoder_layer_ops(layer: nn.Module, layer_index: int) -> _DecoderLayerOps:
    attention, o_proj = _get_projection_parent(layer, "o_proj", layer_index)
    mlp, down_proj = _get_projection_parent(layer, "down_proj", layer_index)
    layer_parameters = inspect.signature(layer.forward).parameters
    if "hidden_states" not in layer_parameters:
        raise ValueError(
            f"Transformers TPSP layer {layer_index} must accept hidden_states"
        )
    attention_parameters = inspect.signature(attention.forward).parameters
    hidden_states_parameter = attention_parameters.get("hidden_states")
    attention_accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in attention_parameters.values()
    )
    if (hidden_states_parameter is None and not attention_accepts_kwargs) or (
        hidden_states_parameter is not None
        and hidden_states_parameter.kind is inspect.Parameter.POSITIONAL_ONLY
    ):
        raise ValueError(
            f"Transformers TPSP layer {layer_index} attention must accept hidden_states"
        )
    return _DecoderLayerOps(
        layer=layer,
        attention=attention,
        o_proj=o_proj,
        input_norm=_get_norm(
            layer,
            ("input_layernorm", "input_layer_norm", "pre_attention_layernorm"),
            "input",
        ),
        post_attention_norm=_get_norm(
            layer,
            ("post_attention_layernorm", "post_attention_layer_norm"),
            "post-attention",
        ),
        mlp=mlp,
        down_proj=down_proj,
    )


def _validate_layer_shapes(
    layers: tuple[_DecoderLayerOps, ...], final_norm: nn.Module
) -> None:
    hidden_size = layers[0].input_norm.weight.numel()
    projection_shapes = {
        "o_proj": tuple(layers[0].o_proj.weight.shape),
        "down_proj": tuple(layers[0].down_proj.weight.shape),
    }
    post_norm_type = get_tpsp_norm_type(layers[0].post_attention_norm)
    down_norms = [
        layers[index + 1].input_norm if index + 1 < len(layers) else final_norm
        for index in range(len(layers))
    ]
    down_norm_type = get_tpsp_norm_type(down_norms[0])

    for index, ops in enumerate(layers):
        if (
            ops.input_norm.weight.numel() != hidden_size
            or ops.post_attention_norm.weight.numel() != hidden_size
            or down_norms[index].weight.numel() != hidden_size
        ):
            raise ValueError(
                f"Transformers TPSP layer {index} has incompatible norm dimensions"
            )
        if get_tpsp_norm_type(ops.post_attention_norm) != post_norm_type:
            raise ValueError("Transformers TPSP layers require matching norm types")
        if get_tpsp_norm_type(down_norms[index]) != down_norm_type:
            raise ValueError("Transformers TPSP layers require matching norm types")
        for name, projection in (
            ("o_proj", ops.o_proj),
            ("down_proj", ops.down_proj),
        ):
            if tuple(projection.weight.shape) != projection_shapes[name]:
                raise ValueError(
                    f"Transformers TPSP layers require matching {name} shapes"
                )
            output_size = getattr(projection, "output_size", None)
            if output_size is None:
                output_size = getattr(projection, "out_features", None)
            if output_size is None:
                output_size = projection.weight.shape[0]
            if output_size != hidden_size:
                raise ValueError(
                    f"Transformers TPSP {name} output does not match hidden size"
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


def _bind_layer_forward(ops: _DecoderLayerOps, next_norm: nn.Module) -> None:
    layer = ops.layer
    original_forward = layer.forward
    layer._tpsp_original_forward = original_forward
    layer_signature = inspect.signature(original_forward)
    attention_signature = inspect.signature(ops.attention.forward)

    def forward(self, *args, **kwargs):
        state = _FORWARD_STATE.get()
        if state is None or not state.active:
            return self._tpsp_original_forward(*args, **kwargs)

        bound = layer_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        hidden_states = arguments["hidden_states"]
        if arguments.get("output_attentions", False):
            raise ValueError("Transformers TPSP does not support output_attentions")
        if arguments.get("use_cache", False):
            raise ValueError("Transformers TPSP does not support use_cache")
        if any(
            arguments.get(name) is not None
            for name in ("past_key_value", "past_key_values")
        ):
            raise ValueError("Transformers TPSP does not support HF KV caches")

        hidden_size = hidden_states.shape[-1]
        if state.residual is None:
            residual = hidden_states.reshape(-1, hidden_size).contiguous()
            hidden_states = ops.input_norm(hidden_states)
        else:
            residual = state.residual

        projection = ops.o_proj
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
        ops.attention(hidden_states=hidden_states, **attention_kwargs)
        attention_input = _take_projection_input(state, projection)

        if state.residual is None:
            residual = tpsp_shard_residual(residual)
        assert state.context is not None
        hidden_states, residual = state.context.backend.fused_gemm_rs_norm_ag(
            state.context.handles["o_proj"],
            attention_input,
            projection,
            residual,
            ops.post_attention_norm,
            norm_type=get_tpsp_norm_type(ops.post_attention_norm),
        )

        projection = ops.down_proj
        state.expected_projection = projection
        ops.mlp(hidden_states.reshape(1, -1, hidden_size))
        mlp_input = _take_projection_input(state, projection)
        hidden_states, residual = state.context.backend.fused_gemm_rs_norm_ag(
            state.context.handles["down_proj"],
            mlp_input,
            projection,
            residual,
            next_norm,
            norm_type=get_tpsp_norm_type(next_norm),
        )
        state.residual = residual
        return hidden_states.reshape(1, -1, hidden_size)

    layer.forward = MethodType(forward, layer)


class TransformersDenseTPSP:
    """Install TPSP for Transformers decoders with supported dense layer paths."""

    def __init__(self, model: nn.Module, vllm_config: "VllmConfig") -> None:
        if vllm_config.quant_config is not None or vllm_config.lora_config is not None:
            raise ValueError("TPSP does not support quantization or LoRA")

        get_decoder = getattr(model, "get_decoder", None)
        decoder = get_decoder() if callable(get_decoder) else model
        if not isinstance(decoder, nn.Module):
            raise ValueError("Transformers model has no supported decoder")
        layers = _decoder_layers(decoder)
        layer_ops = tuple(
            _decoder_layer_ops(layer, index) for index, layer in enumerate(layers)
        )
        final_norm = _final_norm(decoder, model)
        _validate_layer_shapes(layer_ops, final_norm)

        self.model = model
        self.decoder = decoder
        self.layers = layer_ops
        self.final_norm = final_norm
        self.model.do_not_compile = True
        self.max_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.context: TPSPContext | None = None
        self.profile_attempted = False

        for ops in layer_ops:
            for projection in (ops.o_proj, ops.down_proj):
                projection._tpsp_original_forward = projection.forward
                projection.forward = MethodType(_projection_forward, projection)

        final_norm._tpsp_original_forward = final_norm.forward
        final_norm.forward = MethodType(_norm_forward, final_norm)
        for index, ops in enumerate(layer_ops):
            next_norm = (
                layer_ops[index + 1].input_norm
                if index + 1 < len(layer_ops)
                else final_norm
            )
            _bind_layer_forward(ops, next_norm)

    def _maybe_profile(self) -> None:
        if self.profile_attempted:
            return
        self.profile_attempted = True

        from vllm.distributed.parallel_state import get_tp_group
        from vllm.platforms import current_platform

        backend_cls = current_platform.get_tpsp_backend_cls()
        if backend_cls is None:
            logger.warning("Transformers TPSP using regular forward: no backend")
            return

        first_layer = self.layers[0]
        backend = backend_cls(
            get_tp_group().device_group.group_name,
            first_layer.o_proj.weight.device,
        )
        down_norm = (
            self.layers[1].input_norm if len(self.layers) > 1 else self.final_norm
        )
        self.context = backend.profile(
            [
                TPSPOpsGroup(
                    "o_proj",
                    first_layer.o_proj,
                    first_layer.post_attention_norm,
                ),
                TPSPOpsGroup(
                    "down_proj",
                    first_layer.down_proj,
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
            skip_norm=self.final_norm,
        )
        token = _FORWARD_STATE.set(state)
        try:
            yield
        finally:
            _FORWARD_STATE.reset(token)
