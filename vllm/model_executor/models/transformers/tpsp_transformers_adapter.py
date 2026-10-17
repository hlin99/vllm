# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense TPSP adapter for supported Hugging Face decoder architectures."""

import ast
import inspect
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    recover_forward,
)
from vllm.model_executor.models.transformers.tpsp_runtime import (
    TPSPContext,
    TPSPOpsGroup,
    UnsupportedTPSPModelError,
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
    attention_name: str
    o_proj: nn.Module
    input_norm: nn.Module
    post_attention_norm: nn.Module
    post_attention_norm_name: str
    mlp: nn.Module
    mlp_name: str
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
    child_names = {id(child): name for name, child in layer.named_children()}
    attention_name = child_names.get(id(attention))
    mlp_name = child_names.get(id(mlp))
    if attention_name is None or mlp_name is None:
        raise ValueError(f"Transformers TPSP layer {layer_index} has invalid modules")
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
    post_attention_norm = _get_norm(
        layer,
        ("post_attention_layernorm", "post_attention_layer_norm"),
        "post-attention",
    )
    post_attention_norm_name = next(
        (
            name
            for name, module in layer.named_children()
            if module is post_attention_norm
        ),
        None,
    )
    if post_attention_norm_name is None:
        raise ValueError(
            f"Transformers TPSP layer {layer_index} has no direct post-attention norm"
        )
    return _DecoderLayerOps(
        layer=layer,
        attention=attention,
        attention_name=attention_name,
        o_proj=o_proj,
        input_norm=_get_norm(
            layer,
            ("input_layernorm", "input_layer_norm", "pre_attention_layernorm"),
            "input",
        ),
        post_attention_norm=post_attention_norm,
        post_attention_norm_name=post_attention_norm_name,
        mlp=mlp,
        mlp_name=mlp_name,
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
        return projection._original_forward(x)

    if state.projection_input is not None:
        raise RuntimeError("TPSP projection was called more than once in a stage")
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
        state.skip_norm = None
        return x
    return norm._original_forward(x, *args, **kwargs)


def _calls_child(statement: ast.stmt, child_name: str) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == child_name
        for node in ast.walk(statement)
    )


def _residual_add(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    value = statement.value
    if (
        not isinstance(target, ast.Name)
        or target.id != "hidden_states"
        or not isinstance(value, ast.BinOp)
        or not isinstance(value.op, ast.Add)
        or not isinstance(value.left, ast.Name)
        or not isinstance(value.right, ast.Name)
    ):
        return None
    if value.left.id == "hidden_states":
        return value.right.id
    if value.right.id == "hidden_states":
        return value.left.id
    return None


def _assigns_name(statement: ast.stmt, name: str) -> bool:
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and node.id == name
        for node in ast.walk(statement)
    )


def _assigns_from_name(statement: ast.stmt, target: str, source: str) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == target
        and isinstance(statement.value, ast.Name)
        and statement.value.id == source
    )


def _assignment_contains_name(statement: ast.stmt, name: str) -> bool:
    return isinstance(statement, ast.Assign) and any(
        isinstance(node, ast.Name) and node.id == name
        for target in statement.targets
        for node in ast.walk(target)
    )


def _replace_with_stage_call(
    statement: ast.Assign, method_name: str, residual_name: str
) -> None:
    statement.value = ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="self", ctx=ast.Load()),
            attr=method_name,
            ctx=ast.Load(),
        ),
        args=[ast.Name(id=residual_name, ctx=ast.Load())],
        keywords=[],
    )


def _rewrite_layer_forward(
    ops: _DecoderLayerOps,
) -> tuple[ast.FunctionDef, Callable]:
    try:
        funcdef, source = recover_forward(type(ops.layer))
    except (AttributeError, OSError, SyntaxError, TypeError, ValueError) as exc:
        raise UnsupportedTPSPModelError(
            f"cannot read layer forward for {type(ops.layer).__name__}: {exc}"
        ) from exc

    body = funcdef.body
    attention_calls = [
        index
        for index, statement in enumerate(body)
        if _calls_child(statement, ops.attention_name)
    ]
    mlp_calls = [
        index
        for index, statement in enumerate(body)
        if _calls_child(statement, ops.mlp_name)
    ]
    if len(attention_calls) != 1 or len(mlp_calls) != 1:
        raise UnsupportedTPSPModelError(
            f"{type(ops.layer).__name__}.forward must call attention and MLP "
            "exactly once at the top level"
        )
    attention_index, mlp_index = attention_calls[0], mlp_calls[0]
    if attention_index >= mlp_index:
        raise UnsupportedTPSPModelError(
            f"{type(ops.layer).__name__}.forward has an unsupported attention/MLP order"
        )
    if not _assignment_contains_name(body[attention_index], "hidden_states") or not (
        _assignment_contains_name(body[mlp_index], "hidden_states")
    ):
        raise UnsupportedTPSPModelError(
            f"{type(ops.layer).__name__}.forward must assign attention and MLP "
            "outputs to hidden_states"
        )

    attention_adds = [
        index
        for index in range(attention_index + 1, mlp_index)
        if _residual_add(body[index]) is not None
    ]
    mlp_adds = [
        index
        for index in range(mlp_index + 1, len(body))
        if _residual_add(body[index]) is not None
    ]
    if len(attention_adds) != 1 or len(mlp_adds) != 1:
        raise UnsupportedTPSPModelError(
            f"{type(ops.layer).__name__}.forward must have one residual add "
            "after attention and one after MLP"
        )
    attention_add, mlp_add = attention_adds[0], mlp_adds[0]
    if not any(
        _calls_child(body[index], ops.post_attention_norm_name)
        for index in range(attention_add + 1, mlp_index)
    ):
        raise UnsupportedTPSPModelError(
            f"{type(ops.layer).__name__}.forward has no post-attention norm "
            "after its residual add"
        )

    attention_add_statement = body[attention_add]
    mlp_add_statement = body[mlp_add]
    if not isinstance(attention_add_statement, ast.Assign) or not isinstance(
        mlp_add_statement, ast.Assign
    ):
        raise UnsupportedTPSPModelError(
            f"{type(ops.layer).__name__}.forward has unsupported residual statements"
        )
    attention_residual = _residual_add(attention_add_statement)
    mlp_residual = _residual_add(mlp_add_statement)
    assert attention_residual is not None and mlp_residual is not None
    if not any(
        _assigns_from_name(statement, attention_residual, "hidden_states")
        for statement in body[:attention_index]
    ) or not any(
        _assigns_from_name(statement, mlp_residual, "hidden_states")
        for statement in body[attention_add + 1 : mlp_index]
    ):
        raise UnsupportedTPSPModelError(
            f"{type(ops.layer).__name__}.forward has unsupported residual setup"
        )
    if any(
        _assigns_name(body[index], "hidden_states")
        for index in range(attention_index + 1, attention_add)
    ) or any(
        _assigns_name(body[index], "hidden_states")
        for index in range(mlp_index + 1, mlp_add)
    ):
        raise UnsupportedTPSPModelError(
            f"{type(ops.layer).__name__}.forward modifies hidden_states between "
            "the fused projection and residual add"
        )

    def expect_call(name: str) -> ast.Expr:
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr=name,
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        )

    _replace_with_stage_call(
        attention_add_statement, "_finish_o_proj", attention_residual
    )
    _replace_with_stage_call(mlp_add_statement, "_finish_down_proj", mlp_residual)
    body.insert(attention_index, expect_call("_expect_o_proj"))
    if attention_index < mlp_index:
        mlp_index += 1
        mlp_add += 1
    body.insert(mlp_index, expect_call("_expect_down_proj"))

    try:
        return funcdef, compile_forward(funcdef, source)
    except (SyntaxError, TypeError, ValueError) as exc:
        raise UnsupportedTPSPModelError(
            f"cannot rewrite layer forward for {type(ops.layer).__name__}: {exc}"
        ) from exc


def _expect_projection(projection: nn.Module) -> None:
    state = _FORWARD_STATE.get()
    if state is None or not state.active:
        return
    if state.expected_projection is not None or state.projection_input is not None:
        raise RuntimeError("TPSP projection stages overlap")
    state.expected_projection = projection


def _finish_projection(
    projection: nn.Module,
    norm: nn.Module,
    group_name: str,
    residual: torch.Tensor,
) -> torch.Tensor:
    state = _FORWARD_STATE.get()
    if (
        state is None
        or not state.active
        or state.context is None
        or state.expected_projection is not projection
        or state.projection_input is None
    ):
        raise RuntimeError(f"TPSP did not capture {group_name} projection input")

    output_shape = residual.shape
    projection_input = state.projection_input.reshape(
        -1, state.projection_input.shape[-1]
    ).contiguous()
    state.projection_input = None
    state.expected_projection = None
    if state.residual is None:
        residual = tpsp_shard_residual(
            residual.reshape(-1, residual.shape[-1]).contiguous()
        )
    else:
        residual = state.residual
    hidden_states, state.residual = state.context.backend.fused_gemm_rs_norm_ag(
        state.context.handles[group_name],
        projection_input,
        projection,
        residual,
        norm,
        norm_type=get_tpsp_norm_type(norm),
    )
    state.skip_norm = norm
    return hidden_states.reshape(output_shape)


def _bind_layer_forward(
    ops: _DecoderLayerOps,
    next_norm: nn.Module,
    rewritten_forward: Callable,
) -> None:
    layer = ops.layer
    original_forward = layer.forward
    layer._original_forward = original_forward
    signature = inspect.signature(original_forward)
    layer._partial_forward = MethodType(rewritten_forward, layer)

    def bind_stage(name: str, projection: nn.Module, norm: nn.Module, group: str):
        def expect(self):
            _expect_projection(projection)

        def finish(self, residual: torch.Tensor):
            return _finish_projection(projection, norm, group, residual)

        setattr(layer, f"_expect_{name}", MethodType(expect, layer))
        setattr(layer, f"_finish_{name}", MethodType(finish, layer))

    bind_stage("o_proj", ops.o_proj, ops.post_attention_norm, "o_proj")
    bind_stage("down_proj", ops.down_proj, next_norm, "down_proj")

    def forward(self, *args, **kwargs):
        state = _FORWARD_STATE.get()
        if state is None or not state.active:
            return self._original_forward(*args, **kwargs)

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = bound.arguments
        if arguments.get("output_attentions", False):
            raise ValueError("Transformers TPSP does not support output_attentions")
        if arguments.get("use_cache", False):
            raise ValueError("Transformers TPSP does not support use_cache")
        if any(
            arguments.get(name) is not None
            for name in ("past_key_value", "past_key_values")
        ):
            raise ValueError("Transformers TPSP does not support HF KV caches")
        return self._partial_forward(*args, **kwargs)

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
        self.max_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.microchunk_tokens = vllm_config.model_config.tpsp_microchunk_tokens
        self.threshold_tokens = vllm_config.model_config.tpsp_threshold_tokens
        self.context: TPSPContext | None = None
        self.profile_attempted = False

        next_norms = tuple(
            layer_ops[index + 1].input_norm
            if index + 1 < len(layer_ops)
            else final_norm
            for index in range(len(layer_ops))
        )
        rewrites = tuple(_rewrite_layer_forward(ops)[1] for ops in layer_ops)
        patched_modules: set[int] = set()
        for ops in layer_ops:
            for projection in (ops.o_proj, ops.down_proj):
                if id(projection) in patched_modules:
                    continue
                patched_modules.add(id(projection))
                projection._original_forward = projection.forward
                projection.forward = MethodType(_projection_forward, projection)

        norms = {
            id(norm): norm
            for ops in layer_ops
            for norm in (ops.input_norm, ops.post_attention_norm)
        }
        norms[id(final_norm)] = final_norm
        for norm in norms.values():
            norm._original_forward = norm.forward
            norm.forward = MethodType(_norm_forward, norm)

        for ops, next_norm, rewritten_forward in zip(
            layer_ops, next_norms, rewrites, strict=True
        ):
            _bind_layer_forward(ops, next_norm, rewritten_forward)

    @torch.inference_mode()
    def profile(self) -> None:
        if self.profile_attempted:
            return
        self.profile_attempted = True
        self.model.eval()

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
            microchunk_tokens=self.microchunk_tokens,
            threshold_tokens=self.threshold_tokens,
        )

    @contextmanager
    def forward_context(self, num_tokens: int) -> "Iterator[None]":
        context = self.context
        active = context is not None and context.profile.is_active(num_tokens)
        state = _ForwardState(
            context=context,
            active=active,
            residual=None,
        )
        token = _FORWARD_STATE.set(state)
        try:
            yield
        finally:
            _FORWARD_STATE.reset(token)
