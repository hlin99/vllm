# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuse projection/residual/norm regions without owning the decoder forward.

The current HF source supplies every forward. Only the two fused residual adds
are removed; projection and norm calls retain their original call sites.
Unsupported changes inside a fusion region reject installation atomically.
"""

import ast
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.models.transformers.fx_utils import (
    block_chain,
    compile_forward,
    recover_forward,
    single_self_call,
)
from vllm.model_executor.models.transformers.tpsp_runtime import (
    TPSPContext,
    TPSPOpsGroup,
    UnsupportedTPSPModelError,
    get_tpsp_norm_type,
    tpsp_shard_residual,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass
class _ForwardState:
    context: TPSPContext
    expected: nn.Module | None
    residual: torch.Tensor | None = None


_FORWARD_STATE: ContextVar[_ForwardState | None] = ContextVar(
    "transformers_decoder_tpsp_state", default=None
)


def _matches(node: ast.AST, source: str) -> bool:
    return ast.dump(node) == ast.dump(ast.parse(source).body[0])


def _module_name(module: nn.Module, names: tuple[str, ...]) -> str:
    for name in names:
        if isinstance(getattr(module, name, None), nn.Module):
            return name
    raise ValueError(f"{type(module).__name__} requires one of {names}")


def _projection_parent(layer: nn.Module, name: str) -> str:
    parents = [
        child_name
        for child_name, child in layer.named_children()
        if isinstance(getattr(child, name, None), nn.Module)
    ]
    if len(parents) != 1:
        raise ValueError(f"{type(layer).__name__} requires exactly one {name}")
    parent = layer.get_submodule(parents[0])
    body, _ = recover_forward(type(parent))
    call = single_self_call(body, name)
    # A fused projection already returns normalized states. Nothing may consume
    # its result before the residual add and norm that we replace.
    tail = body.body[-1]
    if (
        not isinstance(tail, ast.Return)
        or sum(isinstance(node, ast.Return) for node in ast.walk(body)) != 1
    ):
        raise ValueError(f"{name} must end its parent's forward")
    output = tail.value
    if isinstance(output, ast.Tuple):
        if any(
            not isinstance(value, (ast.Name, ast.Constant)) for value in output.elts[1:]
        ):
            raise ValueError(f"{name} return must not run additional computations")
        output = output.elts[0]
    if output is not call:
        assignment = body.body[-2]
        if not (
            isinstance(assignment, ast.Assign)
            and len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and assignment.value is call
            and isinstance(output, ast.Name)
            and assignment.targets[0].id == output.id
            and sum(
                isinstance(node, ast.Name) and node.id == output.id
                for node in ast.walk(tail)
            )
            == 1
        ):
            raise ValueError(f"{name} output must be returned unchanged")
    return parents[0]


def _rewrite_layer(
    layer: nn.Module, norms: tuple[str, str], blocks: tuple[str, str]
) -> Callable:
    funcdef, source = recover_forward(type(layer))
    body = funcdef.body
    previous_add = -1
    for norm, block in zip(norms, blocks, strict=True):
        norm_call = single_self_call(funcdef, norm)
        chain = block_chain(body, norm_call)
        if len(chain) != 1:
            raise ValueError("TPSP norms must be unconditional")
        norm_index = chain[0][1]
        setup = body[norm_index - 1]
        if not (
            norm_index > 0
            and isinstance(setup, ast.Assign)
            and len(setup.targets) == 1
            and isinstance(setup.targets[0], ast.Name)
            and _matches(setup, f"{setup.targets[0].id} = hidden_states")
            and _matches(
                body[norm_index], f"hidden_states = self.{norm}(hidden_states)"
            )
        ):
            raise ValueError("TPSP requires a residual saved immediately before norm")
        residual = setup.targets[0].id
        if residual == "hidden_states":
            raise ValueError("TPSP requires a separate residual")
        calls = [
            node
            for node in ast.walk(funcdef)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and _matches(ast.Expr(node.func), f"self.{block}")
        ]
        if len(calls) != 1:
            raise ValueError(f"TPSP requires one {block} call")
        call = calls[0]
        chain = block_chain(body, call)
        if len(chain) != 1:
            raise ValueError(f"TPSP requires an unconditional {block} call")
        index = chain[0][1]
        statement = body[index]
        if not (
            index > norm_index
            and isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and statement.value is call
        ):
            raise ValueError(f"TPSP requires a direct {block} assignment")
        target = statement.targets[0]
        if isinstance(target, ast.Tuple):
            if (
                len(target.elts) != 2
                or not isinstance(target.elts[1], ast.Name)
                or target.elts[1].id in ("hidden_states", residual)
            ):
                raise ValueError("TPSP requires an independent attention output")
            target = target.elts[0]
        if not isinstance(target, ast.Name) or target.id != "hidden_states":
            raise ValueError(f"TPSP requires {block} to produce hidden_states")
        add = body[index + 1]
        if not any(
            _matches(add, f"hidden_states = {expression}")
            for expression in (
                f"{residual} + hidden_states",
                f"hidden_states + {residual}",
            )
        ):
            raise ValueError(f"TPSP requires an immediate residual add after {block}")
        prefix = body[previous_add + 1 : norm_index - 1]
        middle = body[norm_index + 1 : index]
        if (
            (previous_add >= 0 and norm_index != previous_add + 2)
            or any(
                isinstance(node, ast.Name) and node.id in ("hidden_states", residual)
                for stmt in prefix + middle
                for node in ast.walk(stmt)
            )
            or any(
                isinstance(node, ast.Name) and node.id == residual
                for node in ast.walk(call)
            )
        ):
            raise ValueError("TPSP cannot move normalization across intermediate uses")
        previous_add = index + 1
        # The projection hook has already performed both the add and next norm.
        body[previous_add] = ast.copy_location(ast.Pass(), add)
    if len(body) != previous_add + 2 or not _matches(body[-1], "return hidden_states"):
        raise ValueError("TPSP requires the final residual to be returned unchanged")
    return compile_forward(funcdef, source)


def _validate_decoder(decoder: nn.Module, layers_name: str, norm_name: str) -> None:
    funcdef, _ = recover_forward(type(decoder))
    norm = single_self_call(funcdef, norm_name)
    chain = block_chain(funcdef.body, norm)
    if len(chain) != 1 or not _matches(
        funcdef.body[chain[0][1]], f"hidden_states = self.{norm_name}(hidden_states)"
    ):
        raise ValueError("TPSP requires a final hidden_states normalization")
    loop = funcdef.body[chain[0][1] - 1]
    if not isinstance(loop, ast.For) or len(loop.body) != 1 or loop.orelse:
        raise ValueError("TPSP requires a plain decoder loop immediately before norm")
    iterable, target = loop.iter, loop.target
    if isinstance(iterable, ast.Call) and isinstance(target, ast.Tuple):
        if (
            not isinstance(iterable.func, ast.Name)
            or iterable.func.id != "enumerate"
            or len(iterable.args) != 1
            or iterable.keywords
            or len(target.elts) != 2
        ):
            raise ValueError("TPSP requires sequential decoder layers")
        iterable, target = iterable.args[0], target.elts[1]
    if not isinstance(target, ast.Name) or not any(
        _matches(ast.Expr(iterable), expression)
        for expression in (
            f"self.{layers_name}",
            f"self.{layers_name}[:self.config.num_hidden_layers]",
        )
    ):
        raise ValueError("TPSP requires sequential decoder layers")
    statement = loop.body[0]
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "hidden_states"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == target.id
        and statement.value.args
        and _matches(ast.Expr(statement.value.args[0]), "hidden_states")
        and sum(
            isinstance(node, ast.Name) and node.id == "hidden_states"
            for node in ast.walk(statement.value)
        )
        == 1
    ):
        raise ValueError("TPSP requires unchanged states between decoder layers")


def _bind(module: nn.Module, active_forward: Callable) -> None:
    original = module.forward

    def forward(*args, **kwargs):
        if _FORWARD_STATE.get() is None:
            return original(*args, **kwargs)
        return active_forward(*args, **kwargs)

    module.forward = forward


def _bind_projection(group: TPSPOpsGroup) -> None:
    projection, norm = group.projection, group.norm

    def forward(x: torch.Tensor) -> torch.Tensor:
        state = _FORWARD_STATE.get()
        assert state is not None
        if state.expected is not projection or state.residual is None:
            raise RuntimeError(f"Unexpected TPSP {group.name} invocation")
        hidden, state.residual = state.context.backend.fused_gemm_rs_norm_ag(
            state.context.handles[group.name],
            x.reshape(-1, x.shape[-1]).contiguous(),
            projection,
            state.residual,
            norm,
            norm_type=get_tpsp_norm_type(norm),
        )
        state.expected = norm
        return hidden.reshape(*x.shape[:-1], hidden.shape[-1])

    _bind(projection, forward)


def _bind_norm(norm: nn.Module, next_projection: nn.Module | None) -> None:
    original = norm.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        state = _FORWARD_STATE.get()
        assert state is not None
        if state.expected is not norm:
            raise RuntimeError("Unexpected TPSP normalization invocation")
        state.expected = next_projection
        if state.residual is None:
            state.residual = tpsp_shard_residual(
                x.reshape(-1, x.shape[-1]).contiguous()
            )
            return original(x)
        return x

    _bind(norm, forward)


class TransformersDenseTPSP:
    """Install two local fusion patterns, retaining the original forward path."""

    def __init__(self, model: nn.Module, vllm_config: "VllmConfig") -> None:
        if vllm_config.quant_config is not None or vllm_config.lora_config is not None:
            raise ValueError("TPSP does not support quantization or LoRA")
        self.model = model
        get_decoder = getattr(model, "get_decoder", None)
        decoder = get_decoder() if callable(get_decoder) else model
        groups: list[TPSPOpsGroup] = []
        rewrites = []
        try:
            if any(
                getattr(getattr(model, "config", None), name, False)
                for name in ("output_attentions", "output_hidden_states")
            ):
                raise ValueError("TPSP does not support intermediate HF outputs")
            layers_name = _module_name(decoder, ("layers", "h", "blocks"))
            layers = getattr(decoder, layers_name)
            if not isinstance(layers, nn.ModuleList) or not layers:
                raise ValueError("TPSP requires a nonempty decoder ModuleList")
            final_name = _module_name(
                decoder, ("norm", "final_norm", "final_layernorm", "final_layer_norm")
            )
            _validate_decoder(decoder, layers_name, final_name)
            input_names = [
                _module_name(
                    layer,
                    ("input_layernorm", "input_layer_norm", "pre_attention_layernorm"),
                )
                for layer in layers
            ]
            norms = [
                layer.get_submodule(name) for layer, name in zip(layers, input_names)
            ]
            norms.append(decoder.get_submodule(final_name))
            self.input_norm = norms[0]
            for index, layer in enumerate(layers):
                post_name = _module_name(
                    layer, ("post_attention_layernorm", "post_attention_layer_norm")
                )
                parents = (
                    _projection_parent(layer, "o_proj"),
                    _projection_parent(layer, "down_proj"),
                )
                rewrites.append(
                    _rewrite_layer(layer, (input_names[index], post_name), parents)
                )
                for name, parent, norm in zip(
                    ("o_proj", "down_proj"),
                    parents,
                    (layer.get_submodule(post_name), norms[index + 1]),
                    strict=True,
                ):
                    groups.append(
                        TPSPOpsGroup(
                            name, layer.get_submodule(f"{parent}.{name}"), norm
                        )
                    )
            hidden_size = self.input_norm.weight.numel()
            get_tpsp_norm_type(self.input_norm)
            for group in groups:
                first = groups[0 if group.name == "o_proj" else 1]
                projection = group.projection
                if (
                    projection.weight.ndim != 2
                    or projection.weight.shape != first.projection.weight.shape
                    or getattr(
                        projection,
                        "output_size",
                        getattr(projection, "out_features", None),
                    )
                    != hidden_size
                    or group.norm.weight.numel() != hidden_size
                    or get_tpsp_norm_type(group.norm) != get_tpsp_norm_type(first.norm)
                ):
                    raise ValueError(
                        f"Incompatible TPSP {group.name} shapes or normalization"
                    )
            modules = [
                self.input_norm,
                *(m for g in groups for m in (g.projection, g.norm)),
            ]
            if len({id(module) for module in modules}) != len(modules):
                raise ValueError("TPSP does not support shared projections or norms")
        except (
            AttributeError,
            IndexError,
            OSError,
            SyntaxError,
            TypeError,
            ValueError,
        ) as exc:
            raise UnsupportedTPSPModelError(
                f"unsupported {type(decoder).__name__} fusion region: {exc}"
            ) from exc

        self.groups = groups[:2]
        self.max_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.microchunk_tokens = vllm_config.model_config.tpsp_microchunk_tokens
        self.threshold_tokens = vllm_config.model_config.tpsp_threshold_tokens
        self.context: TPSPContext | None = None
        self.profile_attempted = False
        _bind_norm(self.input_norm, groups[0].projection)
        for index, group in enumerate(groups):
            _bind_projection(group)
            next_projection = (
                groups[index + 1].projection if index + 1 < len(groups) else None
            )
            _bind_norm(group.norm, next_projection)
        for layer, rewritten in zip(layers, rewrites, strict=True):
            _bind(layer, MethodType(rewritten, layer))

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
        backend = backend_cls(
            get_tp_group().device_group.group_name,
            self.groups[0].projection.weight.device,
        )
        self.context = backend.profile(
            self.groups,
            self.max_batched_tokens,
            microchunk_tokens=self.microchunk_tokens,
            threshold_tokens=self.threshold_tokens,
        )

    @contextmanager
    def forward_context(self, num_tokens: int, **kwargs) -> Iterator[None]:
        context = self.context
        state = None
        if context is not None and context.profile.is_active(num_tokens):
            for name in ("output_attentions", "output_hidden_states"):
                if kwargs.get(name) is not None and kwargs[name] is not False:
                    raise ValueError(f"Transformers TPSP does not support {name}")
            state = _ForwardState(context, self.input_norm)
        token = _FORWARD_STATE.set(state)
        try:
            yield
            if state is not None and state.expected is not None:
                raise RuntimeError("TPSP forward did not consume every fused stage")
        finally:
            _FORWARD_STATE.reset(token)
