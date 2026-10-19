# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural state-operator metadata, available before GPU model construction."""

import copy
import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol

import torch
from cachetools import LRUCache, cached
from torch import nn
from transformers import AutoModel, PreTrainedConfig, PreTrainedModel
from transformers.initialization import no_init_weights

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import MambaSpec


class StateGeometry(Protocol):
    def config(self) -> PreTrainedConfig: ...

    def state_spec(self, path: str, vllm_config: "VllmConfig") -> "MambaSpec": ...


@dataclass(frozen=True)
class StateOperator:
    path: str
    implementation: str
    geometry: StateGeometry


@dataclass(frozen=True)
class GDNGeometry:
    hidden_size: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    conv_kernel_size: int
    activation: str
    norm_eps: float
    allow_neg_eigval: bool = False

    def config(self) -> PreTrainedConfig:
        return PreTrainedConfig(
            hidden_size=self.hidden_size,
            linear_num_key_heads=self.num_k_heads,
            linear_num_value_heads=self.num_v_heads,
            linear_key_head_dim=self.head_k_dim,
            linear_value_head_dim=self.head_v_dim,
            linear_conv_kernel_dim=self.conv_kernel_size,
            hidden_act=self.activation,
            rms_norm_eps=self.norm_eps,
            linear_allow_neg_eigval=self.allow_neg_eigval,
        )

    def state_spec(self, path: str, vllm_config: "VllmConfig") -> "MambaSpec":
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
            MambaStateShapeCalculator,
        )
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        from vllm.v1.kv_cache_interface import MambaSpec

        tp = vllm_config.parallel_config.tensor_parallel_size
        if self.num_k_heads % tp or self.num_v_heads % tp:
            raise ValueError(
                f"State operator {path}: key/value heads "
                f"({self.num_k_heads}, {self.num_v_heads}) must divide TP={tp}."
            )
        if self.num_v_heads % self.num_k_heads:
            raise ValueError(f"State operator {path}: unsupported GDN head ratio.")
        return MambaSpec(
            shapes=MambaStateShapeCalculator.gated_delta_net_state_shape(
                tp,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.conv_kernel_size,
                vllm_config.num_speculative_tokens,
            ),
            dtypes=MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            ),
            block_size=-1,
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        )


@dataclass(frozen=True)
class Mamba1Geometry:
    hidden_size: int
    intermediate_size: int
    state_size: int
    conv_kernel_size: int
    time_step_rank: int
    activation: str
    use_conv_bias: bool
    use_bias: bool
    norm_eps: float | None

    def config(self) -> PreTrainedConfig:
        return PreTrainedConfig(**asdict(self))

    def state_spec(self, path: str, vllm_config: "VllmConfig") -> "MambaSpec":
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
            MambaStateShapeCalculator,
        )
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        from vllm.v1.kv_cache_interface import MambaSpec

        tp = vllm_config.parallel_config.tensor_parallel_size
        if self.intermediate_size % tp:
            raise ValueError(
                f"State operator {path}: intermediate size must divide TP."
            )
        return MambaSpec(
            shapes=MambaStateShapeCalculator.mamba1_state_shape(
                tp, self.intermediate_size, self.state_size, self.conv_kernel_size
            ),
            dtypes=MambaStateDtypeCalculator.mamba1_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            ),
            block_size=-1,
            mamba_type=MambaAttentionBackendEnum.MAMBA1,
        )


@dataclass(frozen=True)
class Mamba2Geometry:
    hidden_size: int
    intermediate_size: int
    state_size: int
    conv_kernel_size: int
    num_heads: int
    head_dim: int
    n_groups: int
    activation: str
    use_conv_bias: bool
    use_bias: bool
    norm_eps: float
    time_step_limit: tuple[float, float]

    def config(self) -> PreTrainedConfig:
        return PreTrainedConfig(**asdict(self))

    def state_spec(self, path: str, vllm_config: "VllmConfig") -> "MambaSpec":
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
            MambaStateShapeCalculator,
        )
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        from vllm.v1.kv_cache_interface import MambaSpec

        tp = vllm_config.parallel_config.tensor_parallel_size
        if self.num_heads % tp or (self.n_groups != 1 and self.n_groups % tp):
            raise ValueError(f"State operator {path}: unsupported Mamba2 TP geometry.")
        return MambaSpec(
            shapes=MambaStateShapeCalculator.mamba2_state_shape(
                tp,
                self.intermediate_size,
                self.n_groups,
                self.num_heads,
                self.head_dim,
                self.state_size,
                self.conv_kernel_size,
                vllm_config.num_speculative_tokens,
            ),
            dtypes=MambaStateDtypeCalculator.mamba2_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            ),
            block_size=-1,
            mamba_type=MambaAttentionBackendEnum.MAMBA2,
        )


@dataclass(frozen=True)
class ShortConvGeometry:
    hidden_size: int
    conv_L_cache: int
    conv_bias: bool

    def config(self) -> PreTrainedConfig:
        return PreTrainedConfig(**asdict(self))

    def state_spec(self, path: str, vllm_config: "VllmConfig") -> "MambaSpec":
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
            MambaStateShapeCalculator,
        )
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
        from vllm.v1.kv_cache_interface import MambaSpec

        tp = vllm_config.parallel_config.tensor_parallel_size
        if self.hidden_size % tp:
            raise ValueError(f"State operator {path}: hidden size must divide TP.")
        return MambaSpec(
            shapes=MambaStateShapeCalculator.short_conv_state_shape(
                tp,
                self.hidden_size,
                self.conv_L_cache,
                vllm_config.num_speculative_tokens,
            ),
            dtypes=MambaStateDtypeCalculator.short_conv_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
            ),
            block_size=-1,
            mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        )


def validate_state_conv(conv: nn.Conv1d, path: str):
    if (
        conv.groups != conv.in_channels
        or conv.out_channels != conv.in_channels
        or conv.dilation != (1,)
        or conv.stride != (1,)
        or conv.padding != (conv.kernel_size[0] - 1,)
    ):
        raise ValueError(
            f"State operator {path}: unsupported non-causal/depthwise convolution."
        )


def describe_mamba1(module: nn.Module, path: str, *, has_norm: bool) -> StateOperator:
    validate_state_conv(module.conv1d, path)
    if module.activation not in ("silu", "swish"):
        raise ValueError(f"State operator {path}: unsupported Mamba activation.")
    norm_eps = module.dt_layernorm.variance_epsilon if has_norm else None
    if has_norm and any(
        norm.variance_epsilon != norm_eps
        for norm in (module.b_layernorm, module.c_layernorm)
    ):
        raise ValueError(f"State operator {path}: inconsistent Mamba projection norms.")
    return StateOperator(
        path,
        "mamba1",
        Mamba1Geometry(
            hidden_size=module.hidden_size,
            intermediate_size=module.intermediate_size,
            state_size=module.ssm_state_size,
            conv_kernel_size=module.conv_kernel_size,
            time_step_rank=module.time_step_rank,
            activation=module.activation,
            use_conv_bias=module.conv1d.bias is not None,
            use_bias=module.in_proj.bias is not None,
            norm_eps=norm_eps,
        ),
    )


def describe_mamba2(module: nn.Module, path: str) -> StateOperator:
    validate_state_conv(module.conv1d, path)
    if (
        module.activation not in ("silu", "swish")
        or module.head_dim * module.num_heads != module.intermediate_size
        or module.norm.group_size != module.intermediate_size // module.n_groups
    ):
        raise ValueError(f"State operator {path}: unsupported Mamba2 geometry or gate.")
    return StateOperator(
        path,
        "mamba2",
        Mamba2Geometry(
            hidden_size=module.hidden_size,
            intermediate_size=module.intermediate_size,
            state_size=module.ssm_state_size,
            conv_kernel_size=module.conv_kernel_size,
            num_heads=module.num_heads,
            head_dim=module.head_dim,
            n_groups=module.n_groups,
            activation=module.activation,
            use_conv_bias=module.conv1d.bias is not None,
            use_bias=module.in_proj.bias is not None,
            norm_eps=module.norm.variance_epsilon,
            time_step_limit=tuple(module.time_step_limit),
        ),
    )


def describe_short_conv(module: nn.Module, path: str) -> StateOperator:
    validate_state_conv(module.conv, path)
    if module.in_proj.bias is not None or module.out_proj.bias is not None:
        raise ValueError(
            f"State operator {path}: unsupported ShortConv projection bias."
        )
    return StateOperator(
        path,
        "short_conv",
        ShortConvGeometry(
            hidden_size=module.in_proj.in_features,
            conv_L_cache=module.conv_kernel_size,
            conv_bias=module.conv.bias is not None,
        ),
    )


def describe_gdn(
    module: nn.Module, path: str, *, implementation: str, norm_name: str
) -> StateOperator:
    norm = getattr(module, norm_name)
    conv = module.conv1d
    if norm.activation != "silu":
        raise ValueError(
            f"State operator {path} ({type(module).__name__}): unsupported "
            f"output gate activation {norm.activation!r}."
        )
    if implementation == "olmo_gdn" and norm.variance_epsilon != 1e-5:
        raise ValueError(
            f"State operator {path} ({type(module).__name__}): unsupported "
            f"output norm epsilon {norm.variance_epsilon}."
        )
    if module.activation not in ("silu", "swish"):
        raise ValueError(
            f"State operator {path} ({type(module).__name__}): unsupported "
            f"convolution activation {module.activation!r}."
        )
    if (
        conv.bias is not None
        or conv.groups != conv.in_channels
        or conv.out_channels != conv.in_channels
        or conv.dilation != (1,)
        or conv.stride != (1,)
        or conv.padding != (module.conv_kernel_size - 1,)
    ):
        raise ValueError(
            f"State operator {path} ({type(module).__name__}): unsupported "
            "non-causal/depthwise convolution geometry or bias."
        )
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.bias is not None:
            raise ValueError(
                f"State operator {path}.{name} ({type(module).__name__}): "
                "projection bias is not implemented."
            )
    return StateOperator(
        path,
        implementation,
        GDNGeometry(
            hidden_size=module.hidden_size,
            num_k_heads=module.num_k_heads,
            num_v_heads=module.num_v_heads,
            head_k_dim=module.head_k_dim,
            head_v_dim=module.head_v_dim,
            conv_kernel_size=module.conv_kernel_size,
            activation=module.activation,
            norm_eps=norm.variance_epsilon,
            allow_neg_eigval=getattr(module, "allow_neg_eigval", False),
        ),
    )


_IMPLEMENTATIONS: dict[tuple[str, str], Callable[[nn.Module, str], StateOperator]] = {
    ("transformers.models.mamba.modeling_mamba", "MambaMixer"): partial(
        describe_mamba1, has_norm=False
    ),
    ("transformers.models.jamba.modeling_jamba", "JambaMambaMixer"): partial(
        describe_mamba1, has_norm=True
    ),
    ("transformers.models.zamba2.modeling_zamba2", "Zamba2MambaMixer"): describe_mamba2,
    ("transformers.models.lfm2.modeling_lfm2", "Lfm2ShortConv"): describe_short_conv,
    ("transformers.models.qwen3_5.modeling_qwen3_5", "Qwen3_5GatedDeltaNet"): partial(
        describe_gdn, implementation="qwen_gdn", norm_name="norm"
    ),
    (
        "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        "Qwen3_5MoeGatedDeltaNet",
    ): partial(describe_gdn, implementation="qwen_gdn", norm_name="norm"),
    (
        "transformers.models.olmo_hybrid.modeling_olmo_hybrid",
        "OlmoHybridGatedDeltaNet",
    ): partial(describe_gdn, implementation="olmo_gdn", norm_name="o_norm"),
}


def describe_state_operators(model: nn.Module) -> tuple[StateOperator, ...]:
    # Match actual implementations, not layer labels or duck-typed attributes.
    operators = []
    seen: dict[int, str] = {}
    for path, module in model.named_modules(remove_duplicate=False):
        cls = type(module)
        key = cls.__module__, cls.__name__
        if key not in _IMPLEMENTATIONS:
            # An unadapted HF recurrent mixer must never run cacheless on
            # packed requests, even if its enclosing block also has attention.
            parameters = inspect.signature(cls.forward).parameters
            cache_container = isinstance(module, PreTrainedModel) or any(
                (type(child).__module__, type(child).__name__) in _IMPLEMENTATIONS
                or "cache_params" in inspect.signature(type(child).forward).parameters
                for child in module.children()
            )
            if ("cache_params" in parameters and not cache_container) or (
                ("cache_params" in parameters or "past_key_values" in parameters)
                and any(isinstance(child, nn.Conv1d) for child in module.children())
            ):
                raise ValueError(
                    f"Unsupported state operator {path} "
                    f"({cls.__module__}.{cls.__name__}): no vLLM state adapter."
                )
            continue
        if getattr(importlib.import_module(key[0]), key[1]) is not cls:
            raise ValueError(
                f"State operator {path}: unrecognized implementation {cls}."
            )
        if id(module) in seen:
            raise ValueError(
                f"State operator {path} ({type(module).__name__}) shares weights "
                f"with {seen[id(module)]}; independent execution-site cache "
                "semantics are not implemented for shared state modules."
            )
        seen[id(module)] = path
        operators.append(_IMPLEMENTATIONS[key](module, path))
    return tuple(operators)


@dataclass(frozen=True)
class ModelStructure:
    operators: tuple[StateOperator, ...]
    decoder: type[PreTrainedModel]
    encoders: dict[str, type[PreTrainedModel]]
    attention_paths: tuple[str, ...]


def find_encoder_classes(model: PreTrainedModel) -> dict[str, type[PreTrainedModel]]:
    encoders = {}
    for modality in ("image", "video", "audio"):
        try:
            encoder = type(model.get_encoder(modality=modality))
        except (TypeError, ValueError):
            continue
        if encoder is not type(model):
            encoders[modality] = encoder
    return encoders


def _inspection_key(config: PreTrainedConfig, trust_remote_code: bool = False):
    data = copy.deepcopy(config.to_dict())

    def remove_execution_options(value):
        if isinstance(value, dict):
            for key in list(value):
                if isinstance(key, str) and (
                    key in ("dtype", "torch_dtype")
                    or key.startswith("_attn_implementation")
                ):
                    del value[key]
                else:
                    remove_execution_options(value[key])

    remove_execution_options(data)
    return (
        type(config),
        json.dumps(data, sort_keys=True, default=str),
        trust_remote_code,
    )


@cached(cache=LRUCache(maxsize=16), key=_inspection_key)
def inspect_model_structure(
    config: PreTrainedConfig, trust_remote_code: bool = False
) -> ModelStructure:
    config = copy.deepcopy(config)
    # No parameters/buffers are allocated, and inspection must not consume the
    # caller's RNG or mutate its config. Cache descriptions, never model objects.
    with torch.random.fork_rng(devices=[]), torch.device("meta"), no_init_weights():
        model = AutoModel.from_config(
            config, trust_remote_code=trust_remote_code, attn_implementation="eager"
        )
    from vllm.model_executor.models.transformers.fusers.attention import interface_call

    decoder_modules = {id(module) for module in model.get_decoder().modules()}
    attention_paths = []
    seen: dict[int, str] = {}
    for path, module in model.named_modules(remove_duplicate=False):
        if (
            id(module) not in decoder_modules
            or interface_call(type(module).forward) is None
        ):
            continue
        if id(module) in seen:
            raise ValueError(
                f"Attention operator {path} ({type(module).__name__}) shares "
                f"execution with {seen[id(module)]}; shared attention-site cache "
                "semantics are not implemented."
            )
        seen[id(module)] = path
        attention_paths.append(path)
    return ModelStructure(
        describe_state_operators(model),
        type(model.get_decoder()),
        find_encoder_classes(model),
        tuple(attention_paths),
    )


def get_state_operators(
    config: PreTrainedConfig, trust_remote_code: bool = False
) -> tuple[StateOperator, ...]:
    return inspect_model_structure(config, trust_remote_code).operators
