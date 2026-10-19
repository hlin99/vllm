# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structural state-operator metadata, available before GPU model construction."""

import copy
import importlib
import inspect
import json
from dataclasses import dataclass
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


_IMPLEMENTATIONS = {
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
            if "cache_params" in parameters or (
                "past_key_values" in parameters
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
