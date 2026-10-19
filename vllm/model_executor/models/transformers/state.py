# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator adapters retaining the Transformers decoder's original forward."""

from copy import copy
from functools import cache
from typing import Literal

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (
    OlmoHybridGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.utils import ShardId, WeightsMapper
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.transformers_utils.hybrid import (
    StateOperator,
    describe_state_operators,
    get_state_operators,
)
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec


def packed_hidden_states(hidden_states, cache_params, attention_mask):
    if cache_params is not None or attention_mask is not None:
        raise ValueError(
            "Transformers state operators require vLLM packed inputs and managed "
            "states, not HF cache_params or attention_mask."
        )
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("Transformers state operators require [1, tokens, hidden].")
    return hidden_states.squeeze(0)


@cache
def state_attention_backend(
    prefill_backend: Literal["triton", "flashinfer", "cutedsl", "aiter_flydsl"],
) -> type[GDNAttentionBackend]:
    # Different kernels can share a state layout but not chunk metadata.
    # Backend classes separate their metadata groups without splitting weights
    # or imposing restrictions on which operators may occur together.
    class Builder(GDNAttentionMetadataBuilder):
        def __init__(
            self,
            kv_cache_spec: MambaSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            local_config = copy(vllm_config)
            local_config.model_config = copy(vllm_config.model_config)
            local_config.model_config.hf_text_config = copy(
                vllm_config.model_config.hf_text_config
            )
            # GDN recurrent state is [value_heads / TP, value_dim, key_dim].
            _, head_v_dim, head_k_dim = kv_cache_spec.shapes[1]
            local_config.model_config.hf_text_config.linear_key_head_dim = head_k_dim
            local_config.model_config.hf_text_config.linear_value_head_dim = head_v_dim
            additional_config = vllm_config.additional_config
            local_config.additional_config = {
                **(additional_config if isinstance(additional_config, dict) else {}),
                "gdn_prefill_backend": prefill_backend,
            }
            super().__init__(kv_cache_spec, layer_names, local_config, device)

    class Backend(GDNAttentionBackend):
        @staticmethod
        def get_builder_cls():
            return Builder

    return Backend


class TransformersQwenGDN(QwenGatedDeltaNetAttention):
    checkpoint_mapping: tuple[tuple[str, str, ShardId], ...] = (
        ("in_proj_qkv", "in_proj_qkvz", (0, 1, 2)),
        ("in_proj_z", "in_proj_qkvz", 3),
        ("in_proj_b", "in_proj_ba", 0),
        ("in_proj_a", "in_proj_ba", 1),
    )
    packed_projections = {
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def get_attn_backend(self):
        return state_attention_backend(self.gdn_prefill_backend)

    def forward(
        self, hidden_states, cache_params=None, attention_mask=None, **kwargs
    ) -> torch.Tensor:
        hidden_states = packed_hidden_states(
            hidden_states, cache_params, attention_mask
        )
        return super().forward(hidden_states).unsqueeze(0)


class TransformersOlmoGDN(OlmoHybridGatedDeltaNetAttention):
    checkpoint_mapping: tuple[tuple[str, str, ShardId], ...] = (
        ("q_proj", "in_proj_qkvg", 0),
        ("k_proj", "in_proj_qkvg", 1),
        ("v_proj", "in_proj_qkvg", 2),
        ("g_proj", "in_proj_qkvg", 3),
        ("q_conv1d", "conv1d", 0),
        ("k_conv1d", "conv1d", 1),
        ("v_conv1d", "conv1d", 2),
    )
    packed_projections = {
        "in_proj_qkvg": ["q_proj", "k_proj", "v_proj", "g_proj"],
    }

    def get_attn_backend(self):
        return state_attention_backend("triton")

    def __init__(self, config, vllm_config, prefix):
        super().__init__(config, vllm_config, prefix)
        # HF checkpoints contain a single Q/K/V-concatenated convolution,
        # unlike the three separate convolution weights in older FLA checkpoints.
        separate_loader = self.conv1d.weight.weight_loader
        combined_loader = mamba_v2_sharded_weight_loader(
            [(self.key_dim, 0, False)] * 2 + [(self.value_dim, 0, False)],
            self.tp_size,
            self.tp_rank,
        )

        def load_conv(param, weight, shard_id=None):
            shard_id = getattr(weight, "shard_id", shard_id)
            if shard_id is None:
                combined_loader(param, weight)
            else:
                separate_loader(param, weight, shard_id)

        self.conv1d.weight.weight_loader = load_conv

    def forward(
        self,
        hidden_states,
        output=None,
        *,
        cache_params=None,
        attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        if output is not None:
            raise ValueError("HF GDN forward must not receive a native output buffer.")
        hidden_states = packed_hidden_states(
            hidden_states, cache_params, attention_mask
        )
        output = torch.empty_like(hidden_states)
        super().forward(hidden_states, output)
        return output.unsqueeze(0)


def state_spec(operator: StateOperator, vllm_config: VllmConfig) -> MambaSpec:
    return operator.geometry.state_spec(operator.path, vllm_config)


_ADAPTERS: dict[str, type[TransformersQwenGDN] | type[TransformersOlmoGDN]] = {
    "qwen_gdn": TransformersQwenGDN,
    "olmo_gdn": TransformersOlmoGDN,
}


class HybridMixin(HasInnerState, IsHybrid, Base):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(IsHybrid, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def recursive_replace(self):
        expected = get_state_operators(self.config, self.model_config.trust_remote_code)
        actual = describe_state_operators(self.model)
        if self.pp_group.world_size == 1 and actual != expected:
            raise ValueError(
                "Runtime state operators differ from early cache metadata."
            )
        # PP has already removed non-local layers. Compare each surviving site.
        expected_by_path = {operator.path: operator for operator in expected}
        stacked: dict[str, tuple[str, ShardId]] = {}
        for operator in actual:
            if expected_by_path.get(operator.path) != operator:
                raise ValueError(
                    f"State operator {operator.path} differs from early cache metadata."
                )
            spec = state_spec(operator, self.vllm_config)
            name = f"model.{operator.path}"
            if operator.implementation not in _ADAPTERS:
                raise ValueError(
                    f"State operator {name}: missing execution adapter "
                    f"{operator.implementation!r}."
                )
            adapter = _ADAPTERS[operator.implementation]
            replacement = adapter(
                operator.geometry.config(), self.vllm_config, prefix=name
            )
            self.packed_modules_mapping.update(adapter.packed_projections)
            if (
                tuple(replacement.get_state_shape()) != spec.shapes
                or replacement.get_state_dtype() != spec.dtypes
            ):
                raise ValueError(f"State operator {name}: runtime cache spec mismatch.")
            parent, _, child = operator.path.rpartition(".")
            setattr(self.model.get_submodule(parent), child, replacement)
            for source, target, shard in adapter.checkpoint_mapping:
                stacked[f"{name}.{source}"] = (f"{name}.{target}", shard)
        if self.quant_config is not None:
            self.quant_config.packed_modules_mapping.update(self.packed_modules_mapping)
        self.hf_to_vllm_mapper |= WeightsMapper(orig_to_new_stacked=stacked)
        super().recursive_replace()

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig):
        operators = get_state_operators(
            vllm_config.model_config.hf_config,
            vllm_config.model_config.trust_remote_code,
        )
        if not operators:
            raise ValueError("Hybrid backend found no supported state operators.")
        return tuple(state_spec(operator, vllm_config) for operator in operators)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_state_copy_funcs(cls, mamba_types):
        supported = {
            MambaAttentionBackendEnum.GDN_ATTN: cls.get_mamba_state_copy_func()
        }
        if missing := mamba_types - supported.keys():
            raise ValueError(f"Missing Transformers state copy adapters for {missing}.")
        return {kind: supported[kind] for kind in mamba_types}


class TextPositionMixin(SupportsMRoPE, Base):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(SupportsMRoPE, self).__init__(vllm_config=vllm_config, prefix=prefix)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        if mm_features:
            raise ValueError("The text backend does not accept multimodal inputs.")
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0
