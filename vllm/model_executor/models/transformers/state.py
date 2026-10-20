# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator adapters retaining the Transformers decoder's original forward."""

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.transformers.gdn import (
    adapt_gdn,
)
from vllm.model_executor.models.transformers.gdn import (
    state_attention_backend as state_attention_backend,
)
from vllm.model_executor.models.utils import ShardId, WeightsMapper
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.transformers_utils.hybrid import (
    StateOperator,
    describe_state_operators,
    get_state_operators,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec


def state_spec(operator: StateOperator, vllm_config: VllmConfig) -> MambaSpec:
    return operator.geometry.state_spec(operator.path, vllm_config)


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
            fuser, kernel = adapt_gdn(
                self.model.get_submodule(operator.path), operator, self.vllm_config
            )
            if fuser is not None:
                self.packed_modules_mapping.update(fuser.packed_modules_mapping)
                stacked.update(fuser.orig_to_new_stacked(name))
            conv_aliases = {
                "olmo_gdn": ("q_conv1d", "k_conv1d", "v_conv1d"),
            }
            for shard, source in enumerate(
                conv_aliases.get(operator.implementation, ())
            ):
                stacked[f"{name}.{source}"] = (f"{name}.conv1d", shard)
            if (
                tuple(kernel.get_state_shape()) != spec.shapes
                or kernel.get_state_dtype() != spec.dtypes
            ):
                raise ValueError(f"State operator {name}: runtime cache spec mismatch.")
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
