# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small HF reference with explicit, independently configured operator sites."""

import copy

import torch
from torch import nn
from transformers import LlamaConfig, OlmoHybridConfig, Qwen3_5TextConfig
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.olmo_hybrid.modeling_olmo_hybrid import OlmoHybridGatedDeltaNet
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet


class HybridTestConfig(LlamaConfig):
    model_type = "hybrid_operator_test"

    def __init__(self, operator_specs=None, **kwargs):
        super().__init__(**kwargs)
        self.operator_specs = operator_specs or []
        self.layer_types = [
            spec["kind"] if spec["kind"].endswith("attention") else "linear_attention"
            for spec in self.operator_specs
        ]


class HybridTestBlock(nn.Module):
    def __init__(self, config, index, spec):
        super().__init__()
        self.kind = spec["kind"]
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LlamaMLP(config)
        if self.kind in ("full_attention", "sliding_attention"):
            self.mixer = LlamaAttention(config, index)
            if hasattr(config, "attention_layer_index"):
                self.mixer.layer_idx = config.attention_layer_index
            self.mixer.sliding_window = (
                config.sliding_window if self.kind == "sliding_attention" else None
            )
        else:
            params = dict(
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_hidden_layers,
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                linear_num_key_heads=2,
                linear_num_value_heads=4,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                layer_types=["linear_attention"] * (config.num_hidden_layers - 1)
                + ["full_attention"],
            )
            params.update({k: v for k, v in spec.items() if k != "kind"})
            if self.kind == "qwen_gdn":
                self.mixer = Qwen3_5GatedDeltaNet(Qwen3_5TextConfig(**params), index)
            else:
                self.mixer = OlmoHybridGatedDeltaNet(OlmoHybridConfig(**params), index)
            if "output_norm_eps" in spec:
                norm = getattr(self.mixer, "norm", None)
                if norm is None:
                    norm = self.mixer.o_norm
                norm.variance_epsilon = spec["output_norm_eps"]

    def forward(self, hidden_states, position_embeddings, attention_mask):
        normalized = self.input_layernorm(hidden_states)
        if self.kind in ("full_attention", "sliding_attention"):
            mixed = self.mixer(
                normalized,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )[0]
        else:
            mixed = self.mixer(normalized)
        hidden_states = hidden_states + mixed
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class HybridTestModel(LlamaPreTrainedModel):
    config_class = HybridTestConfig
    _supports_attention_backend = True

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                HybridTestBlock(copy.deepcopy(config), i, spec)
                for i, spec in enumerate(config.operator_specs)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config)
        self.post_init()

    def get_decoder(self):
        return self

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        return_dict=True,
        **kwargs,
    ):
        if use_cache or past_key_values is not None:
            raise ValueError(
                "This reference model only implements cacheless HF forward."
            )
        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        if position_ids is None:
            position_ids = torch.arange(
                hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }
        for block in self.layers:
            hidden_states = block(
                hidden_states, position_embeddings, masks.get(block.kind)
            )
        hidden_states = self.norm(hidden_states)
        if not return_dict:
            return (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states)


class HybridTestForCausalLM(LlamaPreTrainedModel):
    config_class = HybridTestConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = HybridTestModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, **kwargs):
        output = self.model(input_ids, **kwargs)
        return CausalLMOutputWithPast(logits=self.lm_head(output.last_hidden_state))


HybridTestConfig.register_for_auto_class()
HybridTestModel.register_for_auto_class("AutoModel")
HybridTestForCausalLM.register_for_auto_class("AutoModelForCausalLM")
