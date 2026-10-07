# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental BF16 Qwen3 MoE attention TP/SP adapter for vLLM XPU."""

import torch
import torch.distributed as dist
from deep_symm.async_tp import fused_matmul_reduce_scatter_norm_all_gather

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
)
from vllm.models.tpsp_profile import SPProfile, profile_sp_config, select_sp_config


class TPSPQwen3MoeDecoderLayer(Qwen3MoeDecoderLayer):
    sp_profile: SPProfile | None = None

    def forward(self, positions, hidden_states, residual):
        profile = self.sp_profile
        if profile is None:
            raise RuntimeError("TP/SP Qwen3 MoE requires XPU worker startup profiling")
        if not select_sp_config(profile, hidden_states.size(0)):
            return super().forward(positions, hidden_states, residual)

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        attention = self.self_attn
        qkv, _ = attention.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
        )
        q = attention.q_norm(
            q.view(*q.shape[:-1], q.shape[-1] // attention.head_dim, attention.head_dim)
        ).view(q.shape)
        k = attention.k_norm(
            k.view(*k.shape[:-1], k.shape[-1] // attention.head_dim, attention.head_dim)
        ).view(k.shape)
        q, k = attention.rotary_emb(positions, q, k)
        attn_output = attention.attn(q, k, v)

        group = get_tp_group()
        projection = attention.o_proj
        if (
            attn_output.dtype != torch.bfloat16
            or projection.weight.dtype != torch.bfloat16
        ):
            raise RuntimeError("TP/SP Qwen3 MoE requires BF16 activations and weights")
        if (
            projection.input_size_per_partition not in profile.input_widths
            or profile.hidden_size != self.hidden_size
        ):
            raise RuntimeError(
                "TP/SP Qwen3 MoE projection does not match the startup profile"
            )
        weight = projection.weight
        key = (weight.data_ptr(), weight._version)
        cached = getattr(projection, "_tpsp_transposed_weight", None)
        if cached is None or cached[0] != key:
            cached = (key, weight.T.contiguous())
            projection._tpsp_transposed_weight = cached

        tokens = attn_output.size(0)
        rows = (tokens + group.world_size - 1) // group.world_size
        start = group.rank_in_group * rows
        local_residual = torch.zeros(
            (rows, self.hidden_size), device=residual.device, dtype=residual.dtype
        )
        count = min(rows, max(0, tokens - start))
        if count:
            local_residual[:count] = residual[start : start + count]
        reduced, _, normalized = fused_matmul_reduce_scatter_norm_all_gather(
            attn_output.contiguous(),
            cached[1],
            self.post_attention_layernorm.weight,
            None,
            group.device_group.group_name,
            eps=self.post_attention_layernorm.variance_epsilon,
            norm_type="rms_norm",
            residual=local_residual,
            microchunk_tokens=profile.microchunk_tokens,
        )
        torch.xpu.synchronize()
        padded_residual = torch.empty(
            (group.world_size * rows, self.hidden_size),
            device=reduced.device,
            dtype=reduced.dtype,
        )
        dist.all_gather_into_tensor(
            padded_residual, reduced.contiguous(), group=group.device_group
        )
        return self.mlp(normalized), padded_residual[:tokens].contiguous()


class TPSPQwen3MoeForCausalLM(Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        if vllm_config.quant_config is not None or vllm_config.lora_config is not None:
            raise RuntimeError(
                "TP/SP Qwen3 MoE requires unquantized weights without LoRA"
            )
        if not 2 <= vllm_config.parallel_config.tensor_parallel_size <= 8:
            raise RuntimeError("TP/SP Qwen3 MoE requires TP between 2 and 8")
        if get_pp_group().world_size != 1:
            raise RuntimeError("TP/SP Qwen3 MoE does not support pipeline parallelism")
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=TPSPQwen3MoeDecoderLayer,
        )
        self.sp_profile: SPProfile | None = None

    def profile_tpsp_config(self, max_num_batched_tokens: int) -> None:
        if self.sp_profile is not None:
            return
        group = get_tp_group()
        widths = {
            layer.self_attn.o_proj.input_size_per_partition
            for layer in self.model.layers
        }
        if len(widths) != 1:
            raise RuntimeError(
                "TP/SP Qwen3 MoE attention projection widths differ across layers"
            )
        self.sp_profile = profile_sp_config(
            tp_size=group.world_size,
            hidden_size=self.config.hidden_size,
            max_batched_tokens=max_num_batched_tokens,
            group_name=group.device_group.group_name,
            time_budget_s=180.0,
            input_widths=(widths.pop(),),
            norm_eps=self.config.rms_norm_eps,
            gather_residual_after_native=True,
        )
        for layer in self.model.layers:
            layer.sp_profile = self.sp_profile
