# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental BF16 Llama TP/SP integration for vLLM's eager XPU runner."""

import torch
import torch.distributed as dist
from deep_symm.async_tp import fused_matmul_reduce_scatter_norm_all_gather

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.models.llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
)
from vllm.v1.worker.tpsp_profile import (
    SPProfile,
    profile_sp_config,
    select_sp_config,
)


class TPSPLlamaDecoderLayer(LlamaDecoderLayer):
    def _project_and_normalize(
        self,
        x,
        projection,
        residual,
        norm,
        profile: SPProfile,
        residual_is_sharded: bool,
    ):
        group = get_tp_group()
        tp_size = group.world_size
        if x.dtype != torch.bfloat16 or projection.weight.dtype != torch.bfloat16:
            raise RuntimeError("TP/SP Llama requires BF16 activations and weights")
        if profile.tp_size != tp_size or profile.hidden_size != self.hidden_size:
            raise RuntimeError("TP/SP profile does not match the Llama projection")
        if (
            profile.input_widths
            and projection.input_size_per_partition not in profile.input_widths
        ):
            raise RuntimeError("TP/SP projection input width was not profiled")
        rows = (x.size(0) + tp_size - 1) // tp_size
        if not residual_is_sharded:
            if residual.shape != (x.size(0), self.hidden_size):
                raise RuntimeError("TP/SP full residual has an unexpected shape")
            start = group.rank_in_group * rows
            local_residual = torch.zeros(
                (rows, residual.size(-1)), device=residual.device, dtype=residual.dtype
            )
            count = min(rows, max(0, x.size(0) - start))
            if count:
                local_residual[:count] = residual[start : start + count]
        else:
            if residual.shape != (rows, self.hidden_size):
                raise RuntimeError("TP/SP residual shard has an unexpected shape")
            local_residual = residual

        if select_sp_config(profile, x.size(0)):
            weight = projection.weight
            key = (weight.data_ptr(), weight._version)
            cached = getattr(projection, "_tpsp_transposed_weight", None)
            if cached is None or cached[0] != key:
                cached = (key, weight.T.contiguous())
                projection._tpsp_transposed_weight = cached
            reduced, _, gathered = fused_matmul_reduce_scatter_norm_all_gather(
                x.contiguous(),
                cached[1],
                norm.weight,
                None,
                group.device_group.group_name,
                eps=norm.variance_epsilon,
                norm_type="rms_norm",
                residual=local_residual,
                microchunk_tokens=profile.microchunk_tokens,
            )
            # vLLM's subsequent kernels may consume independent-queue outputs
            # before the native operator's asynchronous writes finish.
            torch.xpu.synchronize()
            return gathered, reduced

        full, _ = projection(x)
        if not residual_is_sharded:
            full_residual = residual.clone()
        else:
            padded = torch.empty(
                (tp_size * rows, self.hidden_size), device=x.device, dtype=x.dtype
            )
            dist.all_gather_into_tensor(
                padded, local_residual.contiguous(), group=group.device_group
            )
            full_residual = padded[: x.size(0)].contiguous()
        gathered, new_residual = norm(full, full_residual)
        reduced = torch.zeros((rows, self.hidden_size), device=x.device, dtype=x.dtype)
        start = group.rank_in_group * rows
        count = min(rows, max(0, x.size(0) - start))
        if count:
            reduced[:count] = new_residual[start : start + count]
        return gathered, reduced

    def forward_sp(
        self,
        positions,
        hidden_states,
        residual,
        next_norm,
        profile: SPProfile,
        residual_is_sharded: bool,
    ):
        attention = self.self_attn
        qkv, _ = attention.qkv_proj(hidden_states)
        q, k, v = qkv.split(
            [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
        )
        q, k = attention.rotary_emb(positions, q, k)
        attn_output = attention.attn(q, k, v)
        hidden_states, residual = self._project_and_normalize(
            attn_output,
            attention.o_proj,
            residual,
            self.post_attention_layernorm,
            profile,
            residual_is_sharded,
        )
        mlp = self.mlp
        hidden_states, _ = mlp.gate_up_proj(hidden_states)
        hidden_states = mlp.act_fn(hidden_states)
        return self._project_and_normalize(
            hidden_states, mlp.down_proj, residual, next_norm, profile, True
        )


class TPSPLlamaModel(LlamaModel):
    def __init__(self, *, vllm_config, prefix="", layer_type=TPSPLlamaDecoderLayer):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        self.sp_profile: SPProfile | None = None

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds=None,
        **extra_layer_kwargs,
    ):
        if get_pp_group().world_size != 1 or intermediate_tensors is not None:
            raise RuntimeError("TP/SP Llama does not support pipeline parallelism")
        if extra_layer_kwargs:
            raise RuntimeError("TP/SP Llama does not support extra layer arguments")
        if self.sp_profile is None:
            raise RuntimeError("TP/SP Llama requires XPU V2 worker startup profiling")
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.embed_input_ids(input_ids)
        )
        residual = hidden_states
        hidden_states = self.layers[0].input_layernorm(hidden_states)
        for idx, layer in enumerate(self.layers):
            next_norm = (
                self.layers[idx + 1].input_layernorm
                if idx + 1 < len(self.layers)
                else self.norm
            )
            hidden_states, residual = layer.forward_sp(
                positions, hidden_states, residual, next_norm, self.sp_profile, idx != 0
            )
        return hidden_states


class TPSPLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, *, vllm_config, prefix="", layer_type=TPSPLlamaDecoderLayer):
        if vllm_config.quant_config is not None:
            raise RuntimeError("TP/SP Llama does not support quantized weights")
        if vllm_config.lora_config is not None:
            raise RuntimeError("TP/SP Llama does not support LoRA")
        if vllm_config.parallel_config.tensor_parallel_size < 2:
            raise RuntimeError("TP/SP Llama requires tensor parallel size >= 2")
        if vllm_config.model_config.hf_config.rms_norm_eps != 1e-5:
            raise RuntimeError("TP/SP Llama profile requires RMSNorm epsilon 1e-5")
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

    def _init_model(self, vllm_config, prefix="", layer_type=TPSPLlamaDecoderLayer):
        return TPSPLlamaModel(
            vllm_config=vllm_config, prefix=prefix, layer_type=layer_type
        )

    def profile_tpsp_config(self, max_num_batched_tokens: int) -> None:
        if self.model.sp_profile is not None:
            return
        group = get_tp_group()
        self.model.sp_profile = profile_sp_config(
            tp_size=group.world_size,
            hidden_size=self.config.hidden_size,
            max_batched_tokens=max_num_batched_tokens,
            group_name=group.device_group.group_name,
            time_budget_s=240.0,
            input_widths=(
                self.config.hidden_size // group.world_size,
                self.config.intermediate_size // group.world_size,
            ),
        )
