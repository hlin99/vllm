# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental BF16 Qwen3 MoE attention TP/SP adapter for vLLM XPU."""

from functools import lru_cache

import torch
import torch.distributed as dist
from deep_symm.async_tp import (
    fused_matmul_reduce_scatter_norm_route_all_gather,
    fused_matmul_reduce_scatter_norm_route_all_gather_with_residual,
    fused_matmul_reduce_scatter_norm_route_all_gather_xccl_ready,
)

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    FusedTopKRouter,
)
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
)
from vllm.v1.worker.tpsp_profile import (
    SPProfile,
    profile_sp_config,
    select_sp_config,
)


@lru_cache(maxsize=8)
def _residual_xccl_stream(device_index: int) -> torch.xpu.Stream:
    return torch.xpu.Stream(device=device_index)


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
        gate = self.mlp.gate
        if gate.weight.dtype != torch.bfloat16:
            raise RuntimeError("TP/SP Qwen3 MoE requires a BF16 router weight")
        residual_ag_mode = profile.residual_ag_mode
        if residual_ag_mode not in ("post", "dual_early", "xccl_early"):
            raise RuntimeError(f"Unsupported residual AG mode: {residual_ag_mode}")
        if residual_ag_mode != "post" and profile.all_gather_mode != "p2p":
            raise RuntimeError("Early residual AG requires P2P hidden all-gather")
        xccl_stream = (
            _residual_xccl_stream(attn_output.device.index)
            if residual_ag_mode == "xccl_early"
            else None
        )
        route_op = (
            fused_matmul_reduce_scatter_norm_route_all_gather_with_residual
            if residual_ag_mode == "dual_early"
            else fused_matmul_reduce_scatter_norm_route_all_gather_xccl_ready
            if xccl_stream is not None
            else fused_matmul_reduce_scatter_norm_route_all_gather
        )
        result = route_op(
            attn_output.contiguous(),
            cached[1],
            self.post_attention_layernorm.weight,
            None,
            gate.weight,
            self.mlp.experts.router.top_k,
            group.device_group.group_name,
            renormalize=self.mlp.experts.router.renormalize,
            eps=self.post_attention_layernorm.variance_epsilon,
            norm_type="rms_norm",
            residual=local_residual,
            microchunk_tokens=profile.microchunk_tokens,
            **({"xccl_stream": xccl_stream} if xccl_stream is not None else {}),
        )
        if residual_ag_mode == "dual_early":
            # Match the late-XCCL path's per-layer barrier so queued outputs can be reused.
            torch.xpu.synchronize()
            reduced, _, normalized, topk_weights, topk_ids, full_residual = result
        else:
            reduced, _, normalized, topk_weights, topk_ids = result
            if xccl_stream is None:
                torch.xpu.synchronize()
            padded_residual = torch.empty(
                (group.world_size * rows, self.hidden_size),
                device=reduced.device,
                dtype=reduced.dtype,
            )
            if xccl_stream is None:
                dist.all_gather_into_tensor(
                    padded_residual, reduced.contiguous(), group=group.device_group
                )
            else:
                with torch.xpu.stream(xccl_stream):
                    dist.all_gather_into_tensor(
                        padded_residual, reduced, group=group.device_group
                    )
                reduced.record_stream(xccl_stream)
                padded_residual.record_stream(xccl_stream)
                torch.xpu.current_stream().wait_stream(xccl_stream)
            full_residual = padded_residual[:tokens].contiguous()
            if xccl_stream is not None:
                torch.xpu.synchronize()
        return (
            self.mlp(normalized, topk_weights=topk_weights, topk_ids=topk_ids),
            full_residual,
        )


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
        width = widths.pop()
        coder_shape = self.config.hidden_size == 6144 and width == 3072
        for layer in self.model.layers:
            if (
                not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock)
                or not isinstance(layer.mlp.experts.router, FusedTopKRouter)
                or layer.mlp.experts.router.scoring_func != "softmax"
                or layer.mlp.experts.router.capture_fn is not None
                or layer.mlp.experts.gate is not layer.mlp.gate
                or layer.mlp.shared_expert is not None
                or layer.mlp.is_sequence_parallel
                or layer.mlp.enable_eplb
                or layer.mlp.experts.is_monolithic
                or layer.mlp.gate.weight.dtype != torch.bfloat16
            ):
                raise RuntimeError(
                    "TP/SP Qwen3 MoE requires modular BF16 softmax routing "
                    "without shared experts, EPLB or sequence parallelism"
                )
        self.sp_profile = profile_sp_config(
            tp_size=group.world_size,
            hidden_size=self.config.hidden_size,
            max_batched_tokens=max_num_batched_tokens,
            group_name=group.device_group.group_name,
            time_budget_s=180.0,
            input_widths=(width,),
            norm_eps=self.config.rms_norm_eps,
            gather_residual_after_native=True,
            check_rtol=0.02 if coder_shape else 0.01,
            check_atol=0.05 if coder_shape else 0.02,
            router_num_experts=self.config.num_experts,
            route_top_k=self.config.num_experts_per_tok,
            route_renormalize=self.config.norm_topk_prob,
        )
        for layer in self.model.layers:
            layer.sp_profile = self.sp_profile
