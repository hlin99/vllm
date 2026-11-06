# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU DeepSeek-V4 attention subclass.

Subclasses the shared ``DeepseekV4Attention`` ABC and provides XPU-native
Triton kernels for decode (FP8 dequant + BF16 attention) and prefill
(BF16 gathered KV + sparse attention).
"""

from dataclasses import dataclass, replace
from typing import cast

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention, DeepseekV4Indexer
from vllm.models.deepseek_v4.common.ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    _SAVE_PARTIAL_STATES_KERNEL,
)
from vllm.models.deepseek_v4.compressor import (
    CompressorBackend,
    CompressorMetadata,
    CompressorMetadataBuilder,
    CompressorStateCache,
    DeepseekCompressor,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLAMetadata,
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.models.deepseek_v4.xpu.pcp_compressor import publish_compressed_cache
from vllm.models.deepseek_v4.xpu.xpu_sparse_decode_fp8 import (
    xpu_sparse_decode_fp8,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadata
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops.xpu_mla_sparse import triton_bf16_mla_sparse_interface
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec
from vllm.v1.worker.workspace import current_workspace_manager


@dataclass
class DeepseekV4XPUCompressorMetadata(CompressorMetadata):
    positions: torch.Tensor | None = None
    boundary_indices: torch.Tensor | None = None
    boundary_positions: torch.Tensor | None = None
    boundary_slots: torch.Tensor | None = None


class DeepseekV4XPUCompressorMetadataBuilder(CompressorMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        cm = common_attn_metadata
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        if pcp_size == 1:
            return super().build(common_prefix_len, cm, fast_build)

        group = get_pcp_group()
        local_slots = cm.slot_mapping.chunk(pcp_size)[group.rank_in_group]
        metadata = super().build(
            common_prefix_len, cm.replace(slot_mapping=local_slots), fast_build
        )
        assert cm.positions is not None
        assert metadata.token_to_req_indices is not None
        padded_tokens = local_slots.shape[0]
        positions = cm.positions.new_zeros(padded_tokens)
        positions[: cm.num_actual_tokens].copy_(cm.positions[: cm.num_actual_tokens])
        token_to_req = metadata.token_to_req_indices.new_zeros(padded_tokens)
        token_to_req[: cm.num_actual_tokens].copy_(metadata.token_to_req_indices)

        if envs.VLLM_XPU_PCP_BOUNDARY_EXCHANGE:
            assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec)
            window = self.kv_cache_spec.sliding_window
            capacity = min(
                padded_tokens,
                2 * self.vllm_config.scheduler_config.max_num_seqs * window,
            )
            rows = torch.arange(padded_tokens, device=positions.device)
            ends = cm.query_start_loc[token_to_req.long() + 1]
            is_tail = (
                (rows < cm.num_actual_tokens)
                & (rows >= ends - window)
                & (local_slots >= 0)
            )
            indices = torch.topk(
                is_tail.to(torch.int32), capacity, sorted=False
            ).indices
            slots = local_slots[indices].masked_fill(~is_tail[indices], -1)
            return DeepseekV4XPUCompressorMetadata(
                block_table=metadata.block_table,
                slot_mapping=local_slots,
                block_size=self.block_size,
                token_to_req_indices=token_to_req,
                positions=positions,
                boundary_indices=indices,
                boundary_positions=group.all_gather(positions[indices], dim=0),
                boundary_slots=group.all_gather(slots, dim=0),
            )

        # PCP can split a request into two local segments. Match PCPManager's
        # table capacity so unequal segment counts still gather uniformly.
        max_reqs = 2 * self.vllm_config.scheduler_config.max_num_seqs
        block_table = metadata.block_table.new_zeros(
            (max_reqs, metadata.block_table.shape[1])
        )
        block_table[: cm.num_reqs].copy_(metadata.block_table)
        token_to_req += group.rank_in_group * max_reqs
        return DeepseekV4XPUCompressorMetadata(
            block_table=group.all_gather(block_table, dim=0),
            slot_mapping=cm.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=group.all_gather(token_to_req, dim=0),
            positions=group.all_gather(positions, dim=0),
        )


class DeepseekV4XPUCompressorBackend(CompressorBackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekV4XPUCompressorMetadataBuilder]:
        return DeepseekV4XPUCompressorMetadataBuilder

    @classmethod
    def supports_pcp(cls) -> bool:
        return True


class DeepseekV4XPUCompressorStateCache(CompressorStateCache):
    def get_attn_backend(self) -> type[CompressorBackend]:
        return DeepseekV4XPUCompressorBackend


class DeepseekV4XPUCompressor(DeepseekCompressor):
    state_cache_cls = DeepseekV4XPUCompressorStateCache

    def forward(
        self,
        kv_score: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        attn_metadata = get_forward_context().attn_metadata
        if get_pcp_group().world_size > 1 and isinstance(attn_metadata, dict):
            metadata = cast(
                DeepseekV4XPUCompressorMetadata,
                attn_metadata[self.state_cache.prefix],
            )
            assert metadata.positions is not None
            if metadata.boundary_indices is not None:
                self._forward_boundary(kv_score, metadata, rotary_emb)
                return
            kv_score = get_pcp_group().all_gather(kv_score.contiguous(), dim=0)
            positions = metadata.positions
        super().forward(kv_score, positions, rotary_emb)

    def _forward_boundary(
        self,
        kv_score: torch.Tensor,
        metadata: DeepseekV4XPUCompressorMetadata,
        rotary_emb,
    ) -> None:
        assert metadata.boundary_indices is not None
        assert metadata.boundary_positions is not None
        assert metadata.boundary_slots is not None
        assert metadata.positions is not None
        group = get_pcp_group()
        tails = group.all_gather(kv_score[metadata.boundary_indices], dim=0)
        kv, score = tails.chunk(2, dim=-1)
        _SAVE_PARTIAL_STATES_KERNEL(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=metadata.boundary_positions,
            state_cache=self.state_cache.kv_cache,
            slot_mapping=metadata.boundary_slots,
            block_size=metadata.block_size,
            state_width=self.coff * self.head_dim,
            compress_ratio=self.compress_ratio,
        )
        context = get_forward_context()
        assert isinstance(context.attn_metadata, dict)
        cache_metadata = context.attn_metadata[self.k_cache_prefix]
        assert isinstance(
            cache_metadata, (DeepseekV4FlashMLAMetadata, DeepseekV32IndexerMetadata)
        )
        local_slots = cache_metadata.slot_mapping.chunk(group.world_size)[
            group.rank_in_group
        ]
        local_metadata = {
            **context.attn_metadata,
            self.k_cache_prefix: replace(cache_metadata, slot_mapping=local_slots),
        }
        with override_forward_context(replace(context, attn_metadata=local_metadata)):
            super().forward(kv_score, metadata.positions, rotary_emb)
        cache = self._static_forward_context[self.k_cache_prefix].kv_cache
        assert cache.dtype == torch.uint8
        publish_compressed_cache(
            cache,
            cache_metadata.slot_mapping,
            self.compress_ratio,
            2 * self.max_num_reqs,
            self._token_stride,
            self._scale_dim,
        )


class DeepseekV4XPUSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        replay_start: torch.Tensor | None = None,
    ) -> DeepseekSparseSWAMetadata:
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        cm = common_attn_metadata
        if pcp_size > 1:
            rank = get_pcp_group().rank_in_group
            cm = cm.replace(slot_mapping=cm.slot_mapping.chunk(pcp_size)[rank])
        metadata = super().build(common_prefix_len, cm, fast_build, replay_start)
        if pcp_size > 1:
            metadata.cache_slot_mapping = common_attn_metadata.slot_mapping
        return metadata


class DeepseekV4XPUSparseSWABackend(DeepseekSparseSWABackend):
    @staticmethod
    def get_builder_cls() -> type[DeepseekV4XPUSparseSWAMetadataBuilder]:
        return DeepseekV4XPUSparseSWAMetadataBuilder

    @classmethod
    def supports_pcp(cls) -> bool:
        return True


class DeepseekV4XPUSparseBackend(DeepseekV4SparseMLABackend):
    @staticmethod
    def get_name() -> str:
        return "XPU_V4_MLA_SPARSE"

    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4XPUSparseMetadataBuilder"]:
        return DeepseekV4XPUSparseMetadataBuilder


class DeepseekV4XPUSparseMetadataBuilder(DeepseekV4SparseMLAMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4FlashMLAMetadata:
        cm = common_attn_metadata
        pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
        if pcp_size > 1:
            group = get_pcp_group()
            local_slots = cm.slot_mapping.chunk(pcp_size)[group.rank_in_group]
            cm = cm.replace(slot_mapping=local_slots)
        metadata = super().build(common_prefix_len, cm, fast_build)
        if pcp_size > 1 and self.compress_ratio > 1:
            metadata.slot_mapping = group.all_gather(
                self.compressed_slot_mapping_buffer[: local_slots.shape[0]], dim=0
            )
        return metadata


class DeepseekV4XPUIndexer(DeepseekV4Indexer):
    def _get_compressor_cls(self) -> type[DeepseekCompressor]:
        return DeepseekV4XPUCompressor


class DeepseekV4XPUAttention(DeepseekV4Attention):
    """XPU sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4XPUSparseBackend
    swa_backend_cls = DeepseekV4XPUSparseSWABackend
    compressor_cls = DeepseekV4XPUCompressor
    use_flashmla_fp8_layout = True

    def _get_indexer_cls(self, vllm_config: VllmConfig) -> type[DeepseekV4Indexer]:
        return DeepseekV4XPUIndexer

    def __init__(self, *args, **kwargs) -> None:
        # torch.cuda.Event() raises RuntimeError on XPU ("dummy base class").
        # The Base and DeepseekV4Indexer both create cuda Events in __init__, so
        # we temporarily redirect torch.cuda.Event → torch.xpu.Event.
        _orig_event = torch.cuda.Event
        torch.cuda.Event = torch.xpu.Event  # type: ignore[misc]
        try:
            super().__init__(*args, **kwargs)
        finally:
            torch.cuda.Event = _orig_event  # type: ignore[misc]

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        from typing import cast

        if not isinstance(attn_metadata, dict):
            # Profile run: no-op, just return q (no padding needed on XPU).
            return q

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        use_pcp = get_pcp_group().world_size > 1
        slot_mapping = swa_metadata.slot_mapping
        if use_pcp:
            assert swa_metadata.cache_slot_mapping is not None
            slot_mapping = swa_metadata.cache_slot_mapping

        from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
            xpu_qnorm_rope_kv_fp8_insert,
        )

        xpu_qnorm_rope_kv_fp8_insert(
            q,
            kv,
            self.swa_cache_layer.kv_cache,
            slot_mapping,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.eps,
            swa_metadata.block_size,
            use_pcp=use_pcp,
        )
        return q

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
            fused_inv_rope_fp8_quant,
        )

        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=False,
        )

        # Precomputed contiguous [G, K, N] weight and [G, K/bs, N/bs] scale.
        wo_a_weight = self.wo_a.bmm_weight
        wo_a_scale = self.wo_a.bmm_scale

        # TODO: optimize fused_inv_rope_fp8_quant for xpu bmm to
        # eliminate o_scale transpose + contiguous
        z = torch.ops.vllm.xpu_fp8_bmm(
            o_fp8.transpose(0, 1),
            wo_a_weight,
            torch.bfloat16,
            o_scale.transpose(0, 1).contiguous(),
            wo_a_scale,
            None,
        )

        return self.wo_b(z.transpose(0, 1).flatten(1))

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: reserve workspace, skip actual kernels.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        if swa_metadata.num_decode_tokens + swa_metadata.num_prefill_tokens == 0:
            output.zero_()
            return

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        assert swa_indices is not None and swa_lens is not None
        xpu_sparse_decode_fp8(
            q=q,
            kv_cache=kv_cache,
            swa_kv_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            attn_sink=self.attn_sink,
            softmax_scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            out=output,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        chunk_size_const = self.PREFILL_CHUNK_SIZE
        num_chunks = (num_prefills + chunk_size_const - 1) // chunk_size_const

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((chunk_size_const, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size_const
            chunk_end = min(chunk_start + chunk_size_const, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )

            kv_ws = kv[:chunk_size].reshape(-1, 1, q.shape[-1])
            out, _, _ = triton_bf16_mla_sparse_interface(
                q=q[query_start:query_end],
                kv=kv_ws,
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                d_v=q.shape[-1],
                block_dpe=0,
            )
            output[query_start:query_end] = out
