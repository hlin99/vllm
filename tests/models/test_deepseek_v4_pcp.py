# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from itertools import accumulate

import numpy as np
import pytest
import torch
from transformers import DeepseekV4Config

from vllm import SamplingParams
from vllm.config import (
    CUDAGraphMode,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    get_pcp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache
from vllm.models.deepseek_v4.compressor import CompressorMetadata, DeepseekCompressor
from vllm.models.deepseek_v4.xpu.model import DeepseekV4Model
from vllm.models.deepseek_v4.xpu.pcp_pipeline import get_pcp_swa_pipeline
from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
    xpu_qnorm_rope_kv_fp8_insert,
)
from vllm.models.deepseek_v4.xpu.xpu_sparse import (
    DeepseekV4XPUAttention,
    DeepseekV4XPUCompressorMetadata,
    DeepseekV4XPUCompressorMetadataBuilder,
    DeepseekV4XPUSparseMetadataBuilder,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.ops.xpu_mla_sparse import triton_paged_mla_sparse_interface
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.pcp_manager import PCPManager, maybe_build_pcp_manager

from .utils import check_logprobs_close


@pytest.fixture
def swa_model_path(tmp_path):
    config = DeepseekV4Config(
        architectures=["DeepseekV4ForCausalLM"],
        num_hidden_layers=2,
        layer_types=["sliding_attention"] * 2,
        mlp_layer_types=["hash_moe"] * 2,
        hidden_size=512,
        vocab_size=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=1,
        moe_intermediate_size=512,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        num_nextn_predict_layers=0,
        max_position_embeddings=512,
        sliding_window=128,
        expert_dtype="fp8",
        quantization_config={
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )
    config.save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.parametrize(
    "pcp_world_size",
    [
        pytest.param(2, marks=pytest.mark.distributed(num_gpus=2), id="pcp2"),
        pytest.param(4, marks=pytest.mark.distributed(num_gpus=4), id="pcp4"),
    ],
)
@pytest.mark.parametrize(
    "layer_types,boundary_exchange,paged_attention,kv_chunk_size,moe_fusion,"
    "expert_parallel",
    [
        (["sliding_attention"] * 2, False, False, 0, False, False),
        (
            ["compressed_sparse_attention", "heavily_compressed_attention"],
            False,
            False,
            0,
            False,
            False,
        ),
        (
            ["compressed_sparse_attention", "heavily_compressed_attention"],
            True,
            False,
            0,
            False,
            False,
        ),
        (["sliding_attention"] * 2, False, True, 0, False, False),
        (
            ["compressed_sparse_attention", "heavily_compressed_attention"],
            False,
            True,
            0,
            False,
            False,
        ),
        (
            ["compressed_sparse_attention", "heavily_compressed_attention"],
            True,
            True,
            0,
            False,
            False,
        ),
        (["sliding_attention"] * 2, False, False, 16, False, False),
        (
            ["compressed_sparse_attention", "heavily_compressed_attention"],
            False,
            False,
            16,
            False,
            False,
        ),
        (
            ["compressed_sparse_attention", "heavily_compressed_attention"],
            True,
            False,
            16,
            False,
            False,
        ),
        (
            ["compressed_sparse_attention", "heavily_compressed_attention"],
            True,
            False,
            16,
            True,
            False,
        ),
        (
            ["compressed_sparse_attention", "heavily_compressed_attention"],
            True,
            False,
            16,
            False,
            True,
        ),
        (
            ["compressed_sparse_attention", "heavily_compressed_attention"],
            True,
            False,
            16,
            True,
            True,
        ),
    ],
    ids=[
        "swa",
        "c4a-c128a",
        "boundary-c4a-c128a",
        "paged-swa",
        "paged-c4a-c128a",
        "paged-boundary-c4a-c128a",
        "pipeline-swa",
        "pipeline-c4a-c128a",
        "pipeline-boundary-c4a-c128a",
        "deepsymm-pipeline-boundary-c4a-c128a",
        "allgather-ep-pipeline-boundary-c4a-c128a",
        "deepep-pipeline-boundary-c4a-c128a",
    ],
)
def test_xpu_pcp_matches_single_rank(
    vllm_runner,
    swa_model_path,
    layer_types,
    boundary_exchange,
    paged_attention,
    kv_chunk_size,
    moe_fusion,
    expert_parallel,
    pcp_world_size,
    monkeypatch,
):
    """Cover chunking, real C4A indexer scoring, decode, and empty peers."""
    monkeypatch.setenv("VLLM_XPU_PCP_BOUNDARY_EXCHANGE", str(int(boundary_exchange)))
    batches = [
        [
            {"prompt_token_ids": [3 + i % 200 for i in range(length)]}
            for length in (19, 257, 1)
        ],
        [{"prompt_token_ids": [7]}],
    ]
    results = []
    for pcp_size in (1, pcp_world_size):
        monkeypatch.setenv(
            "VLLM_XPU_PCP_MOE_FUSION", str(int(moe_fusion and pcp_size > 1))
        )
        monkeypatch.setenv(
            "VLLM_XPU_PCP_KV_CHUNK_SIZE",
            str(kv_chunk_size if pcp_size > 1 else 0),
        )
        monkeypatch.setenv(
            "VLLM_XPU_PAGED_SPARSE_ATTN",
            str(int(paged_attention and pcp_size > 1)),
        )
        with vllm_runner(
            swa_model_path,
            dtype="bfloat16" if kv_chunk_size or moe_fusion else "auto",
            load_format="dummy",
            skip_tokenizer_init=True,
            enforce_eager=True,
            prefill_context_parallel_size=pcp_size,
            enable_expert_parallel=expert_parallel and pcp_size > 1,
            distributed_executor_backend="mp",
            max_model_len=512,
            max_num_batched_tokens=128,
            max_num_seqs=3,
            block_size=256,
            gpu_memory_utilization=0.2,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            hf_overrides={
                "layer_types": layer_types,
                "index_topk": 16,
                "topk_method": "noaux_tc",
                "num_attention_heads": 32 if paged_attention or kv_chunk_size else 4,
                "mlp_layer_types": ["hash_moe", "moe"]
                if moe_fusion or expert_parallel
                else ["hash_moe"] * 2,
            },
        ) as runner:
            batch_results = []
            for prompts in batches:
                outputs = runner.llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=0, max_tokens=4, ignore_eos=True, logprobs=5
                    ),
                )
                assert len(outputs) == len(prompts)
                for output in outputs:
                    completion = output.outputs[0]
                    assert completion.finish_reason == "length"
                    assert len(completion.token_ids) == 4
                    batch_results.append(
                        (
                            list(completion.token_ids),
                            completion.text,
                            completion.logprobs,
                        )
                    )
            results.append(batch_results)
    check_logprobs_close(
        outputs_0_lst=results[0],
        outputs_1_lst=results[1],
        name_0="PCP=1",
        name_1=f"PCP={pcp_world_size}",
        always_check_logprobs=True,
    )


def _check_local_moe_padding(rank, init_method, model_path):
    torch.xpu.set_device(rank)
    device = torch.device("xpu", rank)
    config = EngineArgs(
        model=model_path,
        skip_tokenizer_init=True,
        enforce_eager=True,
        prefill_context_parallel_size=2,
        distributed_executor_backend="mp",
        max_model_len=512,
        max_num_batched_tokens=128,
        max_num_seqs=3,
    ).create_engine_config()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
            backend="xccl",
        )
        initialize_model_parallel(prefill_context_model_parallel_size=2, backend="xccl")
        try:
            manager = maybe_build_pcp_manager(config, device, supports_mm_inputs=False)
            assert manager is not None
            batch = InputBatch.make_dummy(
                num_reqs=3,
                num_tokens=3,
                input_buffers=InputBuffers(3, 128, device),
                max_query_len=1,
                is_padding=False,
            )
            local = manager.partition_batch(
                batch,
                BatchExecutionDescriptor(
                    cg_mode=CUDAGraphMode.NONE, num_tokens=4, num_reqs=None
                ),
            )
            rows = local.num_tokens_after_padding
            hidden = torch.ones(rows, 4, device=device)
            logits = torch.arange(8, device=device, dtype=torch.float32)
            logits = logits.expand(rows, -1).contiguous()
            with set_forward_context(None, config, is_padding=local.is_padding):
                _, ids, _ = fused_topk(hidden, logits, 2, False)
            assert local.num_tokens == (2 if rank == 0 else 1)
            assert (ids[: local.num_tokens] >= 0).all()
            assert (ids[local.num_tokens :] == -1).all()
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
def test_xpu_deepsymm_routes_local_padding(swa_model_path, tmp_path, monkeypatch):
    """Uneven PCP ranks must not apply rank zero's padding to local routing."""
    monkeypatch.setenv("VLLM_XPU_PCP_MOE_FUSION", "1")
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    torch.multiprocessing.spawn(
        _check_local_moe_padding,
        args=(f"file://{tmp_path / 'pcp-init'}", swa_model_path),
        nprocs=2,
    )


def _check_replicated_swa_cache(rank, init_method, pipeline):
    torch.xpu.set_device(rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            prefill_context_parallel_size=2,
            distributed_executor_backend="mp",
        )
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
            backend="xccl",
        )
        initialize_model_parallel(prefill_context_model_parallel_size=2, backend="xccl")
    try:
        torch.manual_seed(0)
        device = torch.device("xpu", rank)
        tokens_per_rank = 9
        num_tokens = 2 * tokens_per_rank
        q = torch.randn(num_tokens, 4, 512, dtype=torch.bfloat16).to(device)
        kv = torch.randn(num_tokens, 512, dtype=torch.bfloat16).to(device)
        angles = torch.randn(64, 32).to(device)
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
        all_slots = torch.randperm(512).to(device)
        slots = all_slots[:num_tokens]
        history_slots = all_slots[num_tokens : num_tokens + 2]
        local_rows = slice(rank * tokens_per_rank, (rank + 1) * tokens_per_rank)
        transport = (
            get_pcp_swa_pipeline(get_pcp_group(), tokens_per_rank, 4)
            if pipeline
            else None
        )
        for active_tokens in (17, 1):
            positions = torch.arange(num_tokens, device=device) + 7
            positions[active_tokens:] = 0
            slot_mapping = slots.clone()
            slot_mapping[active_tokens:] = -1
            cache = torch.full((2, 256, 584), 37, dtype=torch.uint8, device=device)
            if transport is not None:
                quantize_and_insert_k_cache(
                    kv[:2], cache.view(2, -1), history_slots, block_size=256
                )
            expected_cache = cache.clone()
            local_q = q[local_rows].clone()
            expected_q = q.clone()
            if transport is None:
                xpu_qnorm_rope_kv_fp8_insert(
                    local_q,
                    kv[local_rows],
                    cache,
                    slot_mapping,
                    positions[local_rows],
                    cos_sin,
                    1e-6,
                    256,
                    use_pcp=True,
                )
            else:
                queries = min(
                    tokens_per_rank, max(0, active_tokens - rank * tokens_per_rank)
                )
                indices = (
                    torch.cat((history_slots, slots[:active_tokens]))
                    .to(torch.int32)
                    .view(1, 1, -1)
                    .expand(queries, 1, -1)
                    .contiguous()
                )
                lengths = torch.full(
                    (queries,), indices.shape[-1], device=device, dtype=torch.int32
                )
                state = transport.prefill(
                    local_q,
                    kv[local_rows],
                    positions[local_rows],
                    cos_sin,
                    1e-6,
                    cache,
                    slot_mapping,
                    0,
                    queries,
                    indices,
                    lengths,
                    512**-0.5,
                    256,
                )
            xpu_qnorm_rope_kv_fp8_insert(
                expected_q,
                kv,
                expected_cache,
                slot_mapping,
                positions,
                cos_sin,
                1e-6,
                256,
            )
            assert torch.equal(local_q, expected_q[local_rows])
            assert torch.equal(cache, expected_cache)
            if transport is not None:
                expected_out = torch.empty_like(local_q[:queries])
                actual_out = torch.empty_like(expected_out)
                triton_paged_mla_sparse_interface(
                    expected_q[local_rows][:queries],
                    None,
                    expected_cache,
                    None,
                    None,
                    indices,
                    lengths,
                    512**-0.5,
                    expected_out,
                )
                triton_paged_mla_sparse_interface(
                    local_q[:queries],
                    None,
                    cache,
                    None,
                    None,
                    indices[..., :0],
                    lengths,
                    512**-0.5,
                    actual_out,
                    partial_state=state,
                )
                torch.testing.assert_close(
                    actual_out, expected_out, rtol=1e-2, atol=1e-2
                )
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.parametrize("pipeline", [False, True])
def test_xpu_pcp_replicates_cache_with_padding_and_empty_peer(tmp_path, pipeline):
    """Every cache byte and local Q must match an unsharded cache write."""
    torch.multiprocessing.spawn(
        _check_replicated_swa_cache,
        args=(f"file://{tmp_path / 'pcp-init'}", pipeline),
        nprocs=2,
    )


def _check_replicated_compressed_cache(rank, init_method, model_path):
    torch.xpu.set_device(rank)
    config = EngineArgs(
        model=model_path,
        load_format="dummy",
        skip_tokenizer_init=True,
        enforce_eager=True,
        prefill_context_parallel_size=2,
        distributed_executor_backend="mp",
        enable_prefix_caching=False,
        max_model_len=512,
        max_num_batched_tokens=128,
        max_num_seqs=3,
        block_size=256,
        hf_overrides={
            "layer_types": [
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
            "index_topk": 16,
        },
    ).create_engine_config()
    device = torch.device("xpu", rank)
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
            backend="xccl",
        )
        initialize_model_parallel(prefill_context_model_parallel_size=2, backend="xccl")
        with torch.device(device):
            attentions = [
                DeepseekV4XPUAttention(
                    config,
                    prefix=f"model.layers.{layer}.self_attn",
                    topk_indices_buffer=torch.empty((128, 16), dtype=torch.int32),
                )
                for layer in range(2)
            ]
    try:
        targets = [
            (attention.compressor, attention, attention.rotary_emb)
            for attention in attentions
        ]
        for attention in attentions:
            if attention.indexer is not None:
                targets.append(
                    (
                        attention.indexer.compressor,
                        attention.indexer.k_cache,
                        attention.indexer_rotary_emb,
                    )
                )
        for compressor, cache_layer, rotary_emb in targets:
            assert compressor is not None
            state = compressor.state_cache
            state_spec = state.get_kv_cache_spec(config)
            cache_spec = cache_layer.get_kv_cache_spec(config)
            assert cache_spec is not None
            state_blocks = 512 // state_spec.block_size
            state_table = torch.arange(3 * state_blocks).view(3, state_blocks)
            cache_table = torch.arange(6).view(3, 2)
            state.bind_kv_cache(
                torch.zeros(
                    (3 * state_blocks, 1, state_spec.block_size, state.state_dim),
                    dtype=state.dtype,
                    device=device,
                )
            )
            cache_layer.bind_kv_cache(
                torch.full(
                    (
                        6,
                        1,
                        cache_spec.num_states,
                        cache_spec.page_size_bytes // cache_spec.num_states,
                    ),
                    37,
                    dtype=torch.uint8,
                    device=device,
                )
            )
            with torch.no_grad():
                compressor.ape.copy_(
                    torch.linspace(-0.25, 0.25, compressor.ape.numel())
                    .view_as(compressor.ape)
                    .to(device)
                )
                compressor.norm.weight.fill_(1)
            scores = torch.randn(
                (3, 512, state.state_dim),
                dtype=torch.bfloat16,
                generator=torch.Generator().manual_seed(0),
            ).to(device)
            state_builder = DeepseekV4XPUCompressorMetadataBuilder(
                state_spec, [state.prefix], config, device
            )
            cache_builder = DeepseekV4XPUSparseMetadataBuilder(
                cache_spec, [cache_layer.prefix], config, device
            )
            starts = [0, 0, 0]
            phases = [(19, 105, 1), (0, 128, 0), (0, 24, 0)] + [(1, 1, 1)] * 4
            for phase, lengths in enumerate(phases):
                segments: list[list[tuple[int, int, int]]] = [[], []]
                for req, (start, length) in enumerate(zip(starts, lengths)):
                    split = (length + 1) // 2
                    for peer, (begin, end) in enumerate(
                        ((start, start + split), (start + split, start + length))
                    ):
                        if begin < end:
                            if phase == 0 and peer == 0 and req == 1:
                                middle = (begin + end) // 2
                                segments[peer].extend(
                                    [(req, begin, middle), (req, middle, end)]
                                )
                            else:
                                segments[peer].append((req, begin, end))
                for peer in range(2):
                    segments[peer].sort(key=lambda row: row[2] - row[1] > 1)
                    if not segments[peer]:
                        segments[peer] = [(0, 0, 0)]
                padded = max(
                    sum(end - begin for _, begin, end in rows) for rows in segments
                )
                requests, positions, valid = [], [], []
                for rows in segments:
                    peer_positions = [
                        pos for _, begin, end in rows for pos in range(begin, end)
                    ]
                    requests.extend(
                        [req for req, begin, end in rows for _ in range(begin, end)]
                        + [0] * (padded - len(peer_positions))
                    )
                    positions.extend(
                        peer_positions + [0] * (padded - len(peer_positions))
                    )
                    valid.extend(
                        [True] * len(peer_positions)
                        + [False] * (padded - len(peer_positions))
                    )
                req_ids = torch.tensor(requests, device=device)
                pos = torch.tensor(positions, device=device)
                active = torch.tensor(valid, device=device)
                state_slots = (
                    state_table.to(device)[req_ids, pos // state_spec.block_size]
                    * state_spec.block_size
                    + pos % state_spec.block_size
                ).masked_fill_(~active, -1)
                raw_slots = (
                    cache_table.to(device)[req_ids, pos // cache_spec.block_size]
                    * cache_spec.block_size
                    + pos % cache_spec.block_size
                ).masked_fill_(~active, -1)
                rows = segments[rank]
                query_lens = [end - begin for _, begin, end in rows]
                query_start = torch.tensor(
                    [0, *accumulate(query_lens)], dtype=torch.int32
                )
                seq_lens = torch.tensor([end for _, _, end in rows], dtype=torch.int32)
                local = slice(rank * padded, (rank + 1) * padded)
                common = CommonAttentionMetadata(
                    query_start_loc=query_start.to(device),
                    query_start_loc_cpu=query_start,
                    seq_lens=seq_lens.to(device),
                    seq_lens_cpu_upper_bound=seq_lens,
                    num_reqs=len(rows),
                    num_actual_tokens=sum(query_lens),
                    max_query_len=max(query_lens),
                    max_seq_len=max(end for _, _, end in rows),
                    block_table_tensor=state_table[[req for req, _, _ in rows]].to(
                        device
                    ),
                    slot_mapping=state_slots,
                    positions=pos[local],
                )
                state_metadata = state_builder.build(0, common)
                assert isinstance(state_metadata, DeepseekV4XPUCompressorMetadata)
                cache_metadata = cache_builder.build(
                    0,
                    common.replace(
                        block_table_tensor=cache_table[[req for req, _, _ in rows]].to(
                            device
                        ),
                        slot_mapping=raw_slots,
                    ),
                )
                compressed_slots = (
                    raw_slots // cache_spec.block_size * cache_spec.num_states
                    + pos % cache_spec.block_size // compressor.compress_ratio
                ).masked_fill_(
                    ~active | ((pos + 1) % compressor.compress_ratio != 0), -1
                )
                reference = {
                    state.prefix: CompressorMetadata(
                        block_table=state_table.to(device),
                        slot_mapping=state_slots,
                        block_size=state_spec.block_size,
                        token_to_req_indices=req_ids.to(torch.int32),
                    ),
                    cache_layer.prefix: replace(
                        cache_metadata,
                        slot_mapping=compressed_slots,
                    ),
                }
                kv_score = scores[req_ids, pos]
                previous_state = state.kv_cache.clone()
                previous_cache = cache_layer.kv_cache.clone()
                with set_forward_context(reference, config):
                    DeepseekCompressor.forward(compressor, kv_score, pos, rotary_emb)
                expected_state = state.kv_cache.clone()
                expected_cache = cache_layer.kv_cache.clone()
                state.kv_cache.copy_(previous_state)
                cache_layer.kv_cache.copy_(previous_cache)
                with set_forward_context(
                    {state.prefix: state_metadata, cache_layer.prefix: cache_metadata},
                    config,
                ):
                    compressor(kv_score[local], pos[local], rotary_emb)
                if state_metadata.boundary_indices is None:
                    assert torch.equal(state.kv_cache, expected_state)
                else:
                    live_slots = [
                        int(state_table[req, position // state_spec.block_size])
                        * state_spec.block_size
                        + position % state_spec.block_size
                        for req, (start, length) in enumerate(zip(starts, lengths))
                        for position in range(
                            max(0, start + length - state.sliding_window),
                            start + length,
                        )
                    ]
                    actual_rows = state.kv_cache.flatten(0, 1)
                    expected_rows = expected_state.flatten(0, 1)
                    assert torch.equal(
                        actual_rows[live_slots], expected_rows[live_slots]
                    )
                assert torch.equal(cache_layer.kv_cache, expected_cache)
                starts = [start + length for start, length in zip(starts, lengths)]
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
@pytest.mark.parametrize(
    "boundary_exchange", [False, True], ids=["replicated", "boundary"]
)
def test_xpu_pcp_replicates_compressed_states_across_chunks(
    tmp_path, swa_model_path, boundary_exchange, monkeypatch
):
    """C4A/C128A live states and cache bytes match unsharded compression exactly."""
    monkeypatch.setenv("VLLM_XPU_PCP_BOUNDARY_EXCHANGE", str(int(boundary_exchange)))
    torch.multiprocessing.spawn(
        _check_replicated_compressed_cache,
        args=(f"file://{tmp_path / 'compressed-pcp-init'}", swa_model_path),
        nprocs=2,
    )


def _check_replicated_c2_compression(rank, init_method, model_path, pcp_size):
    from vllm.models.deepseek_v41.compressor import (
        CompressorMetadataBuilder as C2MetadataBuilder,
    )
    from vllm.models.deepseek_v41.compressor import (
        DeepseekCompressor as C2Compressor,
    )

    torch.xpu.set_device(rank)
    device = torch.device("xpu", rank)
    # Full-input replication runs the unsharded compressor on every PCP rank.
    config = EngineArgs(
        model=model_path,
        load_format="dummy",
        skip_tokenizer_init=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        max_model_len=512,
        max_num_batched_tokens=256,
        max_num_seqs=3,
        block_size=256,
    ).create_engine_config()
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=pcp_size,
            rank=rank,
            local_rank=rank,
            distributed_init_method=init_method,
            backend="xccl",
        )
        initialize_model_parallel(
            prefill_context_model_parallel_size=pcp_size, backend="xccl"
        )
        with torch.device(device):
            compressors = [
                C2Compressor(
                    config,
                    compress_ratio=2,
                    hidden_size=512,
                    head_dim=512,
                    prefix=prefix,
                )
                for prefix in ("gathered", "reference")
            ]
    try:
        compressor, reference = compressors
        assert compressor.state_cache is not None
        spec = compressor.state_cache.get_kv_cache_spec(config)
        for layer in compressors:
            assert layer.state_cache is not None
            layer.state_cache.bind_kv_cache(
                torch.zeros(
                    (4, 1, spec.block_size, 1024), device=device, dtype=torch.float32
                )
            )
            with torch.no_grad():
                layer.norm.weight.fill_(1)
        builder = C2MetadataBuilder(
            spec, [compressor.state_cache.prefix], config, device
        )
        manager = PCPManager(
            pcp_size,
            rank,
            device,
            shard_decode_requests=True,
            max_num_reqs=3,
            max_num_tokens=256,
        )
        buffers = InputBuffers(3, 256, device)
        scores = torch.randn(
            (3, 512, 1024), generator=torch.Generator().manual_seed(42)
        ).to(device)
        starts = np.zeros(3, dtype=np.int32)
        prompt_lens = np.array([19, 257, 1], dtype=np.int32)
        phases = (
            [(19, 105, 1), (0, 128, 0), (0, 24, 0)]
            + [(1, 1, 1)] * 4
            + [(0, 1, 0), (1, 1, 1)]
        )
        for phase, lengths in enumerate(phases):
            order = np.flatnonzero(lengths)
            if phase % 2:
                order = order[::-1].copy()
            lengths_np = np.array(lengths, dtype=np.int32)[order]
            qsl_np = np.array([0, *accumulate(lengths_np)], dtype=np.int32)
            positions = torch.cat(
                [
                    torch.arange(starts[req], starts[req] + length, device=device)
                    for req, length in zip(order, lengths_np)
                ]
            )
            num_tokens = len(positions)
            req_ids = torch.repeat_interleave(
                torch.tensor(order, device=device),
                torch.tensor(lengths_np, device=device),
            )
            raw = scores[req_ids, positions]
            input_batch = replace(
                InputBatch.make_dummy(len(order), num_tokens, buffers),
                idx_mapping=torch.tensor(order, device=device),
                idx_mapping_np=order,
                num_scheduled_tokens=lengths_np,
                query_start_loc=torch.tensor(qsl_np, device=device),
                query_start_loc_np=qsl_np,
                seq_lens=torch.tensor(starts[order] + lengths_np, device=device),
                seq_lens_cpu_upper_bound=torch.from_numpy(starts[order] + lengths_np),
                num_computed_tokens_np=starts[order].copy(),
                prefill_len_np=prompt_lens[order],
                num_computed_prefill_tokens_np=np.minimum(
                    starts[order], prompt_lens[order]
                ),
                is_prefilling_np=starts[order] < prompt_lens[order],
                has_prefill=bool((starts[order] < prompt_lens[order]).any()),
                input_ids=torch.arange(num_tokens, device=device, dtype=torch.int32),
                positions=positions,
            )
            local = manager.partition_batch(
                input_batch,
                BatchExecutionDescriptor(CUDAGraphMode.NONE, num_tokens, len(order)),
            )
            # C2's ring consumes contiguous request chunks, unlike V4's
            # slot-indexed states: restore order and discard replicated/padded rows.
            gathered_raw = manager.restore_hidden_states(raw[local.input_ids.long()])
            gathered_positions = manager.restore_hidden_states(
                local.positions[:, None]
            ).flatten()
            common = CommonAttentionMetadata(
                query_start_loc=input_batch.query_start_loc,
                query_start_loc_cpu=torch.from_numpy(qsl_np),
                seq_lens=input_batch.seq_lens,
                seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound,
                num_reqs=len(order),
                num_actual_tokens=num_tokens,
                max_query_len=int(lengths_np.max()),
                max_seq_len=int((starts[order] + lengths_np).max()),
                block_table_tensor=torch.tensor(
                    order + 1, dtype=torch.int32, device=device
                )[:, None],
                slot_mapping=torch.arange(num_tokens, device=device),
                positions=gathered_positions,
            )
            metadata = builder.build(0, common)
            with set_forward_context({compressor.state_cache.prefix: metadata}, config):
                actual = compressor(gathered_raw, gathered_positions)
            assert reference.state_cache is not None
            reference_metadata = builder.build(0, common.replace(positions=positions))
            with set_forward_context(
                {reference.state_cache.prefix: reference_metadata}, config
            ):
                expected = reference(raw, positions)
            assert actual is not None and expected is not None
            valid = (positions + 1) % 2 == 0
            assert torch.equal(actual[valid], expected[valid])
            assert torch.equal(
                compressor.state_cache.kv_cache, reference.state_cache.kv_cache
            )
            starts += lengths
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "pcp_size",
    [
        pytest.param(2, marks=pytest.mark.distributed(num_gpus=2), id="pcp2"),
        pytest.param(4, marks=pytest.mark.distributed(num_gpus=4), id="pcp4"),
    ],
)
def test_xpu_pcp_full_input_replication_preserves_c2_compression(
    tmp_path, swa_model_path, pcp_size
):
    """Real C2 latents and ring states match unsharded compression across chunks."""
    torch.multiprocessing.spawn(
        _check_replicated_c2_compression,
        args=(
            f"file://{tmp_path / 'c2-pcp-init'}",
            swa_model_path,
            pcp_size,
        ),
        nprocs=pcp_size,
    )


@pytest.mark.parametrize("pcp_size", [2, 8])
def test_xpu_pcp_rejects_prefix_caching(swa_model_path, pcp_size):
    """Unsupported prefix caching must fail before allocating a partial model."""
    config = EngineArgs(
        model=swa_model_path,
        skip_tokenizer_init=True,
        enforce_eager=True,
        prefill_context_parallel_size=pcp_size,
        distributed_executor_backend="mp",
        enable_prefix_caching=True,
        max_model_len=512,
    ).create_engine_config()
    with pytest.raises(NotImplementedError, match="prefix caching disabled"):
        DeepseekV4Model(vllm_config=config)


def test_xpu_pcp_ep_rejects_unsupported_backend(swa_model_path, monkeypatch):
    """Without native dispatch, PCP EP must use its validated AG/RS backend."""
    monkeypatch.setenv("VLLM_XPU_PCP_MOE_FUSION", "0")
    config = EngineArgs(
        model=swa_model_path,
        skip_tokenizer_init=True,
        enforce_eager=True,
        prefill_context_parallel_size=2,
        enable_expert_parallel=True,
        all2all_backend="deepep_low_latency",
        distributed_executor_backend="mp",
        enable_prefix_caching=False,
        max_model_len=512,
    ).create_engine_config()
    with pytest.raises(
        NotImplementedError, match="requires --all2all-backend allgather_reducescatter"
    ):
        DeepseekV4Model(vllm_config=config)
