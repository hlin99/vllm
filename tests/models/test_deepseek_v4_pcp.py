# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from itertools import accumulate

import pytest
import torch
from transformers import DeepseekV4Config

from vllm import SamplingParams
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.forward_context import set_forward_context
from vllm.models.deepseek_v4.compressor import CompressorMetadata, DeepseekCompressor
from vllm.models.deepseek_v4.xpu.model import DeepseekV4Model
from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
    xpu_qnorm_rope_kv_fp8_insert,
)
from vllm.models.deepseek_v4.xpu.xpu_sparse import (
    DeepseekV4XPUAttention,
    DeepseekV4XPUCompressorMetadataBuilder,
    DeepseekV4XPUSparseMetadataBuilder,
)
from vllm.v1.attention.backend import CommonAttentionMetadata

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
    "layer_types",
    [
        ["sliding_attention"] * 2,
        ["compressed_sparse_attention", "heavily_compressed_attention"],
    ],
    ids=["swa", "c4a-c128a"],
)
def test_xpu_pcp_matches_single_rank(
    vllm_runner, swa_model_path, layer_types, pcp_world_size
):
    """Cover chunking, real C4A indexer scoring, decode, and empty peers."""
    batches = [
        [
            {"prompt_token_ids": [3 + i % 200 for i in range(length)]}
            for length in (19, 257, 1)
        ],
        [{"prompt_token_ids": [7]}],
    ]
    results = []
    for pcp_size in (1, pcp_world_size):
        with vllm_runner(
            swa_model_path,
            load_format="dummy",
            skip_tokenizer_init=True,
            enforce_eager=True,
            prefill_context_parallel_size=pcp_size,
            distributed_executor_backend="mp",
            max_model_len=512,
            max_num_batched_tokens=128,
            max_num_seqs=3,
            block_size=256,
            gpu_memory_utilization=0.2,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            hf_overrides={"layer_types": layer_types, "index_topk": 16},
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


def _check_replicated_swa_cache(rank, init_method):
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
        slots = torch.randperm(512)[:num_tokens].to(device)
        local_rows = slice(rank * tokens_per_rank, (rank + 1) * tokens_per_rank)
        for active_tokens in (17, 1):
            positions = torch.arange(num_tokens, device=device) + 7
            positions[active_tokens:] = 0
            slot_mapping = slots.clone()
            slot_mapping[active_tokens:] = -1
            cache = torch.full((2, 256, 584), 37, dtype=torch.uint8, device=device)
            expected_cache = cache.clone()
            local_q = q[local_rows].clone()
            expected_q = q.clone()
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
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
def test_xpu_pcp_replicates_cache_with_padding_and_empty_peer(tmp_path):
    """Every cache byte and local Q must match an unsharded cache write."""
    torch.multiprocessing.spawn(
        _check_replicated_swa_cache,
        args=(f"file://{tmp_path / 'pcp-init'}",),
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
        for attention in attentions:
            compressor = attention.compressor
            assert compressor is not None
            state = compressor.state_cache
            state_spec = state.get_kv_cache_spec(config)
            cache_spec = attention.get_kv_cache_spec(config)
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
            attention.bind_kv_cache(
                torch.full(
                    (6, 1, cache_spec.num_states, 584),
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
                cache_spec, [attention.prefix], config, device
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
                    attention.prefix: replace(
                        cache_metadata,
                        slot_mapping=compressed_slots,
                    ),
                }
                kv_score = scores[req_ids, pos]
                previous_state = state.kv_cache.clone()
                previous_cache = attention.kv_cache.clone()
                with set_forward_context(reference, config):
                    DeepseekCompressor.forward(
                        compressor, kv_score, pos, attention.rotary_emb
                    )
                expected_state = state.kv_cache.clone()
                expected_cache = attention.kv_cache.clone()
                state.kv_cache.copy_(previous_state)
                attention.kv_cache.copy_(previous_cache)
                with set_forward_context(
                    {state.prefix: state_metadata, attention.prefix: cache_metadata},
                    config,
                ):
                    compressor(kv_score[local], pos[local], attention.rotary_emb)
                assert torch.equal(state.kv_cache, expected_state)
                assert torch.equal(attention.kv_cache, expected_cache)
                starts = [start + length for start, length in zip(starts, lengths)]
    finally:
        cleanup_dist_env_and_memory()


@pytest.mark.distributed(num_gpus=2)
def test_xpu_pcp_replicates_compressed_states_across_chunks(tmp_path, swa_model_path):
    """C4A/C128A states and cache bytes match unsharded compression exactly."""
    torch.multiprocessing.spawn(
        _check_replicated_compressed_cache,
        args=(f"file://{tmp_path / 'compressed-pcp-init'}", swa_model_path),
        nprocs=2,
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
