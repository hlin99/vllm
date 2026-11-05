# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.models.deepseek_v4.xpu.model import DeepseekV4Model
from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import (
    xpu_qnorm_rope_kv_fp8_insert,
)

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
        moe_intermediate_size=256,
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


@pytest.mark.distributed(num_gpus=2)
def test_xpu_swa_pcp_matches_single_rank(vllm_runner, swa_model_path):
    """Replicated KV preserves prefill/decode across chunks and empty peers."""
    batches = [
        [
            {"prompt_token_ids": [3 + i % 200 for i in range(length)]}
            for length in (19, 257, 1)
        ],
        [{"prompt_token_ids": [7]}],
    ]
    results = []
    for pcp_size in (1, 2):
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
        name_1="PCP=2",
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


def test_xpu_pcp_rejects_compressed_layers(swa_model_path):
    """Unsupported compression must fail before allocating a partial model."""
    config = EngineArgs(
        model=swa_model_path,
        skip_tokenizer_init=True,
        enforce_eager=True,
        prefill_context_parallel_size=2,
        distributed_executor_backend="mp",
        enable_prefix_caching=False,
        max_model_len=512,
        hf_overrides={
            "layer_types": [
                "sliding_attention",
                "compressed_sparse_attention",
            ]
        },
    ).create_engine_config()
    with pytest.raises(NotImplementedError, match="SWA-only layers"):
        DeepseekV4Model(vllm_config=config)
