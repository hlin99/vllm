# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPU-only batch-invariance tests for norm, collectives, and sampler behavior."""

import os
import random

import pytest
import torch

from vllm import envs
from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.distributed.device_communicators.xpu_communicator import XpuCommunicator
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="XPU batch-invariance tests"
)


@pytest.fixture
def tiny_qwen3_moe_path(tmp_path):
    """Create a local Qwen3-MoE config for dummy-loading tests."""
    from transformers import Qwen3MoeConfig

    config = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        vocab_size=256,
        hidden_size=512,
        intermediate_size=1024,
        moe_intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=512,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    config.save_pretrained(tmp_path)
    return tmp_path


@pytest.mark.parametrize("source", ["argument", "checkpoint"])
def test_rejects_weight_quantization(tiny_qwen3_moe_path, source):
    """Reject both explicit and checkpoint-inferred quantization before loading."""
    from transformers import Qwen3MoeConfig

    if source == "checkpoint":
        config = Qwen3MoeConfig.from_pretrained(tiny_qwen3_moe_path)
        config.quantization_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
        }
        config.save_pretrained(tiny_qwen3_moe_path)

    model_config = ModelConfig(
        model=str(tiny_qwen3_moe_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        quantization="fp8" if source == "argument" else None,
    )
    assert model_config.quantization == "fp8"
    with pytest.raises(ValueError, match="supports only unquantized models"):
        VllmConfig(
            model_config=model_config,
            load_config=LoadConfig(load_format="dummy"),
        )


@pytest.mark.parametrize("cache_dtype", ["fp8", "int8_per_token_head", "nvfp4"])
def test_rejects_quantized_kv_cache(cache_dtype):
    with pytest.raises(ValueError, match="does not support quantized KV caches"):
        VllmConfig(cache_config=CacheConfig(cache_dtype=cache_dtype))


@pytest.mark.parametrize("cache_dtype", ["auto", "float16", "bfloat16"])
def test_accepts_unquantized_kv_cache(cache_dtype):
    config = VllmConfig(cache_config=CacheConfig(cache_dtype=cache_dtype))
    assert config.cache_config.cache_dtype == cache_dtype


def test_quantized_kv_cache_allowed_without_batch_invariance(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)
    config = VllmConfig(cache_config=CacheConfig(cache_dtype="fp8"))
    assert config.cache_config.cache_dtype == "fp8"


# These dimensions and seeds locally reproduce non-invariant norm mismatches.
# They make the invariant equality assertion exercise a discriminating case.
@pytest.mark.parametrize(
    (
        "norm_name",
        "batch_size",
        "hidden_size",
        "position",
        "weight_seed",
        "input_seed",
        "input_pattern",
    ),
    [
        ("rms", 1024, 16384, 512, 7, 42, "normal"),
        ("gemma", 257, 8192, 255, 1, 2, "normal"),
        ("rms", 5, 8192, 3, 7, 46, "scaled"),
        ("gemma", 5, 8192, 3, 1, 46, "scaled"),
    ],
)
def test_residual_norm_preserves_batch_invariance(
    norm_name,
    batch_size,
    hidden_size,
    position,
    weight_seed,
    input_seed,
    input_pattern,
    monkeypatch,
    default_vllm_config,
):
    """Residual norm output must not change when its row changes batch position."""
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm

    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", True)
    init_batch_invariance()
    norm_cls = RMSNorm if norm_name == "rms" else GemmaRMSNorm
    norm = norm_cls(hidden_size, eps=1e-6).to(device="xpu", dtype=torch.bfloat16)
    with torch.no_grad():
        torch.manual_seed(weight_seed)
        norm.weight.uniform_(-0.5, 0.5)

    if input_pattern == "scaled":
        # Heterogeneous columns expose native mean's batch-dependent rounding.
        generator = torch.Generator(device="cpu").manual_seed(input_seed)
        scale = torch.randn(hidden_size, generator=generator).exp()
        x = (torch.randn(batch_size, hidden_size, generator=generator) * scale).to(
            device="xpu", dtype=torch.bfloat16
        )
        residual = (
            torch.randn(batch_size, hidden_size, generator=generator) * scale
        ).to(device="xpu", dtype=torch.bfloat16)
    else:
        torch.manual_seed(input_seed)
        x = torch.randn(batch_size, hidden_size, device="xpu", dtype=torch.bfloat16)
        residual = torch.randn_like(x)

    single_output, single_residual = norm(
        x[position : position + 1].clone(),
        residual[position : position + 1].clone(),
    )
    batch_output, batch_residual = norm(x.clone(), residual.clone())

    assert torch.equal(single_output, batch_output[position : position + 1])
    assert torch.equal(single_residual, batch_residual[position : position + 1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_rsqrt_preserves_batch_invariance(dtype: torch.dtype, strided: bool) -> None:
    """Identical values must not change with vectorized/tail kernel placement."""
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    init_batch_invariance()
    short = torch.full((762, 2), 0.004544191062450409, dtype=dtype, device="xpu")
    long = torch.full((2048, 2), 0.004544191062450409, dtype=dtype, device="xpu")
    short = short[:, 0] if strided else short[:, 0].contiguous()
    long = long[:, 0] if strided else long[:, 0].contiguous()

    torch.testing.assert_close(
        torch.rsqrt(short), torch.rsqrt(long)[:762], rtol=0, atol=0
    )
    torch.testing.assert_close(torch.rsqrt(short), torch.rsqrt(short.cpu()).to("xpu"))


def test_rsqrt_preserves_empty_input() -> None:
    """The invariant override must retain empty tensor shape and dtype."""
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    init_batch_invariance()
    result = torch.rsqrt(torch.empty((0, 3), dtype=torch.float32, device="xpu"))
    assert result.shape == (0, 3)
    assert result.dtype == torch.float32


def test_rsqrt_preserves_integer_promotion() -> None:
    """Integer inputs must still produce default-dtype floating-point results."""
    from vllm.model_executor.determinism.batch_invariant import init_batch_invariance

    init_batch_invariance()
    result = torch.rsqrt(torch.tensor([1, 4, 16, 64], device="xpu"))
    expected = torch.tensor(
        [1.0, 0.5, 0.25, 0.125], dtype=torch.get_default_dtype(), device="xpu"
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_dense_prefill_preserves_batch_invariant_logprobs() -> None:
    """Residual normalization must preserve logits for 762 vs. 2048 prefill rows."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen3-1.7B"),
        dtype="bfloat16",
        enforce_eager=True,
        attention_config={"backend": "TRITON_ATTN"},
        block_size=16,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        max_num_seqs=2,
        enable_prefix_caching=False,
        gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")),
    )
    try:
        tokenizer = llm.get_tokenizer()
        filler_tokens = tokenizer.encode(
            "Describe how a computer executes unrelated work.",
            add_special_tokens=False,
        )
        filler = TokensPrompt(prompt_token_ids=(filler_tokens * 1286)[:1286])
        params = SamplingParams(
            temperature=0, max_tokens=16, logprobs=5, ignore_eos=True
        )
        for seed in range(4):
            needle = TokensPrompt(
                prompt_token_ids=random.Random(seed).choices(
                    range(1000, tokenizer.vocab_size - 100), k=762
                )
            )
            single = llm.generate([needle], params, use_tqdm=False)[0].outputs[0]
            batched = llm.generate([filler, needle], params, use_tqdm=False)[1].outputs[
                0
            ]

            assert single.token_ids == batched.token_ids
            assert single.logprobs is not None and batched.logprobs is not None
            for token_id, left, right in zip(
                single.token_ids, single.logprobs, batched.logprobs, strict=True
            ):
                assert left is not None and right is not None
                assert left[token_id].logprob == right[token_id].logprob
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.parametrize(
    "operation", ["all_reduce", "reduce_scatter", "reduce_scatterv"]
)
@pytest.mark.parametrize("rank", range(4))
def test_collectives_preserve_rank_order_and_partition(monkeypatch, operation, rank):
    """Cancellation exposes reordered sums; mock ranks require no extra XPUs."""
    from vllm.distributed.device_communicators import xpu_communicator

    inputs = [
        torch.full((8, 3), value, dtype=torch.float32, device="xpu")
        for value in (1e20, -1e20, 3.0, 0.0)
    ]
    inputs[2].add_(torch.arange(24, device="xpu").reshape(8, 3))

    def all_gather_single(output, input_, group):
        torch.testing.assert_close(input_, inputs[rank].reshape(-1), rtol=0, atol=0)
        output.copy_(torch.stack(inputs).reshape(-1))

    monkeypatch.setattr(xpu_communicator.dist, "all_gather_single", all_gather_single)
    communicator = XpuCommunicator.__new__(XpuCommunicator)
    communicator.world_size = 4
    communicator.rank_in_group = rank
    communicator.device_group = object()
    expected = inputs[0].clone()
    for value in inputs[1:]:
        expected.add_(value)
    assert torch.equal(expected, inputs[2])

    if operation == "all_reduce":
        actual = communicator.all_reduce(inputs[rank])
    elif operation == "reduce_scatter":
        actual = communicator.reduce_scatter(inputs[rank], dim=0)
        expected = expected.chunk(4)[rank]
    else:
        sizes = [1, 3, 2, 2]
        actual = communicator.reduce_scatterv(inputs[rank], dim=0, sizes=sizes)
        expected = expected.split(sizes)[rank]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


# This batch size, vocabulary size, and positions locally reproduce an
# unseeded non-invariant XPU sampler mismatch.
@pytest.mark.parametrize("position", [0, 128, 255])
def test_seeded_sampler_preserves_full_distribution_and_rng(position, monkeypatch):
    """Moving a request must preserve its distribution and per-request RNG."""
    from vllm.triton_utils import HAS_TRITON
    from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

    assert HAS_TRITON, "XPU sampling must exercise the Triton top-k/top-p path"
    monkeypatch.setattr(envs, "VLLM_XPU_USE_SAMPLER_KERNEL", True)
    sampler = TopKTopPSampler(logprobs_mode="processed_logprobs")
    torch.manual_seed(42)
    logits = torch.randn(256, 8192, device="xpu", dtype=torch.float32)
    single_rng = torch.Generator(device="xpu").manual_seed(123)
    batch_rng = torch.Generator(device="xpu").manual_seed(123)
    k = torch.full((256,), 32, dtype=torch.int32, device="xpu")
    p = torch.full((256,), 0.9, device="xpu")
    for _ in range(3):
        single_token, single_logprobs = sampler(
            logits[position : position + 1].clone(),
            {0: single_rng},
            k[position : position + 1],
            p[position : position + 1],
        )
        batch_tokens, batch_logprobs = sampler(
            logits.clone(), {position: batch_rng}, k, p
        )
        assert single_logprobs is not None and batch_logprobs is not None
        torch.testing.assert_close(
            single_logprobs[0], batch_logprobs[position], rtol=0, atol=0
        )
        assert single_token.item() == batch_tokens[position].item()
        assert torch.equal(single_rng.get_state(), batch_rng.get_state())
