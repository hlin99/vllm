# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache
from vllm.models.deepseek_v4.xpu.xpu_sparse_decode_fp8 import xpu_sparse_decode_fp8
from vllm.v1.attention.ops.xpu_mla_sparse import (
    triton_bf16_mla_sparse_interface,
    triton_paged_mla_sparse_interface,
)


# https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L7
def _merge_two_lse(
    lse0: torch.Tensor, lse1: torch.Tensor | None, s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )


# Adapted from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/ref.py#L19
def reference_mla_sparse_prefill(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int,
    topk_length: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]

    """
    s_q, h_q, d_qk = q.shape
    s_kv, _, _ = kv.shape
    _, _, topk = indices.shape

    indices = indices.clone().squeeze(1)
    if topk_length is not None:
        mask = torch.arange(topk, device=topk_length.device).unsqueeze(0).broadcast_to(
            s_q, topk
        ) >= topk_length.unsqueeze(1)  # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    q = q.float()
    gathered_kv = (
        kv.index_select(dim=0, index=indices.flatten()).reshape(s_q, topk, d_qk).float()
    )  # [s_q, topk, d_qk]
    P = q @ gathered_kv.transpose(1, 2)  # [s_q, h_q, topk]
    P *= sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    max_logits = P.max(dim=-1).values  # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, attn_sink, s_q, h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float(
        "+inf"
    )  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., :d_v]  # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")  # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(kv.dtype), out, max_logits, orig_lse)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla(device_str, dtype):
    device = torch.device(device_str)
    s_q = 1
    s_kv = 256
    h_q = 64  # kernel expects multiple of 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    for t in range(s_q):
        for h in range(h_kv):
            i_i = torch.randperm(max(1, t))[:topk]
            indices[t, h, : len(i_i)] = i_i

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.shape == (s_q, h_q, d_v)
    assert max_logits.shape == (s_q, h_q)
    assert lse.shape == (s_q, h_q)

    ref_out, ref_out_fp32, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits, ref_max_logits, atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse, ref_lse, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("device_str", ["xpu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(
    not torch.xpu.is_available(),
    reason="XPU is required",
)
def test_bf16_triton_sparse_mla_masked_chunks(device_str, dtype):
    """Rows whose leading BLOCK_N index entries are all masked must not NaN.

    Regression test: with an -inf running max, a fully-masked leading chunk
    produced re_scale = exp2(-inf - -inf) = NaN, permanently poisoning the
    accumulator even though valid keys followed in later chunks.
    """
    device = torch.device(device_str)
    s_q = 3
    s_kv = 256
    h_q = 64
    h_kv = 1
    d_qk = 576
    d_v = 512
    topk = 128  # 8 chunks of BLOCK_N=16

    torch.random.manual_seed(1234)

    q = torch.randn((s_q, h_q, d_qk), dtype=dtype, device=device)
    kv = torch.randn((s_kv, h_kv, d_qk), dtype=dtype, device=device)
    indices = torch.full((s_q, h_kv, topk), -1, dtype=torch.int32, device=device)
    # row 0: valid keys only in chunks 1-2 -> leading AND trailing masked chunks
    indices[0, 0, 16:48] = torch.arange(32, dtype=torch.int32, device=device)
    # row 1: fully valid
    indices[1, 0, :] = torch.arange(topk, dtype=torch.int32, device=device)
    # row 2: no valid key at all

    sm_scale = d_qk**-0.5

    out, max_logits, lse = triton_bf16_mla_sparse_interface(
        q, kv, indices, sm_scale, d_v
    )
    assert out.isfinite().all()

    ref_out, _, ref_max_logits, ref_lse = reference_mla_sparse_prefill(
        q, kv, indices, sm_scale, d_v
    )
    assert torch.allclose(out[:2], ref_out[:2], atol=1e-2, rtol=1e-2)
    assert torch.allclose(max_logits[:2], ref_max_logits[:2], atol=1e-3, rtol=1e-3)
    assert torch.allclose(lse[:2], ref_lse[:2], atol=1e-3, rtol=1e-3)
    # A row with no valid key yields zeros (the reference's convention); its
    # lse/max_logits are large-negative finite rather than the reference's
    # +inf/-inf placeholders, so only the output is compared here.
    assert torch.allclose(out[2], torch.zeros_like(out[2]))


def _make_paged_sparse_cache(block_size, device):
    page_bytes = ((block_size * 584 + 575) // 576) * 576
    storage = torch.full(
        (9, block_size, page_bytes // block_size),
        37,
        dtype=torch.uint8,
        device=device,
    )
    cache = storage[1::2]
    rows = cache.shape[0] * block_size
    values = torch.randn(rows, 512, dtype=torch.bfloat16, device=device)
    values *= torch.logspace(-3, 1, 8, device=device).repeat_interleave(64)
    quantize_and_insert_k_cache(
        values,
        cache.view(cache.shape[0], -1),
        torch.arange(rows, device=device),
        block_size=block_size,
    )
    return cache


@pytest.mark.parametrize("compressed_block_size", [None, 2, 64])
@pytest.mark.parametrize("num_heads", [20, 36, 64])
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
def test_paged_sparse_mla_matches_gathered_kv(
    compressed_block_size, num_heads, monkeypatch
):
    """Paged loads preserve attention with gaps, offset views and masked rows."""
    device = torch.device("xpu")
    torch.manual_seed(42)
    monkeypatch.setenv("VLLM_XPU_PAGED_SPARSE_ATTN", "0")
    swa_cache = _make_paged_sparse_cache(256, device)
    cache = (
        _make_paged_sparse_cache(compressed_block_size, device)
        if compressed_block_size is not None
        else None
    )
    q = torch.randn(11, num_heads + 4, 512, device=device, dtype=torch.bfloat16)[
        1::2, :num_heads
    ]
    out = torch.empty(11, num_heads + 4, 512, device=device, dtype=q.dtype)[
        1::2, :num_heads
    ]
    expected = torch.empty_like(q)
    swa_indices = torch.randint(
        0, 4 * 256, (5, 1, 33), dtype=torch.int32, device=device
    )
    swa_lens = torch.tensor([0, 1, 33, 17, 0], dtype=torch.int32, device=device)
    swa_indices[0] = -1
    swa_indices[1, 0, 1:] = -1
    swa_indices[4] = -1
    topk_indices = None
    topk_lens = None
    if cache is not None:
        topk_indices = torch.randint(
            0,
            cache.shape[0] * cache.shape[1],
            (5, 1, 19),
            dtype=torch.int32,
            device=device,
        )
        topk_lens = torch.tensor([0, 1, 19, 17, 0], dtype=torch.int32, device=device)
    xpu_sparse_decode_fp8(
        q=q,
        kv_cache=cache,
        swa_kv_cache=swa_cache,
        swa_only=cache is None,
        topk_indices=topk_indices,
        topk_lens=topk_lens,
        swa_indices=swa_indices,
        swa_lens=swa_lens,
        attn_sink=torch.zeros(q.shape[1], device=device),
        softmax_scale=512**-0.5,
        head_dim=512,
        nope_head_dim=448,
        rope_head_dim=64,
        out=expected,
    )
    triton_paged_mla_sparse_interface(
        q,
        cache,
        swa_cache,
        topk_indices,
        topk_lens,
        swa_indices,
        swa_lens,
        512**-0.5,
        out,
    )
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("num_heads", [20, 36, 64])
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
def test_paged_sparse_mla_split_kv_preserves_softmax(num_heads):
    """Block-aligned chunks keep FP32 state, including empty chunks and reuse."""
    device = torch.device("xpu")
    torch.manual_seed(43)
    cache = _make_paged_sparse_cache(256, device)
    q = torch.randn(5, num_heads + 4, 512, device=device, dtype=torch.bfloat16)[
        :, :num_heads
    ]
    indices = torch.randint(0, 1024, (5, 1, 33), device=device, dtype=torch.int32)
    lens = torch.tensor([0, 1, 33, 17, 0], device=device, dtype=torch.int32)
    expected = torch.empty_like(q)
    actual = torch.empty_like(q)
    state = torch.full((5, num_heads, 514), float("nan"), device=device)
    empty_indices = indices[..., :0]
    empty_lens = torch.zeros_like(lens)

    def consume(selected, lengths, *, initialize=False, finalize=False):
        triton_paged_mla_sparse_interface(
            q,
            None,
            cache,
            None,
            None,
            selected,
            lengths,
            512**-0.5,
            actual if finalize else None,
            partial_state=state,
            initialize=initialize,
            finalize=finalize,
        )

    for _ in range(2):
        q.neg_()
        triton_paged_mla_sparse_interface(
            q, None, cache, None, None, indices, lens, 512**-0.5, expected
        )
        consume(empty_indices, empty_lens, initialize=True)
        for start in range(0, 33, 16):
            selected = indices[..., start : start + 16]
            consume(selected, (lens - start).clamp(0, selected.shape[-1]))
        consume(empty_indices, empty_lens, finalize=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
