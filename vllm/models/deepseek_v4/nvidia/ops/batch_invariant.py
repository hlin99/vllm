# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-reduction kernels for the DeepSeek V4 FP8 cache layout."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _load_cache(cache, slots, valid, block_size: tl.constexpr, stride: tl.constexpr):
    d = tl.arange(0, 512)
    slots = tl.maximum(slots, 0).to(tl.int64)
    page = cache + slots // block_size * stride
    row = slots % block_size
    data = page[:, None] + row[:, None] * 576
    raw = tl.load(data + d[None, :], valid[:, None] & (d[None, :] < 448), 0)
    scale = tl.load(
        page[:, None] + block_size * 576 + row[:, None] * 8 + d[None, :] // 64,
        valid[:, None] & (d[None, :] < 448),
        127,
    )
    nope = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    nope *= tl.exp2(scale.to(tl.float32) - 127.0)
    rope = tl.load(
        (data + 448).to(tl.pointer_type(tl.bfloat16)) + tl.maximum(d[None, :] - 448, 0),
        valid[:, None] & (d[None, :] >= 448),
        0,
    ).to(tl.float32)
    return tl.where(d[None, :] < 448, nope, rope)


@triton.jit
def _attend_cache(
    q,
    cache,
    indices,
    length,
    m,
    denom,
    acc,
    scale,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STRIDE: tl.constexpr,
):
    for start in range(0, WIDTH, 32):
        cols = start + tl.arange(0, 32)
        slots = tl.load(indices + cols, cols < WIDTH, -1)
        valid = (cols < length) & (slots >= 0)
        kv = _load_cache(cache, slots, valid, BLOCK_SIZE, STRIDE)
        scores = tl.sum(kv * q[None, :], 1) * scale
        scores = tl.where(valid, scores, -float("inf"))
        next_m = tl.maximum(m, tl.max(scores, 0))
        alpha = tl.exp(m - next_m)
        p = tl.exp(scores - next_m)
        acc = acc * alpha + tl.sum(p[:, None] * kv, 0)
        denom = denom * alpha + tl.sum(p, 0)
        m = next_m
    return m, denom, acc


@triton.jit
def _sparse_attention(
    Q,
    SWA,
    EXTRA,
    SI,
    SL,
    EI,
    EL,
    SINK,
    OUT,
    q_stride,
    q_head_stride,
    out_stride,
    out_head_stride,
    si_stride,
    ei_stride,
    scale,
    SWA_WIDTH: tl.constexpr,
    EXTRA_WIDTH: tl.constexpr,
    SWA_BLOCK: tl.constexpr,
    EXTRA_BLOCK: tl.constexpr,
    SWA_STRIDE: tl.constexpr,
    EXTRA_STRIDE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    d = tl.arange(0, 512)
    q = tl.load(Q + row * q_stride + head * q_head_stride + d).to(tl.float32)
    # The sink is an extra softmax entry whose value vector is zero.
    m = tl.load(SINK + head).to(tl.float32)
    denom = 1.0
    acc = tl.full((512,), 0, tl.float32)
    m, denom, acc = _attend_cache(
        q,
        SWA,
        SI + row * si_stride,
        tl.load(SL + row),
        m,
        denom,
        acc,
        scale,
        SWA_WIDTH,
        SWA_BLOCK,
        SWA_STRIDE,
    )
    if EXTRA_WIDTH:
        m, denom, acc = _attend_cache(
            q,
            EXTRA,
            EI + row * ei_stride,
            tl.load(EL + row),
            m,
            denom,
            acc,
            scale,
            EXTRA_WIDTH,
            EXTRA_BLOCK,
            EXTRA_STRIDE,
        )
    tl.store(OUT + row * out_stride + head * out_head_stride + d, acc / denom)


def sparse_attention(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    sparse_indices: torch.Tensor,
    swa_topk_lens: torch.Tensor,
    sinks: torch.Tensor,
    out: torch.Tensor,
    bmm1_scale: float,
    compressed_kv_cache: torch.Tensor | None = None,
    extra_sparse_indices: torch.Tensor | None = None,
    extra_sparse_topk_lens: torch.Tensor | None = None,
) -> None:
    assert query.dtype == torch.bfloat16 and query.shape[-1] == 512
    assert query.stride(-1) == out.stride(-1) == 1
    assert swa_kv_cache.dtype == torch.uint8
    assert swa_kv_cache.shape[-1] == 584
    assert out.shape == query.shape
    extra_width, extra_indices_stride = 0, 0
    extra_block, extra_stride = 1, 0
    if compressed_kv_cache is not None:
        assert compressed_kv_cache.dtype == torch.uint8
        assert compressed_kv_cache.shape[-1] == 584
        assert extra_sparse_indices is not None
        assert extra_sparse_topk_lens is not None
        extra_width = extra_sparse_indices.shape[-1]
        extra_indices_stride = extra_sparse_indices.stride(0)
        extra_block = compressed_kv_cache.shape[1]
        extra_stride = compressed_kv_cache.stride(0)
    if query.shape[0] == 0:
        return
    _sparse_attention[(query.shape[0], query.shape[1])](
        query,
        swa_kv_cache,
        compressed_kv_cache,
        sparse_indices,
        swa_topk_lens,
        extra_sparse_indices,
        extra_sparse_topk_lens,
        sinks,
        out,
        query.stride(0),
        query.stride(1),
        out.stride(0),
        out.stride(1),
        sparse_indices.stride(0),
        extra_indices_stride,
        bmm1_scale,
        SWA_WIDTH=sparse_indices.shape[-1],
        EXTRA_WIDTH=extra_width,
        SWA_BLOCK=swa_kv_cache.shape[1],
        EXTRA_BLOCK=extra_block,
        SWA_STRIDE=swa_kv_cache.stride(0),
        EXTRA_STRIDE=extra_stride,
        num_warps=4,
        enable_fp_fusion=False,
    )
