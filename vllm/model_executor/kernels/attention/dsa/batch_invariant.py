# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-reduction FP8 sparse indexer scoring."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _indexer_logits(
    Q,
    K,
    KS,
    W,
    START,
    END,
    BT,
    OUT,
    qs,
    qhs,
    ks,
    kss,
    ws,
    bts,
    width,
    HEADS: tl.constexpr,
    H: tl.constexpr,
    PAGED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * 32 + tl.arange(0, 32)
    h = tl.arange(0, H)
    d = tl.arange(0, 128)
    end = tl.load(END + row)
    if PAGED:
        pos = col
        valid = (col < end) & (col < width)
        block = tl.load(BT + row * bts + pos // PAGE_SIZE, valid, 0).to(tl.int64)
        page = block * PAGE_STRIDE
        # Each page stores all FP8 keys first, then all FP32 scales.
        offset = page + pos % PAGE_SIZE * 128
        raw = tl.load(K + offset[:, None] + d[None, :], valid[:, None], 0)
        kval = raw.to(tl.float8e4nv, bitcast=True)
        scale_offset = page + PAGE_SIZE * 128 + pos % PAGE_SIZE * 4
        scale = tl.load((K + scale_offset).to(tl.pointer_type(tl.float32)), valid, 0)
    else:
        start = tl.load(START + row)
        pos = col + start
        valid = (pos < end) & (col < width)
        kval = tl.load(K + pos[:, None] * ks + d[None, :], valid[:, None], 0.0)
        scale = tl.load(KS + pos * kss, valid, 0)
    q = tl.load(
        Q + row * qs + h[:, None] * qhs + d[None, :],
        h[:, None] < HEADS,
        0.0,
    )
    scores = tl.dot(q, tl.trans(kval), out_dtype=tl.float32)
    weights = tl.load(W + row * ws + h, h < HEADS, 0).to(tl.float32)
    scores = tl.sum(tl.maximum(scores, 0.0) * weights[:, None], 0) * scale
    tl.store(
        OUT + row * width + col, tl.where(valid, scores, -float("inf")), col < width
    )


def indexer_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    ends: torch.Tensor,
    width: int,
    *,
    scales: torch.Tensor | None = None,
    starts: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score identical FP8 queries with the same reduction in either cache layout."""
    assert q.ndim == 3 and q.shape[-1] == 128
    assert q.dtype == torch.float8_e4m3fn
    assert weights.dtype == torch.float32
    assert ends.numel() == q.shape[0] and ends.is_contiguous()
    paged = block_table is not None
    if paged:
        assert k.dtype == torch.uint8 and k.shape[-1] == 132
    else:
        assert scales is not None and scales.dtype == torch.float32
        assert starts is not None
    out = torch.empty((q.shape[0], width), dtype=torch.float32, device=q.device)
    if q.shape[0] and width:
        _indexer_logits[(q.shape[0], triton.cdiv(width, 32))](
            q,
            k,
            scales,
            weights,
            starts,
            ends,
            block_table,
            out,
            q.stride(0),
            q.stride(1),
            k.stride(1) if paged else k.stride(0),
            scales.stride(0) if scales is not None else 0,
            weights.stride(0),
            block_table.stride(0) if block_table is not None else 0,
            width,
            HEADS=q.shape[1],
            H=max(16, triton.next_power_of_2(q.shape[1])),
            PAGED=paged,
            PAGE_SIZE=k.shape[1] if paged else 1,
            PAGE_STRIDE=k.stride(0) if paged else 0,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
