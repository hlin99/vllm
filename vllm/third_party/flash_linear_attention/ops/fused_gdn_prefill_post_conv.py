# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused post-conv1d preparation for GDN prefill.

Replaces the chain:
    split → rearrange → contiguous * 3 → l2norm * 2 → gating
with a **single Triton kernel** that reads the conv'd mixed_qkv output
and writes directly to q/k/v/g/beta in the target contiguous layout.

"""

from __future__ import annotations

import torch

from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["T"])
def _fused_qkv_l2norm_kernel(
    Q,
    K,
    V,
    Q_out,
    K_out,
    V_out,
    T,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    V_STRIDES: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, BLOCK_D).to(tl.int64)
    qk_rows = (rows // H)[:, None]
    qk_heads = (rows % H)[:, None]
    qk_mask = (rows[:, None] < T * H) & (cols[None, :] < DK)
    q_offsets = (
        qk_rows * Q_STRIDES[0]
        + qk_heads * Q_STRIDES[1]
        + cols[None, :] * Q_STRIDES[2]
    )
    k_offsets = (
        qk_rows * K_STRIDES[0]
        + qk_heads * K_STRIDES[1]
        + cols[None, :] * K_STRIDES[2]
    )
    q = tl.load(Q + q_offsets, qk_mask, other=0).to(tl.float32)
    k = tl.load(K + k_offsets, qk_mask, other=0).to(tl.float32)
    q *= tl.rsqrt(tl.sum(q * q, axis=1) + 1e-6)[:, None]
    k *= tl.rsqrt(tl.sum(k * k, axis=1) + 1e-6)[:, None]
    out_offsets = rows[:, None] * DK + cols[None, :]
    tl.store(Q_out + out_offsets, q, qk_mask)
    tl.store(K_out + out_offsets, k, qk_mask)

    v_mask = (rows[:, None] < T * HV) & (cols[None, :] < DV)
    v_offsets = (
        (rows // HV)[:, None] * V_STRIDES[0]
        + (rows % HV)[:, None] * V_STRIDES[1]
        + cols[None, :] * V_STRIDES[2]
    )
    v = tl.load(V + v_offsets, v_mask, other=0)
    tl.store(V_out + rows[:, None] * DV + cols[None, :], v, v_mask)


def fused_qkv_l2norm(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize strided Q/K and pack Q/K/V without computing or changing gates."""
    if (
        any(x.ndim != 4 or x.shape[0] != 1 for x in (q, k, v))
        or q.shape != k.shape
        or q.shape[1] != v.shape[1]
        or min(*q.shape[2:], *v.shape[2:]) <= 0
        or q.device != k.device
        or q.device != v.device
    ):
        raise ValueError("Q/K/V preparation requires compatible packed [1,T,H,D] inputs.")
    tokens, heads, key_dim = q.shape[1:]
    value_heads, value_dim = v.shape[2:]
    # Already-contiguous Q/K do not benefit from fused packing.
    contiguous_strides = (heads * key_dim, key_dim, 1)
    if q.stride()[1:] == k.stride()[1:] == contiguous_strides:
        return l2norm_fwd(q), l2norm_fwd(k), v.contiguous()
    q_out, k_out, v_out = (
        torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v)
    )
    block_rows = 8
    grid = (triton.cdiv(tokens * max(heads, value_heads), block_rows),)
    _fused_qkv_l2norm_kernel[grid](
        q,
        k,
        v,
        q_out,
        k_out,
        v_out,
        tokens,
        q.stride()[1:],
        k.stride()[1:],
        v.stride()[1:],
        heads,
        value_heads,
        key_dim,
        value_dim,
        block_rows,
        triton.next_power_of_2(max(key_dim, value_dim)),
        num_warps=4,
    )
    return q_out, k_out, v_out


@triton.jit
def _fused_post_conv_kernel(
    # ---- inputs ----
    mixed_qkv_ptr,  # [L, qkv_dim] conv'd output (contiguous)
    a_ptr,  # [L, HV]
    b_ptr,  # [L, HV]
    # ---- params ----
    A_log_ptr,  # [HV]
    dt_bias_ptr,  # [HV]
    # ---- outputs ----
    q_ptr,  # [L, H, K] contiguous
    k_ptr,  # [L, H, K] contiguous
    v_ptr,  # [L, HV, V] contiguous
    g_ptr,  # [L, HV] float32
    beta_ptr,  # [L, HV] float32
    # ---- strides ----
    stride_x_tok,  # qkv_dim
    stride_a_tok,  # HV
    stride_b_tok,  # HV
    stride_q_tok,  # H * K
    stride_k_tok,  # H * K
    stride_v_tok,  # HV * V
    # ---- dims ----
    L,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    APPLY_L2NORM: tl.constexpr,
    L2NORM_EPS: tl.constexpr,
    OUTPUT_G_EXP: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Single fused kernel for post-conv1d preparation.

    Grid: (ceil(L, BLOCK_T), H + HV)
      - program_id(1) in [0, H):    Q/K head processing + l2norm
      - program_id(1) in [H, H+HV): V head processing + gating
    """
    i_tb = tl.program_id(0)
    i_head = tl.program_id(1)

    HK: tl.constexpr = H * K

    offs_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    mask_t = offs_t < L

    if i_head < H:
        # ============ Q/K head processing ============
        i_h = i_head
        offs_k = tl.arange(0, BK)  # [BK]
        mask_k = offs_k < K
        mask_2d = mask_t[:, None] & mask_k[None, :]  # [BLOCK_T, BK]

        # Load Q features: mixed_qkv[t, i_h*K + k]
        q_offsets = offs_t[:, None] * stride_x_tok + i_h * K + offs_k[None, :]
        q_f32 = tl.load(mixed_qkv_ptr + q_offsets, mask=mask_2d, other=0).to(tl.float32)

        # Load K features: mixed_qkv[t, HK + i_h*K + k]
        k_offsets = offs_t[:, None] * stride_x_tok + HK + i_h * K + offs_k[None, :]
        k_f32 = tl.load(mixed_qkv_ptr + k_offsets, mask=mask_2d, other=0).to(tl.float32)

        if APPLY_L2NORM:
            q_sq_sum = tl.sum(q_f32 * q_f32, axis=1)  # [BLOCK_T]
            q_inv = 1.0 / tl.sqrt(q_sq_sum + L2NORM_EPS)
            q_f32 = q_f32 * q_inv[:, None]

            k_sq_sum = tl.sum(k_f32 * k_f32, axis=1)
            k_inv = 1.0 / tl.sqrt(k_sq_sum + L2NORM_EPS)
            k_f32 = k_f32 * k_inv[:, None]

        # Store Q
        q_out = offs_t[:, None] * stride_q_tok + i_h * K + offs_k[None, :]
        tl.store(
            q_ptr + q_out,
            q_f32.to(q_ptr.dtype.element_ty),
            mask=mask_2d,
        )

        # Store K
        k_out = offs_t[:, None] * stride_k_tok + i_h * K + offs_k[None, :]
        tl.store(
            k_ptr + k_out,
            k_f32.to(k_ptr.dtype.element_ty),
            mask=mask_2d,
        )
    else:
        # ============ V head + gating processing ============
        i_hv = i_head - H
        offs_v = tl.arange(0, BV)  # [BV]
        mask_v = offs_v < V
        mask_2d = mask_t[:, None] & mask_v[None, :]  # [BLOCK_T, BV]

        V_OFFSET: tl.constexpr = 2 * H * K

        # Load V features: mixed_qkv[t, 2*H*K + i_hv*V + v]
        v_offsets = (
            offs_t[:, None] * stride_x_tok + V_OFFSET + i_hv * V + offs_v[None, :]
        )
        v_vals = tl.load(mixed_qkv_ptr + v_offsets, mask=mask_2d, other=0)

        # Store V
        v_out = offs_t[:, None] * stride_v_tok + i_hv * V + offs_v[None, :]
        tl.store(v_ptr + v_out, v_vals, mask=mask_2d)

        # Gating: one scalar per (token, v-head)
        A_log_val = tl.load(A_log_ptr + i_hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + i_hv).to(tl.float32)

        a_offsets = offs_t * stride_a_tok + i_hv
        b_offsets = offs_t * stride_b_tok + i_hv
        a_vals = tl.load(a_ptr + a_offsets, mask=mask_t, other=0).to(tl.float32)
        b_vals = tl.load(b_ptr + b_offsets, mask=mask_t, other=0).to(tl.float32)

        # g = -exp(A_log) * softplus(a + dt_bias)
        x = a_vals + dt_bias_val
        sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
        sp = tl.where(x <= SOFTPLUS_THRESHOLD, sp, x)
        g_vals = -tl.exp(A_log_val) * sp

        if OUTPUT_G_EXP:
            g_vals = tl.exp(g_vals)

        beta_vals = tl.sigmoid(b_vals)

        gb_offsets = offs_t * HV + i_hv
        tl.store(g_ptr + gb_offsets, g_vals, mask=mask_t)
        tl.store(beta_ptr + gb_offsets, beta_vals, mask=mask_t)


def fused_post_conv_prep(
    conv_output: torch.Tensor,  # [L, qkv_dim] conv'd mixed_qkv
    a: torch.Tensor,  # [L, HV]
    b: torch.Tensor,  # [L, HV]
    A_log: torch.Tensor,  # [HV]
    dt_bias: torch.Tensor,  # [HV]
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused post-conv1d prep: split + l2norm + gating in one kernel.

    Args:
        conv_output: [L, qkv_dim] contiguous conv'd mixed_qkv
        a: [L, HV] gating input
        b: [L, HV] gating input
        A_log: [HV] log decay parameter
        dt_bias: [HV] dt bias parameter
        num_k_heads: number of K heads (H)
        head_k_dim: dimension per K head (K)
        head_v_dim: dimension per V head (V)
        apply_l2norm: whether to L2-normalize q and k
        output_g_exp: if True, output exp(g) instead of g (for FlashInfer)

    Returns:
        q: [L, H, K] contiguous, optionally l2-normalized
        k: [L, H, K] contiguous, optionally l2-normalized
        v: [L, HV, V] contiguous
        g: [L, HV] float32
        beta: [L, HV] float32
    """
    L = conv_output.shape[0]
    qkv_dim = conv_output.shape[1]
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    HV = A_log.shape[0]
    dtype = conv_output.dtype
    device = conv_output.device

    assert qkv_dim == 2 * H * K + HV * V, (
        f"qkv_dim={qkv_dim} != 2*H*K + HV*V = {2 * H * K + HV * V}"
    )

    # Allocate outputs in target contiguous layout
    q = torch.empty(L, H, K, dtype=dtype, device=device)
    k = torch.empty(L, H, K, dtype=dtype, device=device)
    v = torch.empty(L, HV, V, dtype=dtype, device=device)
    g = torch.empty(L, HV, dtype=torch.float32, device=device)
    beta = torch.empty(L, HV, dtype=torch.float32, device=device)

    if L == 0:
        return q, k, v, g, beta

    # ---- Kernel config ----
    BK = triton.next_power_of_2(K)
    BV = triton.next_power_of_2(V)
    BLOCK_T = 16  # tokens per block

    # Single kernel: blocks [0,H) do Q/K, blocks [H, H+HV) do V+gating
    grid = (triton.cdiv(L, BLOCK_T), H + HV)
    _fused_post_conv_kernel[grid](
        mixed_qkv_ptr=conv_output,
        a_ptr=a,
        b_ptr=b,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        g_ptr=g,
        beta_ptr=beta,
        stride_x_tok=conv_output.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_q_tok=q.stride(0),
        stride_k_tok=k.stride(0),
        stride_v_tok=v.stride(0),
        L=L,
        H=H,
        HV=HV,
        K=K,
        V=V,
        APPLY_L2NORM=apply_l2norm,
        L2NORM_EPS=1e-6,
        OUTPUT_G_EXP=output_g_exp,
        SOFTPLUS_THRESHOLD=20.0,
        BLOCK_T=BLOCK_T,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )

    return q, k, v, g, beta
