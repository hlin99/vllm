# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-reduction kernels for the DeepSeek V4 FP8 cache layout."""

# 提交③：caf5d70e2b31772b81dd85a82d590708baa89ccc
import torch  # 提供张量接口与缓存格式检查。

from vllm.triton_utils import tl, triton  # 使用固定 tile、固定归约的 Triton 内核。


# 提交③：按物理 slot 读取 fp8_ds_mla 缓存，不展开回原始 token。
@triton.jit
def _load_cache(cache, slots, valid, block_size: tl.constexpr, stride: tl.constexpr):
    d = tl.arange(0, 512)  # 448 维非 RoPE + 64 维 RoPE。
    slots = tl.maximum(slots, 0).to(tl.int64)  # 无效 -1 先转安全地址，再用 valid 屏蔽。
    page = (
        cache + slots // block_size * stride
    )  # 使用真实页跨度，兼容共享缓存 padding。
    row = slots % block_size  # token 在物理页中的偏移。
    data = (
        page[:, None] + row[:, None] * 576
    )  # 每 token：448 字节 FP8 + 128 字节 BF16。
    raw = tl.load(data + d[None, :], valid[:, None] & (d[None, :] < 448), 0)
    scale = tl.load(
        page[:, None] + block_size * 576 + row[:, None] * 8 + d[None, :] // 64,
        # scale 区位于整页数据之后；每 token 8 字节，每 64 维使用一个指数。
        valid[:, None] & (d[None, :] < 448),  # 只有非 RoPE 部分需要 scale。
        127,  # UE8M0 指数偏置为 127，默认指数对应比例 1。
    )
    nope = raw.to(tl.float8e4nv, bitcast=True).to(
        tl.float32
    )  # 字节解释为 FP8 再转 FP32。
    nope *= tl.exp2(scale.to(tl.float32) - 127.0)  # 反量化非 RoPE 特征。
    rope = tl.load(
        (data + 448).to(tl.pointer_type(tl.bfloat16)) + tl.maximum(d[None, :] - 448, 0),
        # 后 64 维以 BF16 存储，指针转换后每个偏移对应两个字节。
        valid[:, None] & (d[None, :] >= 448),  # 仅加载 RoPE 部分。
        0,  # 无效 slot 返回零向量。
    ).to(tl.float32)  # 后续点积与 softmax 均使用 FP32。
    return tl.where(d[None, :] < 448, nope, rope)  # 拼成完整 512 维 KV latent。


@triton.jit
def _attend_cache(
    q,  # 当前 query head 的 512 维向量。
    cache,  # 当前分支的分页 KV 缓存。
    indices,  # 当前 query 对应的物理 slot 列表。
    length,  # 列表中实际有效的条目数。
    m,  # 在线 softmax 已处理部分的最大分数。
    denom,  # 以 m 为基准的 softmax 分母。
    acc,  # 以 m 为基准的加权 KV 累积向量。
    scale,  # QK 点积的缩放系数。
    WIDTH: tl.constexpr,  # 索引列表的物理宽度，含 padding。
    BLOCK_SIZE: tl.constexpr,  # 每个 KV 页包含的 token 数。
    STRIDE: tl.constexpr,  # 两页之间的真实字节跨度。
):
    for start in range(0, WIDTH, 32):  # 始终每批处理 32 个 KV，不按请求 batch 改分片。
        cols = start + tl.arange(0, 32)  # 当前固定 tile 的索引列。
        slots = tl.load(indices + cols, cols < WIDTH, -1)  # 末块越界填 -1。
        valid = (cols < length) & (slots >= 0)  # 排除 padding 与无效 slot。
        kv = _load_cache(cache, slots, valid, BLOCK_SIZE, STRIDE)  # 读取并反量化。
        scores = tl.sum(kv * q[None, :], 1) * scale  # 沿固定 512 维计算 QK。
        scores = tl.where(valid, scores, -float("inf"))  # 无效项 softmax 权重为零。
        next_m = tl.maximum(m, tl.max(scores, 0))  # 合并新旧最大值以稳定指数计算。
        alpha = tl.exp(m - next_m)  # 把旧分母和旧累积转换到新最大值基准。
        p = tl.exp(scores - next_m)  # 新 tile 的未归一化概率。
        acc = acc * alpha + tl.sum(p[:, None] * kv, 0)  # 更新 512 维加权和。
        denom = denom * alpha + tl.sum(p, 0)  # 更新归一化分母。
        m = next_m  # 下一 tile 延续同一个 softmax 状态。
    return m, denom, acc  # 允许滑窗与压缩分支共享同一次 softmax。


@triton.jit
def _sparse_attention(
    Q,  # [token, head, 512] query。
    SWA,  # 滑动窗口 KV 缓存。
    EXTRA,  # 可选的压缩 KV 缓存。
    SI,  # 滑窗 KV 的物理 slot 索引。
    SL,  # 每个 query 的有效滑窗条目数。
    EI,  # 压缩 KV 的物理 slot 索引。
    EL,  # 每个 query 的有效压缩条目数。
    SINK,  # 每个 head 的 attention sink 分数。
    OUT,  # attention 输出，与 Q 形状一致。
    q_stride,  # Q 的 token 跨度。
    q_head_stride,  # Q 的 head 跨度。
    out_stride,  # 输出 token 跨度。
    out_head_stride,  # 输出 head 跨度。
    si_stride,  # 滑窗索引的 token 跨度。
    ei_stride,  # 压缩索引的 token 跨度。
    scale,  # 点积缩放值。
    SWA_WIDTH: tl.constexpr,  # 滑窗索引固定宽度。
    EXTRA_WIDTH: tl.constexpr,  # 压缩索引固定宽度；零表示关闭该分支。
    SWA_BLOCK: tl.constexpr,  # 滑窗缓存每页 token 数。
    EXTRA_BLOCK: tl.constexpr,  # 压缩缓存每页 token 数。
    SWA_STRIDE: tl.constexpr,  # 滑窗缓存页跨度。
    EXTRA_STRIDE: tl.constexpr,  # 压缩缓存页跨度。
):
    row = tl.program_id(0).to(tl.int64)  # 每个 token 独立计算，不跨 token 归约。
    head = tl.program_id(1)  # 每个 query head 使用独立 program。
    d = tl.arange(0, 512)  # 固定特征维。
    q = tl.load(Q + row * q_stride + head * q_head_stride + d).to(tl.float32)
    # The sink is an extra softmax entry whose value vector is zero.
    m = tl.load(SINK + head).to(tl.float32)  # 用 sink 初始化 softmax 最大值。
    denom = 1.0  # sink 相对自身的指数为 1，计入分母。
    acc = tl.full((512,), 0, tl.float32)  # sink 的 value 为零，不贡献加权和。
    m, denom, acc = _attend_cache(
        q,  # 当前 query head。
        SWA,  # 先读取滑窗 KV。
        SI + row * si_stride,  # 当前 token 的滑窗索引起点。
        tl.load(SL + row),  # 有效滑窗长度。
        m,  # 延续 sink 初始化的最大值。
        denom,  # 延续 sink 初始化的分母。
        acc,  # 延续零向量加权和。
        scale,  # QK 缩放。
        SWA_WIDTH,  # 滑窗索引宽度。
        SWA_BLOCK,  # 滑窗页大小。
        SWA_STRIDE,  # 滑窗页跨度。
    )
    if EXTRA_WIDTH:  # 编译期判断是否存在压缩 KV 分支。
        m, denom, acc = _attend_cache(
            q,  # 两路使用同一个 query。
            EXTRA,  # 再读取压缩 KV，不另起一次 softmax。
            EI + row * ei_stride,  # 当前 token 的压缩索引。
            tl.load(EL + row),  # 有效压缩条目数。
            m,  # 延续滑窗处理后的最大值。
            denom,  # 延续滑窗处理后的分母。
            acc,  # 延续滑窗处理后的加权和。
            scale,  # 相同 QK 缩放。
            EXTRA_WIDTH,  # 压缩索引宽度。
            EXTRA_BLOCK,  # 压缩缓存页大小。
            EXTRA_STRIDE,  # 压缩缓存页跨度。
        )
    tl.store(OUT + row * out_stride + head * out_head_stride + d, acc / denom)
    # 上一行只在两路都处理完后归一化，保持联合 attention 语义。


def sparse_attention(
    *,
    query: torch.Tensor,  # BF16 query。
    swa_kv_cache: torch.Tensor,  # fp8_ds_mla 滑窗缓存。
    sparse_indices: torch.Tensor,  # 滑窗物理 slot 列表。
    swa_topk_lens: torch.Tensor,  # 每行有效滑窗长度。
    sinks: torch.Tensor,  # 每头 sink 分数。
    out: torch.Tensor,  # 调用方提供的输出缓冲区。
    bmm1_scale: float,  # attention 点积缩放。
    compressed_kv_cache: torch.Tensor | None = None,  # 可选压缩缓存。
    extra_sparse_indices: torch.Tensor | None = None,  # 对应的物理索引。
    extra_sparse_topk_lens: torch.Tensor | None = None,  # 对应的有效长度。
) -> None:
    assert query.dtype == torch.bfloat16 and query.shape[-1] == 512  # 限定输入格式。
    assert query.stride(-1) == out.stride(-1) == 1  # 内核直接按连续特征维访问。
    assert swa_kv_cache.dtype == torch.uint8  # 内核自行解释并反量化缓存字节。
    assert out.shape == query.shape  # 每个 query head 输出一个同维向量。
    extra_width, extra_indices_stride = 0, 0  # 默认没有压缩分支。
    extra_block, extra_stride = 1, 0  # 未使用分支的编译参数占位值。
    if compressed_kv_cache is not None:
        assert compressed_kv_cache.dtype == torch.uint8  # 两路使用相同编码格式。
        assert extra_sparse_indices is not None  # 缓存存在时必须提供索引。
        assert extra_sparse_topk_lens is not None  # 以及每行有效长度。
        extra_width = extra_sparse_indices.shape[-1]  # 读取压缩索引宽度。
        extra_indices_stride = extra_sparse_indices.stride(0)  # 读取索引行跨度。
        extra_block = compressed_kv_cache.shape[1]  # 压缩页大小可以不同于滑窗页。
        extra_stride = compressed_kv_cache.stride(0)  # 不假设缓存页紧凑排列。
    if query.shape[0] == 0:  # 空输入没有输出元素需要写入。
        return
    _sparse_attention[(query.shape[0], query.shape[1])](  # 固定每 token、每 head 一块。
        query,  # Q。
        swa_kv_cache,  # 滑窗缓存。
        compressed_kv_cache,  # 可选压缩缓存。
        sparse_indices,  # 滑窗索引。
        swa_topk_lens,  # 滑窗长度。
        extra_sparse_indices,  # 压缩索引。
        extra_sparse_topk_lens,  # 压缩长度。
        sinks,  # sink 参数。
        out,  # 原地写入输出。
        query.stride(0),  # Q token 跨度。
        query.stride(1),  # Q head 跨度。
        out.stride(0),  # 输出 token 跨度。
        out.stride(1),  # 输出 head 跨度。
        sparse_indices.stride(0),  # 滑窗索引行跨度。
        extra_indices_stride,  # 压缩索引行跨度。
        bmm1_scale,  # QK 缩放。
        SWA_WIDTH=sparse_indices.shape[-1],  # 固定滑窗遍历宽度。
        EXTRA_WIDTH=extra_width,  # 固定压缩遍历宽度。
        SWA_BLOCK=swa_kv_cache.shape[1],  # 滑窗页内 token 数。
        EXTRA_BLOCK=extra_block,  # 压缩页内 token 数。
        SWA_STRIDE=swa_kv_cache.stride(0),  # 滑窗物理页跨度。
        EXTRA_STRIDE=extra_stride,  # 压缩物理页跨度。
        num_warps=4,  # 不随 batch 改变执行配置。
        enable_fp_fusion=False,  # 禁止浮点融合，固定舍入路径。
    )
