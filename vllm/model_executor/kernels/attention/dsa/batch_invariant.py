# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-reduction FP8 sparse indexer scoring."""

import torch  # e426be8173：使用 PyTorch 张量描述输入、输出与数据类型。

from vllm.triton_utils import tl, triton  # Triton 负责 GPU 编译、索引和固定形状归约。


@triton.jit
def _sum_in_dtype(left, right):
    return (left + right).to(left.dtype)


# e426be8173：一个 program 计算一个 query 对 32 个 key 的分数。
# prefill 连续缓存和 decode 分页缓存共用计算形状，避免随 batch 改变归约顺序。
@triton.jit
def _indexer_logits(
    Q,  # FP8 query，逻辑形状为 [query 数, 头数, 128]。
    K,  # 连续 FP8 key 或包含量化数据与 scale 的分页字节缓存。
    KS,  # 连续布局的 FP32 或 packed-UE8M0 key scale；分页布局直接从 K 读取。
    W,  # 每个 query、每个 indexer head 的合并权重。
    START,  # 连续布局中，每个 query 所属请求的 key 起始下标。
    END,  # 每个 query 可见的 key 终点，不包含该位置。
    BT,  # 分页布局的请求块表，把逻辑页映射为物理页。
    OUT,  # 输出 FP32 分数，逻辑形状为 [query 数, width]。
    qs,  # Q 的 query 维步长。
    qhs,  # Q 的 head 维步长。
    ks,  # 连续 K 的 token 维步长。
    kss,  # 连续 key scale 的 token 维步长。
    ws,  # 权重的 query 维步长。
    bts,  # 块表的 query 维步长。
    width,  # 每行输出宽度，也是最后一块的写入边界。
    HEADS: tl.constexpr,  # 实际参与计算的头数。
    H: tl.constexpr,  # 补齐后的固定头数，决定归约形状。
    PAGED: tl.constexpr,  # 编译期选择分页读取或连续读取。
    PAGE_SIZE: tl.constexpr,  # 每个物理页容纳的 key 数。
    PAGE_STRIDE: tl.constexpr,  # 物理页间的字节跨度，保留分配中的 padding。
    ACC_DTYPE: tl.constexpr,  # 调用方选择的累加精度，默认 FP32。
    MX_SCALES: tl.constexpr,  # SM100 的 FP8 scale 以四个相同 UE8M0 字节存放。
):
    row = tl.program_id(0).to(tl.int64)  # 当前 query；64 位地址运算防止大缓存溢出。
    col = tl.program_id(1) * 32 + tl.arange(0, 32)  # 本 program 的 32 个 key。
    h = tl.arange(0, H)  # 固定头维，超出实际头数的部分稍后置零。
    d = tl.arange(0, 128)  # indexer 每个 key/query 的特征维。
    end = tl.load(END + row)  # 读取该 query 的因果可见边界。
    if PAGED:
        pos = col  # 分页布局中列号直接表示请求内逻辑位置。
        valid = (col < end) & (col < width)  # 屏蔽未来位置和末块越界位置。
        block = tl.load(BT + row * bts + pos // PAGE_SIZE, valid, 0).to(tl.int64)
        # 上一行通过块表得到物理页号；下面使用真实页 stride，而非假设页紧邻。
        page = block * PAGE_STRIDE
        # Each page stores all FP8 keys first, then all four-byte scales.
        offset = page + pos % PAGE_SIZE * 128  # 页内先连续存放所有 FP8 key。
        raw = tl.load(K + offset[:, None] + d[None, :], valid[:, None], 0)
        kval = raw.to(
            tl.float8e4nv, bitcast=True
        )  # 把原始字节解释为 FP8，不做数值转换。
        scale_offset = page + PAGE_SIZE * 128 + pos % PAGE_SIZE * 4
        # key 数据之后是每 token 4 字节的 scale；无效位置不读取。
        if MX_SCALES:
            packed = tl.load(
                (K + scale_offset).to(tl.pointer_type(tl.uint32)), valid, 0
            )
            scale = tl.exp2((packed & 255).to(tl.float32) - 127.0)
        else:
            scale = tl.load(
                (K + scale_offset).to(tl.pointer_type(tl.float32)), valid, 0
            )
    else:
        start = tl.load(START + row)  # gather 后多个请求的 key 拼接存放。
        pos = col + start  # 将请求内位置转换为拼接缓存下标。
        valid = (pos < end) & (col < width)  # 只读当前 query 的可见历史。
        kval = tl.load(K + pos[:, None] * ks + d[None, :], valid[:, None], 0.0)
        scale = tl.load(KS + pos * kss, valid, 0)  # 读取同一位置的反量化比例。
        if MX_SCALES:
            scale = tl.exp2((scale.to(tl.uint32) & 255).to(tl.float32) - 127.0)
    q = tl.load(
        Q + row * qs + h[:, None] * qhs + d[None, :],  # 当前 query 的全部头。
        h[:, None] < HEADS,  # padding 头不访问输入。
        0.0,  # padding 头填零，不影响点积。
    )
    weights = tl.load(W + row * ws + h, h < HEADS, 0).to(ACC_DTYPE)
    if tl.float32 == ACC_DTYPE:
        scores = tl.dot(q, tl.trans(kval), out_dtype=tl.float32)
        # 默认路径保留原来的点积和归约顺序，避免改变已有 FP32 结果。
        scores = tl.sum(tl.maximum(scores, 0.0) * weights[:, None], 0) * scale
    else:
        # FP8 dot 不提供 BF16 累加；逐特征计算，避免仅在 FP32 累加后转换结果。
        q = q.to(ACC_DTYPE)
        kval = kval.to(ACC_DTYPE)
        scores = tl.full((H, 32), 0, ACC_DTYPE)
        for feature in range(128):
            q_feature = tl.gather(
                q, tl.full((H, 1), feature, tl.int32), axis=1
            ).reshape((H,))
            k_feature = tl.gather(
                kval, tl.full((32, 1), feature, tl.int32), axis=1
            ).reshape((32,))
            product = (q_feature[:, None] * k_feature[None, :]).to(ACC_DTYPE)
            scores = (scores + product).to(ACC_DTYPE)
        weighted = (tl.maximum(scores, 0.0) * weights[:, None]).to(ACC_DTYPE)
        # 每个归约节点都舍入到指定精度，防止 tl.sum 隐式使用 FP32 累加。
        scores = tl.reduce(weighted, 0, _sum_in_dtype)
        scores = (scores * scale.to(ACC_DTYPE)).to(ACC_DTYPE)
    tl.store(
        OUT + row * width + col,
        tl.where(valid, scores, -float("inf")),
        col < width,
        # 不可见位置写负无穷，防止后续 Top-K 把它选为有效候选。
    )


def indexer_logits(
    q: torch.Tensor,  # [M, heads, 128] FP8 query。
    k: torch.Tensor,  # 连续 key 或分页缓存。
    weights: torch.Tensor,  # [M, heads] 浮点合并权重，输入类型不限定为 FP32。
    ends: torch.Tensor,  # [M] 各 query 可见 key 终点。
    width: int,  # 需要分配、计算的每行候选宽度。
    *,
    scales: torch.Tensor | None = None,  # 连续布局必需的 key scale。
    starts: torch.Tensor | None = None,  # 连续布局必需的请求起始下标。
    block_table: torch.Tensor | None = None,  # 非空时选择分页布局。
    acc_dtype: torch.dtype = torch.float32,  # 可选累加精度，不传时保持 FP32。
    use_mx_scales: bool = False,  # 可选的 SM100 FP8 packed-UE8M0 scale 布局。
) -> torch.Tensor:
    """Score FP8 queries with batch-invariant reductions in either cache layout.

    Args:
        q: FP8 queries of shape [M, heads, 128].
        k: Contiguous FP8 keys or a paged byte cache.
        weights: Head weights of shape [M, heads], in float32, float16 or bfloat16.
        ends: Exclusive visible key endpoints for each query.
        width: Number of output score columns.
        scales: FP32 or packed-UE8M0 int32 key scales for the contiguous layout.
        starts: Request key offsets for the contiguous layout.
        block_table: Logical-to-physical page mapping for the paged layout.
        acc_dtype: Accumulation dtype, independent of the weights' input dtype.
            Supports float32 (default), float16 and bfloat16. Low-precision
            accumulation uses a slower, fixed-order feature loop and rounds
            each product and accumulation to the requested dtype.
        use_mx_scales: Whether key scales use four equal UE8M0 bytes per token
            instead of a single FP32 value. Contiguous scales must be int32
            in this layout, as in the SM100 FP8 indexer cache.

    Returns:
        FP32 scores for the shared Top-K interface, including when accumulation
        uses a lower precision.

    Raises:
        ValueError: If the requested accumulation dtype is unsupported.
        TypeError: If weights do not use a supported floating-point dtype.

    """
    assert q.ndim == 3 and q.shape[-1] == 128  # 内核固定支持 128 维 indexer。
    assert q.dtype == torch.float8_e4m3fn  # 不支持 FP4 query。
    acc_types = {
        torch.float32: tl.float32,
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }
    if acc_dtype not in acc_types:
        raise ValueError(
            f"Unsupported indexer accumulation dtype: {acc_dtype}. "
            "Expected float32, float16 or bfloat16."
        )
    if weights.dtype not in acc_types:
        raise TypeError(
            f"Unsupported indexer weights dtype: {weights.dtype}. "
            "Expected float32, float16 or bfloat16."
        )
    assert ends.numel() == q.shape[0] and ends.is_contiguous()  # 每行一个终点。
    paged = block_table is not None  # 是否需要逻辑页到物理页映射。
    if paged:
        assert k.dtype == torch.uint8 and k.shape[-1] == 132  # 128 字节 key + 4 scale。
    else:
        scale_dtype = torch.int32 if use_mx_scales else torch.float32
        assert scales is not None and scales.dtype == scale_dtype  # 连续 scale 编码。
        assert starts is not None  # 连续缓存必须说明各请求起点。
    out = torch.empty((q.shape[0], width), dtype=torch.float32, device=q.device)
    if q.shape[0] and width:  # 空 query 或空候选集不启动零维 GPU grid。
        _indexer_logits[(q.shape[0], triton.cdiv(width, 32))](
            q,  # 查询张量。
            k,  # key 数据。
            scales,  # 连续 key 的 scale。
            weights,  # 每头合并权重。
            starts,  # 连续请求起点。
            ends,  # 因果可见终点。
            block_table,  # 分页请求块表。
            out,  # 分数输出。
            q.stride(0),  # query 行跨度。
            q.stride(1),  # query 头跨度。
            k.stride(1) if paged else k.stride(0),  # 连续布局使用 token 跨度。
            scales.stride(0) if scales is not None else 0,  # scale 跨度。
            weights.stride(0),  # 权重行跨度。
            block_table.stride(0) if block_table is not None else 0,  # 块表行跨度。
            width,  # 输出行宽。
            HEADS=q.shape[1],  # 实际头数。
            H=max(16, triton.next_power_of_2(q.shape[1])),  # 固定、合法的 dot 头维。
            PAGED=paged,  # 编译期消除另一套缓存读取分支。
            PAGE_SIZE=k.shape[1] if paged else 1,  # 分页缓存每页 token 数。
            PAGE_STRIDE=k.stride(0) if paged else 0,  # 保留真实物理页间隔。
            ACC_DTYPE=acc_types[acc_dtype],  # 显式传入累加精度，供编译期分派。
            MX_SCALES=use_mx_scales,  # 两种缓存布局采用相同的 scale 解码规则。
            num_warps=4,  # 固定执行配置，不按 batch 大小选择。
            enable_fp_fusion=False,  # 禁止融合浮点运算引入不同舍入路径。
        )
    return out  # 返回分数矩阵；Top-K 由调用方执行。
