# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lossless publication of PCP-local compressed cache rows."""

import torch

from vllm.distributed import get_pcp_group
from vllm.triton_utils import tl, triton


@triton.jit
def _copy_packed_cache_rows(
    cache,
    rows,
    slots,
    cache_stride: tl.constexpr,
    cache_block_size: tl.constexpr,
    data_bytes: tl.constexpr,
    scale_bytes: tl.constexpr,
    pack: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    offsets = tl.arange(0, BLOCK)
    width: tl.constexpr = data_bytes + scale_bytes
    block = slot // cache_block_size
    pos = slot % cache_block_size
    cache_offsets = block.to(tl.int64) * cache_stride + tl.where(
        offsets < data_bytes,
        pos * data_bytes + offsets,
        cache_block_size * data_bytes + pos * scale_bytes + offsets - data_bytes,
    )
    valid = (slot >= 0) & (offsets < width)
    if pack:
        values = tl.load(cache + cache_offsets, mask=valid, other=0)
        tl.store(rows + row * width + offsets, values, mask=offsets < width)
    else:
        values = tl.load(rows + row * width + offsets, mask=valid, other=0)
        tl.store(cache + cache_offsets, values, mask=valid)


def publish_compressed_cache(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    max_segments: int,
    data_bytes: int,
    scale_bytes: int,
) -> None:
    group = get_pcp_group()
    slots_by_rank = slot_mapping.view(group.world_size, -1)
    num_tokens = slots_by_rank.shape[1]
    capacity = min(num_tokens, num_tokens // compress_ratio + max_segments)
    order = torch.arange(num_tokens, dtype=torch.int32, device=cache.device)
    keys = torch.where(slots_by_rank >= 0, num_tokens - order, -1 - order)
    indices = torch.topk(keys, capacity, dim=1, sorted=True).indices
    slots = slots_by_rank.gather(1, indices)
    local_slots = slots[group.rank_in_group]
    rows = torch.empty(
        (capacity, data_bytes + scale_bytes), dtype=torch.uint8, device=cache.device
    )
    kwargs = dict(
        cache_stride=cache.stride(0),
        cache_block_size=cache.shape[1],
        data_bytes=data_bytes,
        scale_bytes=scale_bytes,
        BLOCK=triton.next_power_of_2(data_bytes + scale_bytes),
    )
    _copy_packed_cache_rows[(capacity,)](cache, rows, local_slots, pack=True, **kwargs)
    gathered = group.all_gather(rows, dim=0)
    _copy_packed_cache_rows[(group.world_size * capacity,)](
        cache, gathered, slots.flatten(), pack=False, **kwargs
    )
