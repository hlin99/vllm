# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from weakref import WeakKeyDictionary

import torch
import torch.distributed._symmetric_memory as symm_mem

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.models.deepseek_v4.common.ops.cache_utils import quantize_and_insert_k_cache
from vllm.models.deepseek_v4.xpu.xpu_qnorm_rope_kv_fp8_insert import xpu_qnorm_rope
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.xpu_mla_sparse import triton_paged_mla_sparse_interface


@triton.jit
def _mark_slot_chunks(
    slots,
    owners,
    count: tl.constexpr,
    local_tokens: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(slots + row, mask=row < count, other=-1)
    chunk = row % local_tokens // chunk_size + 1
    tl.store(owners + slot, chunk, mask=(row < count) & (slot >= 0))


@triton.jit
def _partition_swa_indices(
    indices,
    lengths,
    owners,
    sorted_indices,
    offsets,
    chunk_lengths,
    num_queries: tl.constexpr,
    width: tl.constexpr,
    stride_indices: tl.constexpr,
    capacity: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query = tl.program_id(0)
    column = tl.arange(0, BLOCK)
    length = tl.minimum(tl.maximum(tl.load(lengths + query), 0), width)
    slots = tl.load(
        indices + query * stride_indices + column,
        mask=column < length,
        other=-1,
    )
    valid = (column < length) & (slots >= 0) & (slots < capacity)
    owner = tl.load(owners + slots, mask=valid, other=num_chunks + 1)
    key = tl.where(valid, owner, num_chunks + 1) * BLOCK + column
    sorted_key = tl.sort(key, descending=False)
    sorted_slots = tl.gather(slots, sorted_key % BLOCK, axis=0)
    tl.store(
        sorted_indices + query * width + column,
        sorted_slots,
        mask=column < width,
    )
    for chunk in tl.static_range(num_chunks + 1):
        begin = tl.sum((valid & (owner < chunk)).to(tl.int32), 0)
        size = tl.sum((valid & (owner == chunk)).to(tl.int32), 0)
        tl.store(offsets + chunk * num_queries + query, begin)
        tl.store(chunk_lengths + chunk * num_queries + query, size)


class PCPSWAPipeline:
    """Shared direct-output IPC arena and chunked SWA attention state."""

    def __init__(self, group: GroupCoordinator, capacity: int, chunk_size: int) -> None:
        self.capacity = capacity
        self.chunk_size = chunk_size
        self.rank = group.rank_in_group
        self.world_size = group.world_size
        self.arena = symm_mem.empty(
            self.world_size, capacity, 512, device=group.device, dtype=torch.bfloat16
        )
        self.handle = symm_mem.rendezvous(self.arena, group.device_group)
        self.peers = [
            self.handle.get_buffer(peer, self.arena.shape, self.arena.dtype)
            for peer in range(self.world_size)
        ]
        self.communication = torch.xpu.Stream(device=group.device)
        self.ready = [
            torch.xpu.Event() for _ in range(triton.cdiv(capacity, chunk_size))
        ]
        self.state_buffer: torch.Tensor | None = None

    def reserve_state(self, tokens: int, heads: int) -> torch.Tensor:
        size = tokens * heads * 514
        if self.state_buffer is None or self.state_buffer.numel() < size:
            self.state_buffer = torch.empty(
                size, device=self.arena.device, dtype=torch.float32
            )
        return self.state_buffer[:size].view(tokens, heads, 514)

    def prefill(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        cos_sin: torch.Tensor,
        eps: float,
        cache: torch.Tensor,
        slots: torch.Tensor,
        prefill_start: int,
        num_queries: int,
        swa_indices: torch.Tensor | None,
        swa_lens: torch.Tensor | None,
        scale: float,
        block_size: int,
    ) -> torch.Tensor:
        """Publish directly generated KV and consume each ready cache chunk."""
        num_tokens = q.shape[0]
        assert 0 < num_tokens <= self.capacity
        assert slots.numel() == self.world_size * num_tokens
        assert q.dtype == torch.bfloat16 and q.device == self.arena.device
        xpu_qnorm_rope(
            q,
            kv,
            positions,
            cos_sin,
            eps,
            kv_out=self.arena[self.rank, :num_tokens],
        )
        query = q[prefill_start : prefill_start + num_queries]
        state = self.reserve_state(num_queries, q.shape[1])
        num_chunks = triton.cdiv(num_tokens, self.chunk_size)
        plan = None
        if num_queries:
            assert swa_indices is not None and swa_lens is not None
            capacity = cache.shape[0] * block_size
            owners = torch.zeros(capacity, device=q.device, dtype=torch.int32)
            _mark_slot_chunks[(triton.cdiv(slots.numel(), 256),)](
                slots, owners, slots.numel(), num_tokens, self.chunk_size, BLOCK=256
            )
            width = swa_indices.shape[-1]
            sorted_indices = torch.empty(
                (num_queries, 1, width), device=q.device, dtype=torch.int32
            )
            offsets = torch.empty(
                (num_chunks + 1, num_queries), device=q.device, dtype=torch.int32
            )
            lengths = torch.empty_like(offsets)
            _partition_swa_indices[(num_queries,)](
                swa_indices,
                swa_lens,
                owners,
                sorted_indices,
                offsets,
                lengths,
                num_queries,
                width,
                swa_indices.stride(0),
                capacity,
                num_chunks,
                BLOCK=triton.next_power_of_2(width),
            )
            plan = sorted_indices, offsets, lengths

        def consume(chunk: int, *, initialize: bool = False) -> None:
            if plan is not None:
                indices, offsets, lengths = plan
                triton_paged_mla_sparse_interface(
                    query,
                    None,
                    cache,
                    None,
                    None,
                    indices,
                    lengths[chunk],
                    scale,
                    None,
                    partial_state=state,
                    initialize=initialize,
                    finalize=False,
                    swa_offsets=offsets[chunk],
                )

        compute = torch.xpu.current_stream(q.device)
        self.handle.barrier(channel=0)
        self.communication.wait_stream(compute)
        consume(0, initialize=True)
        for chunk, start in enumerate(range(0, num_tokens, self.chunk_size)):
            end = min(start + self.chunk_size, num_tokens)
            with torch.xpu.stream(self.communication):
                for step in range(1, self.world_size):
                    peer = (self.rank + step) % self.world_size
                    self.arena[peer, start:end].copy_(
                        self.peers[peer][peer, start:end], non_blocking=True
                    )
                self.ready[chunk].record()
            compute.wait_event(self.ready[chunk])
            for peer in range(self.world_size):
                quantize_and_insert_k_cache(
                    self.arena[peer, start:end],
                    cache.view(cache.shape[0], -1),
                    slots[peer * num_tokens + start : peer * num_tokens + end],
                    block_size=block_size,
                )
            consume(chunk + 1)
        # All peers must finish reading this generation before the arena is reused.
        with torch.xpu.stream(self.communication):
            self.handle.barrier(channel=1)
        compute.wait_stream(self.communication)
        return state


_workspaces: WeakKeyDictionary[GroupCoordinator, PCPSWAPipeline] = WeakKeyDictionary()


def get_pcp_swa_pipeline(
    group: GroupCoordinator, capacity: int, chunk_size: int
) -> PCPSWAPipeline:
    pipeline = _workspaces.get(group)
    if pipeline is None:
        pipeline = PCPSWAPipeline(group, capacity, chunk_size)
        _workspaces[group] = pipeline
    assert pipeline.capacity == capacity and pipeline.chunk_size == chunk_size
    return pipeline
