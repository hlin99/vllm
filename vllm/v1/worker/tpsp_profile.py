"""Startup profiling for the BF16 TP projection / residual RMSNorm chain."""

import logging
import math
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from deep_symm.async_tp import (
    fused_matmul_reduce_scatter_norm_all_gather,
    fused_matmul_reduce_scatter_norm_route_all_gather,
)
from torch.distributed import distributed_c10d as c10d

import vllm._custom_ops as ops

_LOG = logging.getLogger(__name__)
_MODES = ("p2p", "ordered", "independent")
_EPS = 1e-5
_TRIALS = 5
_SCREEN_TRIALS = 5


@dataclass(frozen=True)
class SPMeasurement:
    tokens: int
    conventional_ms: float
    sp_ms: float
    lower_benefit_ms: float


@dataclass(frozen=True)
class SPProfile:
    tp_size: int
    hidden_size: int
    max_batched_tokens: int
    status: str
    reason: str
    threshold_tokens: int | None = None
    microchunk_tokens: int | None = None
    all_gather_mode: str | None = None
    pool_mb: int | None = None
    measurements: tuple[SPMeasurement, ...] = ()
    candidates: tuple[tuple[str, int, float], ...] = ()
    input_widths: tuple[int, ...] = ()
    norm_eps: float = _EPS
    gather_residual_after_native: bool = False
    finalists: tuple[tuple[str, int, float], ...] = ()
    gather_sharded_residual: bool = False
    mode_candidates: tuple[tuple[str, int, float], ...] = ()

    @property
    def enabled(self) -> bool:
        return self.status == "enabled"


def select_sp_config(profile: SPProfile, current_batched_tokens: int) -> bool:
    """Use the fixed startup choice only for batches within its measured range."""
    if not 1 <= current_batched_tokens <= profile.max_batched_tokens:
        raise ValueError("current_batched_tokens must be within the profiled range")
    return (
        profile.enabled
        and profile.threshold_tokens is not None
        and current_batched_tokens >= profile.threshold_tokens
    )


def _token_sizes(max_tokens: int, tp_size: int) -> list[int]:
    sizes = {1, max_tokens}
    sizes.update(
        n
        for n in (tp_size + 1, 16, 128, 512, 2048, 8192, 32768, max_tokens - 1)
        if 1 <= n <= max_tokens
    )
    return sorted(sizes)


def _threshold(measurements: list[SPMeasurement]) -> tuple[str, int | None, str]:
    winning = [
        m.lower_benefit_ms > max(0.05, 0.02 * m.conventional_ms) for m in measurements
    ]
    if not winning[-1]:
        return "disabled", None, "no reliable benefit at max_batched_tokens"
    first = next(i for i in range(len(winning)) if all(winning[i:]))
    if any(winning[:first]):
        return "inconclusive", None, "observed performance reversal"
    return "enabled", measurements[first].tokens, ""


def _screen_score(samples: list[float]) -> float:
    return sorted(samples)[1]


def _search_chunks(
    shard_rows: int, score: Callable[[int], float | None]
) -> list[int] | None:
    chunks = []
    best = float("inf")
    without_improvement = 0
    for chunk in range(64, math.ceil(shard_rows / 64) * 64 + 1, 64):
        value = score(chunk)
        if value is None:
            return None
        chunks.append(chunk)
        if value < best:
            best = value
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement == 32:
            break
    return chunks


def _top_chunks(
    chunks: list[int], scores: dict[tuple[str, int], float], mode: str
) -> list[int]:
    return sorted(chunks, key=lambda chunk: scores[mode, chunk])[:2]


def profile_sp_config(
    tp_size: int,
    hidden_size: int,
    max_batched_tokens: int,
    group_name: str,
    time_budget_s: float,
    input_widths: tuple[int, ...] | None = None,
    norm_eps: float = _EPS,
    gather_residual_after_native: bool = False,
    gather_sharded_residual: bool = False,
    check_rtol: float = 0.01,
    check_atol: float = 0.02,
    router_num_experts: int | None = None,
    route_top_k: int | None = None,
    route_renormalize: bool = True,
) -> SPProfile:
    """Profile the projection chain, including routing when requested."""
    if time_budget_s <= 0:
        raise ValueError("time_budget_s must be positive")

    def unsupported(reason: str) -> SPProfile:
        return SPProfile(
            tp_size, hidden_size, max_batched_tokens, "unsupported", reason
        )

    if (
        not 2 <= tp_size <= 8
        or hidden_size <= 0
        or hidden_size % tp_size
        or max_batched_tokens < 1
    ):
        return unsupported(
            "requires TP in [2, 8], divisible hidden size and positive "
            "max_batched_tokens"
        )
    if input_widths is None:
        input_widths = (hidden_size // tp_size,)
    if not input_widths or any(width <= 0 for width in input_widths):
        raise ValueError(
            "input_widths must contain positive per-rank projection widths"
        )
    if norm_eps <= 0:
        raise ValueError("norm_eps must be positive")
    if check_rtol < 0 or check_atol < 0:
        raise ValueError("numerical check tolerances must be nonnegative")
    if (router_num_experts is None) != (route_top_k is None):
        raise ValueError("Routing requires both router_num_experts and route_top_k")
    if router_num_experts is not None:
        assert route_top_k is not None
        if not 1 <= route_top_k <= min(32, router_num_experts):
            raise ValueError("route_top_k must be in [1, min(32, router_num_experts)]")
    if not torch.xpu.is_available():
        return unsupported("requires XPU")
    group = c10d._resolve_process_group(group_name)
    if dist.get_world_size(group) != tp_size:
        raise ValueError("group_name and tp_size disagree")
    import vllm_xpu_kernels._C  # noqa: F401

    if not torch._C._dispatch_has_kernel_for_dispatch_key(
        "_C::fused_add_rms_norm", "XPU"
    ):
        raise RuntimeError("vLLM XPU fused_add_rms_norm is unavailable")

    device = torch.device("xpu", torch.xpu.current_device())
    rank = dist.get_rank(group)
    shard_rows = math.ceil(max_batched_tokens / tp_size)
    output_mb = math.ceil(max_batched_tokens * hidden_size * 2 / 2**20)
    minimum_pool_mb = math.ceil((2 * output_mb + 256) / 512) * 512
    configured_pool = os.environ.setdefault(
        "ASYNC_TP_OUTPUT_POOL_MB", str(minimum_pool_mb)
    )
    pool_mb = int(configured_pool)
    if pool_mb < minimum_pool_mb:
        raise RuntimeError(
            f"ASYNC_TP_OUTPUT_POOL_MB={pool_mb} is below the profile minimum "
            f"{minimum_pool_mb}"
        )
    deadline = time.monotonic() + time_budget_s

    def expired() -> bool:
        flag = torch.tensor(
            [time.monotonic() >= deadline], dtype=torch.int32, device=device
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
        return bool(flag.item())

    def inputs(tokens: int):
        projections = []
        for width in input_widths:
            generator = torch.Generator(device=device).manual_seed(1831 + rank)
            a = torch.randn(
                tokens, width, dtype=torch.bfloat16, device=device, generator=generator
            )
            b = (
                torch.randn(
                    width,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                / 8
            )
            projections.append((a, b, b.T.contiguous()))
        weight = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        residual = torch.randn(
            tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=torch.Generator(device=device).manual_seed(407),
        )
        rows = math.ceil(tokens / tp_size)
        padded = torch.zeros(
            tp_size * rows, hidden_size, dtype=torch.bfloat16, device=device
        )
        padded[:tokens].copy_(residual)
        router_weight = None
        if router_num_experts is not None:
            router_weight = torch.randn(
                router_num_experts,
                hidden_size,
                device=device,
                dtype=torch.bfloat16,
                generator=torch.Generator(device=device).manual_seed(291),
            ) / math.sqrt(hidden_size)
        return (
            projections,
            weight,
            residual,
            padded.narrow(0, rank * rows, rows).contiguous(),
            router_weight,
        )

    def route_reference(normalized, router_weight):
        assert route_top_k is not None
        logits = torch.mm(normalized, router_weight.T, out_dtype=torch.float32)
        route_weights = torch.empty(
            (normalized.size(0), route_top_k), dtype=torch.float32, device=device
        )
        route_ids = torch.empty(
            (normalized.size(0), route_top_k), dtype=torch.int32, device=device
        )
        token_expert_indices = torch.empty_like(route_ids)
        ops.topk_softmax(
            route_weights,
            route_ids,
            token_expert_indices,
            logits,
            route_renormalize,
        )
        return route_weights, route_ids

    def run(data, candidate):
        projections, weight, residual, local_residual, router_weight = data
        outputs = []

        for index, (a, b, linear_weight) in enumerate(projections):
            if candidate is None:
                partial = F.linear(a, linear_weight)
                full = partial.clone()
                dist.all_reduce(full, op=dist.ReduceOp.SUM, group=group)
                if index or gather_sharded_residual:
                    rows = local_residual.size(0)
                    gathered_residual = torch.empty(
                        (tp_size * rows, hidden_size), device=device, dtype=a.dtype
                    )
                    dist.all_gather_into_tensor(
                        gathered_residual, local_residual, group=group
                    )
                    full_residual = gathered_residual[: a.size(0)].contiguous()
                else:
                    full_residual = residual.clone()
                torch.ops._C.fused_add_rms_norm(full, full_residual, weight, norm_eps)
                outputs.append(full)
                if router_weight is not None:
                    outputs.extend(route_reference(full, router_weight))
            else:
                if router_weight is not None:
                    rows = local_residual.size(0)
                    start = rank * rows
                    runtime_residual = torch.zeros_like(local_residual)
                    count = min(rows, max(0, a.size(0) - start))
                    if count:
                        runtime_residual[:count] = residual[start : start + count]
                else:
                    runtime_residual = local_residual
                if router_weight is None:
                    reduced, _, normalized = (
                        fused_matmul_reduce_scatter_norm_all_gather(
                            a,
                            b,
                            weight,
                            None,
                            group_name,
                            eps=norm_eps,
                            norm_type="rms_norm",
                            residual=runtime_residual,
                            microchunk_tokens=candidate[1],
                        )
                    )
                else:
                    reduced, _, normalized, route_weights, route_ids = (
                        fused_matmul_reduce_scatter_norm_route_all_gather(
                            a,
                            b,
                            weight,
                            None,
                            router_weight,
                            route_top_k,
                            group_name,
                            renormalize=route_renormalize,
                            eps=norm_eps,
                            norm_type="rms_norm",
                            residual=runtime_residual,
                            microchunk_tokens=candidate[1],
                        )
                    )
                if gather_residual_after_native:
                    torch.xpu.synchronize()
                    gathered_residual = torch.empty(
                        (tp_size * reduced.size(0), hidden_size),
                        device=device,
                        dtype=a.dtype,
                    )
                    dist.all_gather_into_tensor(gathered_residual, reduced, group=group)
                outputs.append(normalized)
                if router_weight is not None:
                    outputs.extend((route_weights, route_ids))
        return tuple(outputs)

    def measure(data, candidate) -> float:
        dist.barrier(group=group)
        torch.xpu.synchronize()
        start = time.perf_counter()
        result = run(data, candidate)
        torch.xpu.synchronize()
        elapsed = torch.tensor(
            [(time.perf_counter() - start) * 1000], dtype=torch.float64, device=device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
        value = elapsed.item()
        del result
        return value

    def inconclusive(reason: str, measurements=(), candidate_results=()) -> SPProfile:
        if rank == 0:
            _LOG.warning("TPSP startup profile: status=inconclusive reason=%s", reason)
        return SPProfile(
            tp_size,
            hidden_size,
            max_batched_tokens,
            "inconclusive",
            reason,
            pool_mb=pool_mb,
            measurements=tuple(measurements),
            candidates=tuple(candidate_results),
            input_widths=input_widths,
            norm_eps=norm_eps,
            gather_residual_after_native=gather_residual_after_native,
            gather_sharded_residual=gather_sharded_residual,
        )

    data = inputs(max_batched_tokens)
    samples: dict[tuple[str, int], list[float]] = {}

    def screen_chunk(
        chunk: int,
        results: dict[tuple[str, int], list[float]],
        modes: tuple[str, ...],
    ) -> float | None:
        for mode in modes:
            candidate = (mode, chunk)
            results[candidate] = []
            os.environ["ASYNC_TP_ALL_GATHER_MODE"] = mode
            measure(data, candidate)
            if expired():
                return None
        for trial in range(_SCREEN_TRIALS):
            offset = trial % len(modes)
            for mode in modes[offset:] + modes[:offset]:
                candidate = (mode, chunk)
                os.environ["ASYNC_TP_ALL_GATHER_MODE"] = mode
                results[candidate].append(measure(data, candidate))
                if expired():
                    return None
        score = torch.tensor(
            [min(_screen_score(results[mode, chunk]) for mode in modes)],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(score, op=dist.ReduceOp.MAX, group=group)
        return float(score.item())

    representative = max(64, math.ceil(min(4096, shard_rows) / 64) * 64)
    mode_samples: dict[tuple[str, int], list[float]] = {}
    if screen_chunk(representative, mode_samples, _MODES) is None:
        return inconclusive("communication mode time budget exceeded")
    mode_scores = torch.tensor(
        [_screen_score(mode_samples[mode, representative]) for mode in _MODES],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(mode_scores, op=dist.ReduceOp.MAX, group=group)
    mode_results = tuple(
        (mode, representative, float(score))
        for mode, score in zip(_MODES, mode_scores.tolist())
    )
    selected_mode = min(mode_results, key=lambda item: item[2])[0]
    selected_modes = (selected_mode,)
    chunks = _search_chunks(
        shard_rows,
        lambda chunk: screen_chunk(chunk, samples, selected_modes),
    )
    if chunks is None:
        return inconclusive("screening time budget exceeded")
    candidates = [(selected_mode, chunk) for chunk in chunks]
    scores = torch.tensor(
        [_screen_score(samples[item]) for item in candidates],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(scores, op=dist.ReduceOp.MAX, group=group)
    candidate_results = tuple(
        (mode, chunk, float(score))
        for (mode, chunk), score in zip(candidates, scores.tolist())
    )
    screened_scores = {(mode, chunk): score for mode, chunk, score in candidate_results}
    finalists = _top_chunks(chunks, screened_scores, selected_mode)
    retested: dict[tuple[str, int], list[float]] = {}
    for chunk in finalists:
        if screen_chunk(chunk, retested, selected_modes) is None:
            return inconclusive(
                "finalist time budget exceeded", candidate_results=candidate_results
            )
    finalist_candidates = [(selected_mode, chunk) for chunk in finalists]
    final_scores = torch.tensor(
        [_screen_score(retested[item]) for item in finalist_candidates],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(final_scores, op=dist.ReduceOp.MAX, group=group)
    finalist_results = tuple(
        (mode, chunk, float(score))
        for (mode, chunk), score in zip(finalist_candidates, final_scores.tolist())
    )
    candidate = min(finalist_results, key=lambda item: item[2])[:2]
    os.environ["ASYNC_TP_ALL_GATHER_MODE"] = candidate[0]
    data = None

    check_data = inputs(min(max_batched_tokens, tp_size * 5 + 1))
    conventional = run(check_data, None)
    native = run(check_data, candidate)
    torch.xpu.synchronize()
    if router_num_experts is not None:
        router_weight = check_data[-1]
        assert router_weight is not None

    for index, (actual, expected) in enumerate(zip(native, conventional)):
        if router_num_experts is not None and index % 3 in (1, 2):
            expected = route_reference(native[index - index % 3], router_weight)[
                index % 3 - 1
            ]
        if actual.dtype in (torch.int32, torch.int64):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            torch.testing.assert_close(
                actual, expected, rtol=check_rtol, atol=check_atol
            )
    del check_data, conventional, native

    def measure_size(tokens: int) -> SPMeasurement | None:
        data = inputs(tokens)
        measure(data, None)
        measure(data, candidate)
        paired: list[float] = []
        conventional: list[float] = []
        sp: list[float] = []
        for trial in range(_TRIALS):
            order = (None, candidate) if trial % 2 == 0 else (candidate, None)
            times = {item: measure(data, item) for item in order}
            conventional.append(times[None])
            sp.append(times[candidate])
            paired.append(times[None] - times[candidate])
            if expired():
                return None
        lower = statistics.mean(paired) - 2.776 * statistics.stdev(paired) / math.sqrt(
            _TRIALS
        )
        return SPMeasurement(
            tokens, statistics.median(conventional), statistics.median(sp), lower
        )

    measurements: list[SPMeasurement] = []
    for tokens in _token_sizes(max_batched_tokens, tp_size):
        result = measure_size(tokens)
        if result is None:
            return inconclusive(
                "measurement time budget exceeded", measurements, candidate_results
            )
        measurements.append(result)

    for _ in range(2):
        status, threshold, _ = _threshold(measurements)
        if status != "enabled":
            break
        winner = next(
            i for i, item in enumerate(measurements) if item.tokens == threshold
        )
        if winner == 0 or threshold - measurements[winner - 1].tokens <= max(
            tp_size, 8
        ):
            break
        midpoint = (threshold + measurements[winner - 1].tokens) // 2
        result = measure_size(midpoint)
        if result is None:
            return inconclusive(
                "crossover time budget exceeded", measurements, candidate_results
            )
        measurements.insert(winner, result)

    status, threshold, reason = _threshold(measurements)
    profile = SPProfile(
        tp_size,
        hidden_size,
        max_batched_tokens,
        status,
        reason,
        threshold,
        candidate[1] if status == "enabled" else None,
        candidate[0] if status == "enabled" else None,
        pool_mb,
        tuple(measurements),
        candidate_results,
        input_widths,
        norm_eps,
        gather_residual_after_native,
        finalist_results,
        gather_sharded_residual,
        mode_results,
    )
    if rank == 0:
        _LOG.warning(
            "TPSP startup profile: status=%s threshold=%s chunk=%s mode=%s "
            "widths=%s norm_eps=%s gather_residual=%s "
            "sharded_residual=%s pool_mb=%s "
            "mode_candidates=%s candidates=%s finalists=%s measurements=%s reason=%s",
            status,
            threshold,
            profile.microchunk_tokens,
            profile.all_gather_mode,
            input_widths,
            norm_eps,
            gather_residual_after_native,
            gather_sharded_residual,
            pool_mb,
            mode_results,
            candidate_results,
            finalist_results,
            measurements,
            reason,
        )
    return profile
