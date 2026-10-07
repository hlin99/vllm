"""Compare Qwen3-Coder TP4 projection chains at a fixed 1536-token chunk."""

import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import vllm_xpu_kernels._C  # noqa: F401
from deep_symm.async_tp import fused_matmul_reduce_scatter_norm_all_gather

_HIDDEN = 6144
_WIDTH = 3072
_TOKENS = 32768
_CHUNK = 1536
_TRIALS = 12


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", device_id=torch.device("xpu", rank))
    try:
        group = dist.group.WORLD
        assert dist.get_world_size(group) == 4
        os.environ["ASYNC_TP_ALL_GATHER_MODE"] = "p2p"
        os.environ.setdefault("ASYNC_TP_OUTPUT_POOL_MB", "1024")
        device = torch.device("xpu", rank)
        norm_weight = torch.ones(_HIDDEN, device=device, dtype=torch.bfloat16)

        def inputs(tokens: int):
            generator = torch.Generator(device=device).manual_seed(1831 + rank)
            x = torch.randn(
                tokens,
                _WIDTH,
                device=device,
                dtype=torch.bfloat16,
                generator=generator,
            )
            weight = (
                torch.randn(
                    _WIDTH,
                    _HIDDEN,
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                / 8
            )
            linear_weight = weight.T.contiguous()
            residual = torch.randn(
                tokens,
                _HIDDEN,
                device=device,
                dtype=torch.bfloat16,
                generator=torch.Generator(device=device).manual_seed(407),
            )
            return x, weight, linear_weight, residual

        def run(data, sp: bool):
            x, weight, linear_weight, residual = data
            if sp:
                rows = (x.size(0) + 3) // 4
                start = rank * rows
                local_residual = torch.zeros(
                    (rows, _HIDDEN), device=device, dtype=torch.bfloat16
                )
                count = min(rows, max(0, x.size(0) - start))
                if count:
                    local_residual[:count] = residual[start : start + count]
                reduced, _, normalized = fused_matmul_reduce_scatter_norm_all_gather(
                    x,
                    weight,
                    norm_weight,
                    None,
                    group.group_name,
                    eps=1e-6,
                    norm_type="rms_norm",
                    residual=local_residual,
                    microchunk_tokens=_CHUNK,
                )
                torch.xpu.synchronize()
                gathered = torch.empty(
                    (4 * reduced.size(0), _HIDDEN),
                    device=device,
                    dtype=torch.bfloat16,
                )
                dist.all_gather_into_tensor(gathered, reduced, group=group)
                return normalized, gathered[: x.size(0)].contiguous()
            full = F.linear(x, linear_weight)
            dist.all_reduce(full, group=group)
            new_residual = residual.clone()
            torch.ops._C.fused_add_rms_norm(full, new_residual, norm_weight, 1e-6)
            return full, new_residual

        for tokens in (21, _TOKENS):
            data = inputs(tokens)
            conventional = run(data, False)
            fused = run(data, True)
            torch.testing.assert_close(fused[0], conventional[0], rtol=0.02, atol=0.05)
            # BF16 reduction order affects the pre-norm residual more strongly.
            torch.testing.assert_close(fused[1], conventional[1], rtol=0.02, atol=0.25)
            if rank == 0:
                norm_max = (
                    (fused[0].float() - conventional[0].float()).abs().max().item()
                )
                residual_max = (
                    (fused[1].float() - conventional[1].float()).abs().max().item()
                )
                print(
                    f"QWEN3_CODER_CHECK tokens={tokens} "
                    f"normalized_max_abs={norm_max:.6f} "
                    f"residual_max_abs={residual_max:.6f}",
                    flush=True,
                )
            del data, conventional, fused
        data = inputs(_TOKENS)

        def measure(sp: bool) -> float:
            dist.barrier(group=group)
            torch.xpu.synchronize()
            start = time.perf_counter()
            output = run(data, sp)
            torch.xpu.synchronize()
            elapsed = torch.tensor(
                [(time.perf_counter() - start) * 1000],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=group)
            del output
            return elapsed.item()

        for _ in range(3):
            measure(False)
            measure(True)
        conventional_ms = []
        fused_ms = []
        for trial in range(_TRIALS):
            order = (False, True) if trial % 2 == 0 else (True, False)
            times = {mode: measure(mode) for mode in order}
            conventional_ms.append(times[False])
            fused_ms.append(times[True])
        benefits = [a - b for a, b in zip(conventional_ms, fused_ms)]
        lower = (
            statistics.mean(benefits)
            - 2.201 * statistics.stdev(benefits) / _TRIALS**0.5
        )
        native = statistics.median(conventional_ms)
        tpsp = statistics.median(fused_ms)
        if rank == 0:
            print(
                f"QWEN3_CODER_FIXED_CHUNK tokens={_TOKENS} chunk={_CHUNK} "
                f"native_ms={native:.3f} tpsp_ms={tpsp:.3f} "
                f"lower_benefit_ms={lower:.3f} "
                f"improvement_pct={(native - tpsp) / native * 100:.2f}",
                flush=True,
            )
        assert lower > max(0.05, 0.02 * native), (
            f"fixed-chunk TPSP has no reliable benefit: lower={lower:.3f} ms"
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
