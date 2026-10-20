# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare strided GDN QKV preparation using cold-L2 CUPTI graph timing."""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from cupti import cupti  # noqa: F401
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
    _fused_qkv_l2norm_kernel,
)
from vllm.third_party.flash_linear_attention.ops.l2norm import (
    l2norm_fwd,
    l2norm_fwd_kernel2,
)
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens", nargs="+", type=int, default=[1, 32, 512, 4096, 8192]
    )
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert int(version("cupti-python").split(".")[0]) >= 13
    torch.manual_seed(0)
    rows = []
    for tokens in args.tokens:
        sizes = [args.key_heads * args.key_dim] * 2 + [
            args.value_heads * args.value_dim
        ]
        packed = torch.randn(1, tokens, sum(sizes), device="cuda", dtype=torch.bfloat16)
        q, k, v = packed.split(sizes, dim=-1)
        q = q.view(1, tokens, args.key_heads, args.key_dim)
        k = k.view_as(q)
        v = v.view(1, tokens, args.value_heads, args.value_dim)
        assert args.value_heads % args.key_heads == 0
        ratio = args.value_heads // args.key_heads
        if ratio != 1:
            q = q.repeat_interleave(ratio, dim=2)
            k = k.repeat_interleave(ratio, dim=2)
        reference = (l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous()), v)
        packed_inputs = [
            x
            if x.is_contiguous()
            else torch.empty(x.shape, device=x.device, dtype=x.dtype)
            for x in (q, k, v)
        ]
        outputs = [
            torch.empty(x.shape, device=x.device, dtype=x.dtype) for x in (q, k, v)
        ]
        cases = [
            (method, tile)
            for method in ("unfused", "fused")
            for tile in (8, 16, 32, 64)
        ]
        random.Random(0).shuffle(cases)
        for method, tile in cases:

            def run(
                method=method,
                tile=tile,
                tokens=tokens,
                q=q,
                k=k,
                v=v,
                outputs=outputs,
                packed_inputs=packed_inputs,
            ):
                if method == "fused":
                    return _fused_qkv_l2norm_kernel[
                        (triton.cdiv(tokens * args.value_heads, tile),)
                    ](
                        q,
                        k,
                        v,
                        *outputs,
                        tokens,
                        q.stride()[1:],
                        k.stride()[1:],
                        v.stride()[1:],
                        args.value_heads,
                        args.value_heads,
                        args.key_dim,
                        args.value_dim,
                        tile,
                        triton.next_power_of_2(max(args.key_dim, args.value_dim)),
                        num_warps=4,
                    )
                for source, dest in zip((q, k, v), packed_inputs):
                    if source is not dest:
                        dest.copy_(source)
                for source, dest in zip(packed_inputs[:2], outputs[:2]):
                    l2norm_fwd_kernel2[(triton.cdiv(tokens * args.value_heads, tile),)](
                        source,
                        dest,
                        1e-6,
                        tokens * args.value_heads,
                        args.key_dim,
                        triton.next_power_of_2(args.key_dim),
                        tile,
                        num_warps=4,
                    )

            kernel = run()
            actual = (
                *outputs[:2],
                outputs[2] if method == "fused" else packed_inputs[2],
            )
            for observed, expected in zip(actual, reference):
                torch.testing.assert_close(observed, expected, atol=1e-3, rtol=1e-3)
            if method == "fused" and tile == 8:
                args.output.with_suffix(f".{tokens}.ptx").write_text(kernel.asm["ptx"])
            for _ in range(25):
                run()
            torch.accelerator.synchronize()
            samples = bench_gpu_time_with_cupti(
                run,
                use_cuda_graph=True,
                cold_l2_cache=True,
                dry_run_iters=10,
                repeat_iters=25,
            )
            us = statistics.median(samples) * 1000
            useful_bytes = (
                2
                * (q.numel() + k.numel() + (v.numel() if not v.is_contiguous() else 0))
                * q.element_size()
            )
            row = dict(
                method=method,
                tokens=tokens,
                tile=tile,
                us=us,
                effective_gbps=useful_bytes / (us * 1000),
                samples_ms=samples,
            )
            rows.append(row)
            print(
                method,
                tokens,
                tile,
                round(us, 3),
                round(row["effective_gbps"], 2),
                flush=True,
            )
    args.output.write_text(
        json.dumps(
            {
                "gpu": current_platform.get_device_name(),
                "torch": torch.__version__,
                "triton": triton.__version__,
                "cupti": version("cupti-python"),
                "command": sys.argv,
                "head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "branch": subprocess.check_output(
                    ["git", "branch", "--show-current"], text=True
                ).strip(),
                "env": {
                    key: os.environ.get(key)
                    for key in (
                        "CUDA_VISIBLE_DEVICES",
                        "TRITON_CACHE_DIR",
                        "OMP_NUM_THREADS",
                    )
                },
                "kernel_source_sha256": hashlib.sha256(
                    Path(_fused_qkv_l2norm_kernel.fn.__code__.co_filename).read_bytes()
                ).hexdigest(),
                "args": {**vars(args), "output": str(args.output)},
                "timing": "CUPTI, CUDA graph, cold L2, no allocation",
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
