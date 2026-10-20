# Latest Transformers GDN Status

Updated: 2026-10-20 (UTC).

Branch: `ww43_PR_GDN`, based on
`f305603364e4a507d713d401cb3ff3a6019c6f4f`.

## Summary

Local GDN adaptation and its serving benchmarks are implemented. OLMo Hybrid
and Qwen3.8 text performance are essentially aligned with native vLLM.
Qwen3.5 retains approximately 3-4% batched decode overhead, and its image
first-token latency is not yet aligned. This is not a claim of complete
performance parity or real-checkpoint model quality.

This work was developed with AI assistance from GitHub Copilot. No PR has been
opened for this branch.

## Implementation and design boundary

- Keep the HF model, decoder, and mixer classes and their forward computation.
  Compose projection fusion with narrow convolution/delta-rule call rewrites;
  do not instantiate a native complete GDN mixer or call its forward/core.
- Retain HF gate/beta math, head expansion, output normalization and its epsilon,
  and output gating. Do not monkeypatch HF function globals.
- Reuse native convolution, recurrent/chunked delta-rule kernels, weight-loading
  helpers, TP sharding, cache management, and multimodal integration.
- Derive state geometry from each actual operator. Supported operators may be
  reordered or mixed without a model-family branch, fixed period, or combination
  whitelist. Preserve separate Q/K/V shards for packed QKV checkpoints under TP.
- Keep conservative projection input-stability checks. When fusion is rejected,
  log the reason and retain independent column-parallel projections rather than
  substituting an entire native mixer.
- Register convolution and scan operations as compilation splitting boundaries.
  This avoids incorrect compiled execution around their managed recurrent states.

The previous whole-mixer adapters were removed. This branch does not include
the separate Jamba/Mamba1, Zamba2/Mamba2, LFM2/ShortConv and pure-Mamba extension
commit `ced83ac76c967126bd0cefea96a424fcb5eb7e68`.

## Performance changes

1. Align automatic recurrent-cache precision with the HF text configuration's
   `mamba_ssm_dtype`; an explicit `--mamba-ssm-cache-dtype` takes precedence.
   Early inspection and runtime allocation use the same state specification.
2. Add optional output buffers to convolution and recurrent primitives and reuse
   the existing chunk output buffer. Write directly into the bridge output
   instead of allocating and copying an intermediate result. Existing callers
   keep their default behavior.
3. Fuse strided prefill Q/K normalization and Q/K/V packing. Retain the established
   normalization path for already-contiguous Q/K, where unconditional fusion
   measured slower. Preserve the primitive's epsilon and leave HF gates alone.

On the Qwen3.5 decode profile, output-buffer reuse removed 216 copy kernels over
six profiled steps. The prefill optimization removes three contiguous-copy
kernels and two normalization kernels in favor of one local preparation kernel
for the strided layout. Kernel tuning compared row tiles 8, 16, 32 and 64 using
cold-L2 CUPTI CUDA-graph timing; the strided path uses tile 8.

## Benchmark protocol

All primary comparisons used the same GPUs sequentially for native vLLM and
`model_impl=transformers`:

- NVIDIA L20; TP1 for `Qwen/Qwen3.5-0.8B` and `allenai/Olmo-Hybrid-7B`;
  TP2 on GPUs 0 and 1 for `Qwen/Qwen3.8-27B`.
- BF16 dummy weights, BF16 convolution cache, FP32 recurrent cache, seed 0,
  decoder `FLASH_ATTN`, and Triton GDN prefill.
- Decoder `FULL_AND_PIECEWISE` CUDA graphs, capture sizes 1/2/4/8/16/32,
  maximum model length 8192, maximum sequences 32, and GPU memory utilization 0.8.
- Prefix caching disabled. Fixed token-ID requests, temperature 0, ignored EOS,
  case warmup and a readiness request before each timed round.
- Five timed rounds per implementation/case. Report the median across rounds
  and retain all rounds, including slow ones.
- Shared compilation caches were not cleared. Startup measurements are not
  comparable cold-start measurements.

The text matrix covers 512/4096-token prefill with one output token at concurrency
1/8, and 128/2048-token prompts with 256 output tokens at concurrency 1/8/32.
Chunked cases use 4096 input tokens, 128 output tokens, a 512-token batch budget,
and concurrency 1/8.

Multimodal cases use a fixed generated 224x224 image or a four-frame 224x224 video
at 2 FPS, 128 output tokens, and concurrency 1/8. The multimodal processor cache
is disabled, and per-request media identities prevent encoder-result reuse.
Native/HF prompt and output lengths were checked for equality. The decoder
attention setting does not change HF vision's own attention implementation.

In total, 460 measured round files were validated, producing 80 paired TTFT/TPOT
metrics. Final benchmark logs contained no preemption, OOM or CUDA error reports.

## Results

Positive deltas mean HF is slower; negative deltas mean HF is faster.
TTFT is first-token latency; TPOT is time per output token.

### Text

| Model | TP | Decode TPOT delta, concurrency 8/32 | 4096-token prefill TTFT, concurrency 8 |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | 1 | +2.6% to +4.1% | 404.7 ms native / 404.9 ms HF |
| OLMo-Hybrid-7B | 1 | -1.0% to +0.5% | 3409.0 ms native / 2979.3 ms HF (-12.6%) |
| Qwen3.8-27B | 2 | -1.6% to +0.5% | 10065.4 ms native / 10156.1 ms HF (+0.9%) |

Unpinned Qwen3.5 long-context single-request decode was unstable: its primary
matrix medians were 3.048 ms native and 3.688 ms HF. A separate matched CPU-affinity
experiment on CPUs 32-39 measured 2.781 ms and 2.876 ms (+3.4%). Individual slow
rounds still occurred; affinity improved the median comparison but did not prove
that every source of host-side variation was resolved.

### Chunked prefill

Ranges below cover concurrency 1 and 8.

| Model | TTFT delta | TPOT delta |
| --- | --- | --- |
| Qwen3.5-0.8B | +0.8% to +3.9% | approximately +3.9% |
| OLMo-Hybrid-7B | -6.5% to -5.2% | -2.4% to -0.4% |
| Qwen3.8-27B | +0.3% to +0.4% | -0.5% to +1.1% |

### Multimodal

Ranges below cover concurrency 1 and 8.

| Model | Image TTFT delta | Video TTFT delta | Decode TPOT delta |
| --- | --- | --- | --- |
| Qwen3.5-0.8B | +11.5% to +21.0% | +4.6% to +5.8% | +2.8% to +4.1% |
| Qwen3.8-27B | -2.3% to +5.5% | -3.3% to +1.3% | -1.7% to +1.0% |

## Validation

- Targeted convolution/recurrent output-buffer regressions: **214 passed**.
- Final QKV preparation tests: **72 passed**, covering packed/expanded layouts,
  offset inputs, masked dimensions, FP32/BF16, zero and near-zero vectors, and
  exact preservation of V.
- Mixed-geometry/model/metadata selection: **17 passed**, including irregular
  Qwen/OLMo layouts, sliding/full attention, per-operator states, chunked prefill,
  decode, cache reuse and output-norm epsilon coverage.
- Final Qwen numerical rerun: **3 passed**, covering TP1 eager, TP2 eager and TP1
  compiled execution. These seeded tiny-checkpoint logprob checks do not
  substitute for real-checkpoint quality evaluation.
- Relevant pre-commit hooks, including type checks, and `git diff --check` passed.

Representative commands:

```bash
.venv/bin/python -m pytest tests/kernels/fla/test_fused_gdn_post_conv.py \
  -k strided_qkv_preparation -q
.venv/bin/python -m pytest tests/models/transformers/test_backend.py \
  -k qwen3_5_recurrent_cache_matches_transformers -q
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  benchmarks/kernels/benchmark_gdn_prefill_prepare.py \
  --tokens 1 512 4096 8192 --output /tmp/gdn-prefill-prepare.json
.venv/bin/pre-commit run
```

The standalone kernel benchmark requires the installed FlashInfer/CUPTI tooling.
The three-test numerical rerun overlaps the earlier mixed/model selection; the
counts above are separate runs, not a sum of unique tests.

## Remaining gaps and review notes

- Full performance parity is not achieved. Qwen3.5 retains approximately 3-4%
  batched decode overhead and 11.5-21.0% image TTFT overhead.
- HF vision uses per-sequence PyTorch SDPA, while native vision uses packed
  FlashAttention. Profiles and HF dispatch inspection confirm the difference;
  this is separate from the local GDN scan/convolution substitutions.
- An additional `compile_mm_encoder=true` experiment failed during startup with
  `GuardOnDataDependentSymNode` (`Eq(u7, 0)`). It is not enabled in the reported
  final results. No unsafe guard suppression, HF global monkeypatch or complete
  native block replacement was added to work around it.
- Speculative decoding and internal prefill checkpoint export are explicitly
  rejected. Prefix-cache reuse, PP, quantized GDN, non-CUDA platforms and other
  hardware have not been validated by this work.
- Real-checkpoint full-model quality and model evaluations remain unverified.
  Dummy-weight serving measurements establish timing/execution, not quality.
- The adapter is a substantive refactor. Shared-kernel output buffers are
  optional and compilation changes are limited to the two stateful boundaries;
  no scheduler or model-runner changes were introduced.
- Before an upstream PR, a human must review the changed lines, reproduce the
  relevant validation, perform upstream duplicate-work checks, and document
  missing quality/feature coverage. No upstream PR or review approval is implied.

## Local evidence

Raw logs, source snapshots, commands, traces and analysis are retained locally
under the following directory; they are not bundled into this commit:

```text
/root/.copilot/session-state/13a996fb-275a-4700-bee5-09d5ff05eb5b/files/
```

The main summaries are `gdn-perf-comparison.csv` and `gdn-perf-comparison.json`.
Exact fixed-token serving orchestration is in `benchmark_gdn_serving.py`.
Final data directories are `gdn-perf-final-native`, `gdn-perf-final-hf`,
`gdn-perf-qwen35-host-control`, `gdn-perf-chunked-final` and
`gdn-perf-multimodal-final`. The earlier `gdn-perf-auto-cache-pilot` used mismatched
recurrent precision and is excluded from the reported comparisons.
