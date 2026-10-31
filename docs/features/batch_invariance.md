# Batch Invariance

!!! note
    Batch invariance is currently in beta. Some features are still under active development.
    Track progress and planned improvements at <https://github.com/vllm-project/vllm/issues/27433>

This document shows how to enable batch invariance in vLLM. Batch invariance ensures that the output of a model is deterministic and independent of the batch size or the order of requests in a batch.

## Motivation

Batch invariance is crucial for several use cases:

- **Framework debugging**: Deterministic outputs make it easier to debug issues in the inference framework, as the same input will always produce the same output regardless of batching.
- **Model debugging**: Helps identify issues in model implementations by ensuring consistent behavior across different batch configurations.
- **Reinforcement Learning (RL)**: RL training often requires deterministic rollouts for reproducibility and stable training.
- **Large-scale inference systems**: Systems that use vLLM as a component benefit from deterministic behavior for testing, validation, and consistency guarantees.

## Hardware Requirements

Batch invariance is supported on the following platforms:

- NVIDIA GPUs with compute capability 8.0 or higher.
- AMD Instinct GPUs (gfx90a, gfx942, gfx950).
- Intel XPUs with Triton support.

### Attention Backend Selection for XPU

On XPU, Triton Attention backend is required for batch invariance.
Select this backend using Qwen/Qwen3-1.7B model as an example:

```python
llm = LLM(
    model="Qwen/Qwen3-1.7B",
    attention_config={"backend": "TRITON_ATTN"},
)
```

Or via the CLI:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve Qwen/Qwen3-1.7B \
    --attention-config.backend TRITON_ATTN
```

## Enabling Batch Invariance

Batch invariance can be enabled by setting the `VLLM_BATCH_INVARIANT` environment variable to `1`:

```bash
export VLLM_BATCH_INVARIANT=1
```

### Online Inference (Server Mode)

To start a vLLM server with batch invariance enabled:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

Then use the OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

# These requests will produce deterministic outputs
# regardless of batch size or order
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.7,
    seed=42,
)

print(response.choices[0].text)
```

### Offline Inference

For offline batch inference with batch invariance:

```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

prompts = [
    "The future of AI is",
    "Machine learning enables",
    "Deep learning models can",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100,
    seed=42,
)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
)

# Outputs will be deterministic regardless of batch size
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}")
    print(f"Generated: {generated_text!r}\n")
```

## Tested Models

Batch invariance has been tested and verified on the following models:

- **DeepSeek series**: `deepseek-ai/DeepSeek-V3`, `deepseek-ai/DeepSeek-V3-0324`, `deepseek-ai/DeepSeek-R1`, `deepseek-ai/DeepSeek-V3.1`
- **Qwen3 (Dense)**: `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen3-8B-AWQ`
- **Qwen3-VL (Vision-Language)**: `Qwen/Qwen3-VL-2B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct` (single image and video inputs)
- **Qwen3 (MoE)**: `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Next-80B-A3B-Instruct`, `Qwen/Qwen3-30B-A3B-Thinking-2507-FP8`
- **Qwen2.5**: `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-Coder-7B-Instruct`
- **Llama 3**: Llama3.1 and 3.2 series, `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` for example
- **GPT-OSS**: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`
- **Mistral**: `mistralai/Mistral-7B-v0.3`
- **GLM-4 series**: `zai-org/glm-4-9b-chat`
- **Phi series**: `microsoft/Phi-3.5-mini-instruct`, `microsoft/phi-4`
- **Granite 3.1 (MoE)**: `ibm-granite/granite-3.1-1b-a400m-instruct`, `ibm-granite/granite-3.1-3b-a800m-instruct`
- **Granite 3.1 (Dense)**: `ibm-granite/granite-3.1-2b-instruct`, `ibm-granite/granite-3.1-8b-instruct`
- **EXAONE 4.0 series**: `LGAI-EXAONE/EXAONE-4.0-1.2B`, `LGAI-EXAONE/EXAONE-4.0.1-32B`, `LGAI-EXAONE/EXAONE-4.0-32B`
- **OLMo 2**: `allenai/OLMo-2-0425-1B-Instruct`
- **ERNIE 4.5**: `baidu/ERNIE-4.5-0.3B-PT`
- **SmolLM2**: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- **Gemma 2**: `google/gemma-2-2b-it`
- **PLaMo3**: `pfnet/plamo-3-nict-2b-base`
- **Granite MoE SWA**: `ibm-granite/granite-swash-3b-a600m` (MoE + sliding-window attention; FLASH_ATTN and TRITON_ATTN backends; FLEX_ATTENTION and decode-vs-prefill test excluded due to per-layer SWA KV window differences)
- **Granite SWA**: `ibm-granite/granite-swash-2b` (sliding-window attention; FLASH_ATTN and TRITON_ATTN backends)
- **Qwen3-0.6B**: `Qwen/Qwen3-0.6B`

Other models may also work, but these have been explicitly validated. If you encounter issues with a specific model, please report them on the [GitHub issue tracker](https://github.com/vllm-project/vllm/issues/new/choose).

## Implementation Details

### DeepSeek V4 on SM120

<!-- e33d190dd5：新增的说明限定为单卡 SM120；不是所有 DeepSeek V4 配置的通用保证。 -->

The single-GPU DeepSeek V4 FP8 path uses fixed-reduction Triton sparse attention
for both prefill and decode, FP8 indexer scoring with stable cutoff tie-breaking,
and fixed-reduction mHC projections. It also uses the existing Triton compressor
and batch-invariant linear kernels. MXFP4 experts use the batch-invariant Humming
backend; their weights remain quantized.

<!--
逐行解释：prefill/decode 共用固定归约的稀疏 attention；
FP8 indexer 使用稳定的 cutoff 同分规则；mHC 投影也固定归约形状。
compressor 和线性层复用 Triton 的确定性路径。
MXFP4 专家交给支持批次不变性的 Humming 后端，权重不会因此全量反量化保存。
-->

This path requires `fp8_ds_mla` KV cache (the default), FP8 indexer cache, TP=DP=1,
and no context parallelism or speculative decoding. It is selected automatically
with `VLLM_BATCH_INVARIANT=1` on SM120. Other DeepSeek V4 configurations are not
covered. The reference-oriented kernels can be substantially slower than the
default path and are not intended to reproduce its batch-dependent logits.

<!--
逐行解释：只支持 fp8_ds_mla 主缓存、FP8 indexer 缓存和 TP=DP=1。
不支持上下文并行或推测解码；环境变量开启后在 SM120 自动分派。
这些参考取向内核可能明显更慢；保证自身跨 batch 一致，不保证与原快路径逐位相同。
-->

Validation on a six-layer DeepSeek-V4-Flash checkpoint compares every vocabulary
logit across singleton and mixed batches, request order, prefix-cache hits, and
256-, 257-, 272-, and 8192-token prefill budgets, with eager execution and CUDA
graphs. The 272-token budget forces a single-token prefill tail to exercise the
transition to paged indexer scoring.
This is not a model-quality evaluation of the full checkpoint.

<!--
e33d190dd5 新增六层真实模型回归；569f3f1de3 补入 272-token 预算。
逐行解释：比较全词表 logits，覆盖单请求、混批、换序和前缀缓存命中。
预算 256/272 走 eager，257/8192 允许 CUDA graphs。
2177-token prompt 在预算 272 下留下单 token 尾块，覆盖分页 indexer 路径切换。
此处仅说明确定性回归，不代表完整模型的质量评测。
-->

Run the six-layer regression with one command after installing the development
environment and test dependencies:

```bash
# 569f3f1de3：轻负载使用 3 个 filler，每请求生成 3 步。
sh tests/v1/determinism/run_dsv4_e2e_light.sh /path/to/DeepSeek-V4-Flash-6layers
# 同一测试的重负载使用 15 个 filler，每请求生成 512 步。
sh tests/v1/determinism/run_dsv4_e2e_heavy.sh /path/to/DeepSeek-V4-Flash-6layers
```

| Runner | Filler requests | Generated tokens per request |
| ------ | --------------- | ---------------------------- |
| Light | 3 | 3 |
| Heavy | 15 | 512 |

<!-- 569f3f1de3：两档仅改变负载大小，测试场景与判断标准相同。 -->

Both runners use the same five scenarios and four execution configurations.
They default to `../DeepSeek-V4-Flash-6layers` relative to the repository and
stream model startup logs and per-case progress to the terminal.
Only the target's full-vocabulary logits are retained for comparison.
The readable test cases in `test_batch_invariance.py` compare all
logit bits (including signed zero). Both runners exit nonzero on a mismatch.
The shared `run_dsv4_e2e.sh` entrypoint is retained and defaults to heavy.

<!--
569f3f1de3 逐行解释：两种入口均运行五场景、四配置。
默认权重路径相对仓库根目录解析；模型启动和逐场景进度直接输出到终端。
仅保留目标请求的全词表结果以降低内存；比较原始位模式，连正负零都区分。
任一不一致使脚本非零退出；原公共入口保留，默认使用 heavy 负载。
-->

The underlying regression uses the existing `VLLM_TEST_MODEL` override.
`VLLM_TEST_HF_OVERRIDES` optionally supplies a JSON object for partial checkpoints:

```bash
# e33d190dd5：选择六层权重，并用 JSON 覆盖层数、关闭 MTP、设置各层压缩率。
# 后两行从指定测试文件中只选择新增的 raw-logits 回归。
VLLM_TEST_MODEL=/path/to/DeepSeek-V4-Flash-6layers \
VLLM_TEST_HF_OVERRIDES='{"num_hidden_layers":6,"num_nextn_predict_layers":0,"compress_ratios":[0,0,4,128,4,128]}' \
.venv/bin/python -m pytest tests/v1/determinism/test_batch_invariance.py \
    -k test_raw_logits_bs1_vs_bsN_chunking_and_prefix_cache
```

<!-- e33d190dd5：新增此小标题，把上面的 DSV4 专用限制与下方通用机制分开。 -->

### Shared implementation

When batch invariance is enabled, vLLM:

1. Uses deterministic kernel implementations for attention and other operations
2. Ensures consistent numerical behavior across different batch sizes
3. Disables certain optimizations that may introduce non-determinism
4. Under tensor parallelism, keeps custom all-reduce on with a fixed reduction order (the 1-stage kernel is pinned, and large inputs are reduced in fixed-size chunks), and disables FlashInfer, AITER and QuickReduce all-reduce
5. On CUDA devices with tuned matmul table entries for the model's bf16 unquantized forward linear layers (Ada, Hopper, Blackwell),
   runs without `torch.compile` using breakable CUDA graphs so tile configs follow the runtime batch size; set `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to opt out.
6. Splits the LoRA shrink reduction into 8 fixed K chunks that are summed in a fixed order instead of with atomics
7. Sorts the shared sparse attention indexer's selected logical KV indices before attention, keeping invalid entries at the end. Sorting alone does not resolve selection ties at the top-k cutoff; the DeepSeek V4 path additionally uses stable selection to break ties by logical position.

<!--
e33d190dd5：补充第 7 项。选完后按索引排序只能固定读取顺序，不能固定同分时选谁。
因此 DSV4 还需先稳定选出集合：分数相同的候选按逻辑位置优先。
-->

!!! warning
    Batch invariance under tensor parallelism is not yet supported for all-reduces whose size isn't a multiple of 16 bytes, for example a hidden size that isn't a multiple of 8 in fp16/bf16. Such tensors can switch between NCCL and custom all-reduce depending on batch size. All validated models meet this requirement.

!!! note
    Enabling batch invariance may impact performance compared to the default non-deterministic mode. This trade-off is intentional to guarantee reproducibility.

## Future Improvements

The batch invariance feature is under active development. Planned improvements include:

- Support for additional GPU architectures
- Expanded model coverage
- Performance optimizations
- Additional testing and validation

For the latest status and to contribute ideas, see the [tracking issue](https://github.com/vllm-project/vllm/issues/27433).
