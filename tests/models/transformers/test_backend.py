# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test the functionality of the Transformers modeling backend."""

import ast
import contextlib
import os
import tempfile
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PreTrainedConfig

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.transformers.fusers import AttentionFuser
from vllm.model_executor.models.transformers.fusers.attention import VLLM_ATTN_IMPL
from vllm.model_executor.models.transformers.multimodal import MultiModalMixin
from vllm.model_executor.models.utils import StageMissingLayer

from ...conftest import HfRunner, VllmRunner
from ...utils import multi_gpu_test, prep_prompts
from ..registry import HF_EXAMPLE_MODELS
from ..utils import check_embeddings_close, check_logprobs_close


@pytest.fixture(scope="function", autouse=True)
def enable_pickle(monkeypatch):
    """`LLM.apply_model` requires pickling a function."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


def get_model(arch: str) -> str:
    model_info = HF_EXAMPLE_MODELS.get_hf_info(arch)
    model_info.check_transformers_version(on_fail="skip")
    return model_info.default


def count_modules(model, cls: type) -> int:
    return sum(isinstance(m, cls) for m in model.modules())


def get_num_fused(model) -> tuple[int, int]:
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )

    glu = count_modules(model, MergedColumnParallelLinear)
    qkv = count_modules(model, QKVParallelLinear)
    return glu, qkv


def check_implementation(
    runner_ref: type[HfRunner | VllmRunner],
    runner_test: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    kwargs_ref: dict[str, Any] | None = None,
    kwargs_test: dict[str, Any] | None = None,
    num_fused: tuple[int, int] = (1, 1),
    **kwargs,
):
    if kwargs_ref is None:
        kwargs_ref = {}
    if kwargs_test is None:
        kwargs_test = {}

    max_tokens = 32
    num_logprobs = 5

    args = (example_prompts, max_tokens, num_logprobs)

    with runner_test(model, **kwargs_test, **kwargs) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        num_layers = model_config.hf_config.get_text_config().num_hidden_layers
        expected_glu, expected_qkv = num_fused
        for num_glu, num_qkv in model_test.apply_model(get_num_fused):
            assert num_glu == expected_glu * num_layers
            assert num_qkv == expected_qkv * num_layers

        tp_size = kwargs_test.get("tensor_parallel_size", 1)
        from vllm.model_executor.layers.attention import Attention

        counts = model_test.apply_model(partial(count_modules, cls=Attention))
        assert counts == [num_layers] * tp_size

        outputs_test = model_test.generate_greedy_logprobs(*args)

    with runner_ref(model, **kwargs_ref) as model_ref:
        if isinstance(model_ref, VllmRunner):
            outputs_ref = model_ref.generate_greedy_logprobs(*args)
        else:
            outputs_ref = model_ref.generate_greedy_logprobs_limit(*args)

    check_logprobs_close(
        outputs_0_lst=outputs_ref,
        outputs_1_lst=outputs_test,
        name_0="ref",
        name_1="test",
    )


@pytest.mark.parametrize(
    "model,model_impl,num_fused",
    [
        ("meta-llama/Llama-3.2-1B-Instruct", "transformers", (1, 1)),
        ("hmellor/Ilama-3.2-1B", "auto", (1, 1)),  # CUSTOM CODE
        ("allenai/OLMoE-1B-7B-0924", "transformers", (0, 1)),  # MoE
    ],
)  # trust_remote_code=True by default
def test_models(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    model_impl: str,
    num_fused: tuple[int, int],
) -> None:
    check_implementation(
        hf_runner,
        vllm_runner,
        example_prompts,
        model,
        num_fused=num_fused,
        model_impl=model_impl,
    )


def test_hybrid_attention(vllm_runner: type[VllmRunner]) -> None:
    prompts, _, _ = prep_prompts(4, (800, 801))
    kwargs_ref = {"max_model_len": 8192, "enforce_eager": True}
    kwargs_test = {"model_impl": "transformers", **kwargs_ref}
    check_implementation(
        vllm_runner,
        vllm_runner,
        prompts,
        model="hmellor/tiny-random-Gemma2ForCausalLM",
        kwargs_ref=kwargs_ref,
        kwargs_test=kwargs_test,
    )


@pytest.fixture(scope="module")
def tiny_qwen3_5_path(tmp_path_factory):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        max_position_embeddings=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
        },
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = Qwen3_5ForCausalLM(config)
    path = tmp_path_factory.mktemp("tiny_qwen3_5")
    model.save_pretrained(
        path,
        state_dict={**model.state_dict(), "mtp.aux.weight": torch.zeros(1)},
    )
    return str(path)


@pytest.mark.parametrize(
    ("tp_size", "enforce_eager"), [(1, True), (2, True), (1, False)]
)
def test_qwen3_5_recurrent_cache_matches_transformers(
    vllm_runner: type[VllmRunner],
    tiny_qwen3_5_path,
    tp_size,
    enforce_eager,
    monkeypatch,
):
    """Loaded GDN weights and cached chunked prefill/decode match HF logits."""
    outputs, repeated = _check_recurrent_cache(
        vllm_runner, tiny_qwen3_5_path, tp_size, enforce_eager, monkeypatch
    )
    assert [o.outputs[0].token_ids for o in outputs] == [
        o.outputs[0].token_ids for o in repeated[::-1]
    ]


@pytest.fixture(scope="module")
def tiny_hybrid_path(request, tmp_path_factory):
    from transformers import (
        OlmoHybridConfig,
        OlmoHybridForCausalLM,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
    )

    kind, layer_types, allow_neg_eigval = request.param
    kwargs = dict(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        layer_types=layer_types,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    if kind == "olmo":
        config = OlmoHybridConfig(
            **kwargs,
            linear_allow_neg_eigval=allow_neg_eigval,
            rope_parameters={"rope_type": "default", "rope_theta": 10000},
        )
        model_cls = OlmoHybridForCausalLM
    else:
        config = Qwen3_5TextConfig(
            **kwargs,
            head_dim=32,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [2, 1, 1],
            },
        )
        model_cls = Qwen3_5ForCausalLM
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = model_cls(config)
    path = tmp_path_factory.mktemp(f"tiny_hybrid_{kind}")
    model.save_pretrained(path, save_original_format=False)
    return str(path)


@pytest.mark.parametrize(
    "tiny_hybrid_path",
    [
        pytest.param(
            (
                "qwen",
                [
                    "full_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                ],
                False,
            ),
            id="irregular-qwen",
        ),
        pytest.param(("qwen", ["linear_attention"] * 3, False), id="state-only-qwen"),
        pytest.param(
            ("olmo", ["linear_attention", "full_attention", "linear_attention"], True),
            id="olmo-negative-eigenvalues",
        ),
        pytest.param(
            ("olmo", ["full_attention", "linear_attention", "linear_attention"], False),
            id="olmo-positive-eigenvalues",
        ),
    ],
    indirect=True,
)
def test_hybrid_recurrent_cache_matches_transformers(
    vllm_runner, tiny_hybrid_path, monkeypatch
):
    """Operator order and HF projection layout do not select a model wrapper."""
    _check_recurrent_cache(vllm_runner, tiny_hybrid_path, 1, True, monkeypatch)


@pytest.mark.parametrize(
    "tiny_hybrid_path",
    [("qwen", ["full_attention"] * 3, False)],
    indirect=True,
)
def test_full_attention_still_uses_stateless_transformers_backend(
    vllm_runner, tiny_hybrid_path, monkeypatch
):
    _check_recurrent_cache(
        vllm_runner, tiny_hybrid_path, 1, True, monkeypatch, has_state=False
    )


@pytest.fixture(
    scope="module", params=[1e-5, 0.2], ids=["default-norm", "hf-custom-norm"]
)
def tiny_mixed_operator_path(request, tmp_path_factory):
    from .hybrid_test_model import (
        HybridTestConfig,
        HybridTestForCausalLM,
    )

    config = HybridTestConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        sliding_window=8,
        attention_layer_index=-1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
        auto_map={"AutoModel": "hybrid_test_model.HybridTestModel"},
        operator_specs=[
            {"kind": "qwen_gdn"},
            {"kind": "sliding_attention"},
            {
                "kind": "olmo_gdn",
                "linear_conv_kernel_dim": 3,
                "output_norm_eps": request.param,
            },
            {"kind": "full_attention"},
            {
                "kind": "qwen_gdn",
                "linear_num_value_heads": 2,
                "linear_conv_kernel_dim": 5,
            },
            {"kind": "sliding_attention"},
        ],
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = HybridTestForCausalLM(config)
    path = tmp_path_factory.mktemp("tiny_mixed_operators")
    model.save_pretrained(path, save_original_format=False)
    return str(path)


@pytest.mark.parametrize(
    ("tp_size", "max_num_batched_tokens"), [(1, 32), (1, 128), (2, 32)]
)
def test_mixed_state_geometry_and_sliding_cache_matches_transformers(
    vllm_runner, tiny_mixed_operator_path, monkeypatch, tp_size, max_num_batched_tokens
):
    """Real GDN expressions, different state specs and sliding compose per site."""
    _check_recurrent_cache(
        vllm_runner,
        tiny_mixed_operator_path,
        tp_size,
        True,
        monkeypatch,
        max_num_batched_tokens=max_num_batched_tokens,
    )


def _assert_hf_gdn_boundaries(model):
    import inspect

    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.models.transformers.gdn import GDNKernel

    kernels = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, GDNKernel)
    ]
    assert kernels
    for name, kernel in kernels:
        mixer = model.get_submodule(name.removesuffix("._vllm_gdn"))
        assert not isinstance(mixer, MambaBase)
        assert type(mixer).__module__.startswith("transformers.models.")
        assert not list(kernel.parameters())
        original = inspect.unwrap(type(mixer).forward)
        assert mixer.forward.__func__.__globals__ is original.__globals__
        assert "_vllm_gdn" in mixer.forward.__func__.__code__.co_names


def _check_recurrent_cache(
    vllm_runner,
    checkpoint_path,
    tp_size,
    enforce_eager,
    monkeypatch,
    max_num_batched_tokens=32,
    has_state=True,
):
    from transformers import AutoModelForCausalLM

    from vllm import SamplingParams

    if torch.accelerator.device_count() < tp_size:
        pytest.skip(f"Requires {tp_size} GPUs")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    prompts = [list(range(3, 15)), list(range(30, 113))]
    params = SamplingParams(
        temperature=0,
        max_tokens=4,
        ignore_eos=True,
        detokenize=False,
        logprobs=5,
    )
    with vllm_runner(
        checkpoint_path,
        model_impl="transformers",
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        enforce_eager=enforce_eager,
        skip_tokenizer_init=True,
        max_model_len=128,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=2,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.2,
    ) as runner:
        config = runner.llm.llm_engine.model_config
        assert config.using_transformers_backend()
        assert config.is_hybrid == has_state
        assert config.has_inner_state == has_state
        if has_state:
            runner.apply_model(_assert_hf_gdn_boundaries)
        inputs = [{"prompt_token_ids": prompt} for prompt in prompts]
        outputs = runner.llm.generate(inputs, params, use_tqdm=False)
        repeated = runner.llm.generate(inputs[::-1], params, use_tqdm=False)

    reference = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.float32,
        attn_implementation="eager",
        trust_remote_code=True,
    ).eval()
    for prompt, output in zip(prompts * 2, outputs + repeated[::-1]):
        generated = output.outputs[0]
        assert len(generated.token_ids) == 4
        tokens = torch.tensor([prompt + list(generated.token_ids)])
        with torch.no_grad():
            logits = reference(tokens, use_cache=False).logits[0].float()
        expected = logits[len(prompt) - 1 : -1].log_softmax(-1)
        for step, logprobs in enumerate(generated.logprobs):
            token_ids = list(logprobs)
            actual = torch.tensor([logprobs[t].logprob for t in token_ids])
            torch.testing.assert_close(
                actual, expected[step, token_ids], atol=0.04, rtol=0
            )
    return outputs, repeated


@pytest.mark.parametrize("ssm_cache_dtype", ["auto", "bfloat16"])
@pytest.mark.parametrize("configured_ssm_dtype", [None, "float32"])
def test_state_metadata_uses_each_actual_operator_geometry(
    tiny_mixed_operator_path, ssm_cache_dtype, configured_ssm_dtype
):
    from vllm.model_executor.models.transformers.state import HybridMixin, state_spec
    from vllm.transformers_utils.hybrid import (
        get_state_operators,
        inspect_model_structure,
    )

    config = AutoConfig.from_pretrained(
        tiny_mixed_operator_path, trust_remote_code=True
    )
    config.get_text_config().mamba_ssm_dtype = configured_ssm_dtype
    rng = torch.get_rng_state().clone()
    structure = inspect_model_structure(config, True)
    assert inspect_model_structure(config, True) is structure
    assert torch.equal(torch.get_rng_state(), rng)
    operators = get_state_operators(config, True)
    assert [operator.path for operator in operators] == [
        "layers.0.mixer",
        "layers.2.mixer",
        "layers.4.mixer",
    ]
    fake_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, hf_config=config, trust_remote_code=True
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="auto", mamba_ssm_cache_dtype=ssm_cache_dtype
        ),
        num_speculative_tokens=0,
    )
    specs = [state_spec(operator, fake_config) for operator in operators]
    assert len({spec.shapes for spec in specs}) == 3
    expected_dtype = (
        torch.float32
        if ssm_cache_dtype == "auto" and configured_ssm_dtype == "float32"
        else torch.bfloat16
    )
    assert all(spec.dtypes == (torch.bfloat16, expected_dtype) for spec in specs)
    assert HybridMixin.get_mamba_specs_from_config(fake_config) == tuple(specs)


@pytest.mark.parametrize("rank", [0, 1])
def test_gdn_conv_checkpoint_layouts_keep_qkv_shards_together(rank):
    from vllm.model_executor.models.transformers.gdn import gdn_conv_weight_loader

    sizes = [4, 4, 8]
    weights = torch.arange(sum(sizes) * 3).reshape(-1, 1, 3).float()
    loader = gdn_conv_weight_loader(sizes, 2, rank)
    combined = torch.empty(sum(sizes) // 2, 1, 3)
    separate = torch.empty_like(combined)
    loader(combined, weights)
    expected = []
    for shard, weight in enumerate(weights.split(sizes)):
        loader(separate, weight, shard)
        expected.append(weight.chunk(2)[rank])
    torch.testing.assert_close(combined, torch.cat(expected), rtol=0, atol=0)
    torch.testing.assert_close(separate, combined, rtol=0, atol=0)


def test_state_prefill_kernels_do_not_share_incompatible_metadata(
    tiny_qwen3_5_path, monkeypatch
):
    from vllm.model_executor.models.transformers.state import state_attention_backend
    from vllm.v1.kv_cache_interface import MambaSpec

    triton = state_attention_backend("triton")
    flashinfer = state_attention_backend("flashinfer")
    assert triton is state_attention_backend("triton")
    assert triton is not flashinfer
    assert triton.get_builder_cls() is not flashinfer.get_builder_cls()
    config = VllmConfig(
        model_config=ModelConfig(
            model=str(tiny_qwen3_5_path),
            model_impl="transformers",
            dtype="bfloat16",
            skip_tokenizer_init=True,
        ),
        additional_config={"gdn_prefill_backend": "auto", "keep_option": True},
    )
    text_config = config.model_config.hf_text_config
    original_dims = (
        text_config.linear_key_head_dim,
        text_config.linear_value_head_dim,
    )
    spec = MambaSpec(
        block_size=16,
        shapes=((256, 3), (2, 64, 32)),
        dtypes=(torch.bfloat16, torch.float32),
    )

    def resolve_backend(local_config):
        local_text = local_config.model_config.hf_text_config
        assert (local_text.linear_key_head_dim, local_text.linear_value_head_dim) == (
            32,
            64,
        )
        assert local_config.additional_config["keep_option"]
        backend = local_config.additional_config["gdn_prefill_backend"]
        return backend, backend

    monkeypatch.setattr(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn."
        "_resolve_gdn_prefill_backend",
        resolve_backend,
    )
    for backend, expected in ((triton, "triton"), (flashinfer, "flashinfer")):
        builder = backend.get_builder_cls()(
            spec, ["state"], config, torch.device("cpu")
        )
        assert builder.gdn_prefill_backend == expected
        assert builder.vllm_config is not config
        assert builder.vllm_config.model_config is not config.model_config
        assert builder.vllm_config.model_config.hf_text_config is not text_config
    assert config.additional_config == {
        "gdn_prefill_backend": "auto",
        "keep_option": True,
    }
    assert (text_config.linear_key_head_dim, text_config.linear_value_head_dim) == (
        original_dims
    )


@pytest.mark.parametrize("failure", ["unknown", "unknown-conv", "activation", "shared"])
def test_unadapted_state_semantics_fail_explicitly(failure):
    from transformers import JambaConfig

    from vllm.transformers_utils.hybrid import describe_state_operators

    if failure == "unknown":
        with torch.device("meta"):
            model = AutoModel.from_config(JambaConfig(num_hidden_layers=2))
        match = "layers.0.mamba.*JambaMambaMixer.*no vLLM state adapter"
    elif failure == "unknown-conv":
        from transformers import Lfm2Config

        with torch.device("meta"):
            model = AutoModel.from_config(
                Lfm2Config(num_hidden_layers=2, layer_types=["conv", "full_attention"])
            )
        match = "layers.0.*Lfm2ShortConv.*no vLLM state adapter"
    else:
        from transformers import Qwen3_5TextConfig

        with torch.device("meta"):
            model = AutoModel.from_config(Qwen3_5TextConfig(num_hidden_layers=2))
        if failure == "activation":
            model.layers[0].linear_attn.activation = "relu"
            match = "layers.0.linear_attn.*unsupported convolution activation"
        else:
            model.layers[1].linear_attn = model.layers[0].linear_attn
            match = "layers.1.linear_attn.*shares weights"
    with pytest.raises(ValueError, match=match):
        describe_state_operators(model)


def get_sinks(model) -> dict[int, torch.Tensor]:
    """The sink tensor each attention layer was handed, keyed by layer index."""
    sinks = {}
    for i, (prefix, fuser) in model.attention_fusers.items():
        if (sink := fuser.sinks(model.get_submodule(prefix))) is not None:
            sinks[i] = sink.float().cpu()
    return sinks


def test_sinks(hf_runner: type[HfRunner], vllm_runner: type[VllmRunner]) -> None:
    """Learnable attention sinks must reach the attention layers.

    Only the attention impl can apply them, so if they are not passed to
    `Attention` they are dropped and every softmax is subtly wrong.
    """
    from vllm.platforms import current_platform

    if not current_platform.has_device_capability(90):
        pytest.skip("Attention sinks need FlashAttention 3 or TRTLLM attention")

    model = "tiny-random/gpt-oss-bf16"
    with vllm_runner(
        model, model_impl="transformers", max_model_len=1024, enforce_eager=True
    ) as model_test:
        assert model_test.llm.llm_engine.model_config.using_transformers_backend()
        sinks = model_test.apply_model(get_sinks)[0]

    with hf_runner(model, dtype="bfloat16") as model_ref:
        layers = model_ref.model.model.get_decoder().layers
        expected = {i: layer.self_attn.sinks.float() for i, layer in enumerate(layers)}

    assert sinks.keys() == expected.keys()
    for i, sink in sinks.items():
        torch.testing.assert_close(sink, expected[i].cpu())


def test_mla(vllm_runner: type[VllmRunner], example_prompts: list[str]) -> None:
    model = "hmellor/tiny-random-DeepseekV2ForCausalLM"
    args = (example_prompts, 32, 5)
    kwargs: dict[str, Any] = {"max_model_len": 2048, "enforce_eager": True}

    with vllm_runner(
        model, model_impl="transformers", trust_remote_code=False, **kwargs
    ) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()
        num_layers = model_config.hf_config.get_text_config().num_hidden_layers
        from vllm.model_executor.layers.attention import MLAAttention

        counts = model_test.apply_model(partial(count_modules, cls=MLAAttention))
        assert counts == [num_layers]
        outputs_test = model_test.generate_greedy_logprobs(*args)

    with vllm_runner(model, model_impl="auto") as model_ref:
        outputs_ref = model_ref.generate_greedy_logprobs(*args)

    check_logprobs_close(
        outputs_0_lst=outputs_ref,
        outputs_1_lst=outputs_test,
        name_0="native",
        name_1="transformers",
    )


@multi_gpu_test(num_gpus=2)
def test_distributed(
    hf_runner: type[HfRunner],
    vllm_runner: type[VllmRunner],
    example_prompts,
):
    kwargs = {"model_impl": "transformers", "tensor_parallel_size": 2}
    check_implementation(
        hf_runner,
        vllm_runner,
        example_prompts,
        "meta-llama/Llama-3.2-1B-Instruct",
        kwargs_test=kwargs,
    )


@pytest.mark.parametrize(
    "model, quantization_kwargs",
    [
        ("TheBloke/TinyLlama-1.1B-Chat-v0.3-AWQ", {}),
        ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit", {}),
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_quantization(
    vllm_runner: type[VllmRunner],
    example_prompts: list[str],
    model: str,
    quantization_kwargs: dict[str, str],
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
        model,
        model_impl="auto",
        enforce_eager=True,
        **quantization_kwargs,
    ) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
        )

    with vllm_runner(
        model,
        model_impl="transformers",
        enforce_eager=True,
        **quantization_kwargs,
    ) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        transformers_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens=max_tokens, num_logprobs=num_logprobs
        )

    check_logprobs_close(
        outputs_0_lst=transformers_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="transformers",
        name_1="vllm",
    )


@pytest.mark.parametrize(
    "model",
    [
        # Layers live in `layers`
        "Qwen/Qwen3-Embedding-0.6B",
        # Layers live in `model.layers`
        "meta-llama/Llama-3.2-1B-Instruct",
    ],
)
def test_embed_loading(vllm_runner, model):
    with vllm_runner(
        model,
        max_model_len=1024,
        enforce_eager=True,
        runner="pooling",
        model_impl="transformers",
    ) as model_test:
        model_config = model_test.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()


@pytest.mark.parametrize(
    "arch", ["TransformersEmbeddingModel", "TransformersForSequenceClassification"]
)
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_pooling(
    hf_runner,
    vllm_runner,
    example_prompts,
    arch,
    monkeypatch,
    use_v2_model_runner,
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
    model = get_model(arch)

    vllm_kwargs = dict(max_model_len=None, model_impl="transformers")

    hf_kwargs = dict()
    if arch == "TransformersEmbeddingModel":
        hf_kwargs["is_sentence_transformer"] = True
    elif arch == "TransformersForSequenceClassification":
        from transformers import AutoModelForSequenceClassification

        hf_kwargs["auto_cls"] = AutoModelForSequenceClassification

    # The example_prompts has ending "\n", for example:
    # "Write a short story about a robot that dreams for the first time.\n"
    # sentence_transformers will strip the input texts, see:
    # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
    # This makes the input_ids different between hf_model and vllm_model.
    # So we need to strip the input texts to avoid test failing.
    example_prompts = [str(s).strip() for s in example_prompts]

    with (
        vllm_runner(model, **vllm_kwargs) as vllm_model,
        hf_runner(model, **hf_kwargs) as hf_model,
    ):
        model_config = vllm_model.llm.llm_engine.model_config
        assert model_config.using_transformers_backend()

        if arch == "TransformersEmbeddingModel":
            vllm_outputs = vllm_model.embed(example_prompts)
            hf_outputs = hf_model.encode(example_prompts)
        elif arch == "TransformersForSequenceClassification":
            vllm_outputs = vllm_model.classify(example_prompts)
            hf_outputs = hf_model.classify(example_prompts)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


VOCAB_SIZE = 64
PER_LAYER_VOCAB_SIZE = 32
NUM_POSITIONS = 16
HIDDEN_SIZE = 8
EMBED_SCALE = 3.0


class ScaledWordEmbedding(nn.Embedding):
    """Mirrors Transformers' `*ScaledWordEmbedding` classes."""

    def __init__(
        self, num_embeddings, embedding_dim, padding_idx=None, embed_scale=1.0
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


class ComposedWordEmbedding(nn.Module):
    """Scales embeddings, but wraps `nn.Embedding` instead of inheriting from it."""

    def __init__(self, num_embeddings, embedding_dim, embed_scale=1.0):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids):
        return self.embed(input_ids) * self.embed_scale


@pytest.fixture
def tp_init():
    """Single rank tensor parallel state, so vLLM layers can be constructed."""
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform

    from ...utils import ensure_current_vllm_config

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        with ensure_current_vllm_config():
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend=current_platform.dist_backend,
            )
            initialize_model_parallel(1, 1)
            yield
        cleanup_dist_env_and_memory()
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


@pytest.fixture
def vpe(tp_init):
    """`VocabParallelEmbedding`, imported late so collection does not import vLLM."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    return VocabParallelEmbedding


def replace(embedding):
    """Replace `embedding` and fill the new weights with recognisable values."""
    from vllm.model_executor.models.transformers.utils import replace_embedding_class

    new_embedding = replace_embedding_class(embedding)
    for _, param in new_embedding.named_parameters():
        param.data = torch.arange(param.numel(), dtype=param.dtype).view(param.shape)
    return new_embedding


def assert_scaled(vpe, module):
    """`module`'s output is its own unscaled output times `EMBED_SCALE`."""
    input_ids = torch.arange(VOCAB_SIZE)
    unscaled = vpe.forward(module, input_ids)
    torch.testing.assert_close(module(input_ids), unscaled * EMBED_SCALE)


def test_replace_plain_embedding(vpe):
    """A plain `nn.Embedding` is replaced outright, leaving no subclass behind."""
    assert type(replace(nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE))) is vpe


def test_replace_infers_shape_and_dtype(tp_init):
    """Shape and dtype come from the replaced module, not from the config."""
    embedding = nn.Embedding(VOCAB_SIZE * 2, HIDDEN_SIZE + 1, dtype=torch.float16)
    new_embedding = replace(embedding)

    assert new_embedding.num_embeddings == VOCAB_SIZE * 2
    assert new_embedding.org_vocab_size == VOCAB_SIZE * 2
    assert new_embedding.embedding_dim == HIDDEN_SIZE + 1
    assert new_embedding.weight.dtype == torch.float16


def test_replace_inherited_embedding(vpe):
    """Subclasses keep their extra state and their scaled `forward`."""
    new_embedding = replace(
        ScaledWordEmbedding(
            VOCAB_SIZE, HIDDEN_SIZE, padding_idx=0, embed_scale=EMBED_SCALE
        )
    )

    assert isinstance(new_embedding, vpe)
    assert new_embedding.scalar_embed_scale == EMBED_SCALE
    assert "embed_scale" in new_embedding._non_persistent_buffers_set
    assert_scaled(vpe, new_embedding)


def test_replace_is_idempotent(tp_init):
    """A tied table is reached twice by the recursion, and must not be rebased twice."""
    from vllm.model_executor.models.transformers.utils import replace_embedding_class

    new_embedding = replace(ScaledWordEmbedding(VOCAB_SIZE, HIDDEN_SIZE, embed_scale=2))
    again = replace_embedding_class(new_embedding)

    assert again is new_embedding
    assert type(again) is type(new_embedding)


class FakeVocabModel(nn.Module):
    """The embedding tables a model can hold, only one of which is a vocab table.

    `embed_tokens_per_layer` mirrors Gemma 3n, whose per-layer table is sized by
    `vocab_size_per_layer_input` rather than `vocab_size`.
    """

    def __init__(self, per_layer: bool = False, composed: bool = False):
        super().__init__()
        embed_cls = ComposedWordEmbedding if composed else ScaledWordEmbedding
        self.embed_tokens = embed_cls(VOCAB_SIZE, HIDDEN_SIZE, embed_scale=EMBED_SCALE)
        if per_layer:
            self.embed_tokens_per_layer = ScaledWordEmbedding(
                PER_LAYER_VOCAB_SIZE, HIDDEN_SIZE, embed_scale=EMBED_SCALE
            )
        # Not a vocab table: indexed by position, and read as a whole by models
        # which interpolate it (e.g. `SiglipVisionEmbeddings`)
        self.position_embedding = nn.Embedding(NUM_POSITIONS, HIDDEN_SIZE)

    def get_input_embeddings(self):
        return self.embed_tokens


def replace_vocab_embeddings(model, **config_kwargs):
    """Replace `model`'s vocab tables as `recursive_replace` would, and return them."""
    from vllm.model_executor.models.transformers.utils import attrsetter

    stub = nn.Module()
    stub.model = model
    stub.config = PreTrainedConfig(
        vocab_size=VOCAB_SIZE, num_positions=NUM_POSITIONS, **config_kwargs
    )
    embeddings = Base._vocab_embeddings(stub)

    replaced = []
    for name, module in list(model.named_modules()):
        if module in embeddings:
            replaced.append(replace(module))
            attrsetter(name)(model, replaced[-1])
    return replaced


def test_only_vocab_tables_are_replaced(vpe):
    """Position embeddings share the `nn.Embedding` type, but not the treatment."""
    model = FakeVocabModel()

    assert replace_vocab_embeddings(model) == [model.embed_tokens]
    assert isinstance(model.embed_tokens, vpe)
    assert type(model.position_embedding) is nn.Embedding


def test_per_layer_vocab_table_is_replaced(vpe):
    """A second vocab table is found by its size, without a bespoke accessor."""
    model = FakeVocabModel(per_layer=True)
    replaced = replace_vocab_embeddings(
        model, vocab_size_per_layer_input=PER_LAYER_VOCAB_SIZE
    )

    assert replaced == [model.embed_tokens, model.embed_tokens_per_layer]
    assert all(isinstance(module, vpe) for module in replaced)


def test_composed_input_embeddings_are_replaced(vpe):
    """A wrapper is left alone; only the `nn.Embedding` it composes is replaced.

    `CausalMixin` ties `lm_head` to whichever `VocabParallelEmbedding` it finds under
    `get_input_embeddings()`, reading a `.weight` the wrapper does not have.
    """
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    model = FakeVocabModel(composed=True)
    wrapper = model.embed_tokens
    replaced = replace_vocab_embeddings(model)

    assert model.embed_tokens is wrapper
    assert replaced == [m for m in wrapper.modules() if isinstance(m, vpe)]

    lm_head = ParallelLMHead(VOCAB_SIZE, HIDDEN_SIZE)
    assert lm_head.tie_weights(replaced[0]).weight is replaced[0].weight


def test_missing_input_embeddings_are_skipped(tp_init):
    """Pipeline ranks without the embeddings have a `PPMissingLayer` in their place."""
    from vllm.model_executor.models.utils import PPMissingLayer

    model = FakeVocabModel()
    model.embed_tokens = PPMissingLayer()

    assert replace_vocab_embeddings(model) == []


MULTIMODAL_MODEL = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"


class MarkingStub(SupportsMultiModal, nn.Module):
    """Just enough of the backend to exercise component marking."""

    _mark_model_components = MultiModalMixin._mark_model_components
    _find_encoder_classes = MultiModalMixin._find_encoder_classes
    _from_config_kwargs = Base._from_config_kwargs
    _pre_trained_model_classes = Base._pre_trained_model_classes


def build_marked_model(
    image_limit: int, video_limit: int = 0, skip_tokenizer_init: bool = False
):
    """Build the HF model inside the marking context and return it."""
    model_config = ModelConfig(
        model=MULTIMODAL_MODEL,
        model_impl="transformers",
        limit_mm_per_prompt={"image": image_limit, "video": video_limit},
    )
    # Set after construction: building the config itself needs the tokenizer
    model_config.skip_tokenizer_init = skip_tokenizer_init
    stub = MarkingStub()
    stub.config = AutoConfig.from_pretrained(MULTIMODAL_MODEL)
    stub.model_config = model_config

    vllm_config = VllmConfig(model_config=model_config)
    with stub._mark_model_components(vllm_config), torch.device("meta"):
        stub.model = AutoModel.from_config(**stub._from_config_kwargs)
    return stub.model


@pytest.mark.parametrize(
    ("image_limit", "video_limit", "skipped"),
    [(0, 0, True), (4, 0, False), (0, 4, False)],
)
def test_tower_weights_skipped_when_modality_disabled(
    image_limit, video_limit, skipped
):
    """Only `--limit-mm-per-prompt image=0,video=0` drops the tower they share."""
    vision_tower = build_marked_model(image_limit, video_limit).vision_tower
    assert isinstance(vision_tower, StageMissingLayer) is skipped


def test_marking_skipped_without_tokenizer():
    """Marking needs the HF processor, which needs a tokenizer, so it is skipped.

    The tower is built as normal rather than the model failing to load.
    """
    vision_tower = build_marked_model(0, skip_tokenizer_init=True).vision_tower
    assert not isinstance(vision_tower, StageMissingLayer)


NUM_LAYERS = 4


def build_model(model_type: str, **overrides) -> nn.Module:
    """A tiny HF model of `NUM_LAYERS` layers, built on the meta device."""
    try:
        config = AutoConfig.for_model(
            model_type,
            num_hidden_layers=NUM_LAYERS,
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=VOCAB_SIZE,
            **overrides,
        )
    except ValueError:
        pytest.skip(f"The installed transformers has no {model_type!r} model")
    with torch.device("meta"):
        return AutoModel.from_config(config)


ATTENTION_MODEL_TYPES = [
    # The default `head_size**-0.5` happens to be right here
    "llama",
    # Scales the query by a learnable per-dim weight, so it declares 1.0
    "gemma4_text",
    # MLA, and declares yarn's `mscale`
    "deepseek_v3",
]


@pytest.mark.parametrize("model_type", ATTENTION_MODEL_TYPES)
def test_attention_dispatch_is_matched(model_type: str):
    """Exactly the decoder layers' attention modules match an `AttentionFuser`.

    That match is what `recursive_replace` records, and so what
    `_create_attention_instances` attaches vLLM's attention layer to.
    """
    model = build_model(model_type)
    matched = {
        name
        for name, module in model.named_modules()
        if AttentionFuser.match(None, module) is not None
    }
    assert matched == {f"layers.{i}.self_attn" for i in range(NUM_LAYERS)}


def test_attention_layer_index_is_the_modules_own():
    """The layer served comes from the module, not from its position in the stack.

    LongCat Flash gives each decoder layer two attention sublayers numbered
    `2i` and `2i + 1`, so `num_hidden_layers` is twice the length of the stack
    and the enclosing layer's position is not the index the KV cache is keyed by.
    A vision tower is excluded separately, by `validate`: only the text config is
    patched to dispatch to vLLM, so a tower's attention is left to Transformers.
    """
    model = build_model("longcat_flash", num_layers=2)
    text_config = model.config.get_text_config()
    assert len(model.layers) * 2 == text_config.num_hidden_layers

    attentions = [
        model.get_submodule(f"layers.{i}.self_attn.{j}") for i in (0, 1) for j in (0, 1)
    ]
    fusers = [AttentionFuser.match(None, attn) for attn in attentions]
    assert all(fuser is not None for fuser in fusers)
    assert [f.layer_index(a) for f, a in zip(fusers, attentions)] == [0, 1, 2, 3]

    # Not dispatching to vLLM yet, as a vision tower never would be. The vLLM
    # config is not consulted, only the one the module was built with.
    assert not any(f.validate(a, None) for f, a in zip(fusers, attentions))
    text_config._attn_implementation = VLLM_ATTN_IMPL
    assert all(f.validate(a, None) for f, a in zip(fusers, attentions))


def test_attention_dispatch_is_required(monkeypatch: pytest.MonkeyPatch):
    """Models selected by the permissive registry fail clearly without dispatch."""
    monkeypatch.setattr(
        "vllm.model_executor.models.transformers.base.get_pp_indices",
        lambda *_: (0, 1),
    )
    model = SimpleNamespace(
        model=nn.Identity(),
        text_config=SimpleNamespace(num_hidden_layers=1),
        pp_group=SimpleNamespace(rank_in_group=0, world_size=1),
        attention_fusers={},
        _get_attn_cls=lambda: None,
    )

    with pytest.raises(
        ValueError,
        match="Layer 0 does not dispatch through the Transformers attention interface",
    ):
        Base._create_attention_instances(model)


@pytest.mark.parametrize("model_type", ATTENTION_MODEL_TYPES)
def test_attention_scale_is_the_declared_one(model_type: str):
    """The scale comes off the module, not from the `head_size**-0.5` default.

    Two of the three model types declare something the default would silently
    replace, so they are what makes this more than a tautology.
    """
    attention = build_model(model_type).layers[0].self_attn
    fuser = AttentionFuser.match(None, attention)
    assert fuser is not None
    assert fuser.scale(attention) == attention.scaling


def test_attention_scale_is_the_argument_not_the_attribute():
    """OPT applies the scale to the query itself, then declares a literal 1.0.

    Its `self.scaling` is the `head_size**-0.5` it has already applied, so
    reading the attribute rather than the argument the module hands the
    interface would scale twice.
    """
    model = build_model("opt", ffn_dim=128, word_embed_proj_dim=64)
    attention = model.get_submodule("decoder.layers.0.self_attn")
    fuser = AttentionFuser.match(None, attention)
    assert fuser is not None
    assert fuser.scale(attention) == 1.0 != attention.scaling


def test_attention_scale_rejects_unresolvable_expression():
    """Unresolvable declared scales must not silently use the Llama default."""

    class Attention(nn.Module):
        head_dim = 16

    fuser = AttentionFuser(
        source_cls="Attention",
        scale_expr=ast.parse("self.head_dim**-0.5 * self.factor", mode="eval").body,
    )

    with pytest.raises(ValueError, match="Cannot resolve attention scaling expression"):
        fuser.scale(Attention())


class _ImageOnlyMRoPEModel:
    """`get_rope_index` without `video_grid_thw` or `**kwargs` (e.g. HunYuanVL)."""

    def get_rope_index(self, input_ids, image_grid_thw):
        seq_len = input_ids.shape[-1]
        positions = torch.arange(seq_len).view(1, 1, -1).expand(4, 1, -1)
        return positions, torch.tensor([0])


def test_get_mrope_input_positions_omits_unsupported_grid_kwargs():
    """Optional grids the model can't accept must not be passed when empty."""
    mixin = SimpleNamespace(model=_ImageOnlyMRoPEModel())

    positions, delta = MultiModalMixin.get_mrope_input_positions(mixin, [1, 2, 3], [])

    assert positions.shape == (4, 3)
    assert delta == 0
