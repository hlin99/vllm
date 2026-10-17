# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavioral tests of the local projection/residual/norm adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm.model_executor.models.transformers import tpsp_transformers_adapter
from vllm.model_executor.models.transformers.tpsp_runtime import (
    TPSPContext,
    TPSPProfile,
    UnsupportedTPSPModelError,
)
from vllm.model_executor.models.transformers.tpsp_transformers_adapter import (
    TransformersDenseTPSP,
)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(16, 16)

    def forward(self, hidden_states, scale):
        output = self.o_proj(hidden_states * scale)
        return output, None


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(16, 32)
        self.down_proj = nn.Linear(32, 16)

    def forward(self, x):
        return self.down_proj(torch.tanh(self.up_proj(x)))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(16)
        self.post_attention_layernorm = nn.LayerNorm(16)
        self.attention = Attention()
        self.mlp = MLP()
        self.calls = 0

    def forward(self, hidden_states, scale):
        self.calls += 1
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.attention(hidden_states, scale=scale)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])
        self.norm = nn.LayerNorm(16)

    def forward(self, hidden_states, scale=1.7):
        hidden_states = hidden_states.sin()
        for layer in self.layers:
            hidden_states = layer(hidden_states, scale=scale)
        hidden_states = self.norm(hidden_states)
        return hidden_states.cos()


class ReferenceBackend:
    """Evaluate the fused contract on CPU, independently of the installed hooks."""

    def __init__(self, shared_output=False):
        self.calls = []
        self.shared_output = shared_output
        self.output = None
        self.open = Mock(return_value=True)
        self.close = Mock()
        self.prepare_weights = Mock()
        self.profile = Mock(
            return_value=TPSPContext(
                TPSPProfile(True, 4, 32), self, {"o_proj": "o", "down_proj": "down"}
            )
        )

    def fused_gemm_rs_norm_ag(
        self, handle, x, projection, residual, norm, *, norm_type
    ):
        self.calls.append(handle)
        assert norm_type == "layer_norm"
        residual = F.linear(x, projection.weight, projection.bias) + residual
        hidden = F.layer_norm(
            residual, norm.normalized_shape, norm.weight, norm.bias, norm.eps
        )
        if self.shared_output:
            if self.output is None:
                self.output = torch.empty_like(hidden)
            hidden = self.output.copy_(hidden)
        return hidden, residual


@pytest.fixture
def config():
    return SimpleNamespace(
        quant_config=None,
        lora_config=None,
        device_config=SimpleNamespace(device=torch.device("cpu")),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        model_config=SimpleNamespace(
            dtype=torch.float32, tpsp_microchunk_tokens=4, tpsp_threshold_tokens=4
        ),
    )


@pytest.fixture
def install(config, monkeypatch):
    from vllm.distributed import parallel_state
    from vllm.platforms import current_platform

    # The adapter is independent of TP transport; emulate a one-rank shard here.
    monkeypatch.setattr(tpsp_transformers_adapter, "tpsp_shard_residual", torch.clone)
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=1, device_group=SimpleNamespace(group_name="tp")
        ),
    )

    def install(model, shared_output=False, *, profile=True):
        backend = ReferenceBackend(shared_output)
        monkeypatch.setattr(
            current_platform, "get_tpsp_backend_cls", lambda: lambda *args: backend
        )
        adapter = TransformersDenseTPSP(model, config)
        if profile:
            adapter.profile()
        return adapter, backend

    return install


@pytest.mark.parametrize("device_type", ["cuda", "xpu"])
def test_adapter_preserves_configured_accelerator_type(
    install, config, monkeypatch, device_type
):
    config.device_config.device = torch.device(device_type)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    _, backend = install(Decoder(), profile=False)
    assert backend.open.call_args.kwargs["device"] == torch.device(device_type, 3)


def test_adapter_opens_before_weights_and_profiles_only_once(install):
    with torch.device("meta"):
        model = Decoder()
    adapter, backend = install(model, profile=False)
    backend.open.assert_called_once_with(
        dtype=torch.float32,
        tp_size=1,
        hidden_size=16,
        max_batched_tokens=32,
        group_name="tp",
        device=torch.device("cpu"),
    )
    backend.profile.assert_not_called()
    backend.prepare_weights.assert_not_called()
    model.to_empty(device="cpu")
    adapter.profile()
    adapter.profile()
    backend.profile.assert_called_once()
    backend.prepare_weights.assert_called_once_with(
        [
            projection
            for layer in model.layers
            for projection in (layer.attention.o_proj, layer.mlp.down_proj)
        ]
    )
    backend.open.assert_called_once()


@pytest.mark.parametrize("raises", [False, True])
def test_adapter_closes_backend_when_profiling_fails(install, raises):
    adapter, backend = install(Decoder(), profile=False)
    if raises:
        backend.profile.side_effect = RuntimeError("profile failed")
        with pytest.raises(RuntimeError, match="profile failed"):
            adapter.profile()
    else:
        backend.profile.return_value = None
        adapter.profile()
    backend.close.assert_called_once()
    adapter.profile()
    backend.profile.assert_called_once()
    assert adapter.context is None and adapter.backend is None


@pytest.mark.parametrize("tokens", [2, 7, 8])
@pytest.mark.parametrize("shared_output", [False, True])
def test_original_computation_survives_outside_fused_regions(
    install, tokens, shared_output
):
    torch.manual_seed(0)
    model = Decoder()
    x = torch.randn(1, tokens, 16)
    expected = model(x, scale=2.3)
    adapter, backend = install(model, shared_output)
    for _ in range(2):
        with adapter.forward_context(tokens):
            actual = model(x, scale=2.3)
        torch.testing.assert_close(actual, expected)
    assert backend.calls == (["o", "down"] * 4 if tokens >= 4 else [])
    assert [layer.calls for layer in model.layers] == [3, 3]
    torch.testing.assert_close(model(x, scale=2.3), expected)


@pytest.fixture
def cuda_tpsp_backend(monkeypatch):
    from vllm.distributed import parallel_state
    from vllm.platforms import tpsp_cuda

    group, control_group = object(), object()
    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(device_group=group, cpu_group=control_group),
    )
    monkeypatch.setattr(tpsp_cuda.c10d, "_resolve_process_group", lambda name: group)
    monkeypatch.setattr(torch.accelerator, "device_count", lambda: 2)
    for name, value in (("get_world_size", 2), ("get_rank", 0)):
        monkeypatch.setattr(tpsp_cuda.dist, name, Mock(return_value=value))
    for name in ("all_reduce", "all_gather", "barrier"):
        monkeypatch.setattr(tpsp_cuda.dist, name, Mock())
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    native = SimpleNamespace(
        init_tpsp_p2p=Mock(return_value=123),
        get_tpsp_p2p_handles=Mock(return_value=torch.zeros(8, dtype=torch.uint8)),
        map_tpsp_p2p=Mock(return_value=True),
        destroy_tpsp_p2p=Mock(),
        tpsp_fused_matmul_reduce_scatter_norm_all_gather=Mock(),
    )
    monkeypatch.setattr(torch.ops, "_C", native)
    backend = tpsp_cuda.CudaTPSPBackend("tp", torch.device("cuda:0"))
    settings = dict(
        dtype=torch.bfloat16,
        tp_size=2,
        hidden_size=16,
        max_batched_tokens=32,
        group_name="tp",
        device=backend.device,
    )
    yield backend, settings, native
    backend.close()


def test_cuda_tpsp_reuses_workspace_without_sharing_configs(cuda_tpsp_backend):
    backend, settings, native = cuda_tpsp_backend
    assert backend.open(**settings)
    o_proj = backend._create_context()
    down_proj = backend._create_context()
    assert o_proj is not down_proj
    native.init_tpsp_p2p.assert_called_once_with(0, 2, 0, 16, 16, 32)
    native.get_tpsp_p2p_handles.assert_called_once_with(123)
    native.map_tpsp_p2p.assert_called_once()
    backend.set_config(o_proj, 4)
    backend.set_config(down_proj, 8)
    assert o_proj.config.microchunk_tokens_per_rank == 4
    assert down_proj.config.microchunk_tokens_per_rank == 8

    backend.close(o_proj)
    native.destroy_tpsp_p2p.assert_not_called()
    backend.set_config(down_proj, 2)
    reopened = backend._create_context()
    backend.set_config(reopened, 8)
    native.init_tpsp_p2p.assert_called_once()
    with pytest.raises(ValueError, match="another backend"):
        backend.set_config(o_proj, 4)

    backend.close()
    backend.close()
    native.destroy_tpsp_p2p.assert_called_once_with(123)
    with pytest.raises(RuntimeError, match="closed"):
        backend.set_config(down_proj, 4)
    with pytest.raises(RuntimeError, match="closed"):
        backend.open(**settings)


def test_cuda_tpsp_configures_context_without_fetching_workspace(
    cuda_tpsp_backend, monkeypatch
):
    backend, settings, _ = cuda_tpsp_backend
    backend.open(**settings)
    fetch = Mock(side_effect=AssertionError("configuration must not fetch workspace"))
    monkeypatch.setattr(backend, "_get_workspace", fetch)
    context = backend._create_context()
    backend.set_config(context, 4)
    assert context.config.microchunk_tokens_per_rank == 4
    fetch.assert_not_called()


def test_cuda_tpsp_reuses_prepared_transposed_weights(cuda_tpsp_backend):
    backend, settings, native = cuda_tpsp_backend
    backend.open(**settings)
    context = backend._create_context()
    backend.set_config(context, 4)
    projection = nn.Linear(8, 16)
    norm = nn.LayerNorm(16)
    backend.prepare_weights([projection])
    transposed = projection._tpsp_transposed_weight
    torch.testing.assert_close(transposed, projection.weight.T)
    assert transposed.is_contiguous() and not transposed.requires_grad
    native.tpsp_fused_matmul_reduce_scatter_norm_all_gather.return_value = (
        torch.empty(4, 16),
        None,
        torch.empty(8, 16),
    )
    for _ in range(2):
        backend.fused_gemm_rs_norm_ag(
            context,
            torch.randn(8, 8),
            projection,
            torch.randn(4, 16),
            norm,
            norm_type="layer_norm",
        )
    for call in native.tpsp_fused_matmul_reduce_scatter_norm_all_gather.call_args_list:
        assert call.args[1] is transposed


def test_cuda_tpsp_requires_open_before_configuring_ops(cuda_tpsp_backend):
    backend, _, native = cuda_tpsp_backend
    with pytest.raises(RuntimeError, match="opened before"):
        backend._create_context()
    native.init_tpsp_p2p.assert_not_called()


def test_cuda_tpsp_rejects_repeated_open(cuda_tpsp_backend):
    backend, settings, native = cuda_tpsp_backend
    backend.open(**settings)
    with pytest.raises(RuntimeError, match="already open"):
        backend.open(**settings)
    native.init_tpsp_p2p.assert_called_once()
    native.destroy_tpsp_p2p.assert_not_called()


@pytest.mark.parametrize("failure", ["allocation", "mapping"])
def test_cuda_tpsp_failed_workspace_is_not_reused(cuda_tpsp_backend, failure):
    backend, settings, native = cuda_tpsp_backend
    if failure == "allocation":
        native.init_tpsp_p2p.return_value = 0
    else:
        native.map_tpsp_p2p.return_value = False
    assert not backend.open(**settings)
    assert native.destroy_tpsp_p2p.call_count == (failure == "mapping")
    native.init_tpsp_p2p.return_value = 123
    native.map_tpsp_p2p.return_value = True
    assert backend.open(**settings)
    assert native.init_tpsp_p2p.call_count == 2


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
@pytest.mark.parametrize("shared_output", [False, True])
def test_current_hf_forward_is_preserved(install, architecture, shared_output):
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.for_model(
        architecture,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        vocab_size=64,
    )
    config._attn_implementation = "eager"
    model = AutoModel.from_config(config).eval()
    for layer in model.layers:
        layer.input_layernorm = nn.LayerNorm(32)
        layer.post_attention_layernorm = nn.LayerNorm(32)
    model.norm = nn.LayerNorm(32)
    inputs = torch.randint(0, 64, (1, 7))
    with torch.inference_mode():
        expected = model(inputs, use_cache=False).last_hidden_state
        adapter, backend = install(model, shared_output)
        with adapter.forward_context(7):
            actual = model(inputs, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected)
    assert backend.calls == ["o", "down"] * 2


class ScaledAttention(Attention):
    def forward(self, hidden_states, scale):
        return self.o_proj(hidden_states * scale) * 2, None


class ScaledMLP(MLP):
    def forward(self, x):
        return self.down_proj(self.up_proj(x)) * 2


class ScaledLayer(Layer):
    def forward(self, hidden_states, scale):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.attention(hidden_states, scale=scale)
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states * 2


class InterLayerComputation(Decoder):
    def forward(self, hidden_states, scale=1.7):
        for layer in self.layers:
            hidden_states = layer(hidden_states, scale=scale)
            hidden_states = hidden_states * 2
        hidden_states = self.norm(hidden_states)
        return hidden_states


@pytest.mark.parametrize(
    "region",
    ["attention", "mlp", "layer", "decoder", "norm", "norm_shape", "shared_norm"],
)
def test_unsafe_fusion_is_rejected_before_any_forward_is_patched(config, region):
    model = InterLayerComputation() if region == "decoder" else Decoder()
    if region == "attention":
        model.layers[-1].attention = ScaledAttention()
    elif region == "mlp":
        model.layers[-1].mlp = ScaledMLP()
    elif region == "layer":
        model.layers[-1] = ScaledLayer()
    elif region == "norm":
        model.norm = nn.Identity()
    elif region == "norm_shape":
        model.norm = nn.LayerNorm(8)
    elif region == "shared_norm":
        model.norm = model.layers[-1].post_attention_layernorm
    forwards = [(module, module.forward) for module in model.modules()]
    with pytest.raises(UnsupportedTPSPModelError):
        TransformersDenseTPSP(model, config)
    assert all(module.forward == original for module, original in forwards)


def test_norm_validation_runs_once_at_installation_not_during_forward(
    install, monkeypatch
):
    validate = Mock(wraps=tpsp_transformers_adapter.get_tpsp_norm_type)
    monkeypatch.setattr(tpsp_transformers_adapter, "get_tpsp_norm_type", validate)
    model = Decoder()
    norms = {
        id(module) for module in model.modules() if isinstance(module, nn.LayerNorm)
    }
    adapter, _ = install(model)
    assert validate.call_count == len(norms)
    assert {id(call.args[0]) for call in validate.call_args_list} == norms
    validate.reset_mock()
    for _ in range(2):
        with adapter.forward_context(7):
            model(torch.randn(1, 7, 16))
    validate.assert_not_called()


@pytest.mark.parametrize("custom_norm", [False, True])
def test_profiling_preserves_single_input_norm_calls_after_installation(
    install, default_vllm_config, custom_norm
):
    from vllm.model_executor.models.transformers.tpsp_runtime import (
        _ProjectionInputs,
        _run_conventional_projection,
    )

    model = Decoder()
    if custom_norm:
        from vllm.model_executor.layers.layernorm import LayerNorm

        model.layers[0].post_attention_layernorm = LayerNorm(16)
    adapter, _ = install(model)
    group = adapter.groups[0]
    x, residual = torch.randn(7, 16), torch.randn(7, 16)
    expected = group.norm(group.projection(x) + residual)
    inputs = _ProjectionInputs(x, residual.clone(), residual.clone(), residual.clone())
    torch.testing.assert_close(
        _run_conventional_projection(group.projection, group.norm, inputs), expected
    )


def test_incomplete_forward_raises_and_restores_original_path(install):
    model = Decoder()
    x = torch.randn(1, 7, 16)
    expected = model(x)
    adapter, _ = install(model)
    with (
        pytest.raises(RuntimeError, match="did not consume"),
        adapter.forward_context(7),
    ):
        model.layers[0](x, scale=1.7)
    torch.testing.assert_close(model(x), expected)
    with pytest.raises(RuntimeError, match="test failure"), adapter.forward_context(7):
        raise RuntimeError("test failure")
    torch.testing.assert_close(model(x), expected)


@pytest.mark.parametrize("name", ["output_attentions", "output_hidden_states"])
def test_intermediate_outputs_are_rejected_when_active(install, name):
    adapter, _ = install(Decoder())
    with (
        pytest.raises(ValueError, match=name),
        adapter.forward_context(7, **{name: True}),
    ):
        pytest.fail("unsupported request entered the fused forward")


def test_cache_options_do_not_disable_local_fusion(install):
    model = Decoder()
    x = torch.randn(1, 7, 16)
    expected = model(x)
    adapter, backend = install(model)
    with adapter.forward_context(
        7, use_cache=True, past_key_value=object(), past_key_values=object()
    ):
        actual = model(x)
    torch.testing.assert_close(actual, expected)
    assert backend.calls == ["o", "down"] * 2
