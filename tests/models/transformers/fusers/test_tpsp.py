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

    def __init__(self):
        self.calls = []

    def fused_gemm_rs_norm_ag(
        self, handle, x, projection, residual, norm, *, norm_type
    ):
        self.calls.append(handle)
        assert norm_type == "layer_norm"
        residual = F.linear(x, projection.weight, projection.bias) + residual
        hidden = F.layer_norm(
            residual, norm.normalized_shape, norm.weight, norm.bias, norm.eps
        )
        return hidden, residual


@pytest.fixture
def config():
    return SimpleNamespace(
        quant_config=None,
        lora_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
        model_config=SimpleNamespace(tpsp_microchunk_tokens=4, tpsp_threshold_tokens=4),
    )


@pytest.fixture
def install(config, monkeypatch):
    # The adapter is independent of TP transport; emulate a one-rank shard here.
    monkeypatch.setattr(tpsp_transformers_adapter, "tpsp_shard_residual", torch.clone)

    def install(model):
        adapter = TransformersDenseTPSP(model, config)
        backend = ReferenceBackend()
        adapter.context = TPSPContext(
            TPSPProfile(True, 4, 32), backend, {"o_proj": "o", "down_proj": "down"}
        )
        return adapter, backend

    return install


@pytest.mark.parametrize("tokens", [2, 7, 8])
def test_original_computation_survives_outside_fused_regions(install, tokens):
    torch.manual_seed(0)
    model = Decoder()
    x = torch.randn(1, tokens, 16)
    expected = model(x, scale=2.3)
    adapter, backend = install(model)
    for _ in range(2):
        with adapter.forward_context(tokens):
            actual = model(x, scale=2.3)
        torch.testing.assert_close(actual, expected)
    assert backend.calls == (["o", "down"] * 4 if tokens >= 4 else [])
    assert [layer.calls for layer in model.layers] == [3, 3]
    torch.testing.assert_close(model(x, scale=2.3), expected)


@pytest.mark.parametrize("architecture", ["llama", "qwen3"])
def test_current_hf_forward_is_preserved(install, architecture):
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
        adapter, backend = install(model)
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
