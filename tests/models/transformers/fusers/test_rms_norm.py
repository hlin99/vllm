# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Transformers modeling backend's RMSNorm fuser."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.models.transformers.fuser import get_fuser
from vllm.model_executor.models.transformers.fusers import RMSNormFuser


class RMSNorm(nn.Module):
    """The canonical HF RMSNorm: `weight * x * rsqrt(mean(x**2) + eps)`."""

    def __init__(self, hidden: int = 16, eps: float = 1e-5, weight: bool = True):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.ones(hidden))
        self.variance_epsilon = eps

    def _rms(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)

    def forward(self, x):
        return self.weight * self._rms(x.to(torch.float32)).to(x.dtype)


class GemmaRMSNorm(RMSNorm):
    """Zero-centered weight: `(1 + weight) * normalized`."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__(hidden, eps)
        self.weight = nn.Parameter(torch.zeros(hidden))

    def forward(self, x):
        return (1.0 + self.weight) * self._rms(x.to(torch.float32)).to(x.dtype)


class WeightlessRMSNorm(RMSNorm):
    """No scale parameter (e.g. Gemma3n `with_scale=False`)."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__(hidden, eps, weight=False)

    def forward(self, x):
        return self._rms(x.to(torch.float32)).to(x.dtype)


class Gemma4RMSNorm(RMSNorm):
    """`pow(v, -0.5)` where the others write `rsqrt(v)` (HF `Gemma4RMSNorm._norm`).

    Gemma 4 spells the reciprocal square root as `torch.pow(..., -0.5)` to keep
    Torch and JAX in agreement, and -- unlike Gemma 1-3 -- its weight is *not*
    zero-centered, so this fuses to a plain `RMSNorm`.
    """

    def _rms(self, x):
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        return x * torch.pow(mean_squared, -0.5)

    def forward(self, x):
        return (self._rms(x.float()) * self.weight.float()).type_as(x)


class PowOperatorRMSNorm(Gemma4RMSNorm):
    """The `v ** -0.5` operator spelling of the same reciprocal square root."""

    def _rms(self, x):
        return x * (x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon) ** -0.5


class KwargPowRMSNorm(Gemma4RMSNorm):
    """The keyword spelling `torch.pow(v, exponent=-0.5)` of the same rsqrt."""

    def _rms(self, x):
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        return x * torch.pow(mean_squared, exponent=-0.5)


class KwargBasePowRMSNorm(Gemma4RMSNorm):
    """The fully functional spelling `torch.pow(input=v, exponent=-0.5)`."""

    def _rms(self, x):
        mean_squared = x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        return x * torch.pow(input=mean_squared, exponent=-0.5)


class WeightlessGemma4RMSNorm(Gemma4RMSNorm):
    """Gemma 4 with `with_scale=False`: the `pow` spelling and no scale parameter."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__(hidden, eps, weight=False)

    def forward(self, x):
        return self._rms(x.float()).type_as(x)


class LayerNorm(RMSNorm):
    """An RMSNorm not named `*RMSNorm`, keeping the input dtype (no upcast)."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__(hidden, eps)

    def forward(self, x):
        return self.weight * self._rms(x)


class NamedEpsRMSNorm(nn.Module):
    """Holds eps under `attr`, with no `variance_epsilon` to find instead."""

    attr = "eps"

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        setattr(self, self.attr, eps)

    def forward(self, x):
        eps = getattr(self, self.attr)
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, True) + eps))


class ToleranceRMSNorm(NamedEpsRMSNorm):
    """Names eps something other than `eps`, so only its value identifies it."""

    attr = "tolerance"


class LiteralEpsRMSNorm(RMSNorm):
    """eps is a literal in the source, and an unrelated attribute happens to
    hold the same value: matching on the value alone would bind to it."""

    def __init__(self, hidden: int = 16, eps: float = 1e-5):
        super().__init__(hidden, eps)
        self.epsilon_lookalike = 1e-4

    def forward(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, True) + 1e-4))


class AmbiguousEpsRMSNorm(nn.Module):
    """Two attributes hold the eps value, and the forward reads the second, so
    their order cannot pick it."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.decoy_eps = eps
        self.variance_epsilon = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, True)
        return self.weight * (x * torch.rsqrt(var + self.variance_epsilon))


class NotAnRMSNorm(RMSNorm):
    """Mean-subtracting LayerNorm-like math -> not an RMSNorm."""

    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__(hidden, eps)

    def forward(self, x):
        x = x - x.mean(-1, keepdim=True)
        variance = x.var(-1, keepdim=True)
        return self.weight * x / torch.sqrt(variance + self.variance_epsilon)


class GatedRMSNorm(RMSNorm):
    """Second input and tail compute -> not an RMSNorm."""

    def forward(self, x, gate=None):
        normed = self.weight * self._rms(x.to(torch.float32)).to(x.dtype)
        return normed * F.silu(gate)


class GatedFusedRMSNorm(nn.Module):
    """Same as GatedRMSNorm, but built on the fused `rms_norm` op -> not an RMSNorm."""

    def __init__(self, hidden: int = 16, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = eps

    def forward(self, x, gate=None):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps) * F.silu(gate)


class UntraceableGatedRMSNorm(RMSNorm):
    """Tracer can't see tail compute in forward, but still has a second input (gate)."""

    def forward(self, x, gate=None):
        normed = self.weight * self._rms(x.to(torch.float32)).to(x.dtype)
        if gate.sum() > 0:  # untraceable -> partial graph, no visible tail
            normed = normed * F.silu(gate)
        return normed


@pytest.mark.parametrize(
    "cls,eps,zero_centered",
    [
        (RMSNorm, 1e-5, False),
        (GemmaRMSNorm, 1e-6, True),
        (WeightlessRMSNorm, 1e-6, False),
        (Gemma4RMSNorm, 1e-6, False),
        (PowOperatorRMSNorm, 1e-6, False),
        (KwargPowRMSNorm, 1e-6, False),
        (KwargBasePowRMSNorm, 1e-6, False),
        (WeightlessGemma4RMSNorm, 1e-6, False),
        (LayerNorm, 1e-6, False),
        (torch.nn.RMSNorm, 1e-5, False),  # fused `F.rms_norm` op
    ],
)
def test_detects_rms_norm_variants(cls, eps, zero_centered):
    with torch.device("meta"):
        fuser = get_fuser(cls(16, eps=eps), RMSNormFuser)
    assert isinstance(fuser, RMSNormFuser)
    assert fuser.zero_centered == zero_centered


@pytest.mark.parametrize("cls", [NotAnRMSNorm, nn.LayerNorm, nn.SiLU])
def test_non_rms_norms_are_not_matched(cls):
    with torch.device("meta"):
        module = cls(16) if cls is nn.LayerNorm else cls()
    assert get_fuser(module, RMSNormFuser) is None


def test_tpsp_norm_type_accepts_known_norm_implementations(default_vllm_config):
    from vllm.model_executor.layers.layernorm import (
        LayerNorm as VllmLayerNorm,
    )
    from vllm.model_executor.layers.layernorm import (
        RMSNorm as VllmRMSNorm,
    )
    from vllm.model_executor.models.transformers.fusers.rms_norm import (
        TPAwareRMSNorm,
    )
    from vllm.model_executor.models.transformers.tpsp_runtime import (
        get_tpsp_norm_type,
    )

    with torch.device("meta"):
        norms = (
            (nn.LayerNorm(16), "layer_norm"),
            (VllmLayerNorm(16), "layer_norm"),
            (VllmRMSNorm(16), "rms_norm"),
            (TPAwareRMSNorm(16), "rms_norm"),
        )
    for norm, expected in norms:
        assert get_tpsp_norm_type(norm) == expected


def test_tpsp_norm_type_rejects_unsupported_implementations(default_vllm_config):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.models.transformers.tpsp_runtime import (
        UnsupportedTPSPNormError,
        get_tpsp_norm_type,
    )

    with torch.device("meta"):
        norms = (nn.RMSNorm(16), GemmaRMSNorm(16))
    for norm in norms:
        with pytest.raises(UnsupportedTPSPNormError):
            get_tpsp_norm_type(norm)


def test_cuda_tpsp_declares_supported_norm_types():
    from vllm.platforms.tpsp_cuda import CudaTPSPBackend

    assert CudaTPSPBackend.supported_norm_types == {"layer_norm", "rms_norm"}
    assert "rms_norm_offset_weight" not in CudaTPSPBackend.supported_norm_types


def test_tpsp_rejects_norm_types_outside_runtime_scope(default_vllm_config):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.models.transformers.tpsp_runtime import (
        UnsupportedTPSPNormError,
        get_tpsp_norm_type,
    )

    with torch.device("meta"):
        norm = GemmaRMSNorm(16)

    with pytest.raises(UnsupportedTPSPNormError):
        get_tpsp_norm_type(norm)


@pytest.mark.parametrize(
    "microchunk_tokens,threshold_tokens,expected_microchunk_per_rank,"
    "expected_threshold,default_calls",
    [
        (None, None, 64, 512, 1),
        (1000, None, 250, 512, 1),
        (None, 800, 64, 800, 1),
        (2000, 2000, 250, 1000, 0),
    ],
)
def test_tpsp_profile_resolves_missing_token_settings(
    monkeypatch,
    microchunk_tokens,
    threshold_tokens,
    expected_microchunk_per_rank,
    expected_threshold,
    default_calls,
):
    from vllm.distributed import parallel_state
    from vllm.model_executor.models.transformers.tpsp_runtime import (
        TPSPBackend,
        TPSPOpsGroup,
    )

    class Backend(TPSPBackend):
        supported_norm_types = frozenset({"layer_norm"})

        def __init__(self):
            super().__init__("tp", torch.device("cpu"))
            self.defaults_called = 0
            self.configs = []

        def get_default_profile_settings(self):
            self.defaults_called += 1
            return 256, 512

        def open(self, **kwargs):
            return object()

        def set_config(self, handle, config):
            self.configs.append(config)

        def fused_gemm_rs_norm_ag(self, *args, **kwargs):
            raise NotImplementedError

        def close(self, context=None):
            pass

    monkeypatch.setattr(
        parallel_state,
        "get_tp_group",
        lambda: SimpleNamespace(
            world_size=4, device_group=SimpleNamespace(group_name="tp")
        ),
    )
    backend = Backend()
    with torch.device("meta"):
        ops_group = TPSPOpsGroup("proj", nn.Linear(8, 8, bias=False), nn.LayerNorm(8))

    context = backend.profile(
        [ops_group],
        1000,
        microchunk_tokens=microchunk_tokens,
        threshold_tokens=threshold_tokens,
    )

    assert context is not None
    assert context.profile.threshold_tokens == expected_threshold
    assert backend.configs == [expected_microchunk_per_rank]
    assert backend.defaults_called == default_calls


def test_tpsp_microchunk_tokens_convert_to_per_rank_with_ceiling():
    from vllm.model_executor.models.transformers.tpsp_runtime import (
        _microchunk_tokens_per_rank,
    )
    from vllm.platforms.tpsp_cuda import CudaTPSPBackend

    backend = CudaTPSPBackend("test", torch.device("cuda"))

    assert _microchunk_tokens_per_rank(backend, 65, 4) == 17


def test_cuda_tpsp_profiles_when_a_token_setting_is_missing(monkeypatch):
    from vllm.distributed import parallel_state
    from vllm.platforms import tpsp_cuda
    from vllm.platforms.tpsp_cuda import CudaTPSPBackend

    backend = CudaTPSPBackend("tp", torch.device("cpu"))
    result = object()
    calls = []

    def profile(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(tpsp_cuda, "profile_tpsp", profile)
    monkeypatch.setattr(
        parallel_state, "get_tp_group", lambda: SimpleNamespace(world_size=4)
    )

    assert (
        backend.profile(
            [],
            1000,
            microchunk_tokens=100,
            threshold_tokens=None,
        )
        is result
    )
    assert len(calls) == 1
    assert calls[0][1]["microchunk_tokens"] == 100
    assert calls[0][1]["threshold_tokens"] is None


def test_cuda_tpsp_uses_fixed_settings_without_scanning(monkeypatch):
    from vllm.model_executor.models.transformers import tpsp_runtime
    from vllm.platforms import tpsp_cuda
    from vllm.platforms.tpsp_cuda import CudaTPSPBackend

    backend = CudaTPSPBackend("tp", torch.device("cpu"))
    result = object()
    monkeypatch.setattr(
        tpsp_runtime,
        "_build_static_tpsp_context",
        lambda *args, **kwargs: result,
    )
    monkeypatch.setattr(
        tpsp_cuda,
        "profile_tpsp",
        lambda *args, **kwargs: pytest.fail("fixed settings must not scan"),
    )

    assert (
        backend.profile(
            [],
            1000,
            microchunk_tokens=100,
            threshold_tokens=200,
        )
        is result
    )


@pytest.mark.parametrize(
    "cls", [GatedRMSNorm, GatedFusedRMSNorm, UntraceableGatedRMSNorm]
)
def test_gated_rms_norm_is_not_fused(cls):
    with torch.device("meta"):
        assert get_fuser(cls(), RMSNormFuser) is None


@pytest.mark.parametrize(
    "cls,expected,zero_centered",
    [
        (RMSNorm, "RMSNorm", False),
        (GemmaRMSNorm, "GemmaRMSNorm", True),
        (WeightlessRMSNorm, "RMSNorm", False),
        (Gemma4RMSNorm, "RMSNorm", False),
        (KwargPowRMSNorm, "RMSNorm", False),
        (KwargBasePowRMSNorm, "RMSNorm", False),
        (WeightlessGemma4RMSNorm, "RMSNorm", False),
    ],
)
def test_rms_norm_builds_vllm_class(cls, expected, zero_centered, default_vllm_config):
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm as VLLMGemmaRMSNorm
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm

    with torch.device("meta"):
        module = cls()
        fuser = get_fuser(module, RMSNormFuser)
        built = fuser.fuse(module, "norm", default_vllm_config)
    from vllm.model_executor.models.transformers.fusers.rms_norm import (
        TPAwareNormMixin,
    )

    types_by_name = {"RMSNorm": VLLMRMSNorm, "GemmaRMSNorm": VLLMGemmaRMSNorm}
    assert isinstance(built, types_by_name[expected])
    assert isinstance(built, TPAwareNormMixin)
    assert built.variance_epsilon == module.variance_epsilon
    # The weight states the hidden size (and is what the TP check reads); a norm
    # without one has none to state.
    weight = getattr(module, "weight", None)
    assert isinstance(built.weight, nn.Parameter) == (weight is not None)
    assert built.weight.shape[0] == (weight.size(0) if weight is not None else 0)


@pytest.mark.parametrize(
    "cls", [Gemma4RMSNorm, PowOperatorRMSNorm, KwargPowRMSNorm, KwargBasePowRMSNorm]
)
def test_pow_spelled_norm_fuses_to_equivalent_math(cls, default_vllm_config):
    """Matching `pow(v, -0.5)` is only sound if it really is the reciprocal square
    root: the fused norm must reproduce the unfused forward, not just replace it.
    `pow` and `rsqrt` take different paths to the same value, so compare at float32
    tolerance rather than bit-for-bit."""
    torch.manual_seed(0)
    module = cls(16)
    with torch.no_grad():
        module.weight.copy_(torch.randn(16))
    built = get_fuser(module, RMSNormFuser).fuse(module, "norm", default_vllm_config)
    with torch.no_grad():
        built.weight.copy_(module.weight)
    x = torch.randn(4, 16)
    torch.testing.assert_close(built(x), module(x), rtol=1e-5, atol=1e-6)


def test_weightless_norm_has_no_hidden_size(default_vllm_config):
    """A weightless norm states no hidden size, and the model's (LM) hidden size
    would be wrong for one on a sub-dimension: Llama 4's `qk_norm` normalizes
    head_dim, so sizing it at hidden_size made the TP gather reject its input.
    It fuses with none, and so normalizes the width it is given at any TP size,
    matching the unfused norm and vLLM's native `RMSNorm(hidden_size=head_dim)`.
    """
    module = WeightlessRMSNorm(72)
    built = get_fuser(module, RMSNormFuser).fuse(module, "norm", default_vllm_config)
    assert built.hidden_size == 0

    built.tp_size = 2  # emulate TP=2 without a real process group
    x = torch.randn(4, 72)
    torch.testing.assert_close(built(x), module(x))


def test_fused_rms_norm_op_default_eps(default_vllm_config):
    """`torch.nn.RMSNorm` (a single `F.rms_norm` call) matches via the fast path;
    its default `eps=None` resolves to `finfo(dtype).eps` in `fuse`."""
    from vllm.model_executor.layers.layernorm import RMSNorm as VLLMRMSNorm

    with torch.device("meta"):
        module = torch.nn.RMSNorm(16)  # forward is a single `F.rms_norm` call
        fuser = get_fuser(module, RMSNormFuser)
        assert isinstance(fuser, RMSNormFuser)
        assert not fuser.zero_centered
        vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float32))
        built = fuser.fuse(module, "norm", vllm_config)
    assert isinstance(built, VLLMRMSNorm)
    assert built.variance_epsilon == torch.finfo(torch.float32).eps


def test_eps_is_derived_per_instance(default_vllm_config):
    """Two instances of the same norm class with different eps must fuse to their
    own eps: the type-cached fuser holds only structure, not this value."""
    with torch.device("meta"):
        for eps in (1e-5, 1e-6):
            module = RMSNorm(16, eps=eps)
            fuser = get_fuser(module, RMSNormFuser)
            built = fuser.fuse(module, "norm", default_vllm_config)
            assert built.variance_epsilon == eps


@pytest.mark.parametrize("cls", [NamedEpsRMSNorm, ToleranceRMSNorm])
def test_eps_attr_is_found_by_value_not_name(cls, default_vllm_config):
    """The eps attribute is identified by holding the traced value, so a norm
    stays per-instance correct whatever it names it."""
    with torch.device("meta"):
        for eps in (1e-5, 1e-6):
            module = cls(16, eps=eps)
            fuser = get_fuser(module, RMSNormFuser)
            assert fuser.eps_attr == cls.attr
            built = fuser.fuse(module, "norm", default_vllm_config)
            assert built.variance_epsilon == eps


def test_literal_eps_is_not_mistaken_for_an_attribute(default_vllm_config, caplog):
    """A literal eps is recognised as coming from no attribute, even when one
    holds the same value, and is taken from the traced source instead."""
    logger = "vllm.model_executor.models.transformers.fusers.rms_norm"
    with caplog.at_level("DEBUG", logger=logger), torch.device("meta"):
        module = LiteralEpsRMSNorm()
        fuser = get_fuser(module, RMSNormFuser)
        built = fuser.fuse(module, "norm", default_vllm_config)
    assert fuser.eps_attr is None
    assert built.variance_epsilon == 1e-4
    assert "does not hold its eps" in caplog.text


def test_ambiguous_eps_attrs_are_disambiguated(default_vllm_config):
    """When several attributes hold the eps value, the one the forward actually
    reads is identified, and they are all left as they were found."""
    with torch.device("meta"):
        module = AmbiguousEpsRMSNorm(16, eps=1e-6)
        before = dict(vars(module))
        fuser = get_fuser(module, RMSNormFuser)
        assert fuser.eps_attr == "variance_epsilon"
        assert vars(module) == before

        other = AmbiguousEpsRMSNorm(16, eps=1e-5)
        other.decoy_eps = 1e-3
        built = fuser.fuse(other, "norm", default_vllm_config)
    assert built.variance_epsilon == 1e-5


def test_fused_norm_is_gather_capable(default_vllm_config):
    """Every weighted fused norm is emitted gather-capable, so a norm on a head-sharded
    projection (OLMoE-style) self-corrects at runtime with no QKV-specific
    plumbing. A full-width input skips the gather and equals a plain norm."""
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
    from vllm.model_executor.models.transformers.fusers import rms_norm

    torch.manual_seed(0)
    x = torch.randn(4, 16)
    for gathered_cls, plain_cls in [
        (rms_norm.TPAwareRMSNorm, RMSNorm),
        (rms_norm.TPAwareGemmaRMSNorm, GemmaRMSNorm),
    ]:
        gathered = gathered_cls(hidden_size=16, eps=1e-6)
        assert isinstance(gathered, rms_norm.TPAwareNormMixin)
        plain = plain_cls(hidden_size=16, eps=1e-6)
        with torch.no_grad():
            weight = torch.randn(16)
            gathered.weight.copy_(weight)
            plain.weight.copy_(weight)
        torch.testing.assert_close(gathered(x), plain(x))


def test_gathered_norm_rejects_uneven_sharding(default_vllm_config):
    """A sharded input (narrower than the full-width weight) that does not tile
    the weight evenly across ranks is rejected before any collective."""
    from vllm.model_executor.models.transformers.fusers import rms_norm

    norm = rms_norm.TPAwareRMSNorm(hidden_size=8, eps=1e-6)
    norm.tp_size = 2  # emulate TP=2 without a real process group
    with pytest.raises(ValueError, match="does not tile it evenly"):
        norm(torch.randn(2, 3))  # 3 * 2 != 8
