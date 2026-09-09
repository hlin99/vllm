# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for XPU batch-invariant-aware attention backend selection."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

# XPU-specific attention backend selection tests
pytestmark = pytest.mark.skipif(
    not current_platform.is_xpu(), reason="XPU-specific tests"
)


def _make_config(**overrides) -> AttentionSelectorConfig:
    defaults = dict(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype=None,
        block_size=16,
    )
    defaults.update(overrides)
    return AttentionSelectorConfig(**defaults)


def test_batch_invariant_defaults_to_triton_attn():
    """With no explicit backend requested, batch invariance should route to
    Triton Attention rather than Flash Attention, since only Triton Attention
    has been validated for batch-invariant kernels on XPU."""
    from vllm.platforms.xpu import XPUPlatform

    config = _make_config(use_batch_invariant=True)
    backend_path = XPUPlatform.get_attn_backend_cls(
        selected_backend=None,
        attn_selector_config=config,
    )
    assert backend_path == AttentionBackendEnum.TRITON_ATTN.get_path()


def test_batch_invariant_honors_explicit_flash_attn_request():
    """An explicit Flash Attention request should still be honored (with a
    warning), matching the existing mm_prefix/float32 fallback pattern."""
    from vllm.platforms.xpu import XPUPlatform

    config = _make_config(use_batch_invariant=True)
    backend_path = XPUPlatform.get_attn_backend_cls(
        selected_backend=AttentionBackendEnum.FLASH_ATTN,
        attn_selector_config=config,
    )
    assert backend_path == AttentionBackendEnum.FLASH_ATTN.get_path()


def test_no_batch_invariant_defaults_to_flash_attn():
    """Without batch invariance, the default XPU backend remains unchanged."""
    from vllm.platforms.xpu import XPUPlatform

    config = _make_config(use_batch_invariant=False)
    backend_path = XPUPlatform.get_attn_backend_cls(
        selected_backend=None,
        attn_selector_config=config,
    )
    assert backend_path == AttentionBackendEnum.FLASH_ATTN.get_path()
