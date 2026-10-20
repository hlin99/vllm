# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Local GDN kernels and managed states for otherwise HF-driven mixers."""

import ast
from copy import copy
from functools import cache
from types import MethodType
from typing import Literal

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    ChunkGatedDeltaRule,
)
from vllm.model_executor.layers.mamba.mamba_mixer2 import mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.transformers.fusers.merged_column import (
    MergedColumnParallelFuser,
)
from vllm.model_executor.models.transformers.fx_utils import (
    compile_forward,
    recover_forward,
    single_self_call,
)
from vllm.model_executor.models.transformers.utils import replace_linear_class
from vllm.third_party.flash_linear_attention.ops import fused_recurrent_gated_delta_rule
from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
    fused_qkv_l2norm,
)
from vllm.transformers_utils.hybrid import GDNGeometry, StateOperator
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

logger = init_logger(__name__)


def gdn_conv_weight_loader(sizes: list[int], tp: int, rank: int):
    combined = mamba_v2_sharded_weight_loader(
        [(size, 0, False) for size in sizes], tp, rank
    )
    separate = [
        mamba_v2_sharded_weight_loader([(size, 0, False)], tp, rank) for size in sizes
    ]

    def load(param, weight, shard_id=None):
        shard_id = getattr(weight, "shard_id", shard_id)
        if shard_id is None:
            combined(param, weight)
        else:
            if shard_id not in range(len(sizes)):
                raise ValueError(f"Invalid GDN convolution shard: {shard_id}.")
            target = param.narrow(0, sum(sizes[:shard_id]) // tp, sizes[shard_id] // tp)
            separate[shard_id](target, weight)

    return load


@cache
def state_attention_backend(
    prefill_backend: Literal["triton", "flashinfer", "cutedsl", "aiter_flydsl"],
) -> type[GDNAttentionBackend]:
    class Builder(GDNAttentionMetadataBuilder):
        def __init__(
            self,
            kv_cache_spec: MambaSpec,
            layer_names: list[str],
            vllm_config: VllmConfig,
            device: torch.device,
        ):
            local_config = copy(vllm_config)
            local_config.model_config = copy(vllm_config.model_config)
            local_config.model_config.hf_text_config = copy(
                vllm_config.model_config.hf_text_config
            )
            _, head_v_dim, head_k_dim = kv_cache_spec.shapes[1]
            local_config.model_config.hf_text_config.linear_key_head_dim = head_k_dim
            local_config.model_config.hf_text_config.linear_value_head_dim = head_v_dim
            additional_config = vllm_config.additional_config
            local_config.additional_config = {
                **(additional_config if isinstance(additional_config, dict) else {}),
                "gdn_prefill_backend": prefill_backend,
            }
            super().__init__(kv_cache_spec, layer_names, local_config, device)

    class Backend(GDNAttentionBackend):
        @staticmethod
        def get_builder_cls():
            return Builder

    return Backend


class GDNKernel(nn.Module, MambaBase):
    """Own only convolution/recurrence kernels and their vLLM-managed states."""

    def __init__(self, geometry: GDNGeometry, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "HF-driven GDN kernels do not support speculative decoding."
            )
        self.prefix = prefix
        self.spec = geometry.state_spec(prefix, vllm_config)
        self.chunk = ChunkGatedDeltaRule(geometry.config())
        self.kv_cache = ()
        layers = vllm_config.compilation_config.static_forward_context
        if prefix in layers:
            raise ValueError(f"Duplicate layer name: {prefix}")
        layers[prefix] = self

    @property
    def mamba_type(self):
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_shape(self):
        return self.spec.shapes

    def get_state_dtype(self):
        return self.spec.dtypes

    def get_attn_backend(self):
        return state_attention_backend(self.chunk.gdn_prefill_backend)

    def convolution(self, x, weight, bias=None, activation=None, **kwargs):
        if x.ndim != 3 or x.shape[0] != 1:
            raise ValueError("GDN convolution requires packed [1, channels, tokens].")
        x = x.squeeze(0).transpose(0, 1)
        out = torch.zeros_like(x)
        torch.ops.vllm.transformers_gdn_conv(
            x, weight, bias, out, activation, self.prefix
        )
        return out.transpose(0, 1).unsqueeze(0)

    def delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        cu_seqlens=None,
        **kwargs,
    ):
        if initial_state is not None or output_final_state or cu_seqlens is not None:
            raise ValueError(
                "GDN requires vLLM-managed states and sequence boundaries."
            )
        if not use_qk_l2norm_in_kernel:
            raise ValueError("HF GDN must request kernel Q/K normalization.")
        out = torch.zeros_like(v)
        torch.ops.vllm.transformers_gdn_scan(q, k, v, g, beta, out, self.prefix)
        return out, None

    def _metadata(self):
        all_metadata = get_forward_context().attn_metadata
        if all_metadata is None:
            return None
        assert isinstance(all_metadata, dict)
        metadata = all_metadata[self.prefix]
        assert isinstance(metadata, GDNAttentionMetadata)
        if metadata.num_spec_decodes or metadata.checkpoint is not None:
            raise ValueError(
                "HF-driven GDN does not support speculative/internal checkpoint states."
            )
        return metadata

    def conv_forward(self, x, weight, bias, out, activation):
        metadata = self._metadata()
        if metadata is None:
            return
        n = metadata.num_actual_tokens
        state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            state = state.transpose(-1, -2)
        if metadata.num_prefills:
            causal_conv1d_fn(
                x[:n].transpose(0, 1),
                weight,
                bias,
                conv_states=state,
                query_start_loc=metadata.non_spec_query_start_loc,
                cache_indices=metadata.non_spec_state_indices_tensor,
                has_initial_state=metadata.has_initial_state,
                activation=activation,
                metadata=metadata,
                out=out[:n].transpose(0, 1),
            )
        else:
            causal_conv1d_update(
                x[:n],
                state,
                weight,
                bias,
                activation=activation,
                conv_state_indices=metadata.non_spec_state_indices_tensor[:n],
                validate_data=True,
                out=out[:n],
            )

    def scan_forward(self, q, k, v, g, beta, out):
        metadata = self._metadata()
        if metadata is None:
            # Warm the prefill primitive before allocating the state cache.
            state = v.new_zeros((1, *self.spec.shapes[1]), dtype=self.spec.dtypes[1])
            q, k, v = fused_qkv_l2norm(q[:, :1], k[:, :1], v[:, :1])
            self.chunk(
                q,
                k,
                v,
                g[:, :1],
                beta[:, :1],
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
            return
        state = self.kv_cache[1]
        n_decode = metadata.num_decode_tokens
        if metadata.num_decodes:
            fused_recurrent_gated_delta_rule(
                q[:, :n_decode],
                k[:, :n_decode],
                v[:, :n_decode],
                g[:, :n_decode],
                beta[:, :n_decode],
                initial_state=state,
                cu_seqlens=metadata.non_spec_query_start_loc[
                    : metadata.num_decodes + 1
                ],
                ssm_state_indices=metadata.non_spec_state_indices_tensor[
                    : metadata.num_decodes
                ],
                use_qk_l2norm_in_kernel=True,
                out=out[:, :n_decode],
            )
        if metadata.num_prefills:
            indices = metadata.prefill_state_indices
            initial = state[indices]
            initial.masked_fill_(
                ~metadata.prefill_has_initial_state[:, None, None, None], 0
            )
            n = metadata.num_actual_tokens
            q, k, v = fused_qkv_l2norm(
                q[:, n_decode:n], k[:, n_decode:n], v[:, n_decode:n]
            )
            _, final = self.chunk(
                q,
                k,
                v,
                g[:, n_decode:n],
                beta[:, n_decode:n],
                initial_state=initial,
                output_final_state=True,
                cu_seqlens=metadata.prefill_query_start_loc,
                chunk_indices=metadata.chunk_indices,
                chunk_offsets=metadata.chunk_offsets,
                use_qk_l2norm_in_kernel=False,
                aiter_prefill_metadata=metadata.aiter_prefill_metadata,
                core_attn_out=out[:, n_decode:n].reshape(-1),
            )
            state[indices] = final.to(state.dtype)


def transformers_gdn_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    activation: str | None,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.conv_forward(x, weight, bias, out, activation)


def transformers_gdn_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer.scan_forward(q, k, v, g, beta, out)


direct_register_custom_op(
    op_name="transformers_gdn_conv",
    op_func=transformers_gdn_conv,
    mutates_args=["out"],
)
direct_register_custom_op(
    op_name="transformers_gdn_scan",
    op_func=transformers_gdn_scan,
    mutates_args=["out"],
)


def adapt_gdn(
    module: nn.Module, operator: StateOperator, vllm_config: VllmConfig
) -> tuple[MergedColumnParallelFuser | None, GDNKernel]:
    """Fuse projections and bind local kernel interfaces, not a native mixer."""
    geometry = operator.geometry
    if not isinstance(geometry, GDNGeometry):
        raise ValueError(f"Unsupported state geometry at {operator.path}.")
    prefix = f"model.{operator.path}"
    sizes = [geometry.num_k_heads * geometry.head_k_dim] * 2 + [
        geometry.num_v_heads * geometry.head_v_dim
    ]
    funcdef, _ = recover_forward(type(module))
    input_name = funcdef.args.args[1].arg
    projections = []
    packed = []
    outputs = []
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            call = single_self_call(funcdef, name)
            if isinstance(call.args[0], ast.Name) and call.args[0].id == input_name:
                if child.out_features == sum(sizes):
                    packed.append(name)
                else:
                    projections.append(name)
            else:
                outputs.append(name)
    if len(outputs) != 1:
        raise ValueError(f"GDN {prefix}: expected one output projection.")
    candidate = MergedColumnParallelFuser(
        source_cls=type(module).__name__, linear_names=tuple(projections)
    )
    if not candidate.validate(module, vllm_config):
        raise ValueError(f"GDN {prefix}: unsupported projection fusion.")
    fuser = None
    try:
        funcdef, fn = candidate.rewrite_forward(module)
        fuser = candidate
    except ValueError as exc:
        logger.info(
            "GDN %s: leaving projections unfused (%s); using column-parallel linears.",
            prefix,
            exc,
        )
        funcdef, fn = recover_forward(type(module))
    replacements = {
        "causal_conv1d_fn": "convolution",
        "torch_chunk_gated_delta_rule": "delta_rule",
        "torch_recurrent_gated_delta_rule": "delta_rule",
    }
    found = set()
    for node in ast.walk(funcdef):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in replacements
        ):
            found.add(node.func.id)
            node.func = ast.Attribute(
                value=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="_vllm_gdn",
                    ctx=ast.Load(),
                ),
                attr=replacements[node.func.id],
                ctx=ast.Load(),
            )
    if found != replacements.keys():
        raise ValueError(f"GDN {prefix}: unsupported HF kernel call sites.")
    forward = compile_forward(funcdef, fn)
    kernel = GDNKernel(geometry, vllm_config, f"{prefix}._vllm_gdn")
    if fuser is not None:
        fuser.fused_forward = forward
        fuser.fuse(module, prefix, vllm_config)
    else:
        for name in projections:
            setattr(
                module,
                name,
                replace_linear_class(
                    getattr(module, name),
                    "colwise",
                    vllm_config.quant_config,
                    prefix=f"{prefix}.{name}",
                ),
            )
        module.forward = MethodType(forward, module)
    for name in packed:
        linear = getattr(module, name)
        setattr(
            module,
            name,
            MergedColumnParallelLinear(
                input_size=linear.in_features,
                output_sizes=sizes,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.{name}",
                return_bias=False,
            ),
        )
    output = outputs[0]
    setattr(
        module,
        output,
        replace_linear_class(
            getattr(module, output),
            "rowwise",
            vllm_config.quant_config,
            prefix=f"{prefix}.{output}",
        ),
    )
    tp = vllm_config.parallel_config.tensor_parallel_size
    rank = get_tensor_model_parallel_rank()
    device = vllm_config.device_config.device
    conv = module.conv1d
    conv.weight = nn.Parameter(
        torch.empty(
            sum(sizes) // tp,
            1,
            geometry.conv_kernel_size,
            dtype=conv.weight.dtype,
            device=device,
        ),
        requires_grad=False,
    )
    conv.weight.weight_loader = gdn_conv_weight_loader(sizes, tp, rank)
    conv.in_channels = conv.out_channels = conv.groups = sum(sizes) // tp
    for name in ("A_log", "dt_bias"):
        param = nn.Parameter(
            torch.empty(geometry.num_v_heads // tp, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        param.weight_loader = sharded_weight_loader(0)
        setattr(module, name, param)
    for name in ("key_dim", "value_dim", "conv_dim", "num_k_heads", "num_v_heads"):
        setattr(module, name, getattr(module, name) // tp)
    module._vllm_gdn = kernel
    return fuser, kernel
