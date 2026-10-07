"""DeepSeek-V3 dummy-model startup scanner; inference remains native vLLM."""

from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MLP,
    DeepseekV3ForCausalLM,
)
from vllm.v1.worker.tpsp_profile import SPProfile, profile_sp_config


class DeepseekV3ProfileOnlyForCausalLM(DeepseekV3ForCausalLM):
    def profile_tpsp_config(self, max_num_batched_tokens: int) -> None:
        if hasattr(self, "tpsp_profiles"):
            return
        if self.quant_config is not None:
            raise RuntimeError(
                "DeepSeek-V3 TPSP shape scan requires BF16 dummy weights"
            )
        if get_pp_group().world_size != 1:
            raise RuntimeError("DeepSeek-V3 TPSP shape scan requires PP=1")

        group = get_tp_group()
        attention_widths = {
            layer.self_attn.o_proj.input_size_per_partition
            for layer in self.model.layers
        }
        dense_widths = {
            layer.mlp.down_proj.input_size_per_partition
            for layer in self.model.layers
            if isinstance(layer.mlp, DeepseekV2MLP)
        }
        if len(attention_widths) != 1 or len(dense_widths) != 1:
            raise RuntimeError("DeepSeek-V3 projection widths differ across layers")

        profiles: dict[str, SPProfile] = {}
        for name, widths, sharded in (
            ("o_proj", attention_widths, False),
            ("dense_down_proj", dense_widths, True),
        ):
            width = widths.pop()
            profile = profile_sp_config(
                tp_size=group.world_size,
                hidden_size=self.config.hidden_size,
                max_batched_tokens=max_num_batched_tokens,
                group_name=group.device_group.group_name,
                time_budget_s=240,
                input_widths=(width,),
                norm_eps=self.config.rms_norm_eps,
                gather_sharded_residual=sharded,
            )
            profiles[name] = profile
            if group.rank_in_group == 0:
                print(
                    f"DEEPSEEK_V3_DUMMY_STARTUP_PROFILE projection={name} "
                    f"hidden={profile.hidden_size} width={width} "
                    f"status={profile.status} mode={profile.all_gather_mode} "
                    f"chunk={profile.microchunk_tokens} "
                    f"threshold={profile.threshold_tokens}",
                    flush=True,
                )
        self.tpsp_profiles = profiles
