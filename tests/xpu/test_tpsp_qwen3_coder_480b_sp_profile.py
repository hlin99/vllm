"""Profile the Qwen3-Coder-480B-A35B TP4 attention projection shape."""

import os

import torch
import torch.distributed as dist

from vllm.v1.worker.tpsp_profile import profile_sp_config


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", device_id=torch.device("xpu", rank))
    try:
        profile = profile_sp_config(
            tp_size=4,
            hidden_size=6144,
            max_batched_tokens=65536,
            group_name=dist.group.WORLD.group_name,
            time_budget_s=360,
            input_widths=(3072,),
            norm_eps=1e-6,
            gather_residual_after_native=True,
            check_rtol=0.02,
            check_atol=0.05,
        )
        assert profile.input_widths == (3072,)
        assert profile.gather_residual_after_native
        assert profile.status in ("enabled", "disabled"), profile.reason
        assert profile.candidates
        assert profile.finalists
        upper = next(m for m in profile.measurements if m.tokens == 65536)
        if rank == 0:
            print(
                f"QWEN3_CODER_480B_TP4_PROFILE status={profile.status} "
                f"reason={profile.reason!r} threshold={profile.threshold_tokens} "
                f"chunk={profile.microchunk_tokens} mode={profile.all_gather_mode} "
                f"conventional_ms={upper.conventional_ms:.3f} "
                f"sp_ms={upper.sp_ms:.3f} "
                f"lower_benefit_ms={upper.lower_benefit_ms:.3f}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
