"""Profile the Llama 3.1 8B TP4 projection shapes used by the XPU worker."""

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
            hidden_size=4096,
            max_batched_tokens=65536,
            group_name=dist.group.WORLD.group_name,
            time_budget_s=240,
            input_widths=(1024, 3584),
        )
        assert profile.input_widths == (1024, 3584)
        assert profile.norm_eps == 1e-5
        assert not profile.gather_residual_after_native
        assert profile.candidates
        assert {mode for mode, _, _ in profile.candidates} == {
            "p2p",
            "ordered",
            "independent",
        }
        assert all(chunk % 64 == 0 for _, chunk, _ in profile.candidates)
        upper = next(m for m in profile.measurements if m.tokens == 65536)
        if rank == 0:
            best = min(profile.candidates, key=lambda item: item[2])
            print(
                f"LLAMA_TP4_PROFILE status={profile.status} "
                f"threshold={profile.threshold_tokens} "
                f"chunk={profile.microchunk_tokens} "
                f"mode={profile.all_gather_mode} "
                f"conventional_ms={upper.conventional_ms:.3f} "
                f"sp_ms={upper.sp_ms:.3f} "
                f"lower_benefit_ms={upper.lower_benefit_ms:.3f} "
                f"screened_best={best} reason={profile.reason}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
