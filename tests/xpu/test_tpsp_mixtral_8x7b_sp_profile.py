"""Profile the Mixtral-8x7B TP4 attention projection and MoE residual shape."""

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
            max_batched_tokens=32768,
            group_name=dist.group.WORLD.group_name,
            time_budget_s=240,
            input_widths=(1024,),
            norm_eps=1e-5,
            gather_residual_after_native=True,
        )
        assert profile.input_widths == (1024,)
        assert profile.hidden_size == 4096
        assert profile.gather_residual_after_native
        assert {mode for mode, _, _ in profile.mode_candidates} == {
            "p2p",
            "ordered",
            "independent",
        }
        assert profile.candidates
        assert profile.finalists
        best_mode, best_chunk, best_ms = min(
            profile.finalists, key=lambda item: item[2]
        )
        upper = next(m for m in profile.measurements if m.tokens == 32768)
        if rank == 0:
            print(
                f"MIXTRAL_8X7B_TP4_PROFILE status={profile.status} "
                f"reason={profile.reason!r} threshold={profile.threshold_tokens} "
                f"chunk={profile.microchunk_tokens} mode={profile.all_gather_mode} "
                f"best_mode={best_mode} best_chunk={best_chunk} "
                f"best_chunk_ms={best_ms:.3f} "
                f"conventional_ms={upper.conventional_ms:.3f} "
                f"sp_ms={upper.sp_ms:.3f} "
                f"lower_benefit_ms={upper.lower_benefit_ms:.3f}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
