"""Profile the Llama 3.1 8B TP4 projection shapes used by the XPU worker."""

import os

import torch
import torch.distributed as dist

from vllm.v1.worker.tpsp_profile import profile_sp_config, select_sp_config


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", device_id=torch.device("xpu", rank))
    try:
        for name, width, sharded in (
            ("o_proj", 1024, False),
            ("down_proj", 3584, True),
        ):
            profile = profile_sp_config(
                tp_size=4,
                hidden_size=4096,
                max_batched_tokens=65536,
                group_name=dist.group.WORLD.group_name,
                time_budget_s=240,
                input_widths=(width,),
                gather_sharded_residual=sharded,
            )
            assert profile.input_widths == (width,)
            assert profile.gather_sharded_residual == sharded
            assert profile.norm_eps == 1e-5
            assert not profile.gather_residual_after_native
            assert {mode for mode, _, _ in profile.mode_candidates} == {
                "p2p",
                "ordered",
                "independent",
            }
            assert {chunk for _, chunk, _ in profile.mode_candidates} == {4096}
            assert profile.candidates
            selected_mode = min(profile.mode_candidates, key=lambda item: item[2])[0]
            assert {mode for mode, _, _ in profile.candidates} == {selected_mode}
            assert all(chunk % 64 == 0 for _, chunk, _ in profile.candidates)
            assert len({chunk for _, chunk, _ in profile.finalists}) == 2
            assert {mode for mode, _, _ in profile.finalists} == {selected_mode}
            if profile.enabled:
                assert (profile.all_gather_mode, profile.microchunk_tokens) == min(
                    profile.finalists, key=lambda item: item[2]
                )[:2]
            assert select_sp_config(profile, 65536) == profile.enabled
            upper = next(m for m in profile.measurements if m.tokens == 65536)
            if rank == 0:
                print(
                    f"LLAMA_TP4_PROFILE projection={name} status={profile.status} "
                    f"threshold={profile.threshold_tokens} "
                    f"chunk={profile.microchunk_tokens} "
                    f"mode={profile.all_gather_mode} "
                    f"conventional_ms={upper.conventional_ms:.3f} "
                    f"sp_ms={upper.sp_ms:.3f} "
                    f"lower_benefit_ms={upper.lower_benefit_ms:.3f} "
                    f"reason={profile.reason}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
