"""Profile the DeepSeek-V3 TP4 dense projection shapes (hidden size 7168)."""

import os

import torch
import torch.distributed as dist

from vllm.v1.worker.tpsp_profile import profile_sp_config


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.xpu.set_device(rank)
    dist.init_process_group("xccl", device_id=torch.device("xpu", rank))
    try:
        for name, width, sharded in (
            ("o_proj", 4096, False),
            ("dense_down_proj", 4608, True),
        ):
            profile = profile_sp_config(
                tp_size=4,
                hidden_size=7168,
                max_batched_tokens=32768,
                group_name=dist.group.WORLD.group_name,
                time_budget_s=240,
                input_widths=(width,),
                gather_sharded_residual=sharded,
            )
            assert profile.input_widths == (width,)
            assert profile.gather_sharded_residual == sharded
            assert {mode for mode, _, _ in profile.mode_candidates} == {
                "p2p",
                "ordered",
                "independent",
            }
            assert {chunk for _, chunk, _ in profile.mode_candidates} == {4096}
            assert profile.candidates
            assert profile.finalists
            assert profile.enabled, profile.reason
            upper = next(m for m in profile.measurements if m.tokens == 32768)
            if rank == 0:
                print(
                    f"DEEPSEEK_V3_TP4_PROFILE projection={name} "
                    f"status={profile.status} threshold={profile.threshold_tokens} "
                    f"chunk={profile.microchunk_tokens} "
                    f"mode={profile.all_gather_mode} "
                    f"conventional_ms={upper.conventional_ms:.3f} "
                    f"sp_ms={upper.sp_ms:.3f} "
                    f"lower_benefit_ms={upper.lower_benefit_ms:.3f}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
