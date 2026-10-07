from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.v1.worker import cpu_worker, gpu_worker


@pytest.mark.parametrize("supported", [False, True])
def test_gpu_scan_precedes_memory_profile(supported: bool):
    calls = []
    model = SimpleNamespace()
    if supported:
        model.profile_tpsp_config = lambda tokens: calls.append(("scan", tokens))
    runner = SimpleNamespace(
        model=model,
        max_num_tokens=32,
        profile_run=lambda **kwargs: calls.append(("memory", kwargs)),
    )
    worker = SimpleNamespace(
        model_runner=runner,
        cache_config=SimpleNamespace(kv_cache_memory_bytes=1024),
        vllm_config=object(),
        randomize_dummy_inputs=False,
        init_snapshot=SimpleNamespace(free_memory=4096),
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(_api_process_count=1),
    )
    with (
        patch.object(gpu_worker, "maybe_apply_startup_plan"),
        patch.object(gpu_worker, "set_current_vllm_config", return_value=nullcontext()),
        patch.object(gpu_worker, "reserve_mm_ipc_gpu_memory", return_value=1024),
    ):
        assert gpu_worker.Worker.determine_available_memory(worker) == 1024
    assert calls == (
        [("scan", 32), ("memory", {"randomize_inputs": False})]
        if supported
        else [("memory", {"randomize_inputs": False})]
    )


def test_cpu_scan_precedes_memory_measurement():
    calls = []
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=SimpleNamespace(
                profile_tpsp_config=lambda tokens: calls.append(("scan", tokens))
            ),
            max_num_tokens=32,
        ),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=1024),
        _should_warm_up_model=lambda: False,
    )

    def measure_memory(_):
        calls.append(("memory", None))
        return SimpleNamespace(available_memory=4096, total_memory=4096)

    with (
        patch.object(
            cpu_worker,
            "get_allowed_cpu_list",
            return_value=[SimpleNamespace(numa_node=0)],
        ),
        patch.object(cpu_worker, "get_memory_node_info", side_effect=measure_memory),
    ):
        assert cpu_worker.CPUWorker.determine_available_memory(worker) == 1024
    assert calls == [("scan", 32), ("memory", None)]


def test_chunk_search_starts_at_half_shard_and_refines_on_64_rows():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _search_chunks

    visited = []

    def score(chunk: int) -> float:
        visited.append(chunk)
        return abs(chunk - 2048)

    chunks = _search_chunks(16384, score)
    assert chunks is not None
    assert visited[0] == 8192
    assert 2048 in chunks
    assert 128 in chunks
    assert 16384 in chunks
    assert all(chunk % 64 == 0 for chunk in chunks)


def test_chunk_search_respects_budget_and_small_shards():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _search_chunks

    assert _search_chunks(17, lambda chunk: float(chunk)) == [64]
    assert _search_chunks(8192, lambda chunk: None) is None
    assert 2048 in _search_chunks(8192, lambda chunk: abs(chunk - 2048))


def test_chunk_search_128k_tp4_starts_at_half_shard():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _search_chunks

    visited = []

    def score(chunk: int) -> float:
        visited.append(chunk)
        return abs(chunk - 2048)

    chunks = _search_chunks(32768, score)
    assert chunks is not None
    assert visited[:6] == [16384, 256, 32768, 8384, 16512, 24640]
    assert 2048 in chunks
    assert all(chunk % 64 == 0 for chunk in chunks)


def test_chunk_search_can_miss_global_minimum():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _search_chunks

    def score(chunk: int) -> float:
        return -10 if chunk == 1024 else abs(chunk - 3072)

    chunks = _search_chunks(16384, score)
    assert chunks is not None
    assert 1024 not in chunks


def test_screen_score_uses_second_fastest_of_five():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _screen_score

    assert _screen_score([5.0, 40.0, 3.0, 2.0, 4.0]) == 3.0
