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


def test_chunk_search_scans_from_64_through_32_non_improvements():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _search_chunks

    chunks = _search_chunks(16384, lambda chunk: abs(chunk - 2048))
    assert chunks == list(range(64, 4096 + 1, 64))
    assert 2048 in chunks


def test_chunk_search_respects_budget_and_small_shards():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _search_chunks

    assert _search_chunks(17, lambda chunk: float(chunk)) == [64]
    assert _search_chunks(8192, lambda chunk: None) is None
    assert _search_chunks(8192, lambda chunk: -float(chunk)) == list(
        range(64, 8192 + 1, 64)
    )


def test_chunk_search_128k_tp4_resets_lookahead_after_improvement():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _search_chunks

    def score(chunk: int) -> float:
        return -2 if chunk == 2560 else -1 if chunk == 1024 else 0

    chunks = _search_chunks(32768, score)
    assert chunks == list(range(64, 4608 + 1, 64))


def test_chunk_search_can_stop_before_later_improvement():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _search_chunks

    def score(chunk: int) -> float:
        return -10 if chunk == 4096 else float(chunk)

    chunks = _search_chunks(16384, score)
    assert chunks == list(range(64, 2112 + 1, 64))


def test_top_chunks_selects_distinct_global_front_runners():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _top_chunks

    scores = {
        (mode, chunk): value
        for chunk, value in ((64, 3.0), (128, 1.0), (192, 2.0))
        for mode in ("p2p", "ordered", "independent")
    }
    scores["p2p", 128] = 5.0
    assert _top_chunks([64, 128, 192], scores, "p2p") == [192, 64]
    assert _top_chunks([64], scores, "p2p") == [64]


def test_screen_score_uses_second_fastest_of_five():
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import _screen_score

    assert _screen_score([5.0, 40.0, 3.0, 2.0, 4.0]) == 3.0


@pytest.mark.parametrize(
    ("configured", "legacy", "expected"),
    [
        (None, None, "post"),
        ("post", None, "post"),
        ("dual_early", None, "dual_early"),
        ("xccl_early", None, "xccl_early"),
        (None, "1", "dual_early"),
        ("dual_early", "1", "dual_early"),
    ],
)
def test_residual_ag_mode(monkeypatch, configured, legacy, expected):
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import configured_residual_ag_mode

    for name, value in (
        ("ASYNC_TP_RESIDUAL_AG_MODE", configured),
        ("ASYNC_TP_DUAL_EARLY_AG", legacy),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert configured_residual_ag_mode() == expected


@pytest.mark.parametrize(
    ("configured", "legacy"),
    [("invalid", None), ("post", "1"), ("xccl_early", "1"), (None, "invalid")],
)
def test_residual_ag_mode_rejects_invalid_config(monkeypatch, configured, legacy):
    pytest.importorskip("deep_symm")
    from vllm.v1.worker.tpsp_profile import configured_residual_ag_mode

    for name, value in (
        ("ASYNC_TP_RESIDUAL_AG_MODE", configured),
        ("ASYNC_TP_DUAL_EARLY_AG", legacy),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="ASYNC_TP"):
        configured_residual_ag_mode()
