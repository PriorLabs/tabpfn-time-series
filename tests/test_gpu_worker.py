"""GPUParallelWorker wiring, tested without a GPU by mocking CUDA."""

from concurrent.futures import Future

import pandas as pd
import pytest
import tabpfn
import torch

from tabpfn_time_series import TimeSeriesDataFrame
from tabpfn_time_series.predictor import TimeSeriesPredictor
from tabpfn_time_series.worker.parallel_workers import gpu_worker
from tabpfn_time_series.worker.parallel_workers.gpu_worker import GPUParallelWorker


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


def test_from_tabpfn_family_passes_num_workers_per_gpu(fake_cuda):
    predictor = TimeSeriesPredictor.from_tabpfn_family(
        tabpfn_class=tabpfn.TabPFNRegressor, num_workers_per_gpu=4
    )
    assert isinstance(predictor._worker, GPUParallelWorker)
    assert predictor._worker.total_num_workers == 4


def test_num_workers_per_gpu_ignored_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    predictor = TimeSeriesPredictor.from_tabpfn_family(
        tabpfn_class=tabpfn.TabPFNRegressor, num_workers_per_gpu=4
    )
    assert not isinstance(predictor._worker, GPUParallelWorker)


class _InlineExecutor:
    """Runs submitted jobs in-process so the test can record them."""

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def _tsdf(num_series):
    return TimeSeriesDataFrame(
        pd.DataFrame(
            {
                "item_id": [i for i in range(num_series) for _ in range(3)],
                "timestamp": list(pd.date_range("2024-01-01", periods=3)) * num_series,
                "target": 1.0,
            }
        )
    )


@pytest.mark.parametrize(
    ("num_gpus", "num_series", "expected_calls"),
    [
        (1, 1, [(0, [0])]),
        (1, 3, [(0, [0]), (0, [1]), (0, [2])]),
        (1, 6, [(0, [0, 4]), (0, [1, 5]), (0, [2]), (0, [3])]),
        (2, 3, [(0, [0]), (1, [1]), (0, [2])]),
    ],
)
def test_series_split_round_robin_over_per_gpu_pools(
    fake_cuda, monkeypatch, num_gpus, num_series, expected_calls
):
    """Series always go to the pools, even one, so the parent never shares the
    GPU with the workers. Each chunk is sent to the fixed pool of its GPU, so a
    process stays on one device, and gets the inference kwargs (e.g. quantiles)."""
    executor, pools, calls = _InlineExecutor(), set(), []

    def fake_get_pool(gpu_id, num_workers, num_threads):
        pools.add((gpu_id, num_workers))
        return executor

    def fake_routine(self, train_tsdf, test_tsdf, gpu_id, **kwargs):
        calls.append((gpu_id, list(train_tsdf.item_ids), kwargs))
        return pd.DataFrame(test_tsdf)

    monkeypatch.setattr(gpu_worker, "_get_pool", fake_get_pool)
    monkeypatch.setattr(GPUParallelWorker, "_prediction_routine_per_gpu", fake_routine)

    tsdf = _tsdf(num_series)
    worker = GPUParallelWorker(
        inference_routine=None, num_gpus=num_gpus, num_workers_per_gpu=4 // num_gpus
    )
    out = worker.predict(tsdf, tsdf, quantiles=[0.1, 0.9])

    assert list(out.item_ids) == list(range(num_series))
    assert [(gpu, chunk) for gpu, chunk, _ in calls] == expected_calls
    assert pools == {(gpu, 4 // num_gpus) for gpu, _ in expected_calls}
    assert all(kw == {"quantiles": [0.1, 0.9]} for _, _, kw in calls)
