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
    ("num_series", "expected_chunks"),
    [(3, [[0], [1], [2]]), (6, [[0, 4], [1, 5], [2], [3]])],
)
def test_series_split_round_robin_over_a_fixed_pool(
    fake_cuda, monkeypatch, num_series, expected_chunks
):
    """Series are split even when fewer than the workers. Each chunk gets the
    inference kwargs (e.g. quantiles) and 1/K of the GPU, and the pool is always
    requested at full size so it is reused across calls."""
    executor, pool_sizes, calls = _InlineExecutor(), [], []

    def fake_get_executor(max_workers, **_):
        pool_sizes.append(max_workers)
        return executor

    def fake_routine(self, train_tsdf, test_tsdf, gpu_id, **kwargs):
        calls.append((list(train_tsdf.item_ids), kwargs))
        return pd.DataFrame(test_tsdf)

    monkeypatch.setattr(gpu_worker, "get_reusable_executor", fake_get_executor)
    monkeypatch.setattr(GPUParallelWorker, "_prediction_routine_per_gpu", fake_routine)

    tsdf = _tsdf(num_series)
    worker = GPUParallelWorker(inference_routine=None, num_workers_per_gpu=4)
    out = worker.predict(tsdf, tsdf, quantiles=[0.1, 0.9])

    assert list(out.item_ids) == list(range(num_series))
    assert pool_sizes == [4]
    assert [chunk for chunk, _ in calls] == expected_chunks
    assert all(
        kw == {"memory_fraction": 0.25, "quantiles": [0.1, 0.9]} for _, kw in calls
    )
