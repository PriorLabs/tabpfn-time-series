"""CPU-based parallel worker for time series prediction."""

import pandas as pd
from typing import Callable
from joblib import Parallel, delayed
from tqdm import tqdm

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.worker.parallel_workers.base import ParallelWorker


class CPUParallelWorker(ParallelWorker):
    """Parallel worker that distributes time series predictions across CPU cores."""

    _DEFAULT_NUM_WORKERS = 8

    def __init__(
        self,
        inference_routine: Callable,
        num_workers: int = _DEFAULT_NUM_WORKERS,
    ):
        """Initialize CPU parallel worker.

        Args:
            inference_routine: Callable that performs inference on a single time series
            num_workers: Number of parallel workers to use (default: 8)
        """
        super().__init__(inference_routine)
        self.num_workers = num_workers

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ):
        """Predict on multiple time series in parallel using CPU cores.

        Args:
            train_tsdf: Training time series data
            test_tsdf: Test time series data
            **kwargs: Additional arguments passed to inference routine

        Returns:
            TimeSeriesDataFrame with predictions
        """
        predictions = Parallel(
            n_jobs=min(self.num_workers, len(train_tsdf.item_ids)),
            backend="loky",
        )(
            delayed(self._prediction_routine)(
                item_id,
                train_tsdf.loc[item_id],
                test_tsdf.loc[item_id],
                **kwargs,
            )
            for item_id in tqdm(train_tsdf.item_ids, desc="Predicting time series")
        )

        predictions = pd.concat(predictions)

        # Sort predictions according to original item_ids order
        predictions = predictions.loc[train_tsdf.item_ids]

        return TimeSeriesDataFrame(predictions)
