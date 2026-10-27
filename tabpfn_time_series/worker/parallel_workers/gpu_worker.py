"""GPU-based parallel worker for time series prediction."""

import os

import torch
import pandas as pd
from typing import Callable
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.worker.parallel_workers.base import ParallelWorker


def _init_process(num_threads: int) -> None:
    torch.set_num_threads(num_threads)


class GPUParallelWorker(ParallelWorker):
    """Parallel worker that distributes time series predictions across GPUs."""

    def __init__(
        self,
        inference_routine: Callable,
        num_gpus: int = None,
        num_workers_per_gpu: int = 1,
    ):
        """Initialize GPU parallel worker.

        Args:
            inference_routine: Callable that performs inference on a single time series
            num_gpus: Number of GPUs to use (default: all available)
            num_workers_per_gpu: Number of workers per GPU (default: 1)

        Raises:
            ValueError: If GPU is not available
        """
        super().__init__(inference_routine)

        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.num_workers_per_gpu = num_workers_per_gpu
        self.total_num_workers = self.num_gpus * self.num_workers_per_gpu

        if not torch.cuda.is_available():
            raise ValueError("GPU is required for GPU parallel inference")

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ):
        """Predict on multiple time series in parallel using GPUs.

        Args:
            train_tsdf: Training time series data
            test_tsdf: Test time series data
            **kwargs: Additional arguments passed to inference routine

        Returns:
            TimeSeriesDataFrame with predictions
        """
        item_ids = train_tsdf.item_ids
        num_chunks = min(self.total_num_workers, len(item_ids))
        if num_chunks == 1:
            predictions = self._prediction_routine_per_gpu(
                train_tsdf,
                test_tsdf,
                gpu_id=0,
                **kwargs,
            )
            return TimeSeriesDataFrame(predictions)

        # The pool is reused across calls (no idle timeout, fixed size), so each
        # process loads and compiles TabPFN once rather than once per call.
        executor = get_reusable_executor(
            max_workers=self.total_num_workers,
            timeout=None,
            initializer=_init_process,
            initargs=(max(1, (os.cpu_count() or 1) // self.total_num_workers),),
        )
        # Round-robin, so long and short series spread evenly over the workers.
        chunks = [item_ids[i::num_chunks] for i in range(num_chunks)]
        futures = [
            executor.submit(
                self._prediction_routine_per_gpu,
                train_tsdf.loc[chunk],
                test_tsdf.loc[chunk],
                gpu_id=i % self.num_gpus,  # Alternate between available GPUs
                memory_fraction=1 / self.num_workers_per_gpu,
                **kwargs,
            )
            for i, chunk in enumerate(chunks)
        ]
        predictions = pd.concat([f.result() for f in futures])

        # Sort predictions according to original item_ids order
        predictions = predictions.loc[item_ids]

        return TimeSeriesDataFrame(predictions)

    def _prediction_routine_per_gpu(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        gpu_id: int,
        memory_fraction: float | None = None,
        **kwargs,
    ):
        """Run predictions on a specific GPU.

        Args:
            train_tsdf: Training time series data subset
            test_tsdf: Test time series data subset
            gpu_id: ID of the GPU to use
            memory_fraction: Share of the GPU memory this process may use, set when
                several workers share the GPU (default: no limit)
            **kwargs: Additional arguments passed to inference routine

        Returns:
            Concatenated DataFrame with predictions
        """
        # Set GPU
        torch.cuda.set_device(gpu_id)
        if memory_fraction is not None:
            # Otherwise each worker's caching allocator holds several times what
            # TabPFN uses and starves the others sharing the GPU.
            torch.cuda.set_per_process_memory_fraction(memory_fraction, gpu_id)

        all_pred = []
        for item_id in tqdm(train_tsdf.item_ids, desc=f"GPU {gpu_id}:"):
            predictions = self._prediction_routine(
                item_id,
                train_tsdf.loc[item_id],
                test_tsdf.loc[item_id],
                **kwargs,
            )
            all_pred.append(predictions)

        # Clear GPU cache
        torch.cuda.empty_cache()

        return pd.concat(all_pred)
