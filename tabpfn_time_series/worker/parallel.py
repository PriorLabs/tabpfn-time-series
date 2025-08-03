import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from abc import ABC, abstractmethod
from typing import Callable
from joblib import Parallel, delayed

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import split_time_series_to_X_y


class ParallelWorker(ABC):
    def __init__(
        self,
        inference_routine: Callable,
    ):
        self.inference_routine = inference_routine

    @abstractmethod
    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ):
        pass

    def _prediction_routine(
        self,
        item_id: str,
        single_train_tsdf: TimeSeriesDataFrame,
        single_test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        test_index = single_test_tsdf.index
        train_X, train_y = split_time_series_to_X_y(single_train_tsdf.copy())
        test_X, _ = split_time_series_to_X_y(single_test_tsdf.copy())
        train_y = train_y.squeeze()

        # TODO: solve the issue of constant target

        results = self.inference_routine(train_X, train_y, test_X, **kwargs)
        self._assert_valid_inference_output(results)

        result = pd.DataFrame(results, index=test_index)
        result["item_id"] = item_id
        result.set_index(["item_id", result.index], inplace=True)
        return result

    def _assert_valid_inference_output(self, inference_output: dict[str, np.ndarray]):
        if not isinstance(inference_output, dict):
            raise ValueError("Inference output must be a dictionary")

        if "target" not in inference_output:
            raise ValueError("Inference output must contain a 'target' key")

        if not isinstance(inference_output["target"], np.ndarray):
            raise ValueError("Inference output 'target' must be a numpy array")

        for q, q_pred in inference_output.items():
            if q != "target":
                if not isinstance(q_pred, np.ndarray):
                    raise ValueError(f"Inference output '{q}' must be a numpy array")
                if q_pred.shape != inference_output["target"].shape:
                    raise ValueError(
                        f"Inference output '{q}' must have the same shape as the target"
                    )


class GPUParallelWorker(ParallelWorker):
    def __init__(
        self,
        inference_routine: Callable,
        num_gpus: int = None,
        num_workers_per_gpu: int = 1,
    ):
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
        if (
            self.total_num_workers == 1
            or len(train_tsdf.item_ids) < self.total_num_workers
        ):
            predictions = self._prediction_routine_per_gpu(
                train_tsdf,
                test_tsdf,
                gpu_id=0,
                **kwargs,
            )
            return TimeSeriesDataFrame(predictions)

        # Split data into chunks for parallel inference on each GPU
        #   since the time series are of different lengths, we shuffle
        #   the item_ids s.t. the workload is distributed evenly across GPUs
        # Also, using 'min' since num_workers could be larger than the number of time series
        np.random.seed(0)
        item_ids_chunks = np.array_split(
            np.random.permutation(train_tsdf.item_ids),
            min(self.total_num_workers, len(train_tsdf.item_ids)),
        )

        # Run predictions in parallel
        predictions = Parallel(
            n_jobs=len(item_ids_chunks),
            backend="loky",
        )(
            delayed(self._prediction_routine_per_gpu)(
                train_tsdf.loc[chunk],
                test_tsdf.loc[chunk],
                gpu_id=i % self.num_gpus,  # Alternate between available GPUs
            )
            for i, chunk in enumerate(item_ids_chunks)
        )

        predictions = pd.concat(predictions)

        # Sort predictions according to original item_ids order
        predictions = predictions.loc[train_tsdf.item_ids]

        return TimeSeriesDataFrame(predictions)

    def _prediction_routine_per_gpu(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        gpu_id: int,
        **kwargs,
    ):
        # Set GPU
        torch.cuda.set_device(gpu_id)

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


class CPUParallelWorker(ParallelWorker):
    _DEFAULT_NUM_WORKERS = 8

    def __init__(
        self,
        inference_routine: Callable,
        num_workers: int = _DEFAULT_NUM_WORKERS,
    ):
        super().__init__(inference_routine)
        self.num_workers = num_workers

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        **kwargs,
    ):
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
