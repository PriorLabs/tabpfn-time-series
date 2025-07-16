import logging
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from scipy.stats import norm

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import split_time_series_to_X_y
from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_QUANTILE_CONFIG

logger = logging.getLogger(__name__)


class TabPFNWorker(ABC):
    def __init__(
        self,
        config: dict = {},
        num_workers: int = 1,
    ):
        self.config = config
        self.num_workers = num_workers

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
    ):
        predictions = Parallel(
            n_jobs=self.num_workers,
            backend="loky",
        )(
            delayed(self._prediction_routine)(
                item_id,
                train_tsdf.loc[item_id],
                test_tsdf.loc[item_id],
            )
            for item_id in tqdm(train_tsdf.item_ids, desc="Predicting time series")
        )

        predictions = pd.concat(predictions)

        # Sort predictions according to original item_ids order (important for MASE and WQL calculation)
        predictions = predictions.loc[train_tsdf.item_ids]

        return TimeSeriesDataFrame(predictions)

    def _prediction_routine(
        self,
        item_id: str,
        single_train_tsdf: TimeSeriesDataFrame,
        single_test_tsdf: TimeSeriesDataFrame,
    ) -> pd.DataFrame:
        # logger.debug(f"Predicting on item_id: {item_id}")

        test_index = single_test_tsdf.index
        train_X, train_y = split_time_series_to_X_y(single_train_tsdf.copy())
        test_X, _ = split_time_series_to_X_y(single_test_tsdf.copy())
        train_y = train_y.squeeze()

        train_y_has_constant_value = train_y.nunique() == 1
        if train_y_has_constant_value:
            logger.info("Found time-series with constant target")
            result = self._predict_on_constant_train_target(
                single_train_tsdf, single_test_tsdf
            )
        else:
            tabpfn = self._get_tabpfn_engine()
            tabpfn.fit(train_X, train_y)
            full_pred = tabpfn.predict(test_X, output_type="main")

            result = {"target": full_pred[self.config["tabpfn_output_selection"]]}
            result.update(
                {
                    q: q_pred
                    for q, q_pred in zip(
                        TABPFN_TS_DEFAULT_QUANTILE_CONFIG, full_pred["quantiles"]
                    )
                }
            )

        result = pd.DataFrame(result, index=test_index)
        result["item_id"] = item_id
        result.set_index(["item_id", result.index], inplace=True)
        return result

    @abstractmethod
    def _get_tabpfn_engine(self):
        pass

    def _predict_on_constant_train_target(
        self,
        single_train_tsdf: TimeSeriesDataFrame,
        single_test_tsdf: TimeSeriesDataFrame,
    ) -> pd.DataFrame:
        # If train_y is constant, we return the constant value from the training set
        mean_constant = single_train_tsdf.target.iloc[0]
        result = {"target": np.full(len(single_test_tsdf), mean_constant)}

        # For quantile prediction, we assume that the uncertainty follows a standard normal distribution
        quantile_pred_with_uncertainty = norm.ppf(
            TABPFN_TS_DEFAULT_QUANTILE_CONFIG, loc=mean_constant, scale=1
        )
        result.update(
            {
                q: np.full(len(single_test_tsdf), v)
                for q, v in zip(
                    TABPFN_TS_DEFAULT_QUANTILE_CONFIG, quantile_pred_with_uncertainty
                )
            }
        )

        return result


class TabPFNClient(TabPFNWorker):
    def __init__(
        self,
        config: dict = {},
        num_workers: int = 2,
    ):
        # Initialize the TabPFN client (e.g. sign up, login, etc.)
        from tabpfn_client import init

        init()

        # Parse the model name (only needed for TabPFNClient)
        config = config.copy()
        config["tabpfn_internal"]["model_path"] = self._parse_model_name(
            config["tabpfn_internal"]["model_path"]
        )

        super().__init__(config, num_workers)

    def _get_tabpfn_engine(self):
        from tabpfn_client import TabPFNRegressor

        return TabPFNRegressor(**self.config["tabpfn_internal"])

    def _parse_model_name(self, model_name: str) -> str:
        from tabpfn_client import TabPFNRegressor

        available_models = TabPFNRegressor.list_available_models()

        for m in available_models:
            # Model names from tabpfn_client are abbreviated
            # e.g. "tabpfn-v2-regressor-2noar4o2.ckpt" -> "2noar4o2"
            if m in model_name:
                return m
        raise ValueError(
            f"Model {model_name} not found. Available models: {available_models}."
            "Note that model names from tabpfn_client are abbreviated (e.g. 'tabpfn-v2-regressor-2noar4o2.ckpt' -> '2noar4o2')"
        )


class LocalTabPFN(TabPFNWorker):
    def __init__(
        self,
        config: dict = {},
        num_workers_per_gpu: int = 4,  # per GPU
    ):
        self.num_workers_per_gpu = num_workers_per_gpu

        # Only support GPU for now (inference on CPU takes too long)
        if not torch.cuda.is_available():
            raise ValueError("GPU is required for local TabPFN inference")

        super().__init__(
            config, num_workers=torch.cuda.device_count() * self.num_workers_per_gpu
        )

        # Download the model specified in the config
        self._download_model(self.config["tabpfn_internal"]["model_path"])

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
    ):
        total_num_workers = torch.cuda.device_count() * self.num_workers_per_gpu

        # Split data into chunks for parallel inference on each GPU
        #   since the time series are of different lengths, we shuffle
        #   the item_ids s.t. the workload is distributed evenly across GPUs
        # Also, using 'min' since num_workers could be larger than the number of time series
        np.random.seed(0)
        item_ids_chunks = np.array_split(
            np.random.permutation(train_tsdf.item_ids),
            min(total_num_workers, len(train_tsdf.item_ids)),
        )

        # Run predictions in parallel
        predictions = Parallel(n_jobs=len(item_ids_chunks), backend="loky")(
            delayed(self._prediction_routine_per_gpu)(
                train_tsdf.loc[chunk],
                test_tsdf.loc[chunk],
                gpu_id=i
                % torch.cuda.device_count(),  # Alternate between available GPUs
            )
            for i, chunk in enumerate(item_ids_chunks)
        )

        predictions = pd.concat(predictions)

        # Sort predictions according to original item_ids order
        predictions = predictions.loc[train_tsdf.item_ids]

        return TimeSeriesDataFrame(predictions)

    @staticmethod
    def _download_model(model_name: str):
        from tabpfn.model.loading import resolve_model_path, download_model

        # Resolve the model path
        # If the model path is not specified, this resolves to the default model path
        model_path, _, model_name, which = resolve_model_path(
            model_name,
            which="regressor",
        )

        if not model_path.exists():
            download_model(
                to=model_path,
                which=which,
                version="v2",
                model_name=model_name,
            )

    def _get_tabpfn_engine(self):
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(**self.config["tabpfn_internal"], random_state=0)

    def _prediction_routine_per_gpu(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        gpu_id: int,
    ):
        # Set GPU
        torch.cuda.set_device(gpu_id)

        all_pred = []
        for item_id in tqdm(train_tsdf.item_ids, desc=f"GPU {gpu_id}:"):
            predictions = self._prediction_routine(
                item_id,
                train_tsdf.loc[item_id],
                test_tsdf.loc[item_id],
            )
            all_pred.append(predictions)

        # Clear GPU cache
        torch.cuda.empty_cache()

        return pd.concat(all_pred)


class MockTabPFN(TabPFNWorker):
    """
    Mock TabPFN worker that returns random values for predictions.
    Can be used for testing or debugging.
    """

    class MockTabPFNRegressor:
        TABPFN_QUANTILE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def predict(self, test_X, output_type="main", **kwargs):
            if output_type != "main":
                raise NotImplementedError(
                    "Only main output is supported for mock TabPFN"
                )

            return {
                "mean": np.random.rand(len(test_X)),
                "median": np.random.rand(len(test_X)),
                "mode": np.random.rand(len(test_X)),
                "quantiles": [
                    np.random.rand(len(test_X)) for _ in self.TABPFN_QUANTILE
                ],
            }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_tabpfn_engine(self):
        return self.MockTabPFNRegressor()
