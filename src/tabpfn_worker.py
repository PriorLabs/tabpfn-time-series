import logging
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
from scipy.stats import norm
from autogluon.timeseries import TimeSeriesDataFrame

from src.data_preparation import split_time_series_to_X_y
from src.defaults import TABPFN_DEFAULT_QUANTILE_CONFIG

logger = logging.getLogger(__name__)


class TabPFNWorker(ABC):
    def __init__(
        self,
        tabpfn_config: dict = {},
        num_workers: int = 1,
    ):
        self.tabpfn_config = tabpfn_config
        self.num_workers = num_workers

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        quantile_config: list[float],
    ):
        predictions = Parallel(
            n_jobs=self.num_workers,
            backend="loky",
        )(
            delayed(self._prediction_routine)(
                item_id,
                train_tsdf.loc[item_id],
                test_tsdf.loc[item_id],
                quantile_config,
            )
            for item_id in train_tsdf.item_ids
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
        quantile_config: list[float],
    ) -> pd.DataFrame:
        test_index = single_test_tsdf.index
        train_X, train_y = split_time_series_to_X_y(single_train_tsdf.copy())
        test_X, _ = split_time_series_to_X_y(single_test_tsdf.copy())
        train_y = train_y.squeeze()

        train_y_has_constant_value = train_y.nunique() == 1
        if train_y_has_constant_value:
            logger.info("Found time-series with constant target")
            result = self._predict_on_constant_train_target(
                single_train_tsdf, single_test_tsdf, quantile_config
            )
        else:
            # Call worker-specific prediction routine
            result = self._worker_specific_prediction_routine(
                train_X,
                train_y,
                test_X,
                quantile_config,
            )

        result = pd.DataFrame(result, index=test_index)
        result["item_id"] = item_id
        result.set_index(["item_id", result.index], inplace=True)
        return result

    @abstractmethod
    def _worker_specific_prediction_routine(
        self,
        train_X: pd.DataFrame,
        train_y: pd.Series,
        test_X: pd.DataFrame,
        quantile_config: list[float],
    ) -> pd.DataFrame:
        pass

    def _predict_on_constant_train_target(
        self,
        single_train_tsdf: TimeSeriesDataFrame,
        single_test_tsdf: TimeSeriesDataFrame,
        quantile_config: list[float],
    ) -> pd.DataFrame:
        # If train_y is constant, we return the constant value from the training set
        mean_constant = single_train_tsdf.target.iloc[0]
        result = {"target": np.full(len(single_test_tsdf), mean_constant)}

        # For quantile prediction, we assume that the uncertainty follows a standard normal distribution
        quantile_pred_with_uncertainty = norm.ppf(
            quantile_config, loc=mean_constant, scale=1
        )
        result.update(
            {
                q: np.full(len(single_test_tsdf), v)
                for q, v in zip(quantile_config, quantile_pred_with_uncertainty)
            }
        )

        return result


class TabPFNClient(TabPFNWorker):
    def __init__(
        self,
        tabpfn_config: dict = {},
        num_workers: int = 2,
    ):
        super().__init__(tabpfn_config, num_workers)

        # Initialize the TabPFN client (e.g. sign up, login, etc.)
        from tabpfn_client import init

        init()

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        quantile_config: list[float],
    ):
        if not set(quantile_config).issubset(set(TABPFN_DEFAULT_QUANTILE_CONFIG)):
            raise NotImplementedError(
                f"TabPFNClient currently only supports {TABPFN_DEFAULT_QUANTILE_CONFIG} for quantile prediction,"
                f" but got {quantile_config}."
            )

        return super().predict(train_tsdf, test_tsdf, quantile_config)

    def _worker_specific_prediction_routine(
        self,
        train_X: pd.DataFrame,
        train_y: pd.Series,
        test_X: pd.DataFrame,
        quantile_config: list[float],
    ) -> pd.DataFrame:
        from tabpfn_client import TabPFNRegressor

        tabpfn = TabPFNRegressor(**self.tabpfn_config)
        tabpfn.fit(train_X, train_y)
        full_pred = tabpfn.predict_full(test_X)

        result = {"target": full_pred[self._get_optimization_mode()]}
        result.update({q: full_pred[f"quantile_{q:.2f}"] for q in quantile_config})

        return result

    def _get_optimization_mode(self):
        if (
            "optimize_metric" not in self.tabpfn_config
            or self.tabpfn_config["optimize_metric"] is None
        ):
            return "mean"
        elif self.tabpfn_config["optimize_metric"] in ["rmse", "mse", "r2", "mean"]:
            return "mean"
        elif self.tabpfn_config["optimize_metric"] in ["mae", "median"]:
            return "median"
        elif self.tabpfn_config["optimize_metric"] in ["mode", "exact_match"]:
            return "mode"
        else:
            raise ValueError(f"Unknown metric {self.tabpfn_config['optimize_metric']}")


class LocalTabPFN(TabPFNWorker):
    def __init__(
        self,
        tabpfn_config: dict = {},
    ):
        # Local TabPFN has a different interface for declaring the model
        if "model" in tabpfn_config:
            config = tabpfn_config.copy()
            config["model_path"] = self._parse_model_path(config["model"])
            del config["model"]
            tabpfn_config = config

        super().__init__(tabpfn_config, num_workers=1)

    def _worker_specific_prediction_routine(
        self,
        train_X: pd.DataFrame,
        train_y: pd.Series,
        test_X: pd.DataFrame,
        quantile_config: list[float],
    ) -> pd.DataFrame:
        from tabpfn import TabPFNRegressor

        tabpfn = TabPFNRegressor(**self.tabpfn_config)
        tabpfn.fit(train_X, train_y)
        full_pred = tabpfn.predict_full(test_X)

        result = {"target": full_pred[tabpfn.get_optimization_mode()]}
        if set(quantile_config).issubset(set(TABPFN_DEFAULT_QUANTILE_CONFIG)):
            result.update({q: full_pred[f"quantile_{q:.2f}"] for q in quantile_config})
        else:
            import torch

            criterion = full_pred["criterion"]
            logits = torch.tensor(full_pred["logits"])
            result.update({q: criterion.icdf(logits, q) for q in quantile_config})

        return result

    def _parse_model_path(self, model_name: str) -> str:
        from pathlib import Path
        import importlib.util

        tabpfn_path = Path(importlib.util.find_spec("tabpfn").origin).parent
        return str(
            tabpfn_path / "model_cache" / f"model_hans_regression_{model_name}.ckpt"
        )
