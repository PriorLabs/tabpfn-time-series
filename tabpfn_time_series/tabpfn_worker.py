import logging
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
from scipy.stats import norm
from autogluon.timeseries import TimeSeriesDataFrame

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
        quantile_config: list[float],
    ):
        if not set(quantile_config).issubset(set(TABPFN_TS_DEFAULT_QUANTILE_CONFIG)):
            raise NotImplementedError(
                f"We currently only supports {TABPFN_TS_DEFAULT_QUANTILE_CONFIG} for quantile prediction,"
                f" but got {quantile_config}."
            )

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
            tabpfn = self._get_tabpfn_engine()
            tabpfn.fit(train_X, train_y)
            full_pred = tabpfn.predict(test_X, output_type="main")

            result = {"target": full_pred[self.config["tabpfn_output_selection"]]}
            result.update(
                {
                    q: q_pred
                    for q, q_pred in zip(quantile_config, full_pred["quantiles"])
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
        config: dict = {},
        num_workers: int = 2,
    ):
        super().__init__(config, num_workers)

        # Initialize the TabPFN client (e.g. sign up, login, etc.)
        from tabpfn_client import init

        init()

    def _get_tabpfn_engine(self):
        from tabpfn_client import TabPFNRegressor

        return TabPFNRegressor(**self.config["tabpfn_internal"])


class LocalTabPFN(TabPFNWorker):
    def __init__(
        self,
        config: dict = {},
    ):
        super().__init__(config, num_workers=1)

    def _get_tabpfn_engine(self):
        from tabpfn import TabPFNRegressor

        if "model_path" in self.config["tabpfn_internal"]:
            config = self.config["tabpfn_internal"].copy()
            config["model_path"] = self._parse_model_path(config["model_path"])

        return TabPFNRegressor(**config)

    def _parse_model_path(self, model_name: str) -> str:
        return f"tabpfn-v2-regressor-{model_name}.ckpt"
