import logging
from enum import Enum
from typing import Type, Dict, Any

import torch
from sklearn.base import RegressorMixin

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.defaults import (
    TABPFN_DEFAULT_CONFIG,
    DEFAULT_QUANTILE_CONFIG,
)
from tabpfn_time_series.worker import (
    ParallelWorker,
    CPUParallelWorker,
    GPUParallelWorker,
    TabPFNModelAdapter,
    BaseModelAdapter,
    PointPredictionModelAdapter,
)
from tabpfn_common_utils.telemetry import set_extension


logger = logging.getLogger(__name__)


class TabPFNMode(str, Enum):
    LOCAL = "local"
    CLIENT = "client"


@set_extension("time-series")
class TimeSeriesPredictor:
    def __init__(
        self,
        model_adapter: Type[BaseModelAdapter],
        worker_class: Type[ParallelWorker] = None,
        worker_kwargs: dict = {},
    ):
        worker_class = worker_class or (
            GPUParallelWorker if torch.cuda.is_available() else CPUParallelWorker
        )
        self._worker = worker_class(
            inference_routine=model_adapter.predict,
            **worker_kwargs,
        )

    @classmethod
    def from_tabpfn_family(
        cls,
        tabpfn_class: Type[RegressorMixin],
        tabpfn_config: Dict[str, Any] = {},
        tabpfn_output_selection: str = "median",  # mean or median
    ):
        from tabpfn import TabPFNRegressor
        from tabpfn_client import TabPFNRegressor as TabPFNClientRegressor

        model_adapter = TabPFNModelAdapter(
            model_class=tabpfn_class,
            model_config=tabpfn_config,
            tabpfn_output_selection=tabpfn_output_selection,
        )

        worker_class = None
        if tabpfn_class == TabPFNClientRegressor:
            from tabpfn_time_series.worker.parallel_workers import (
                TabPFNClientCPUParallelWorker,
            )

            worker_class = TabPFNClientCPUParallelWorker
        elif tabpfn_class == TabPFNRegressor and torch.cuda.is_available():
            worker_class = GPUParallelWorker
        elif tabpfn_class == TabPFNRegressor and not torch.cuda.is_available():
            worker_class = CPUParallelWorker
        else:
            raise ValueError(f"Expected TabPFN-family regressor, got {tabpfn_class}")

        return cls(model_adapter=model_adapter, worker_class=worker_class)

    @classmethod
    def from_point_prediction_regressor(
        cls,
        regressor_class: Type[RegressorMixin],
        regressor_config: Dict[str, Any] = {},
        regressor_fit_config: Dict[str, Any] = {},
        regressor_predict_config: Dict[str, Any] = {},
    ):
        model_adapter = PointPredictionModelAdapter(
            model_class=regressor_class,
            model_config=regressor_config,
            inference_config={
                "fit": regressor_fit_config,
                "predict": regressor_predict_config,
            },
        )

        return cls(model_adapter=model_adapter)

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        quantiles: list[float] = DEFAULT_QUANTILE_CONFIG,
    ) -> TimeSeriesDataFrame:
        """
        Generates predictions for each time series in `test_tsdf`, using `train_tsdf` for training context.

        Args:
            train_tsdf: TimeSeriesDataFrame containing training data for each time series.
            test_tsdf: TimeSeriesDataFrame containing the forecasting horizon for each time series.
            quantiles: List of quantiles to compute for probabilistic prediction. The returned predictions
                will include forecast quantiles for each value in this list.

        Returns:
            TimeSeriesDataFrame containing deterministic and/or probabilistic predictions for each item
            in `test_tsdf`. The specific structure and additional columns depend on the model adapter.

        Notes:
            - The output format may include columns for point forecasts (such as "mean" or "median")
              and columns for each quantile requested.
            - The structure and naming of columns are determined by the model adapter and worker used;
              consult the implementation for details.
        """

        self._validate_quantiles(quantiles)

        return self._worker.predict(
            train_tsdf=train_tsdf,
            test_tsdf=test_tsdf,
            quantiles=quantiles,
        )

    def _validate_quantiles(self, quantiles: list[float]):
        """Validate the quantiles."""
        if not isinstance(quantiles, list):
            raise ValueError("Quantiles must be a list")
        if not all(isinstance(q, float) for q in quantiles):
            raise ValueError("Quantiles must be a list of floats")
        if not all(0 <= q <= 1 for q in quantiles):
            raise ValueError("Quantiles must be between 0 and 1")


class TabPFNTimeSeriesPredictor(TimeSeriesPredictor):
    """
    A TabPFN-based time series predictor.
    Keeping this class for backward compatibility and as an interface for evaluation.

    Designed for TabPFNClient and TabPFNRegressor.
    """

    def __new__(
        cls,
        tabpfn_mode: TabPFNMode = TabPFNMode.CLIENT,
        tabpfn_config: dict = TABPFN_DEFAULT_CONFIG,
        tabpfn_output_selection: str = "median",  # mean or median
    ):
        from tabpfn import TabPFNRegressor
        from tabpfn_client import TabPFNRegressor as TabPFNClientRegressor

        if tabpfn_mode == TabPFNMode.CLIENT:
            tabpfn_class = TabPFNClientRegressor
        elif tabpfn_mode == TabPFNMode.LOCAL:
            tabpfn_class = TabPFNRegressor
        else:
            raise ValueError(f"Invalid tabpfn_mode: {tabpfn_mode}")

        return TimeSeriesPredictor.from_tabpfn_family(
            tabpfn_class=tabpfn_class,
            tabpfn_config=tabpfn_config,
            tabpfn_output_selection=tabpfn_output_selection,
        )
