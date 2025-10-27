import logging
from enum import Enum
from typing import Type, Dict, Any

import torch
from sklearn.base import RegressorMixin

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.defaults import (
    TABPFN_TS_DEFAULT_CONFIG,
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


class TabPFNMode(Enum):
    LOCAL = "tabpfn-local"
    CLIENT = "tabpfn-client"


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
        elif tabpfn_class == TabPFNRegressor:
            worker_class = GPUParallelWorker
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
    ) -> TimeSeriesDataFrame:
        """
        Predict on each time series individually (local forecasting).

        Args:
            train_tsdf: TimeSeriesDataFrame containing training data
            test_tsdf: TimeSeriesDataFrame containing test data
            use_probabilistic_output: Whether to use probabilistic output
            quantiles: List of quantiles to use for probabilistic output

        Returns:
            TimeSeriesDataFrame containing predictions

        Note:
            If use_probabilistic_output, the prediction output from the model/model_adapter
                should be dictionary with the following keys:
                    - "mean": mean prediction
                    - "std": standard deviation of the prediction
                    - "quantiles": dictionary with quantiles as keys and values as arrays of predictions

            If use_probabilistic_output is False, the prediction output from the model/model_adapter
                should be a single array of predictions.
        """

        return self._worker.predict(
            train_tsdf=train_tsdf,
            test_tsdf=test_tsdf,
            quantiles=DEFAULT_QUANTILE_CONFIG,
        )


class TabPFNTimeSeriesPredictor(TimeSeriesPredictor):
    """
    A TabPFN-based time series predictor.
    Keeping this class for backward compatibility and as an interface for evaluation.

    Designed for TabPFNClient and TabPFNRegressor.
    """

    def __new__(
        cls,
        tabpfn_mode: TabPFNMode = TabPFNMode.CLIENT,
        tabpfn_config: dict = TABPFN_TS_DEFAULT_CONFIG,
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
