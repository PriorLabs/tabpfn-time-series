import logging
from enum import Enum
import time
from typing import Type, TypeAlias, Dict, Any

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.defaults import TABPFN_TS_DEFAULT_CONFIG
from tabpfn_time_series.worker.parallel import (
    ParallelWorker,
    CPUParallelWorker,
    GPUParallelWorker,
)
from tabpfn_time_series.worker.tabpfn_model_adapter import (
    ModelAdapter,
    TabPFNClientModelAdapter,
    LocalTabPFNModelAdapter,
)


logger = logging.getLogger(__name__)


class TabPFNMode(Enum):
    LOCAL = "tabpfn-local"
    CLIENT = "tabpfn-client"


class TimeSeriesPredictor:
    """
    Given a TimeSeriesDataFrame (multiple time series), perform prediction on each time series individually.
    """

    def __init__(
        self,
        model_adapter: Type[ModelAdapter],
        worker_class: Type[ParallelWorker],
        worker_kwargs: dict = {},
    ):
        self.worker = worker_class(model_adapter.predict, **worker_kwargs)

    def predict(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
    ) -> TimeSeriesDataFrame:
        """
        Predict on each time series individually (local forecasting).
        """

        logger.info(f"Predicting {len(train_tsdf.item_ids)} time series...")

        start_time = time.time()
        predictions = self.worker.predict(train_tsdf, test_tsdf)
        end_time = time.time()
        logger.info(f"Prediction time: {end_time - start_time} seconds")

        return predictions


class TabPFNTimeSeriesPredictor(TimeSeriesPredictor):
    """
    A TabPFN-based time series predictor.

    Designed for TabPFNClient and TabPFNRegressor.
    """

    def __init__(
        self,
        tabpfn_mode: TabPFNMode = TabPFNMode.LOCAL,
        tabpfn_config: dict = TABPFN_TS_DEFAULT_CONFIG,
        tabpfn_output_selection: str = "median",  # mean or median
    ) -> None:
        model_adapter_and_worker_mapping = {
            TabPFNMode.CLIENT: (TabPFNClientModelAdapter, CPUParallelWorker),
            TabPFNMode.LOCAL: (LocalTabPFNModelAdapter, GPUParallelWorker),
        }
        model_adapter, worker_class = model_adapter_and_worker_mapping[tabpfn_mode]

        model_adapter = model_adapter(
            tabpfn_config,
            tabpfn_output_selection,
        )

        super().__init__(
            model_adapter=model_adapter,
            worker_class=worker_class,
        )


ModelInferenceConfig: TypeAlias = Dict[str, Any]
"""
Configuration dictionary for model adapters.

Expected structure:
{
    "fit": {
        "param1": value1,
        "param2": value2,
        ...
    },
    "predict": {
        "param1": value1,
        "param2": value2,
        ...
    }
}
"""


class GenericTimeSeriesPredictor(TimeSeriesPredictor):
    def __init__(
        self,
        model_adapter_class: Type[ModelAdapter],
        model_adapter_config: Dict[str, Any],
        worker_class: Type[ParallelWorker] = GPUParallelWorker,
        worker_config: Dict[str, Any] = {},
    ):
        model_adapter = model_adapter_class(**model_adapter_config)

        super().__init__(
            model_adapter=model_adapter,
            worker_class=worker_class,
            worker_kwargs=worker_config,
        )
