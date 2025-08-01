from typing import Type, Dict, Any

from tabpfn_time_series.predictor import TimeSeriesPredictor
from tabpfn_time_series.worker.model_adapter import BaseModelAdapter
from tabpfn_time_series.worker.parallel import ParallelWorker, GPUParallelWorker


class GenericTimeSeriesPredictor(TimeSeriesPredictor):
    def __init__(
        self,
        model_adapter_class: Type[BaseModelAdapter],
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
