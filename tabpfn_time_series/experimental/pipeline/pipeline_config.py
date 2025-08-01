import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Type, ClassVar

from tabpfn_time_series import TabPFNTimeSeriesPredictor
from tabpfn_time_series.experimental.noisy_transform.tabpfn_noisy_transform_predictor import (
    TabPFNNoisyTranformPredictor,
)
from tabpfn_time_series.experimental.evaluation.predictor import (
    GenericTimeSeriesPredictor,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    predictor_name: str
    predictor_config: dict
    features: dict
    context_length: int
    slice_before_featurization: bool = True
    pipeline_name: str = "TimeSeriesEvalPipeline"
    additional_pipeline_config: dict = field(default_factory=dict)
    use_covariates: bool = False
    _PREDICTOR_NAME_TO_CLASS: ClassVar[Dict[str, Type]] = {
        "TabPFNTimeSeriesPredictor": TabPFNTimeSeriesPredictor,
        "TabPFNNoisyTranformPredictor": TabPFNNoisyTranformPredictor,
        "GenericTimeSeriesPredictor": GenericTimeSeriesPredictor,
    }

    # Mapping of string model adapter names to actual classes
    _MODEL_ADAPTER_MAPPING: ClassVar[Dict[str, str]] = {
        "TabDPTModelAdapter": "tabpfn_time_series.experimental.other_tfm.tabdpt.TabDPTModelAdapter",
        "MitraModelAdapter": "tabpfn_time_series.experimental.other_tfm.mitra.MitraModelAdapter",
        "LinearRegressionModelAdapter": "tabpfn_time_series.experimental.other_tfm.linear_regression.LinearRegressionModelAdapter",
    }

    # Mapping of string worker class names to actual classes
    _WORKER_CLASS_MAPPING: ClassVar[Dict[str, str]] = {
        "CPUParallelWorker": "tabpfn_time_series.worker.parallel.CPUParallelWorker",
        "GPUParallelWorker": "tabpfn_time_series.worker.parallel.GPUParallelWorker",
    }

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            config = json.load(f)
        instance = cls(**config)
        cls.resolve_model_adapter_imports(instance.predictor_config)
        cls.resolve_worker_class_imports(instance.predictor_config)
        return instance

    @classmethod
    def get_predictor_class(cls, predictor_name: str) -> Type:
        """Get a predictor class by name."""
        return cls._PREDICTOR_NAME_TO_CLASS.get(predictor_name)

    @classmethod
    def get_pipeline_class(cls, pipeline_name: str) -> Type:
        """Get a pipeline class by name."""
        from tabpfn_time_series.experimental.pipeline.pipeline_mapping import (
            PIPELINE_MAPPING,
        )

        logger.debug(f"Looking for pipeline: {pipeline_name}")
        logger.debug(f"Available pipelines: {PIPELINE_MAPPING}")
        return PIPELINE_MAPPING.get(pipeline_name)

    @classmethod
    def resolve_model_adapter_imports(cls, predictor_config: dict) -> None:
        """
        Resolve string model adapter names to actual class objects in predictor_config.

        Modifies the predictor_config dict in place by replacing string model_adapter_class
        values with their corresponding imported class objects.

        Args:
            predictor_config: Configuration dictionary that may contain a 'model_adapter_class' key

        Raises:
            ValueError: If an unknown model class is specified
            ImportError: If the specified model class cannot be imported
        """
        if "model_adapter_class" not in predictor_config:
            return

        model_adapter_class = predictor_config["model_adapter_class"]
        if not isinstance(model_adapter_class, str):
            # Already resolved or is an actual class
            return

        if model_adapter_class not in cls._MODEL_ADAPTER_MAPPING:
            available_classes = list(cls._MODEL_ADAPTER_MAPPING.keys())
            raise ValueError(
                f"Unknown model adapter class: {model_adapter_class}. "
                f"Available classes: {available_classes}"
            )

        # Import the class dynamically
        module_path = cls._MODEL_ADAPTER_MAPPING[model_adapter_class]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            module = __import__(module_name, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            predictor_config["model_adapter_class"] = class_obj
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import {model_adapter_class} from {module_path}: {e}"
            )

    @classmethod
    def resolve_worker_class_imports(cls, predictor_config: dict) -> None:
        """
        Resolve string worker class names to actual class objects in predictor_config.

        Modifies the predictor_config dict in place by replacing string worker_class
        values with their corresponding imported class objects.

        Args:
            predictor_config: Configuration dictionary that may contain a 'worker_class' key

        Raises:
            ValueError: If an unknown worker class is specified
            ImportError: If the specified worker class cannot be imported
        """
        if "worker_class" not in predictor_config:
            return

        worker_class = predictor_config["worker_class"]
        if not isinstance(worker_class, str):
            # Already resolved or is an actual class
            return

        if worker_class not in cls._WORKER_CLASS_MAPPING:
            available_classes = list(cls._WORKER_CLASS_MAPPING.keys())
            raise ValueError(
                f"Unknown worker class: {worker_class}. "
                f"Available classes: {available_classes}"
            )

        # Import the class dynamically
        module_path = cls._WORKER_CLASS_MAPPING[worker_class]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            module = __import__(module_name, fromlist=[class_name])
            class_obj = getattr(module, class_name)
            predictor_config["worker_class"] = class_obj
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import {worker_class} from {module_path}: {e}"
            )
