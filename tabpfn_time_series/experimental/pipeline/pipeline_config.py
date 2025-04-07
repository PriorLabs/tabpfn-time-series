import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Type, ClassVar

from tabpfn_time_series import TabPFNTimeSeriesPredictor
from tabpfn_time_series.experimental.noisy_transform.tabpfn_noisy_transform_predictor import (
    TabPFNNoisyTranformPredictor,
)


logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    predictor_name: str
    predictor_config: dict
    features: dict
    context_length: int
    slice_before_featurization: bool = True
    pipeline_name: str = "TabPFNTSPipeline"
    additional_pipeline_config: dict = field(default_factory=dict)

    _PREDICTOR_NAME_TO_CLASS: ClassVar[Dict[str, Type]] = {
        "TabPFNTimeSeriesPredictor": TabPFNTimeSeriesPredictor,
        "TabPFNNoisyTranformPredictor": TabPFNNoisyTranformPredictor,
    }

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            config = json.load(f)
        return cls(**config)

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

        logger.info(f"Looking for pipeline: {pipeline_name}")
        logger.info(f"Available pipelines: {PIPELINE_MAPPING}")
        return PIPELINE_MAPPING.get(pipeline_name)
