from tabpfn_time_series.experimental.pipeline.pipeline import TabPFNTSPipeline
from tabpfn_time_series.experimental.features.dataset_seasonality_pipeline import (
    DatasetSeasonalityPipeline,
)

PIPELINE_MAPPING = {
    "TabPFNTSPipeline": TabPFNTSPipeline,
    "DatasetSeasonalityPipeline": DatasetSeasonalityPipeline,
}
