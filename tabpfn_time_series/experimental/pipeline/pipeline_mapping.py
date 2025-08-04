from tabpfn_time_series.experimental.pipeline.pipeline import TimeSeriesEvalPipeline
from tabpfn_time_series.experimental.features.dataset_seasonality_pipeline import (
    DatasetSeasonalityPipeline,
)

PIPELINE_MAPPING = {
    "TimeSeriesEvalPipeline": TimeSeriesEvalPipeline,
    "DatasetSeasonalityPipeline": DatasetSeasonalityPipeline,
}
