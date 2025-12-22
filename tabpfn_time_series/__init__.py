from importlib import metadata

try:
    __version__ = metadata.version("tabpfn-time-series")
except metadata.PackageNotFoundError:
    # package is not installed from PyPI (e.g. from source)
    __version__ = "0.0.0"

from tabpfn_time_series.features import FeatureTransformer
from tabpfn_time_series.predictor import (
    TimeSeriesPredictor,
    TabPFNTimeSeriesPredictor,
    TabPFNMode,
)
from tabpfn_time_series.defaults import DEFAULT_QUANTILE_CONFIG, TABPFN_DEFAULT_CONFIG
from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.pipeline import TabPFNTSPipeline, TABPFN_TS_DEFAULT_FEATURES

__all__ = [
    "FeatureTransformer",
    "TimeSeriesPredictor",
    "TabPFNTimeSeriesPredictor",
    "TabPFNMode",
    "TimeSeriesDataFrame",
    "TabPFNTSPipeline",
    # Constants and defaults
    "DEFAULT_QUANTILE_CONFIG",
    "TABPFN_DEFAULT_CONFIG",
    "TABPFN_TS_DEFAULT_FEATURES",
]
