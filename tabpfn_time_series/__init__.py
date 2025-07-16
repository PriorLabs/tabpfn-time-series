from importlib import metadata

try:
    __version__ = metadata.version("tabpfn-time-series")
except metadata.PackageNotFoundError:
    # package is not installed from PyPI (e.g. from source)
    __version__ = "0.0.0"

from .features import FeatureTransformer
from .predictor import TabPFNTimeSeriesPredictor, TabPFNMode
from .defaults import TABPFN_TS_DEFAULT_QUANTILE_CONFIG
from .ts_dataframe import TimeSeriesDataFrame

__all__ = [
    "FeatureTransformer",
    "TabPFNTimeSeriesPredictor",
    "TabPFNMode",
    "TABPFN_TS_DEFAULT_QUANTILE_CONFIG",
    "TimeSeriesDataFrame",
]
