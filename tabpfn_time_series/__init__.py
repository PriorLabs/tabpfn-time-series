from pathlib import Path

from importlib import metadata

try:
    __version__ = metadata.version("tabpfn-time-series")
except metadata.PackageNotFoundError:
    # package is not installed from PyPI (e.g. from source)
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # Python <3.11, requires 'tomli' package

    with (Path(__file__).parent.parent / "pyproject.toml").open("rb") as f:
        __version__ = tomllib.load(f)["project"]["version"]


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
