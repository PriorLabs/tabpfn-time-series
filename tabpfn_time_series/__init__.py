from pathlib import Path

try:
    # For Python 3.11+
    import tomllib

    # For Python <3.11
    # import tomli as tomllib
except ImportError:
    import tomli as tomllib

with (Path(__file__).parent.parent / "pyproject.toml").open("rb") as f:
    __version__ = tomllib.load(f)["project"]["version"]


from .features import FeatureTransformer
from .predictor import TabPFNTimeSeriesPredictor, TabPFNMode
from .defaults import TABPFN_TS_DEFAULT_QUANTILE_CONFIG

__all__ = [
    "FeatureTransformer",
    "TabPFNTimeSeriesPredictor",
    "TabPFNMode",
    "TABPFN_TS_DEFAULT_QUANTILE_CONFIG",
]
