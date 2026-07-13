from abc import ABC, abstractmethod

import pandas as pd


class FeatureGenerator(ABC):
    """Abstract base class for feature generators"""

    #: Whether this generator must see one time series at a time.
    #:
    #: When True (default), the transformer calls ``generate`` once per series
    #: (grouped by ``item_id``) — required for features derived from a single
    #: series' values, e.g. FFT-based seasonality.
    #:
    #: When False, the generator is safe to run once on the whole multi-series
    #: frame (it depends only on per-row inputs like the timestamp, or handles
    #: the ``item_id`` index level itself). This avoids a redundant Python-level
    #: pass per series and is a large speedup for many-series datasets. Such a
    #: generator MUST NOT depend on columns produced by other generators (only on
    #: the target/covariates/timestamp), so it can be hoisted out of the per-series
    #: loop without changing output.
    PER_SERIES: bool = True

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features for the given dataframe"""
        pass

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.generate(df)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.__dict__}"

    def __repr__(self) -> str:
        return self.__str__()
