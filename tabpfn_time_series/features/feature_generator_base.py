from abc import ABC, abstractmethod

import pandas as pd


class FeatureGenerator(ABC):
    """Abstract base class for feature generators.

    ``generate`` receives the whole ``(item_id, timestamp)``-indexed frame (all
    series at once) and must produce features per series where relevant by grouping
    on the ``item_id`` index level (see the built-in generators for examples).
    """

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
