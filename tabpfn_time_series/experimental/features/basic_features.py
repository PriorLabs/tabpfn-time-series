import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple

import gluonts.time_feature

from tabpfn_time_series.experimental.features.feature_generator_base import (
    FeatureGenerator,
)


class RunningIndexFeature(FeatureGenerator):
    
    _NAME = "running_index"

    def generate(
        self,
        df: pd.DataFrame,
        use_relative_index: bool = False,
        timestamp_col_name: str = "timestamp",
    ) -> Tuple[pd.DataFrame, str]:
        df = df.copy()

        # Check if the timestamp_col_name is in the index
        if timestamp_col_name in df.index.names:
            timestamps = df.index.get_level_values(timestamp_col_name)
        else:
            timestamps = df[timestamp_col_name]

        # Check if timestamp is regular
        if use_relative_index and not RunningIndexFeature._is_timestamp_regular(timestamps):
            first_timestamp = timestamps.min()
            df[self._NAME] = (timestamps - first_timestamp).dt.days

        else:
            df[self._NAME] = timestamps.rank(method="dense")

        return df, self._NAME

    @staticmethod
    def _is_timestamp_regular(timestamps: pd.Series) -> bool:
        deltas = timestamps.diff().dropna()
        return deltas.nunique() == 1
    

class CalendarFeature(FeatureGenerator):
    def __init__(
        self,
        components: Optional[List[str]] = None,
        seasonal_features: Optional[Dict[str, List[float]]] = None,
    ):
        self.components = components or ["year"]
        self.seasonal_features = seasonal_features or {
            # (feature, natural seasonality)
            "hour_of_day": [24],
            "day_of_week": [7],
            "day_of_month": [30.5],
            "day_of_year": [365],
            "week_of_year": [52],
            "month_of_year": [12],
        }

    def generate(
        self,
        df: pd.DataFrame,
        timestamp_col_name: str = "timestamp",
    ) -> Tuple[pd.DataFrame, List[str]]:
        raw_df = df
        df = df.copy()

        # Check if timestamp_col_name is in the index
        if timestamp_col_name in df.index.names:
            timestamps = df.index.get_level_values(timestamp_col_name)
        else:
            timestamps = df[timestamp_col_name].dt

        # Add basic calendar components
        for component in self.components:
            df[component] = getattr(timestamps, component)

        # Add seasonal features
        for feature_name, periods in self.seasonal_features.items():
            feature_func = getattr(gluonts.time_feature, f"{feature_name}_index")
            feature = feature_func(timestamps).astype(np.int32)

            if periods is not None:
                for period in periods:
                    period = period - 1  # Adjust for 0-based indexing
                    df[f"{feature_name}_sin"] = np.sin(2 * np.pi * feature / period)
                    df[f"{feature_name}_cos"] = np.cos(2 * np.pi * feature / period)
            else:
                df[feature_name] = feature

        # Find names of the columns that were added
        added_columns = [col for col in df.columns if col not in raw_df.columns]

        return df, added_columns


class AdditionalCalendarFeature(CalendarFeature):
    def __init__(
        self,
        components: Optional[List[str]] = None,
        additional_seasonal_features: Optional[Dict[str, List[float]]] = None,
    ):
        super().__init__(components=components)

        self.seasonal_features = {
            **additional_seasonal_features,
            **self.seasonal_features,
        }


class PeriodicSinCosineFeature(FeatureGenerator):
    def __init__(self, periods: List[float], name_suffix: str = None):
        self.periods = periods
        self.name_suffix = name_suffix

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for i, period in enumerate(self.periods):
            name_suffix = f"{self.name_suffix}_{i}" if self.name_suffix else f"{period}"
            df[f"sin_{name_suffix}"] = np.sin(2 * np.pi * np.arange(len(df)) / period)
            df[f"cos_{name_suffix}"] = np.cos(2 * np.pi * np.arange(len(df)) / period)

        return df
