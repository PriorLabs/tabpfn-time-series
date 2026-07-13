import numpy as np
import pandas as pd
from typing import List, Dict, Optional

import gluonts.time_feature

from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)


class RunningIndexFeature(FeatureGenerator):
    # Safe on the whole frame: a per-series 0..n-1 counter, computed via a single
    # grouped cumcount instead of a Python loop over series.
    PER_SERIES = False

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "item_id" in (df.index.names or []):
            # Per-series counter over the whole multi-series frame in one pass.
            df["running_index"] = df.groupby(level="item_id", sort=False).cumcount()
        else:
            # Single-series frame (item_id already dropped).
            df["running_index"] = range(len(df))
        return df


class CalendarFeature(FeatureGenerator):
    # Safe on the whole frame: every feature is a pure function of the per-row
    # timestamp, independent of series boundaries.
    PER_SERIES = False

    def __init__(
        self,
        components: Optional[List[str]] = None,
        seasonal_features: Optional[Dict[str, List[float]]] = None,
    ):
        self.components = components or ["year"]
        self.seasonal_features = seasonal_features or {
            # (feature, natural seasonality)
            "second_of_minute": [60],
            "minute_of_hour": [60],
            "hour_of_day": [24],
            "day_of_week": [7],
            "day_of_month": [30.5],
            "day_of_year": [365],
            "week_of_year": [52],
            "month_of_year": [12],
        }

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        timestamps = df.index.get_level_values("timestamp")

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

        return df


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
        # Build all sin/cos columns as one block and attach in a single concat,
        # rather than inserting them one at a time (each single-column insert
        # reallocates the whole frame).
        idx = np.arange(len(df))
        columns = {}
        for i, period in enumerate(self.periods):
            name_suffix = f"{self.name_suffix}_{i}" if self.name_suffix else f"{period}"
            angle = 2 * np.pi * idx / period
            columns[f"sin_{name_suffix}"] = np.sin(angle)
            columns[f"cos_{name_suffix}"] = np.cos(angle)

        if not columns:
            return df.copy()
        return pd.concat([df, pd.DataFrame(columns, index=df.index)], axis=1)
