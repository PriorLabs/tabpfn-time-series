import numpy as np
import pandas as pd
from typing import List, Dict, Optional

import gluonts.time_feature

from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)


class RunningIndexFeature(FeatureGenerator):
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["running_index"] = df.groupby(level="item_id", sort=False).cumcount()
        return df


class CalendarFeature(FeatureGenerator):
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
        # One concat, not per-column assignment: each single-column insert would
        # reallocate the whole block manager.
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


def _offset_parts(freq) -> tuple[pd.DateOffset, str, int]:
    off = pd.tseries.frequencies.to_offset(freq)
    return off, off.name.split("-")[0].upper(), max(1, getattr(off, "n", 1))


def _infer_freq(timestamps: pd.DatetimeIndex, item_ids: pd.Index) -> Optional[str]:
    """Sampling frequency of the frame, from the first series pandas can infer it on.

    Context rows with NaN targets are dropped upstream, so single series can have gaps;
    fall back to the most common step of the first series when no series is regular.
    """
    by_item = pd.Series(timestamps, index=item_ids).groupby(level=0, sort=False)
    first_steps = None
    for _, ts in by_item:
        ts = pd.DatetimeIndex(ts.unique()).sort_values()
        if len(ts) < 3:
            continue
        freq = pd.infer_freq(ts)
        if freq is not None:
            return freq
        if first_steps is None:
            first_steps = ts[1:] - ts[:-1]
    if first_steps is None:
        return None
    step = pd.Series(first_steps).mode().iloc[0]
    if pd.Timedelta(days=28) <= step <= pd.Timedelta(days=31):
        return "MS"
    if pd.Timedelta(days=89) <= step <= pd.Timedelta(days=92):
        return "QS"
    return pd.tseries.frequencies.to_offset(step).freqstr


class FrequencyFourierFeature(FeatureGenerator):
    """Sin/cos of the natural periods implied by the sampling frequency.

    Periods are in steps (e.g. hourly -> 24 and 168, daily -> 7 and 365.25), each with
    harmonics k = 1..harmonics up to the Nyquist limit. The phase is computed on absolute
    time, so it is the same across series and continuous from context into horizon,
    unlike PeriodicSinCosineFeature which runs on the row position.
    """

    def __init__(self, freq: Optional[str] = None, harmonics: int = 3):
        self.freq = freq
        self.harmonics = harmonics

    @staticmethod
    def freq_periods(freq) -> List[float]:
        """Natural periods, in steps, implied by the sampling frequency."""
        _, name, n = _offset_parts(freq)
        if name in {"T", "MIN"}:
            return [1440 / n, 10080 / n]
        if name == "H":
            return [24 / n, 168 / n]
        if name in {"D", "B"}:
            return [7 / n, 365.25 / n]
        if name == "W":
            return [52.1775 / n]
        if name in {"M", "MS", "ME"}:
            return [12 / n]
        if name in {"Q", "QS", "QE"}:
            return [4 / n]
        return []

    @staticmethod
    def _absolute_steps(timestamps: pd.DatetimeIndex, freq) -> np.ndarray:
        """Absolute time in units of the sampling step (common origin for every series)."""
        _, name, n = _offset_parts(freq)
        if name in {"M", "MS", "ME"}:
            months = timestamps.year * 12 + timestamps.month - 1
            return (months + (timestamps.day - 1) / 31.0).to_numpy(float) / n
        if name in {"Q", "QS", "QE"}:
            quarters = timestamps.year * 4 + timestamps.quarter - 1
            return (quarters + (timestamps.month - 1) % 3 / 3.0).to_numpy(float) / n
        if name == "W":
            step = pd.Timedelta(weeks=n)
        else:
            step = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
        return ((timestamps - pd.Timestamp("2000-01-03")) / step).to_numpy(float)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        timestamps = pd.DatetimeIndex(df.index.get_level_values("timestamp"))
        freq = self.freq or _infer_freq(
            timestamps, df.index.get_level_values("item_id")
        )
        periods = self.freq_periods(freq) if freq is not None else []
        if not periods:
            return df.copy()

        t = self._absolute_steps(timestamps, freq)
        columns = {}
        for period in periods:
            for k in range(1, self.harmonics + 1):
                if 2 * k > period:
                    break
                angle = 2 * np.pi * k * t / period
                columns[f"fourier_sin_{period:g}_{k}"] = np.sin(angle)
                columns[f"fourier_cos_{period:g}_{k}"] = np.cos(angle)

        if not columns:
            return df.copy()
        return pd.concat([df, pd.DataFrame(columns, index=df.index)], axis=1)
