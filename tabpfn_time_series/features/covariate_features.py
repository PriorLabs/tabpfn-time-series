from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)


def _offset_name(freq) -> tuple[str, int]:
    off = pd.tseries.frequencies.to_offset(freq)
    return off.name.split("-")[0].upper(), max(1, getattr(off, "n", 1))


def season_from_freq(freq) -> int:
    """Window length, in steps, for covariate lags / means at a sampling frequency.

    One day for sub-daily data, one week for daily, one year for monthly / quarterly.
    Weekly data uses 4 (about a month): the value benchmarked, since a 52-step window
    exceeds most weekly contexts. Unknown frequencies fall back to 4.
    """
    name, n = _offset_name(freq)
    if name in {"T", "MIN"}:
        season = 1440 / n
    elif name == "H":
        season = 24 / n
    elif name in {"D", "B"}:
        season = 7 / n
    elif name in {"M", "MS", "ME"}:
        season = 12 / n
    elif name in {"Q", "QS", "QE"}:
        season = 4 / n
    else:
        season = 4
    return max(4, int(round(season)))


def _freq_from_step(step: pd.Timedelta):
    """Frequency for a typical timestamp step; calendar steps snap to M / Q."""
    days = step / pd.Timedelta(days=1)
    if 28 <= days <= 31:
        return "MS"
    if 89 <= days <= 92:
        return "QS"
    return pd.tseries.frequencies.to_offset(step)


def _infer_freq(timestamps: pd.DatetimeIndex, item_ids: pd.Index):
    """Sampling frequency from the first series pandas can infer it on.

    pd.infer_freq needs a gap-free series, and the pipeline drops NaN-target context
    rows, so when no series is gap-free fall back to the median step over all series.
    """
    by_item = pd.Series(timestamps, index=item_ids).groupby(level=0, sort=False)
    steps = []
    for _, ts in by_item:
        ts = pd.DatetimeIndex(ts.unique()).sort_values()
        if len(ts) >= 3:
            freq = pd.infer_freq(ts)
            if freq is not None:
                return freq
        if len(ts) >= 2:
            steps.append(ts.to_series().diff().dropna())
    if not steps:
        return None
    step = pd.concat(steps).median()
    return _freq_from_step(step) if step > pd.Timedelta(0) else None


class KnownCovariateLagFeature(FeatureGenerator):
    """Lags and trailing means of the known covariates.

    Known covariates are available over the context and the forecast horizon, and the
    FeatureTransformer passes both as one frame, so lag k at a horizon row reads the
    covariate k steps earlier (from the horizon or the context) without touching the
    target. For every numeric / boolean known covariate this adds lags {1, season}
    and a trailing mean over `season` steps (row-based, per series).

    `columns=None` selects the numeric / boolean columns, other than the target, that
    have values in the horizon rows (past-only covariates are NaN there and are left
    alone); it must then run before other generators, which add their own columns.
    `season=None` derives the window from the sampling frequency (`season_from_freq`).
    """

    def __init__(
        self,
        columns: Optional[Sequence[str]] = None,
        season: Optional[int] = None,
        lags: Optional[Sequence[int]] = None,
        means: Optional[Sequence[int]] = None,
        target_column: str = "target",
    ):
        self.columns = list(columns) if columns is not None else None
        self.season = season
        self.lags = list(lags) if lags is not None else None
        self.means = list(means) if means is not None else None
        self.target_column = target_column

    def _known_columns(self, df: pd.DataFrame) -> List[str]:
        if self.columns is not None:
            return [c for c in self.columns if c in df.columns]
        horizon = ~df["_is_train"] if "_is_train" in df.columns else None
        cols = []
        for c in df.columns:
            if c in (self.target_column, "_is_train"):
                continue
            if not (
                pd.api.types.is_numeric_dtype(df[c])
                or pd.api.types.is_bool_dtype(df[c])
            ):
                continue
            if horizon is not None and not df.loc[horizon, c].notna().any():
                continue
            cols.append(c)
        return cols

    def _season(self, df: pd.DataFrame) -> int:
        if self.season is not None:
            return max(1, int(self.season))
        freq = _infer_freq(
            pd.DatetimeIndex(df.index.get_level_values("timestamp")),
            df.index.get_level_values("item_id"),
        )
        return season_from_freq(freq) if freq is not None else 4

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self._known_columns(df)
        if not cols:
            return df.copy()
        season = self._season(df)
        lags = self.lags if self.lags is not None else sorted({1, season})
        means = self.means if self.means is not None else [season]

        x = df[cols].astype(float)
        grouped = x.groupby(level="item_id", sort=False)
        columns = {}
        for k in lags:
            shifted = grouped.shift(k)
            for c in cols:
                columns[f"{c}_lag_{k}"] = shifted[c].to_numpy()
        for k in means:
            rolled = grouped.rolling(k, min_periods=1).mean().droplevel(0)
            # groupby-rolling returns rows in group order; realign to the frame
            rolled = rolled.reindex(df.index)
            for c in cols:
                columns[f"{c}_mean_{k}"] = rolled[c].to_numpy()
        return pd.concat(
            [df, pd.DataFrame(columns, index=df.index, dtype=np.float64)], axis=1
        )
