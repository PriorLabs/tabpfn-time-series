"""Regression tests for CalendarFeature's cyclical (sin/cos) encoding."""

import math

import numpy as np
import pandas as pd
import pytest

from tabpfn_time_series.features.basic_features import CalendarFeature


def _generate(feature_name, period, freq, raw):
    idx = pd.date_range("2026-01-05", periods=period * 2, freq=freq)
    mi = pd.MultiIndex.from_product([["s0"], idx], names=["item_id", "timestamp"])
    df = pd.DataFrame(index=mi)
    out = CalendarFeature(seasonal_features={feature_name: [period]}).generate(df)
    raw_values = raw(idx)

    def point(v):
        i = int(np.argmax(raw_values == v))
        return np.array(
            [out[f"{feature_name}_sin"].iloc[i], out[f"{feature_name}_cos"].iloc[i]]
        )

    return point


@pytest.mark.parametrize(
    "feature_name,period,freq,raw",
    [
        ("hour_of_day", 24, "h", lambda idx: idx.hour),
        ("day_of_week", 7, "D", lambda idx: idx.dayofweek),
        ("minute_of_hour", 60, "min", lambda idx: idx.minute),
    ],
)
def test_cyclical_encoding_is_injective_with_equal_chords(
    feature_name, period, freq, raw
):
    point = _generate(feature_name, period, freq, raw)
    points = [tuple(round(c, 9) for c in point(v)) for v in range(period)]

    assert len(set(points)) == period

    expected_chord = 2 * math.sin(math.pi / period)
    d_interior = np.linalg.norm(point(1) - point(0))
    d_wrap = np.linalg.norm(point(period - 1) - point(0))
    assert d_interior == pytest.approx(expected_chord, abs=1e-9)
    assert d_wrap == pytest.approx(expected_chord, abs=1e-9)
