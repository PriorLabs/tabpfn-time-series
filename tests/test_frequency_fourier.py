import numpy as np
import pandas as pd
import pytest

from tabpfn_time_series.features import FrequencyFourierFeature


def make_frame(freq: str, periods: int = 60, starts=("2024-01-01", "2024-03-05")):
    parts = []
    for item, start in enumerate(starts):
        ts = pd.date_range(start, periods=periods, freq=freq)
        parts.append(
            pd.DataFrame(
                {"item_id": item, "timestamp": ts, "target": np.arange(periods)}
            )
        )
    return pd.concat(parts).set_index(["item_id", "timestamp"])


@pytest.mark.parametrize(
    "freq, n_cols",
    [("h", 12), ("15min", 12), ("D", 12), ("W-SUN", 6), ("MS", 6), ("QS", 4)],
)
def test_columns_per_frequency(freq, n_cols):
    out = FrequencyFourierFeature()(make_frame(freq))
    fourier = [c for c in out.columns if c.startswith("fourier_")]
    assert len(fourier) == n_cols
    assert np.isfinite(out[fourier].to_numpy()).all()


def test_unsupported_frequency_is_noop():
    df = make_frame("YS")
    pd.testing.assert_frame_equal(FrequencyFourierFeature()(df), df)


def test_phase_is_absolute_time():
    # Same timestamp -> same features, whichever series and row position it sits at.
    out = FrequencyFourierFeature()(
        make_frame("D", starts=("2024-01-01", "2024-01-11"))
    )
    a = out.xs(0, level="item_id").loc["2024-01-20":"2024-01-30"]
    b = out.xs(1, level="item_id").loc["2024-01-20":"2024-01-30"]
    pd.testing.assert_frame_equal(a.filter(like="fourier_"), b.filter(like="fourier_"))
    assert np.allclose(
        out["fourier_sin_7_1"].to_numpy()[:8:7], out["fourier_sin_7_1"].iloc[0]
    )


def test_inferred_freq_matches_explicit_with_gaps():
    df = make_frame("h", periods=200)
    df = df.drop(df.index[[3, 50, 51, 120]])  # NaN-target rows dropped upstream
    df = df.iloc[
        np.random.default_rng(0).permutation(len(df))
    ]  # train/test concat order
    inferred = FrequencyFourierFeature()(df)
    explicit = FrequencyFourierFeature(freq="h")(df)
    pd.testing.assert_frame_equal(inferred, explicit)
