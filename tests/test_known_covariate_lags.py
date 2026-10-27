import numpy as np
import pandas as pd
import pytest

from tabpfn_time_series.features import FeatureTransformer, KnownCovariateLagFeature
from tabpfn_time_series.features.covariate_features import season_from_freq
from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame

N_CONTEXT, N_HORIZON = 40, 12


def make_frames(freq: str = "h", lengths=(N_CONTEXT, N_CONTEXT - 7)):
    rng = np.random.default_rng(0)
    ctx, fut = [], []
    for item, n in enumerate(lengths):
        ts = pd.date_range("2024-01-01", periods=n + N_HORIZON, freq=freq)
        temp = rng.normal(size=n + N_HORIZON)
        flag = rng.integers(0, 2, n + N_HORIZON).astype(bool)
        ctx.append(
            pd.DataFrame(
                {
                    "item_id": item,
                    "timestamp": ts[:n],
                    "target": rng.normal(size=n),
                    "temp": temp[:n],
                    "flag": flag[:n],
                    "past_only": rng.normal(size=n),
                    "label": "x",
                }
            )
        )
        fut.append(
            pd.DataFrame(
                {
                    "item_id": item,
                    "timestamp": ts[n:],
                    "target": np.nan,
                    "temp": temp[n:],
                    "flag": flag[n:],
                    "label": "x",
                }
            )
        )
    return pd.concat(ctx), pd.concat(fut)


def joint_frame(ctx: pd.DataFrame, fut: pd.DataFrame) -> pd.DataFrame:
    """Context rows of every series, then horizon rows: FeatureTransformer's order."""
    df = pd.concat([ctx.assign(_is_train=True), fut.assign(_is_train=False)])
    return df.set_index(["item_id", "timestamp"])


def test_no_covariates_is_noop():
    ctx, fut = make_frames()
    df = joint_frame(
        ctx[["item_id", "timestamp", "target"]], fut[["item_id", "timestamp", "target"]]
    )
    pd.testing.assert_frame_equal(KnownCovariateLagFeature()(df), df)


def test_lags_and_means_cross_the_cutoff():
    ctx, fut = make_frames()
    df = joint_frame(ctx, fut)
    out = KnownCovariateLagFeature(season=24)(df)
    new = [c for c in out.columns if c not in df.columns]
    assert sorted(new) == sorted(
        f"{c}_{kind}"
        for c in ("temp", "flag")
        for kind in ("lag_1", "lag_24", "mean_24")
    )
    for item in (0, 1):
        series = out.xs(item, level="item_id").sort_index()
        for c in ("temp", "flag"):
            x = series[c].astype(float)
            np.testing.assert_array_equal(series[f"{c}_lag_1"], x.shift(1))
            np.testing.assert_array_equal(series[f"{c}_lag_24"], x.shift(24))
            np.testing.assert_allclose(
                series[f"{c}_mean_24"], x.rolling(24, min_periods=1).mean()
            )
        # first horizon row reads the last context value
        first = series[~series["_is_train"]].iloc[0]
        last_ctx = series[series["_is_train"]].iloc[-1]
        assert first["temp_lag_1"] == last_ctx["temp"]


def test_past_only_and_string_covariates_are_skipped():
    ctx, fut = make_frames()
    out = KnownCovariateLagFeature(season=24)(joint_frame(ctx, fut))
    assert not any(c.startswith(("past_only_", "label_")) for c in out.columns)


@pytest.mark.parametrize(
    "freq, season",
    [
        ("15min", 96),
        ("30min", 48),
        ("h", 24),
        ("D", 7),
        ("W-SUN", 4),
        ("ME", 12),
        ("QS", 4),
    ],
)
def test_season_from_freq(freq, season):
    assert season_from_freq(freq) == season


def test_season_inferred_from_timestamps():
    ctx, fut = make_frames(freq="D")
    out = KnownCovariateLagFeature()(joint_frame(ctx, fut))
    assert {"temp_lag_1", "temp_lag_7", "temp_mean_7"} <= set(out.columns)


@pytest.mark.parametrize("freq,season", [("h", 24), ("D", 7), ("MS", 12)])
def test_season_inferred_with_gaps(freq, season):
    # The pipeline drops NaN-target context rows; one gap per series must not
    # send the season to the fallback of 4.
    ctx, fut = make_frames(freq=freq)
    ctx = ctx[ctx.groupby("item_id").cumcount() != 5]
    out = KnownCovariateLagFeature()(joint_frame(ctx, fut))
    assert {f"temp_lag_{season}", f"temp_mean_{season}"} <= set(out.columns)


def test_feature_transformer_split():
    ctx, fut = make_frames()
    ctx = ctx.drop(columns=["past_only", "label"])
    fut = fut.drop(columns=["label"])
    train, test = FeatureTransformer([KnownCovariateLagFeature()]).transform(
        TimeSeriesDataFrame.from_data_frame(ctx),
        TimeSeriesDataFrame.from_data_frame(fut),
    )
    assert len(train) == len(ctx) and len(test) == len(fut)
    assert {"temp_lag_1", "temp_lag_24", "temp_mean_24"} <= set(test.columns)
    assert test["temp_lag_1"].notna().all()
