"""Tests for the explainability wrapper.

Most tests use a fake inference routine (a known linear model) together with the
*real* feature transformer, so they are fast, hermetic, and let us assert exact
properties of PDP / Shapley / decomposition. One small integration test exercises
the whole path against a local TabPFN model.
"""

import numpy as np
import pandas as pd
import pytest

from tabpfn_time_series import (
    TABPFN_TS_DEFAULT_FEATURES,
    FeatureTransformer,
    TabPFNMode,
    TabPFNTSPipeline,
)
from tabpfn_time_series.pipeline import _featurize_context_future
from tabpfn_time_series.explainability import TabPFNTSExplainer


def make_series(n=288, freq="h", with_covariate=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    dates = pd.date_range("2023-01-01", periods=n, freq=freq)
    target = 10 + 0.02 * idx + 3 * np.sin(2 * np.pi * idx / 24) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"timestamp": dates, "target": target})
    if with_covariate:
        df["temp"] = 15 + 5 * np.sin(2 * np.pi * idx / 24) + rng.normal(0, 0.2, n)
    return df


class FakePredictor:
    def __init__(self, fn):
        self.inference_routine = fn


class FakePipeline:
    """Real featurizer + an injectable inference routine (no TabPFN needed)."""

    def __init__(self, infer_fn, max_context_length=10**9):
        self.feature_transformer = FeatureTransformer(TABPFN_TS_DEFAULT_FEATURES)
        self.predictor = FakePredictor(infer_fn)
        self.max_context_length = max_context_length

    def featurize(self, context_tsdf, future_tsdf):
        return _featurize_context_future(
            context_tsdf,
            future_tsdf,
            self.feature_transformer,
            self.max_context_length,
        )


def linear_infer(weights):
    """A deterministic linear model over named feature columns."""

    def fn(train_X, train_y, test_X, quantiles=None):
        target = np.zeros(len(test_X), dtype=float)
        for col, w in weights.items():
            target = target + w * test_X[col].to_numpy()
        out = {"target": target}
        out.update({q: target for q in (quantiles or [])})
        return out

    return fn


@pytest.fixture
def explainer_factory():
    def build(weights, max_context_length=10**9):
        return TabPFNTSExplainer(
            FakePipeline(linear_infer(weights), max_context_length)
        )

    return build


def test_feature_groups_detects_calendar_trend_auto_and_covariate(explainer_factory):
    exp = explainer_factory({"hour_of_day_sin": 1.0})
    windows = exp._windows(
        make_series(),
        prediction_length=12,
        context_length=72,
        n_windows=1,
        item_id=None,
    )
    groups = exp.feature_groups(windows[0]["train_X"].columns)

    assert groups["hour_of_day"] == ["hour_of_day_sin", "hour_of_day_cos"]
    assert "trend" in groups and "running_index" in groups["trend"]
    assert "auto_seasonal" in groups
    assert groups["temp"] == ["temp"]  # covariate becomes its own group


def test_pdp_hour_recovers_sine_shape(explainer_factory):
    exp = explainer_factory({"hour_of_day_sin": 1.0})
    pdp = exp.partial_dependence(
        make_series(),
        "hour_of_day",
        prediction_length=12,
        context_length=72,
        n_contexts=2,
    )

    assert list(pdp["hour_of_day"]) == list(range(24))
    expected = np.sin(2 * np.pi * np.arange(24) / 23)  # divisor matches CalendarFeature
    # Linear model on the sin column => PDP is exactly that sine (up to mean shift).
    corr = np.corrcoef(pdp["mean"], expected)[0, 1]
    assert corr > 0.999


def test_pdp_covariate_is_monotonic_for_positive_weight(explainer_factory):
    exp = explainer_factory({"temp": 2.0})
    pdp = exp.partial_dependence(
        make_series(), "temp", prediction_length=12, context_length=72, n_contexts=2
    )

    assert np.all(np.diff(pdp["mean"]) > 0)


def test_pdp_constant_for_unused_feature(explainer_factory):
    exp = explainer_factory({"temp": 2.0})  # day_of_week has zero weight
    pdp = exp.partial_dependence(
        make_series(),
        "day_of_week",
        prediction_length=12,
        context_length=72,
        n_contexts=2,
    )

    assert pdp["mean"].std() < 1e-9


def test_pdp_unknown_feature_raises(explainer_factory):
    exp = explainer_factory({"temp": 1.0})
    with pytest.raises(ValueError, match="Unknown feature"):
        exp.partial_dependence(
            make_series(),
            "nonexistent",
            prediction_length=12,
            context_length=72,
            n_contexts=1,
        )


def test_window_shap_shape_and_attribution(explainer_factory):
    exp = explainer_factory({"hour_of_day_sin": 1.0})
    shap_df = exp.window_shap(
        make_series(),
        prediction_length=12,
        context_length=72,
        n_windows=4,
    )

    assert shap_df.shape[1] == 4  # one column per window
    assert "hour_of_day" in shap_df.index
    # Only the weighted group should carry attribution; the rest are ~0
    # (KernelSHAP's regression leaves ~1e-8 numerical noise on the zeros).
    mean_abs = shap_df.abs().mean(axis=1)
    assert mean_abs["hour_of_day"] == mean_abs.max()
    others = mean_abs.drop("hour_of_day")
    assert others.max() < 1e-6


def test_window_shap_efficiency_sums_to_prediction_gap(explainer_factory):
    """Shapley values must sum to f(full) - f(baseline) (efficiency axiom)."""
    exp = explainer_factory({"hour_of_day_sin": 1.5, "temp": 0.5, "running_index": 0.1})
    windows = exp._windows(
        make_series(),
        prediction_length=12,
        context_length=72,
        n_windows=1,
        item_id=None,
    )
    w = windows[0]
    group_map = exp.feature_groups(w["train_X"].columns)
    phi = exp._shapley(w, group_map, budget=None, min_std=1e-6, random_state=0)

    base = w["test_X"]
    train_X, train_y = w["train_X"], w["train_y"]
    baseline_X = base.copy()
    for col in train_X.columns:
        baseline_X[col] = train_X[col].mean()
    f_full = exp._predict_blocks(train_X, train_y, [base])[0].mean()
    f_base = exp._predict_blocks(train_X, train_y, [baseline_X])[0].mean()

    assert sum(phi.values()) == pytest.approx(f_full - f_base, abs=1e-6)


def test_predict_blocks_chunking_matches_single_call(explainer_factory):
    exp = explainer_factory({"hour_of_day_sin": 1.0, "temp": 0.5})
    w = exp._windows(
        make_series(),
        prediction_length=12,
        context_length=72,
        n_windows=1,
        item_id=None,
    )[0]
    blocks = [w["test_X"], w["test_X"] * 1.1, w["test_X"] * 0.9, w["test_X"] + 1]

    single = exp._predict_blocks(w["train_X"], w["train_y"], blocks)
    exp.max_batch_rows = 12  # horizon=12 -> exactly one block per chunk
    chunked = exp._predict_blocks(w["train_X"], w["train_y"], blocks)

    assert single.shape == (4, 12)
    np.testing.assert_allclose(single, chunked)


def test_decompose_extracts_hourly_cycle(explainer_factory):
    exp = explainer_factory({})  # decomposition is model-free
    n = 240
    idx = np.arange(n)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n, freq="h"),
            "target": 5 + 0.01 * idx + 2 * np.sin(2 * np.pi * idx / 24),
        }
    )
    decomp = exp.decompose(df, features=["hour_of_day"], period=24)

    assert list(decomp.columns) == ["observed", "trend", "hour_of_day", "residual"]
    # The hourly component recovers the injected sine shape (not just sums back, which
    # is true by construction), leaving little in the residual.
    truth = 2 * np.sin(2 * np.pi * idx / 24)
    assert np.corrcoef(decomp["hour_of_day"], truth)[0, 1] > 0.99
    assert decomp["hour_of_day"].abs().max() > 1.5
    assert np.median(np.abs(decomp["residual"])) < 0.1


def test_short_series_raises(explainer_factory):
    exp = explainer_factory({"temp": 1.0})
    short = make_series(n=40)
    with pytest.raises(ValueError, match="too short"):
        exp._windows(
            short, prediction_length=12, context_length=72, n_windows=2, item_id=None
        )


def test_multi_item_warns_and_selects_first(explainer_factory):
    exp = explainer_factory({"temp": 1.0})
    a, b = make_series(n=120, seed=0), make_series(n=120, seed=1)
    a["item_id"], b["item_id"] = "A", "B"
    df = pd.concat([a, b], ignore_index=True)
    with pytest.warns(UserWarning, match="Multiple item_ids"):
        exp._windows(
            df, prediction_length=12, context_length=72, n_windows=1, item_id=None
        )


def test_window_collapse_warns(explainer_factory):
    exp = explainer_factory({"temp": 1.0})
    # n=85, context=72, pred=12 -> valid cutoffs collapse to fewer than requested.
    df = make_series(n=85)
    with pytest.warns(UserWarning, match="distinct windows"):
        exp._windows(
            df, prediction_length=12, context_length=72, n_windows=4, item_id=None
        )


def test_integration_local_tabpfn():
    """End-to-end against a local TabPFN model on a tiny series."""
    df = make_series(n=180)
    context_length, horizon = 96, 6
    pipeline = TabPFNTSPipeline(
        tabpfn_mode=TabPFNMode.LOCAL,
        max_context_length=120,
        tabpfn_model_config={"random_state": 0},  # deterministic for the parity checks
    )
    exp = TabPFNTSExplainer(pipeline)

    pdp = exp.partial_dependence(
        df,
        "hour_of_day",
        prediction_length=horizon,
        context_length=context_length,
        n_contexts=1,
    )
    assert len(pdp) == 24 and np.isfinite(pdp["mean"]).all()

    shap_df = exp.window_shap(
        df,
        prediction_length=horizon,
        context_length=context_length,
        n_windows=2,
    )
    assert shap_df.shape[1] == 2 and np.isfinite(shap_df.values).all()

    decomp = exp.decompose(df, features=["hour_of_day", "day_of_week"], period=24)
    assert list(decomp.columns) == [
        "observed",
        "trend",
        "hour_of_day",
        "day_of_week",
        "residual",
    ]
    assert np.isfinite(decomp.to_numpy()).all()

    w = exp._windows(
        df,
        prediction_length=horizon,
        context_length=context_length,
        n_windows=1,
        item_id=None,
    )[0]

    # Load-bearing assumption: batching perturbed horizons into one inference call
    # must equal evaluating them separately (TabPFN test rows are independent).
    base = w["test_X"]
    variants = [base, base.assign(**{base.columns[0]: 0.0}), base * 1.01]
    batched = exp._predict_blocks(w["train_X"], w["train_y"], variants)
    per_row = np.vstack(
        [exp._predict_blocks(w["train_X"], w["train_y"], [v])[0] for v in variants]
    )
    np.testing.assert_allclose(batched, per_row, rtol=1e-4, atol=1e-4)

    # The explainer must explain the *same* point forecast the pipeline produces.
    d = df.reset_index(drop=True)
    c = exp._cutoffs(d, horizon, context_length, 1)[0]
    ctx_df = d.iloc[c - context_length : c][["timestamp", "target", "temp"]]
    fut_df = d.iloc[c : c + horizon][["timestamp", "temp"]]
    pipe_pred = pipeline.predict_df(ctx_df, future_df=fut_df)["target"].to_numpy()
    explainer_pred = exp._predict_blocks(w["train_X"], w["train_y"], [base])[0]
    np.testing.assert_allclose(explainer_pred, pipe_pred, rtol=1e-3, atol=1e-3)
