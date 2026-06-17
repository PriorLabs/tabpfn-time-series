"""Explainability wrapper around :class:`TabPFNTSPipeline`.

The wrapper exposes three lightweight explanations of a TabPFN-TS forecast:

- ``partial_dependence``: PDP over calendar concepts (hour-of-day, day-of-week,
  ...) and known covariates. Calendar PDPs are computed in the *original*
  feature space: we sweep the raw calendar value (e.g. hour 0..23), re-encode it
  into the sin/cos columns the model actually consumes, and propagate it through.
- ``window_shap``: grouped Shapley attributions of the forecast to feature
  groups (via ``shapiq``'s KernelSHAP), computed across several rolling windows of
  the series. Stacked across windows this gives a (feature x time) heatmap, a bit
  like a spectrogram (the "window" framing follows arxiv.org/abs/2211.06507).
- ``decompose``: a classic model-free additive decomposition of the *target signal
  itself* (no model involved) into trend + per-calendar-feature seasonal components
  + residual. It explains the data, not the forecast, and serves as the simplest
  baseline alongside the two model-based explanations above.

All model-based explanations target the *point* forecast (the pipeline's selected
mean/median), so no quantiles are computed.

Efficiency comes from one observation about the pipeline: featurization happens
*before* the model, and the model treats every non-``target`` column as a plain
feature. A perturbation therefore only changes the *horizon* (test) rows, never
the context. TabPFN evaluates each test row independently of the others given the
context, so we fit the model once per window and reuse that fit across every
perturbation by batching them into a single prediction call (the number of TabPFN
fits equals the number of windows, not the number of perturbations). Batches are
split to keep each call under ``max_batch_rows`` test rows; this matters in CLIENT
mode, where one window can otherwise produce a very large single request.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import split_time_series_to_X_y
from tabpfn_time_series.features.basic_features import CalendarFeature
from tabpfn_time_series.explainability._deps import require

# Natural seasonality of each calendar concept, as used by CalendarFeature.
CALENDAR_PERIODS = {
    name: p[0] for name, p in CalendarFeature().seasonal_features.items()
}
TREND_COLUMNS = ["running_index", "year"]

# Raw calendar value of a timestamp, for the classic seasonal-means decomposition.
CALENDAR_ACCESSORS = {
    "hour_of_day": lambda idx: idx.hour,
    "day_of_week": lambda idx: idx.dayofweek,
    "day_of_month": lambda idx: idx.day,
    "week_of_year": lambda idx: idx.isocalendar().week.to_numpy(),
    "month_of_year": lambda idx: idx.month,
    "day_of_year": lambda idx: idx.dayofyear,
}


def _calendar_sin_cos(name: str, value: float) -> tuple[float, float]:
    """Encode a raw calendar value the exact same way CalendarFeature does."""
    divisor = CALENDAR_PERIODS[name] - 1  # 0-based adjustment, matches the featurizer
    angle = 2 * np.pi * value / divisor
    return np.sin(angle), np.cos(angle)


class TabPFNTSExplainer:
    """Post-hoc explanations for a (single-series) TabPFN-TS forecast.

    Args:
        pipeline: a (fitted) ``TabPFNTSPipeline`` to explain.
        max_batch_rows: cap on the number of test rows per inference call. Larger
            batches mean fewer model fits but bigger requests; lower this if you hit
            payload/memory limits (notably in CLIENT mode).
    """

    def __init__(self, pipeline, max_batch_rows: int = 10000):
        self.pipeline = pipeline
        self.max_batch_rows = max_batch_rows

    # -- core engine ---------------------------------------------------------

    @property
    def _inference_routine(self):
        # (train_X, train_y, test_X, quantiles=...) -> dict, the per-series routine
        return self.pipeline.predictor.inference_routine

    def _select_item(self, df: pd.DataFrame, item_id) -> pd.DataFrame:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if "item_id" not in df.columns:
            return df
        ids = df["item_id"].unique()
        if item_id is None:
            if len(ids) > 1:
                warnings.warn(
                    f"Multiple item_ids found ({list(ids)}); explaining the first "
                    f"('{ids[0]}'). Pass item_id to select a specific series.",
                    stacklevel=2,
                )
            item_id = ids[0]
        elif item_id not in ids:
            raise ValueError(
                f"item_id '{item_id}' not found in DataFrame. Available IDs: {list(ids)}"
            )
        return df[df["item_id"] == item_id]

    def _featurize(self, ctx_df: pd.DataFrame, fut_df: pd.DataFrame):
        """Run the pipeline's own preprocessing + featurization on one window."""
        ctx_df = ctx_df.copy()
        fut_df = fut_df.drop(columns="target", errors="ignore").copy()
        ctx_df["item_id"] = 0
        fut_df["item_id"] = 0

        ctx = TimeSeriesDataFrame.from_data_frame(ctx_df)
        fut = TimeSeriesDataFrame.from_data_frame(fut_df)
        return self.pipeline.featurize(ctx, fut)

    def _make_window(self, df, cutoff, context_length, prediction_length):
        ctx_df = df.iloc[cutoff - context_length : cutoff]
        fut_df = df.iloc[cutoff : cutoff + prediction_length]
        train, test = self._featurize(ctx_df, fut_df)
        train_X, train_y = split_time_series_to_X_y(train)
        test_X, _ = split_time_series_to_X_y(test)
        return {
            "train_X": train_X,
            "train_y": train_y.squeeze(),
            "test_X": test_X,
            "origin": df["timestamp"].iloc[cutoff - 1],
            "horizon_index": test.index.get_level_values("timestamp"),
        }

    def _cutoffs(self, df, prediction_length, context_length, n_windows):
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}.")
        n = len(df)
        first, last = context_length, n - prediction_length
        if last < first:
            raise ValueError(
                f"Series too short: need >= context_length + prediction_length "
                f"({context_length} + {prediction_length}), got {n}."
            )
        cutoffs = np.unique(np.linspace(first, last, n_windows).round().astype(int))
        if len(cutoffs) < n_windows:
            warnings.warn(
                f"Series too short for {n_windows} distinct windows; "
                f"using {len(cutoffs)}.",
                stacklevel=2,
            )
        return cutoffs

    def _windows(
        self, context_df, prediction_length, context_length, n_windows, item_id
    ):
        """Build rolling-origin windows; each forecasts an in-series horizon."""
        df = self._select_item(context_df, item_id).reset_index(drop=True)
        cutoffs = self._cutoffs(df, prediction_length, context_length, n_windows)
        return [
            self._make_window(df, c, context_length, prediction_length) for c in cutoffs
        ]

    def _predict_blocks(self, train_X, train_y, blocks):
        """Fit once per chunk, predict a list of horizon matrices.

        Blocks are concatenated into as few inference calls as possible while
        keeping each call's test matrix under ``max_batch_rows`` rows (one model fit
        per call). Only the point forecast is used, so no quantiles are computed.
        Returns an array of shape (n_blocks, horizon) of point forecasts.
        """
        cols = train_X.columns
        horizon = len(blocks[0])
        per_chunk = max(1, self.max_batch_rows // horizon)
        out = []
        for i in range(0, len(blocks), per_chunk):
            chunk = blocks[i : i + per_chunk]
            big = pd.concat([b[cols] for b in chunk], ignore_index=True)
            pred = self._inference_routine(train_X, train_y, big, quantiles=())
            out.append(np.asarray(pred["target"]).reshape(len(chunk), horizon))
        return np.vstack(out)

    def feature_groups(self, feature_cols) -> dict[str, list[str]]:
        """Map interpretable group names to the featurized columns they own.

        Window SHAP attributes to these groups, not to raw columns, so a calendar
        concept reads as one row instead of an opaque sin/cos pair. The groupings:

        - one group per calendar concept (``hour_of_day``, ``day_of_week``, ...),
          owning its ``{concept}_sin`` and ``{concept}_cos`` columns;
        - ``trend``: ``running_index`` and ``year`` (the non-periodic drift terms);
        - ``auto_seasonal``: the auto-detected Fourier columns (``sin_#.../cos_#...``);
        - one group per remaining column, treated as a user covariate.

        Pass an explicit ``groups`` mapping to ``window_shap`` to override this.
        """
        feature_cols = list(feature_cols)
        groups: dict[str, list[str]] = {}
        for name in CALENDAR_PERIODS:
            cols = [f"{name}_sin", f"{name}_cos"]
            if all(c in feature_cols for c in cols):
                groups[name] = cols
        trend = [c for c in TREND_COLUMNS if c in feature_cols]
        if trend:
            groups["trend"] = trend
        auto = [c for c in feature_cols if c.startswith(("sin_#", "cos_#"))]
        if auto:
            groups["auto_seasonal"] = auto

        owned = {c for cols in groups.values() for c in cols}
        for c in feature_cols:
            if c not in owned:  # leftover columns are user covariates
                groups[c] = [c]
        return groups

    # -- partial dependence --------------------------------------------------

    def partial_dependence(
        self,
        context_df: pd.DataFrame,
        feature: str,
        grid=None,
        prediction_length: int = 24,
        n_contexts: int = 3,
        context_length: int = 168,
        item_id=None,
        n_grid: int = 12,
    ) -> pd.DataFrame:
        """Partial dependence of the (point) forecast on ``feature``.

        ``feature`` is either a calendar concept (e.g. ``"hour_of_day"``,
        ``"day_of_week"``) or a covariate column name. Calendar features are swept
        in the original integer space and re-encoded into sin/cos; covariates are
        swept directly over their observed range.

        The PDP is averaged over the forecast horizon and over ``n_contexts``
        rolling windows (only ``n_contexts`` model fits in total). The returned
        ``std`` column is the spread *across those windows*, not a predictive
        interval.
        """
        windows = self._windows(
            context_df, prediction_length, context_length, n_contexts, item_id
        )
        grid, override = self._make_override(
            feature, windows[0]["test_X"], grid, n_grid
        )

        per_window = []
        for w in windows:
            blocks = [override(w["test_X"], v) for v in grid]
            preds = self._predict_blocks(w["train_X"], w["train_y"], blocks)
            per_window.append(preds.mean(axis=1))  # mean over horizon -> (n_grid,)
        arr = np.vstack(per_window)  # (n_contexts, n_grid)

        return pd.DataFrame(
            {feature: grid, "mean": arr.mean(axis=0), "std": arr.std(axis=0)}
        )

    def _make_override(self, feature, test_X, grid, n_grid):
        """Return (grid, fn) where fn(test_X, value) -> perturbed test_X."""
        if grid is not None and len(grid) == 0:
            raise ValueError("grid cannot be empty.")
        if feature in CALENDAR_PERIODS:
            sin_col, cos_col = f"{feature}_sin", f"{feature}_cos"
            if sin_col not in test_X.columns:
                raise ValueError(f"Calendar feature '{feature}' not in feature space.")
            if grid is None:
                grid = list(range(int(CALENDAR_PERIODS[feature])))

            def fn(x, value):
                x = x.copy()
                s, c = _calendar_sin_cos(feature, value)
                x[sin_col] = s
                x[cos_col] = c
                return x

            return grid, fn

        if feature not in test_X.columns:
            raise ValueError(
                f"Unknown feature '{feature}'. Expected a calendar concept "
                f"({sorted(CALENDAR_PERIODS)}) or a covariate column."
            )
        if grid is None:
            lo, hi = test_X[feature].min(), test_X[feature].max()
            grid = list(np.linspace(lo, hi, n_grid))

        def fn(x, value):
            x = x.copy()
            x[feature] = value
            return x

        return grid, fn

    # -- window SHAP (spectrogram) ------------------------------------------

    def window_shap(
        self,
        context_df: pd.DataFrame,
        prediction_length: int = 24,
        n_windows: int = 8,
        context_length: int = 168,
        groups=None,
        item_id=None,
        budget=None,
        min_std: float = 1e-6,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """Grouped Shapley attributions of the (point) forecast across rolling windows.

        Returns a DataFrame indexed by feature group (see :meth:`feature_groups`),
        with one column per window named by its forecast-origin timestamp. Values
        are the group's mean Shapley contribution to the horizon forecast. Plot as a
        heatmap for a spectrogram-like view of how feature importance shifts over
        time. The "window" framing follows WindowSHAP (https://arxiv.org/abs/2211.06507):
        Shapley values are recomputed over successive rolling forecast windows.

        Shapley values are estimated with ``shapiq``'s ``KernelSHAP`` over the feature
        groups as players. ``budget`` caps the number of coalitions evaluated per
        window; ``None`` evaluates all ``2**n_groups`` coalitions (exact). Since the
        groups are few, all coalitions of a window are batched into a single fit, so
        cost stays at roughly one TabPFN fit per window.

        Attributions are interventional Shapley values against a baseline where each
        group's columns are set to their context mean. Be aware this makes the
        ``trend`` group (running_index/year) swing hard: the horizon's running_index
        lies well outside the context range, so replacing it with the mean is a large
        intervention and trend tends to dominate the heatmap. Pass an explicit
        ``groups`` mapping to drop or regroup it.
        """
        windows = self._windows(
            context_df, prediction_length, context_length, n_windows, item_id
        )
        group_map = groups or self.feature_groups(windows[0]["train_X"].columns)

        columns = {}
        for w in windows:
            phi = self._shapley(w, group_map, budget, min_std, random_state)
            columns[w["origin"]] = phi
        return pd.DataFrame(columns).fillna(0.0)

    def _shapley(self, window, group_map, budget, min_std, random_state):
        """Grouped interventional Shapley values for one window, via shapiq KernelSHAP."""
        KernelSHAP = require("shapiq").KernelSHAP
        train_X, train_y, base = window["train_X"], window["train_y"], window["test_X"]

        # Keep only groups whose columns actually vary in the context.
        stds = train_X.std()
        active = [
            g
            for g, cols in group_map.items()
            if any(stds.get(c, 0.0) > min_std for c in cols)
        ]
        m = len(active)
        if m == 0:
            return {}

        baseline_vals = train_X.mean()

        def value_function(coalitions):
            # coalitions: (n_coalitions, m) bool. An excluded group's columns are
            # set to their context mean; included groups keep their horizon values.
            blocks = []
            for coalition in coalitions:
                x = base.copy()
                for gi, included in enumerate(coalition):
                    if not included:
                        for c in group_map[active[gi]]:
                            x[c] = baseline_vals[c]
                blocks.append(x)
            return self._predict_blocks(train_X, train_y, blocks).mean(axis=1)

        # All coalitions evaluate in a single batched call, so KernelSHAP costs one
        # fit per window regardless of m. None -> exact (KernelSHAP caps at 2**m).
        budget = 2**m if budget is None else budget
        phi = KernelSHAP(n=m, random_state=random_state).approximate(
            budget=budget, game=value_function
        )
        return dict(zip(active, phi.get_n_order_values(1)))

    # -- simple autoregressive decomposition --------------------------------

    def decompose(
        self,
        context_df: pd.DataFrame,
        features=("hour_of_day", "day_of_week"),
        period: int = 24,
        item_id=None,
    ) -> pd.DataFrame:
        """Classic additive decomposition of the target signal itself.

        This explains the *data*, not the forecast: a plain seasonal-means
        decomposition of the series with no model involved, offered as the simplest
        baseline alongside the model-based explanations above. The series is split as:

            observed = trend + sum(seasonal feature components) + residual

        ``trend`` is a centered moving average over ``period`` steps. Each seasonal
        component is the mean of the de-trended signal grouped by that calendar
        value (hour-of-day, day-of-week, ...) and is extracted sequentially, so the
        columns sum back to ``observed`` exactly. Because extraction is sequential,
        the split is order-dependent (earlier features absorb shared variance), and
        edge windows of the centered trend are one-sided. ``period`` defaults to 24,
        i.e. hourly data; set it to the dominant cycle length (in steps) of your series.
        """
        if period < 1:
            raise ValueError(f"period must be >= 1, got {period}.")
        unknown = [f for f in features if f not in CALENDAR_ACCESSORS]
        if unknown:
            raise ValueError(
                f"Unsupported feature(s) {unknown}. "
                f"Supported: {sorted(CALENDAR_ACCESSORS)}."
            )
        df = self._select_item(context_df, item_id).sort_values("timestamp")
        ts = pd.DatetimeIndex(df["timestamp"])
        observed = df["target"].to_numpy(dtype=float)
        trend = (
            pd.Series(observed)
            .rolling(period, center=True, min_periods=1)
            .mean()
            .to_numpy()
        )

        out = pd.DataFrame({"observed": observed, "trend": trend}, index=ts)
        residual = observed - trend
        for feature in features:
            key = np.asarray(CALENDAR_ACCESSORS[feature](ts))
            component = pd.Series(residual).groupby(key).transform("mean").to_numpy()
            out[feature] = component
            residual = residual - component
        out["residual"] = residual
        return out
