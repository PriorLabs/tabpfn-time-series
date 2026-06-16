"""Explainability demo on the electricity-price dataset (German day-ahead price).

Runs the three explanations exposed by ``TabPFNTSExplainer`` and saves the
figures to ``examples/explainability_outputs/``:

1. Partial dependence over calendar concepts (in the original feature space) and
   over the known covariates.
2. Window SHAP: grouped Shapley attributions across rolling windows, plotted as a
   feature x time spectrogram.
3. A simple additive forecast decomposition isolating the autoregressive/trend term.

The series is subsampled aggressively so the whole demo runs in well under a
minute on CPU/MPS. Efficiency comes from reusing a single context fit across all
perturbations of a window (see TabPFNTSExplainer), so only one TabPFN fit is paid
per window.

Interpreting the results, and the main limitation
--------------------------------------------------
The lagged target is *not* exposed to the model as a covariate, so these tools
cannot directly attribute the forecast to "what the series did recently". PDP and
window SHAP therefore explain the influence of the calendar/trend features and the
known covariates, not of past target values. The time-series components (the
``trend`` and ``auto_seasonal`` groups) act as a partial proxy for that
autoregressive signal. The model-free ``decompose`` is currently the most direct
view of the target's own structure; richer (more mechanistic) attributions of the
autoregressive component are future work.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline
from tabpfn_time_series.explainability import (
    TabPFNTSExplainer,
    plot_decomposition,
    plot_pdp_grid,
    plot_window_shap_spectrogram,
)

OUT = Path(__file__).parent / "explainability_outputs"
OUT.mkdir(exist_ok=True)

COVARIATES = ["Ampirion Load Forecast", "PV+Wind Forecast"]


def load_electricity(n=360):
    url = "https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet"
    df = pd.read_parquet(url)
    df = df[df["id"] == "DE"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True).head(n)
    return df.drop(columns="id")


def main():
    df = load_electricity(n=360)
    print(
        f"Loaded {len(df)} hourly rows ({df.timestamp.min()} -> {df.timestamp.max()})"
    )

    # device="auto" -> CUDA / MPS / CPU as available.
    pipeline = TabPFNTSPipeline(
        tabpfn_mode=TabPFNMode.LOCAL,
        max_context_length=168,
        tabpfn_model_config={"device": "auto"},
    )
    explainer = TabPFNTSExplainer(pipeline)

    horizon, context_length = 24, 168

    # 1. Partial dependence: calendar concepts (original space) + covariates.
    print("Computing partial dependence ...")
    pdps = {}
    for feature in ["hour_of_day", "day_of_week", *COVARIATES]:
        pdps[feature] = explainer.partial_dependence(
            df,
            feature,
            prediction_length=horizon,
            context_length=context_length,
            n_contexts=3,
        )
    fig = plot_pdp_grid(pdps, ncols=2)
    fig.savefig(OUT / "pdp.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 2. Window SHAP spectrogram.
    print("Computing window SHAP ...")
    shap_df = explainer.window_shap(
        df,
        prediction_length=horizon,
        context_length=context_length,
        n_windows=8,
        budget=64,
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    plot_window_shap_spectrogram(shap_df, ax=ax)
    fig.savefig(OUT / "window_shap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # 3. Classic additive decomposition of the autoregressive signal.
    print("Computing series decomposition ...")
    decomp = explainer.decompose(df, features=["hour_of_day", "day_of_week"], period=24)
    fig = plot_decomposition(decomp)
    fig.savefig(OUT / "decomposition.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figures to {OUT}")
    print("\nWindow SHAP (mean |contribution| per group):")
    print(shap_df.abs().mean(axis=1).sort_values(ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
