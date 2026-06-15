"""Plotting helpers for :class:`TabPFNTSExplainer` outputs.

Each helper draws onto a provided ``ax`` (or creates one) and returns it, so they
compose in notebooks and scripts alike. None of them call ``plt.show()``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_pdp(pdp_df: pd.DataFrame, feature: str | None = None, ax=None):
    """Plot a partial-dependence curve with a +/-1 std band across windows."""
    feature = feature or pdp_df.columns[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))

    x = pdp_df[feature]
    mean, std = pdp_df["mean"], pdp_df["std"]
    ax.plot(
        x, mean, color="tomato", marker="o", markersize=3, linewidth=1.8,
        label="mean over windows",
    )
    ax.fill_between(
        x, mean - std, mean + std, color="tomato", alpha=0.2,
        label="±1 std across windows",
    )
    ax.set_xlabel(feature)
    ax.set_ylabel("mean forecast")
    ax.set_title(f"Partial dependence: {feature}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    return ax


def plot_pdp_grid(pdps: dict[str, pd.DataFrame], ncols: int = 3):
    """Small multiples of several PDP curves."""
    n = len(pdps)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (feature, df) in zip(axes, pdps.items()):
        plot_pdp(df, feature, ax=ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_window_shap_spectrogram(shap_df: pd.DataFrame, ax=None):
    """Heatmap of grouped Shapley values: feature groups (y) x window time (x)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    values = shap_df.values
    limit = np.abs(values).max() or 1.0
    im = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(shap_df.index)))
    ax.set_yticklabels(shap_df.index)
    times = [pd.Timestamp(c).strftime("%m-%d %Hh") for c in shap_df.columns]
    ax.set_xticks(range(len(times)))
    ax.set_xticklabels(times, rotation=45, ha="right")
    ax.set_xlabel("forecast origin")
    ax.set_title("Window SHAP (contribution to forecast)")
    ax.figure.colorbar(im, ax=ax, label="Shapley value")
    return ax


def plot_decomposition(decomp_df: pd.DataFrame):
    """Traditional decomposition: one subplot per component (observed, trend, ...)."""
    cols = list(decomp_df.columns)
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 1.7 * len(cols)), sharex=True)
    x = decomp_df.index
    for ax, col in zip(axes, cols):
        if col == "residual":
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.scatter(x, decomp_df[col], s=6, color="gray", alpha=0.7)
        else:
            ax.plot(x, decomp_df[col], color="royalblue", linewidth=1.4)
        ax.set_ylabel(col, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
    axes[0].set_title("Series decomposition (additive)")
    axes[-1].set_xlabel("timestamp")
    fig.tight_layout()
    return fig
