"""Explainability tools for TabPFN-TS forecasts.

``TabPFNTSExplainer`` exposes partial dependence, window SHAP, and a model-free
series decomposition. ``window_shap`` needs ``shapiq`` and the ``plot_*`` helpers
need ``matplotlib``; both are optional (the ``explainability`` extra) and imported
only when those code paths run, so ``import tabpfn_time_series`` stays lightweight.
"""

from tabpfn_time_series.explainability.explainer import TabPFNTSExplainer
from tabpfn_time_series.explainability.plot import (
    plot_decomposition,
    plot_pdp,
    plot_pdp_grid,
    plot_window_shap_spectrogram,
)

__all__ = [
    "TabPFNTSExplainer",
    "plot_decomposition",
    "plot_pdp",
    "plot_pdp_grid",
    "plot_window_shap_spectrogram",
]
