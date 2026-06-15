"""Explainability tools for TabPFN-TS forecasts.

``TabPFNTSExplainer`` has no plotting dependency. The ``plot_*`` helpers require
matplotlib and are imported lazily so the core explainer stays importable without it.
"""

from tabpfn_time_series.explainability.explainer import TabPFNTSExplainer

_PLOTTERS = {
    "plot_pdp",
    "plot_pdp_grid",
    "plot_window_shap_spectrogram",
    "plot_decomposition",
}

__all__ = ["TabPFNTSExplainer", *sorted(_PLOTTERS)]


def __getattr__(name):
    if name in _PLOTTERS:
        from tabpfn_time_series.explainability import plot

        return getattr(plot, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
