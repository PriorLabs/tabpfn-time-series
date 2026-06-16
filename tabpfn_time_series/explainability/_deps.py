"""Lazy access to the optional explainability dependencies.

``shapiq`` (window SHAP) and ``matplotlib`` (plotting) are only needed by parts of
this subpackage, so they live behind the ``explainability`` extra and are imported
on first use through :func:`require`, which raises a friendly install hint instead
of a bare ``ImportError``.
"""

from __future__ import annotations

import importlib

_INSTALL_HINT = "pip install 'tabpfn-time-series[explainability]'"


def require(module: str):
    """Import ``module`` lazily, with an actionable error if it is missing."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"'{module}' is needed for tabpfn_time_series explainability. "
            f"Install the optional dependencies with `{_INSTALL_HINT}`."
        ) from e
