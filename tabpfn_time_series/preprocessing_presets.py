"""Inference-time X/y preprocessing presets for TabPFN-TS.

TabPFN applies a sequence of feature and target transforms inside the
regressor (quantile mapping, outlier clipping, …) before every forward pass.
These transforms are configured via the ``inference_config`` argument of
``TabPFNRegressor``, which accepts a list of ``PreprocessorConfig`` objects
under ``PREPROCESS_TRANSFORMS`` and a list of optional Y transforms under
``REGRESSION_Y_PREPROCESS_TRANSFORMS``.

Finding the right preset is fiddly for library users — it requires digging
through the tabpfn internals. This module exposes a small enum of
battle-tested presets that we let :class:`TabPFNTSPipeline` accept directly
via a ``preprocessing=`` kwarg.

Empirically we've seen measurable SQL skill differences between these presets
on small datasets (≤2000 steps), so it's useful to expose them.
"""

from __future__ import annotations

from typing import Literal


PreprocessingPreset = Literal["default", "none", "squashing_scaler"]


def build_preprocessing_inference_config(preset: PreprocessingPreset) -> dict | None:
    """Return the ``inference_config`` dict for the given preprocessing preset.

    Returns ``None`` for ``"default"`` — the caller should omit
    ``inference_config`` entirely so tabpfn falls back to its library
    defaults. Otherwise returns a dict suitable for
    ``TabPFNRegressor(inference_config=...)``.
    """
    if preset == "default":
        return None

    # Lazy import: tabpfn.preprocessing.configs pulls torch through the chain
    # and we'd like this module to stay importable without it.
    from tabpfn.preprocessing.configs import PreprocessorConfig

    if preset == "none":
        return {
            "PREPROCESS_TRANSFORMS": [PreprocessorConfig("none")],
            "REGRESSION_Y_PREPROCESS_TRANSFORMS": [None],
        }

    if preset == "squashing_scaler":
        return {
            "PREPROCESS_TRANSFORMS": [
                PreprocessorConfig(
                    "squashing_scaler_max10",
                    append_original=False,
                    categorical_name="ordinal_very_common_categories_shuffled",
                    global_transformer_name="svd_quarter_components",
                ),
                PreprocessorConfig("none", categorical_name="numeric"),
            ],
            "REGRESSION_Y_PREPROCESS_TRANSFORMS": [None],
        }

    raise ValueError(
        f"Unknown preprocessing preset {preset!r}. "
        f"Expected one of: 'default', 'none', 'squashing_scaler'."
    )
