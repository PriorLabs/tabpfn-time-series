"""Tests for the preprocessing preset enum plumbed into TabPFNTSPipeline."""

from __future__ import annotations

import pytest

from tabpfn_time_series.preprocessing_presets import (
    build_preprocessing_inference_config,
)


def test_default_returns_none():
    # "default" means: omit inference_config so tabpfn picks its own.
    assert build_preprocessing_inference_config("default") is None


def test_none_disables_preprocessing():
    cfg = build_preprocessing_inference_config("none")
    assert cfg is not None
    assert "PREPROCESS_TRANSFORMS" in cfg
    assert "REGRESSION_Y_PREPROCESS_TRANSFORMS" in cfg
    assert len(cfg["PREPROCESS_TRANSFORMS"]) == 1
    assert cfg["PREPROCESS_TRANSFORMS"][0].name == "none"
    assert cfg["REGRESSION_Y_PREPROCESS_TRANSFORMS"] == [None]


def test_squashing_scaler_config_shape():
    cfg = build_preprocessing_inference_config("squashing_scaler")
    assert cfg is not None
    transforms = cfg["PREPROCESS_TRANSFORMS"]
    assert len(transforms) == 2
    assert transforms[0].name == "squashing_scaler_max10"
    assert transforms[0].append_original is False
    assert transforms[0].global_transformer_name == "svd_quarter_components"
    assert transforms[1].name == "none"
    assert transforms[1].categorical_name == "numeric"
    assert cfg["REGRESSION_Y_PREPROCESS_TRANSFORMS"] == [None]


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preprocessing preset"):
        build_preprocessing_inference_config("made_up")  # type: ignore[arg-type]


def test_pipeline_injects_preset_into_config():
    # Smoke-test the integration with TabPFNTSPipeline without actually
    # constructing a predictor (we monkeypatch from_tabpfn_family).
    from unittest.mock import patch

    import tabpfn_time_series.pipeline as pipeline_mod

    captured = {}

    def fake_from_tabpfn_family(*, tabpfn_class, tabpfn_config, tabpfn_output_selection):
        captured["tabpfn_config"] = tabpfn_config
        class _Stub:
            _worker = None
        return _Stub()

    with patch.object(
        pipeline_mod.TimeSeriesPredictor,
        "from_tabpfn_family",
        staticmethod(fake_from_tabpfn_family),
    ):
        pipeline_mod.TabPFNTSPipeline(preprocessing="none")

    cfg = captured["tabpfn_config"]
    assert "inference_config" in cfg
    assert cfg["inference_config"]["PREPROCESS_TRANSFORMS"][0].name == "none"


def test_pipeline_respects_explicit_inference_config():
    """User-provided inference_config wins over the preset, with a warning."""
    from unittest.mock import patch
    import warnings

    import tabpfn_time_series.pipeline as pipeline_mod

    captured = {}

    def fake_from_tabpfn_family(*, tabpfn_class, tabpfn_config, tabpfn_output_selection):
        captured["tabpfn_config"] = tabpfn_config
        class _Stub:
            _worker = None
        return _Stub()

    explicit_cfg = {"inference_config": {"PREPROCESS_TRANSFORMS": ["sentinel"]}}

    with patch.object(
        pipeline_mod.TimeSeriesPredictor,
        "from_tabpfn_family",
        staticmethod(fake_from_tabpfn_family),
    ), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pipeline_mod.TabPFNTSPipeline(
            tabpfn_model_config=explicit_cfg,
            preprocessing="none",
        )

    # Preset ignored; explicit cfg preserved.
    assert captured["tabpfn_config"]["inference_config"] == {
        "PREPROCESS_TRANSFORMS": ["sentinel"]
    }
    # And we should have warned the user about the conflict.
    assert any("preprocessing" in str(w.message) for w in caught)
