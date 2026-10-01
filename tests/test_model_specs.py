from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from tabpfn.architectures import tabpfn_v3_5
from tabpfn.base import ModelSpecs
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing import PreprocessorConfig

from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline, TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series.features import RunningIndexFeature


@pytest.fixture
def model_specs() -> ModelSpecs:
    config = tabpfn_v3_5.TabPFNV3p5Config(
        max_num_classes=3,
        num_buckets=10,
        embed_dim=16,
        nlayers=1,
        icl_num_heads=2,
        icl_num_kv_heads_test=1,
        dist_embed_num_heads=2,
        dist_embed_num_blocks=1,
        dist_embed_num_inducing_points=4,
        feat_agg_num_heads=2,
        feat_agg_num_blocks=1,
        feat_agg_num_cls_tokens=1,
    )
    return ModelSpecs(
        model=tabpfn_v3_5.get_architecture(config).eval(),
        architecture_config=config,
        inference_config=InferenceConfig(
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name="none")],
        ),
    )


@pytest.mark.parametrize("path_as_string", [False, True])
def test_pipeline_model_specs_matches_checkpoint(
    model_specs: ModelSpecs, tmp_path: Path, path_as_string: bool
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": model_specs.model.state_dict(),
            "architecture_name": "tabpfn_v3_5",
            "config": asdict(model_specs.architecture_config),
            "inference_config": asdict(model_specs.inference_config),
        },
        checkpoint,
    )
    context = TimeSeriesDataFrame(
        pd.DataFrame(
            {
                "item_id": np.repeat([0, 1], 20),
                "timestamp": list(pd.date_range("2025-01-01", periods=20)) * 2,
                "target": np.random.default_rng(0).normal(size=40),
            }
        )
    )
    future = generate_test_X(context, prediction_length=3)
    config = {"device": "cpu", "n_estimators": 1, "random_state": 0}
    reference = TabPFNTSPipeline(
        temporal_features=[RunningIndexFeature()],
        tabpfn_model_config={
            **config,
            "model_path": str(checkpoint) if path_as_string else checkpoint,
        },
    )
    reference.predictor._worker.num_workers = 1
    expected = reference.predict(context, future, quantiles=[0.1, 0.5, 0.9])

    forward_calls = []

    def record_forward(model: torch.nn.Module, args: tuple[Any, ...]) -> None:
        assert model is model_specs.model
        assert next(model.parameters()).device == torch.device("cpu")
        forward_calls.append(model)

    live_config = {**config, "model_path": model_specs}
    with (
        patch(
            "tabpfn.base.load_model_criterion_config",
            side_effect=AssertionError("In-memory models must not load checkpoints"),
        ),
        patch(
            "tabpfn.model_loading.download_model",
            side_effect=AssertionError(
                "In-memory models must not download checkpoints"
            ),
        ),
        patch(
            "tabpfn_time_series.predictor._select_local_worker_class",
            side_effect=AssertionError("Live models must use the caller's device"),
        ),
        patch.object(
            model_specs.model,
            "__deepcopy__",
            side_effect=AssertionError("Live weights must not be copied"),
            create=True,
        ),
    ):
        handle = model_specs.model.register_forward_pre_hook(record_forward)
        try:
            pipeline = TabPFNTSPipeline(
                temporal_features=[RunningIndexFeature()],
                tabpfn_model_config=live_config,
            )
            result = pipeline.predict(context, future, quantiles=[0.1, 0.5, 0.9])
        finally:
            handle.remove()

    assert live_config == {**config, "model_path": model_specs}
    assert len(forward_calls) >= 2
    assert len(result) == 6
    assert np.isfinite(result.to_numpy()).all()
    pd.testing.assert_frame_equal(result, expected, rtol=1e-6, atol=1e-6)


def test_pipeline_rejects_model_specs_in_client_mode(model_specs: ModelSpecs) -> None:
    with (
        patch(
            "tabpfn_time_series.worker.model_adapters.tabpfn_adapter.tabpfn_client_init",
            side_effect=AssertionError("Invalid config must not start authentication"),
        ),
        pytest.raises(ValueError, match="ModelSpecs requires local inference"),
    ):
        TabPFNTSPipeline(
            tabpfn_mode=TabPFNMode.CLIENT,
            tabpfn_model_config={"model_path": model_specs},
        )
