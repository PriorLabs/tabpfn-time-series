import os

import numpy as np
import pandas as pd
import pytest

import tabpfn_client
import fev
import datasets

from tabpfn_time_series import (
    TabPFNMode,
    TabPFNTSPipeline,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.data_preparation import generate_test_X


def create_context_tsdf():
    """Create a simple context TimeSeriesDataFrame for testing."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    item_ids = [0, 1]

    data = []
    for item in item_ids:
        for date in dates:
            data.append(
                {
                    "item_id": item,
                    "timestamp": date,
                    "target": np.random.rand(),
                }
            )

    return TimeSeriesDataFrame(
        pd.DataFrame(data),
        id_column="item_id",
        timestamp_column="timestamp",
    )


def create_context_df():
    """Create a simple context DataFrame for testing predict_df."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "item_id": [0] * 10 + [1] * 10,
            "timestamp": list(dates) * 2,
            "target": np.random.rand(20),
        }
    )


def maybe_setup_tabpfn_client_on_github_actions() -> None:
    if not os.getenv("GITHUB_ACTIONS"):
        return

    access_token = os.getenv("TABPFN_CLIENT_API_KEY")
    assert access_token is not None, "TABPFN_CLIENT_API_KEY is not set"

    tabpfn_client.set_access_token(access_token)


def create_toy_fev_task() -> fev.Task:
    # Create a toy dataset with a single time series
    ts = {
        "id": "A",
        "timestamp": pd.date_range("2025-01-01", freq="D", periods=10),
        "target": list(range(10)),
    }
    ds = datasets.Dataset.from_list([ts])
    dataset_path = "/tmp/toy_dataset.parquet"
    ds.to_parquet(dataset_path)

    return fev.Task(
        dataset_path=dataset_path,
        horizon=3,
        num_windows=2,
    )


class TestTabPFNTSPipeline:
    def test_predict_local_mode(self):
        """Test predict method with local TabPFN mode."""
        context_tsdf = create_context_tsdf()
        future_tsdf = generate_test_X(context_tsdf, prediction_length=3)

        pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)
        result = pipeline.predict(context_tsdf, future_tsdf)

        assert result is not None
        assert len(result) == 6  # 2 items * 3 prediction steps
        assert "target" in result.columns

    @pytest.mark.uses_tabpfn_client
    def test_predict_client_mode(self):
        """Test predict method with client TabPFN mode."""
        maybe_setup_tabpfn_client_on_github_actions()

        context_tsdf = create_context_tsdf()
        future_tsdf = generate_test_X(context_tsdf, prediction_length=3)

        pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.CLIENT)
        result = pipeline.predict(context_tsdf, future_tsdf)

        assert result is not None
        assert len(result) == 6
        assert "target" in result.columns

    def test_predict_df_with_prediction_length(self):
        """Test predict_df using prediction_length parameter."""
        context_df = create_context_df()

        pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)
        result = pipeline.predict_df(context_df, prediction_length=3)

        assert result is not None
        assert len(result) == 6  # 2 items * 3 prediction steps

    def test_predict_df_with_future_df(self):
        """Test predict_df using explicit future_df."""
        context_df = create_context_df()
        future_dates = pd.date_range(start="2023-01-11", periods=3, freq="D")
        future_df = pd.DataFrame(
            {
                "item_id": [0] * 3 + [1] * 3,
                "timestamp": list(future_dates) * 2,
            }
        )

        pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)
        result = pipeline.predict_df(context_df, future_df=future_df)

        assert result is not None
        assert len(result) == 6

    def test_predict_df_single_series(self):
        """Test predict_df with a single time series (no item_id column)."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        context_df = pd.DataFrame(
            {
                "timestamp": dates,
                "target": np.random.rand(10),
            }
        )

        pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)
        result = pipeline.predict_df(context_df, prediction_length=3)

        assert result is not None
        assert len(result) == 3

    def test_predict_df_mutually_exclusive_args(self):
        """Test that future_df and prediction_length are mutually exclusive."""
        context_df = create_context_df()
        future_dates = pd.date_range(start="2023-01-11", periods=3, freq="D")
        future_df = pd.DataFrame(
            {
                "item_id": [0] * 3,
                "timestamp": future_dates,
            }
        )

        pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)

        # Both provided - should raise
        with pytest.raises(ValueError, match="exactly one"):
            pipeline.predict_df(context_df, future_df=future_df, prediction_length=3)

        # Neither provided - should raise
        with pytest.raises(ValueError, match="exactly one"):
            pipeline.predict_df(context_df)

    def test_custom_quantiles(self):
        """Test that custom quantiles are returned in the result."""
        context_tsdf = create_context_tsdf()
        future_tsdf = generate_test_X(context_tsdf, prediction_length=3)

        pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)
        custom_quantiles = [0.1, 0.5, 0.9]
        result = pipeline.predict(context_tsdf, future_tsdf, quantiles=custom_quantiles)

        for q in custom_quantiles:
            assert q in result.columns, f"Quantile {q} not in result columns"

    def test_predict_fev(self):
        """Test predict_fev method with a real fev task."""
        task = create_toy_fev_task()
        pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)

        predictions_per_window, inference_time_s = pipeline.predict_fev(task)

        # Verify results
        assert isinstance(predictions_per_window, list)
        assert len(predictions_per_window) == task.num_windows
        assert isinstance(inference_time_s, float)
        assert inference_time_s >= 0

        # Verify each prediction window has the expected structure
        for pred in predictions_per_window:
            assert isinstance(pred, datasets.DatasetDict)
            assert "target" in pred
