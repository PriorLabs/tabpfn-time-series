import numpy as np
import pandas as pd
import pytest

from tabpfn_time_series import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import generate_test_X


def create_train_tsdf(item_ids, dates):
    """Helper to create a TimeSeriesDataFrame for testing."""
    train_data = []
    for item in item_ids:
        for date in dates:
            train_data.append(
                {
                    "item_id": item,
                    "timestamp": date,
                    "target": np.random.rand(),
                }
            )

    return TimeSeriesDataFrame(
        pd.DataFrame(train_data),
        id_column="item_id",
        timestamp_column="timestamp",
    )


class TestGenerateTestX:
    @pytest.mark.parametrize("prediction_length", [1, 5, 10])
    @pytest.mark.parametrize(
        "freq,periods",
        [
            ("D", 10),  # Daily
            ("h", 24),  # Hourly
            ("W", 8),  # Weekly
            ("ME", 6),  # Month end
        ],
    )
    def test_basic_functionality(self, prediction_length, freq, periods):
        """Test generate_test_X produces correct output structure."""
        dates = pd.date_range(start="2023-01-01", periods=periods, freq=freq)
        item_ids = [0, 1, 2]
        train_tsdf = create_train_tsdf(item_ids, dates)

        result = generate_test_X(train_tsdf, prediction_length)

        # Check it's a TimeSeriesDataFrame
        assert isinstance(result, TimeSeriesDataFrame)

        # Check item_ids match
        assert result.item_ids.equals(train_tsdf.item_ids)

        # Check length
        assert len(result) == len(item_ids) * prediction_length

        # Check all targets are NaN
        assert result["target"].isna().all()

    def test_timestamps_start_after_train(self):
        """Test that test timestamps start immediately after train data."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        item_ids = [0, 1]
        train_tsdf = create_train_tsdf(item_ids, dates)
        prediction_length = 5

        result = generate_test_X(train_tsdf, prediction_length)

        # Check timestamps start after train data for each item
        for item_id in item_ids:
            train_last_ts = train_tsdf.xs(item_id, level="item_id").index.max()
            test_timestamps = result.xs(item_id, level="item_id").index

            # First test timestamp should be exactly one freq after last train timestamp
            expected_first_ts = train_last_ts + pd.Timedelta(days=1)
            assert test_timestamps.min() == expected_first_ts

            # Should have prediction_length timestamps
            assert len(test_timestamps) == prediction_length

    def test_with_string_item_ids(self):
        """Test with string item IDs."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        item_ids = ["item_a", "item_b", "item_c"]
        train_tsdf = create_train_tsdf(item_ids, dates)

        result = generate_test_X(train_tsdf, prediction_length=5)

        assert isinstance(result, TimeSeriesDataFrame)
        assert result.item_ids.equals(train_tsdf.item_ids)
        assert len(result) == len(item_ids) * 5

    def test_with_single_item(self):
        """Test with a single time series."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        item_ids = [0]
        train_tsdf = create_train_tsdf(item_ids, dates)

        result = generate_test_X(train_tsdf, prediction_length=5)

        assert isinstance(result, TimeSeriesDataFrame)
        assert len(result) == 5
        assert result["target"].isna().all()

    def test_with_many_items(self):
        """Test with many time series."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        item_ids = list(range(100))
        train_tsdf = create_train_tsdf(item_ids, dates)

        result = generate_test_X(train_tsdf, prediction_length=5)

        assert isinstance(result, TimeSeriesDataFrame)
        assert result.item_ids.equals(train_tsdf.item_ids)
        assert len(result) == 100 * 5

    @pytest.mark.parametrize("freq", ["D", "h", "W"])
    def test_preserves_frequency(self, freq):
        """Test that output maintains the same frequency as input."""
        dates = pd.date_range(start="2023-01-01", periods=10, freq=freq)
        train_tsdf = create_train_tsdf([0], dates)

        result = generate_test_X(train_tsdf, prediction_length=5)

        # Check the timestamps are evenly spaced with correct frequency
        test_timestamps = result.xs(0, level="item_id").index
        inferred_freq = pd.infer_freq(test_timestamps)
        # Normalize both to offset objects for comparison (W == W-SUN, etc.)
        assert pd.tseries.frequencies.to_offset(
            inferred_freq
        ) == pd.tseries.frequencies.to_offset(freq)
