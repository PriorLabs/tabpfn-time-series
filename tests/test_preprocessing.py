import numpy as np
import pandas as pd
import pytest
from tabpfn_time_series import TimeSeriesDataFrame

from tabpfn_time_series.pipeline import _handle_missing_values
from tabpfn_time_series.features.feature_transformer import FeatureTransformer


@pytest.fixture
def sample_tsdf_with_nans():
    """
    Creates a TimeSeriesDataFrame with 3 items to test all missing value conditions:
    - item_0: 0 valid targets (all NaNs)
    - item_1: 1 valid target (rest NaNs)
    - item_2: 2 valid targets (1 NaN)
    """
    dates = pd.date_range("2024-01-01", periods=3, freq="D")

    df = pd.DataFrame(
        {
            "item_id": ["item_0"] * 3 + ["item_1"] * 3 + ["item_2"] * 3,
            "timestamp": list(dates) * 3,
            "target": [
                np.nan,
                np.nan,
                np.nan,  # item_0: <= 1 valid
                10.0,
                np.nan,
                np.nan,  # item_1: <= 1 valid
                20.0,
                21.0,
                np.nan,  # item_2: > 1 valid
            ],
            "covariate": [1, 2, 3] * 3,
        }
    )

    tsdf = TimeSeriesDataFrame.from_data_frame(df)

    static_df = pd.DataFrame(
        {"item_id": ["item_0", "item_1", "item_2"], "category": ["A", "B", "C"]}
    ).set_index("item_id")

    tsdf.static_features = static_df
    return tsdf


def test_handle_missing_values_logic(sample_tsdf_with_nans):
    """Test that NaNs are filled with 0 for <=1 valid targets, and dropped otherwise."""
    result = _handle_missing_values(sample_tsdf_with_nans)

    # item_0: all 3 NaNs should become 0
    item_0 = result.loc["item_0", "target"]
    assert len(item_0) == 3
    assert (item_0 == 0.0).all()

    # item_1: the 2 NaNs should become 0, the 10.0 should remain
    item_1 = result.loc["item_1", "target"]
    assert len(item_1) == 3
    assert item_1.iloc[0] == 10.0
    assert (item_1.iloc[1:] == 0.0).all()

    # item_2: the 1 NaN row should be completely dropped
    item_2 = result.loc["item_2"]
    assert len(item_2) == 2
    assert "target" in item_2
    assert not item_2["target"].isna().any()


def test_handle_missing_values_preserves_static_features(sample_tsdf_with_nans):
    """Crucial check: ensure the vectorized operations don't strip metadata."""
    original_static = sample_tsdf_with_nans.static_features.copy()

    result = _handle_missing_values(sample_tsdf_with_nans)

    assert result.static_features is not None, "static_features were stripped!"
    pd.testing.assert_frame_equal(result.static_features, original_static)


class DummyFeatureGenerator:
    """A mock generator that just adds a column without groupby.apply."""

    def __call__(self, tsdf: pd.DataFrame) -> pd.DataFrame:
        tsdf = tsdf.copy()
        tsdf["dummy_feature"] = 99
        return tsdf


def test_feature_transformer_preserves_metadata(sample_tsdf_with_nans):
    """Test that concatenating and slicing preserves TSDF types and static features."""

    # Split the fixture into train and test (e.g., first 2 timesteps = train, last 1 = test)
    train_tsdf = sample_tsdf_with_nans.slice_by_timestep(None, 2)
    test_tsdf = sample_tsdf_with_nans.slice_by_timestep(2, None)

    # Ensure static features are present before transform
    assert train_tsdf.static_features is not None
    original_static = train_tsdf.static_features.copy()

    # Initialize transformer with our dummy generator
    transformer = FeatureTransformer([DummyFeatureGenerator()])

    # Safely prepare train data: The transformer expects NO NaNs in the train target
    train_tsdf_clean = train_tsdf.copy()
    train_tsdf_clean["target"] = train_tsdf_clean["target"].fillna(0)

    # Safely prepare test data: The transformer expects ALL NaNs in the test target
    test_tsdf_clean = test_tsdf.copy()
    test_tsdf_clean["target"] = np.nan

    transformed_train, transformed_test = transformer.transform(
        train_tsdf=train_tsdf_clean, test_tsdf=test_tsdf_clean
    )

    # 1. Check that they are still TimeSeriesDataFrames
    from tabpfn_time_series import TimeSeriesDataFrame

    assert isinstance(transformed_train, TimeSeriesDataFrame), (
        "Failed to cast back to TimeSeriesDataFrame!"
    )
    assert isinstance(transformed_test, TimeSeriesDataFrame), (
        "Failed to cast back to TimeSeriesDataFrame!"
    )

    # 2. Check that the dummy generator logic was actually applied
    assert "dummy_feature" in transformed_train.columns
    assert "dummy_feature" in transformed_test.columns

    # 3. Check that static features survived the pd.concat and slicing operations
    assert transformed_train.static_features is not None
    assert transformed_test.static_features is not None
    pd.testing.assert_frame_equal(transformed_train.static_features, original_static)
    pd.testing.assert_frame_equal(transformed_test.static_features, original_static)
