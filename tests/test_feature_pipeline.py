import pandas as pd
import numpy as np

# from tabpfn_time_series.features.feature_pipeline import (
#     RunningIndexFeatureTransformer,
#     FeatureTransformer,
# )
import pytest
from autogluon.timeseries import TimeSeriesDataFrame

from tabpfn_time_series.features import (
    FeatureTransformer,
    RunningIndexFeatureTransformer,
    CalendarFeatureTransformer,
    AutoSeasonalFeatureTransformer,
    train_test_split_time_series,
)
# from tabpfn_time_series.features.utils_pipeline import (
#     from_autogluon_tsdf_to_df,
#     from_df_to_autogluon_tsdf,
# )


# Run the test
# test_per_item_mode()
# test_global_mode()


def is_autogluon_tsdf(df):
    """
    Checks if the input is an AutoGluon TimeSeriesDataFrame.

    Args:
        df (any): The object to check.

    Returns:
        bool: True if the object is a TimeSeriesDataFrame, False otherwise.
    """
    return isinstance(df, TimeSeriesDataFrame)


def is_pure_pandas_df(df):
    import pandas as pd

    """
    Checks if the input is a pure pandas DataFrame and not a subclass.

    Since autogluon.timeseries.TimeSeriesDataFrame is a subclass of
    pandas.DataFrame, a simple isinstance(df, pd.DataFrame) would return
    True for both. This function checks for the exact type.

    Args:
        df (any): The object to check.

    Returns:
        bool: True if the object is exactly a pandas.DataFrame, False otherwise.
    """
    return type(df) is pd.DataFrame


# --- Pytest Fixture ---
# Fixtures are a pytest feature for setting up resources that tests need.
# This fixture replaces the setUp method from unittest.


@pytest.fixture(params=[0, 1, 2, 3])
def loaded_tsdf(request):
    """
    A pytest fixture that loads the initial TimeSeriesDataFrame.
    Test functions that declare 'loaded_tsdf' as an argument will receive
    the return value of this function.
    """

    from tabpfn_time_series.features.utils import (
        load_data,
    )

    # Define the datasets of interest (metadata)
    dataset_metadata = {
        "monash_tourism_monthly": {"prediction_length": 24},
        "m4_hourly": {"prediction_length": 48},
    }

    # For now, we only have one dataset of interest
    dataset_choice = "monash_tourism_monthly"
    num_time_series_subset = 1

    # Loading Time Series Data Frames
    tsdf, train_tsdf, test_tsdf_ground_truth, test_tsdf = load_data(
        dataset_choice, num_time_series_subset, dataset_metadata
    )
    # Create a tuple of the four dataframes
    all_tsdfs = (tsdf, train_tsdf, test_tsdf_ground_truth, test_tsdf)

    # Return the dataframe corresponding to the current parameter (0, 1, 2, or 3)
    return all_tsdfs[request.param]


# --- Pytest Test Functions ---
# Tests are now simple functions that use the standard 'assert' statement.


def test_loaded_data_is_tsdf(loaded_tsdf):
    """
    Tests if the loaded data is correctly identified as a TimeSeriesDataFrame
    and not as a pure pandas DataFrame.
    """
    # The 'loaded_tsdf' argument is automatically supplied by the fixture above.
    assert is_autogluon_tsdf(loaded_tsdf)
    assert not is_pure_pandas_df(loaded_tsdf)


def test_conversion_to_pandas(loaded_tsdf):
    """
    Tests the conversion from TimeSeriesDataFrame to a pure pandas DataFrame.
    """
    from tabpfn_time_series.features.utils import (
        from_autogluon_tsdf_to_df,
    )

    # Convert the TSDF to a standard DataFrame
    pandas_df = from_autogluon_tsdf_to_df(loaded_tsdf)

    # The result should be a pure pandas DataFrame
    assert is_pure_pandas_df(pandas_df)
    assert not is_autogluon_tsdf(pandas_df)


def test_conversion_to_tsdf():
    """
    Tests the conversion from a pure pandas DataFrame back to a TimeSeriesDataFrame.
    This test doesn't need the fixture since it creates its own data.
    """
    from tabpfn_time_series.features.utils import (
        from_df_to_autogluon_tsdf,
    )

    # First, create a pure pandas DataFrame
    pure_df = pd.DataFrame(
        {
            "item_id": [0, 0, 0, 0],
            "timestamp": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "target": [1, 2, 3, 4],
        }
    )

    # Convert it to a TimeSeriesDataFrame
    new_tsdf = from_df_to_autogluon_tsdf(pure_df)

    # The result should now be a TimeSeriesDataFrame
    assert is_autogluon_tsdf(new_tsdf)
    assert not is_pure_pandas_df(new_tsdf)


def test_feature_pipeline():
    # old implementation
    from tabpfn_time_series.features_old import (
        RunningIndexFeature,
        CalendarFeature,
        AutoSeasonalFeature,
    )
    from tabpfn_time_series.features.utils import (
        from_autogluon_tsdf_to_df,
        from_df_to_autogluon_tsdf,
        load_data,
    )

    dataset_metadata = {
        "monash_tourism_monthly": {"prediction_length": 24},
        "m4_hourly": {"prediction_length": 48},
    }

    dataset_choice = "monash_tourism_monthly"
    num_time_series_subset = 2
    prediction_length = dataset_metadata[dataset_choice]["prediction_length"]

    # Loading Time Series Data Frames
    tsdf, train_tsdf, test_tsdf_ground_truth, test_tsdf = load_data(
        dataset_choice, num_time_series_subset, dataset_metadata
    )

    # Original Feature Transformer
    from tabpfn_time_series.features_old import FeatureTransformer

    selected_features = [
        RunningIndexFeature(),
        CalendarFeature(),
        AutoSeasonalFeature(),
    ]

    # Transform the data using the original feature transformer
    feature_transformer = FeatureTransformer(selected_features)
    train_tsdf_original, test_tsdf_original = feature_transformer.transform(
        train_tsdf, test_tsdf
    )

    import torch

    if torch.cuda.is_available():
        # Your code to run on GPU
        print("CUDA is available! Running predictions on GPU.")
        # Predict using TabPFNTimeSeriesPredictor
        from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode

        predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=TabPFNMode.LOCAL,
        )

        pred = predictor.predict(train_tsdf_original, test_tsdf_original)

        from tabpfn_time_series.features.utils import (
            quick_mase_evaluation,
        )

        _, average_mase_original = quick_mase_evaluation(
            train_tsdf, test_tsdf_ground_truth, pred, prediction_length
        )

    # New Feature Transformer
    from tabpfn_time_series.features import (
        RunningIndexFeatureTransformer,
        FeatureTransformer,
    )

    # load data
    tsdf, train_tsdf, test_tsdf_ground_truth, test_tsdf = load_data(
        dataset_choice, num_time_series_subset, dataset_metadata
    )

    # convert to pandas dataframe
    df = from_autogluon_tsdf_to_df(tsdf)
    assert not is_autogluon_tsdf(df)
    assert is_pure_pandas_df(df)

    # convert tsdf to pandas dataframe
    train_df, test_df, ground_truth = train_test_split_time_series(
        df, prediction_length
    )

    # Set up pipeline
    pipeline = [
        ("running_index", RunningIndexFeatureTransformer()),
        ("calendar", CalendarFeatureTransformer()),
        ("auto_seasonal", AutoSeasonalFeatureTransformer()),
    ]

    feat_transformer = FeatureTransformer(pipeline_steps=pipeline)

    # 1. Fit on the training data AND transform it in one step
    train_feat_PDs = feat_transformer.fit_transform(train_df)

    # 2. ONLY transform the test data using what was learned from train
    test_feat_PDs = feat_transformer.transform(test_df)

    train_feat_tsdf = from_df_to_autogluon_tsdf(train_feat_PDs)
    test_feat_tsdf = from_df_to_autogluon_tsdf(test_feat_PDs)

    assert set(train_tsdf_original.columns) == set(train_feat_tsdf.columns)
    assert set(test_tsdf_original.columns) == set(test_feat_tsdf.columns)

    assert len(train_feat_tsdf) == len(train_tsdf)
    assert len(test_feat_tsdf) == len(test_tsdf)

    def test_column_values(tsdf_original, tsdf_new):
        is_same = True
        for col in tsdf_new.columns:
            if col in tsdf_original.columns:
                is_same &= True
                # check if the column is the same
                if tsdf_original[col].equals(tsdf_new[col]):
                    is_same &= True
                else:
                    is_same &= False
                    raise ValueError(f"Column {col} is not the same")
            else:
                is_same &= False
                raise ValueError(f"Column {col} is not in tsdf_original")

        return is_same

    assert test_column_values(train_tsdf_original, train_feat_tsdf)
    assert test_column_values(test_tsdf_original, test_feat_tsdf)

    assert train_feat_tsdf.equals(train_tsdf_original)
    assert test_feat_tsdf.equals(test_tsdf_original)

    import torch

    if torch.cuda.is_available():
        print("CUDA is available! Running predictions on GPU.")
        predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=TabPFNMode.LOCAL,
        )

        pred = predictor.predict(train_feat_tsdf, test_feat_tsdf)

        from tabpfn_time_series.features.utils import (
            quick_mase_evaluation,
        )

        _, average_mase_new = quick_mase_evaluation(
            train_tsdf, test_tsdf_ground_truth, pred, prediction_length
        )

        assert average_mase_original == average_mase_new


# --- Test Data ---
# We'll use this DataFrame for both tests.
# The timestamps are intentionally mixed to ensure sorting logic is tested.
test_data = (
    pd.DataFrame(
        {
            "item_id": ["A", "B", "A", "B", "A"],
            "timestamp": pd.to_datetime(
                ["2023-01-02", "2023-01-04", "2023-01-01", "2023-01-05", "2023-01-03"]
            ),
            "target": [10, 20, 12, 22, 15],
        }
    )
    .sort_values("timestamp")
    .reset_index(drop=True)
)

# Reorder to make the test non-trivial
test_data = pd.DataFrame(
    {
        "item_id": ["A", "B", "A", "B", "A"],
        "timestamp": pd.to_datetime(
            ["2023-01-02", "2023-01-04", "2023-01-01", "2023-01-05", "2023-01-03"]
        ),
        "target": [10, 20, 12, 22, 15],
    },
    index=[10, 11, 12, 13, 14],
)  # Use a non-standard index


def test_per_item_mode():
    """Tests the default 'per_item' grouping behavior."""
    print("--- Running Test Case 1: Per-Item Mode ---")

    # 1. Define the pipeline
    per_item_pipeline = [("running_index", RunningIndexFeatureTransformer())]

    # 2. Instantiate the main transformer (uses group_by_column='item_id' by default)
    grouped_transformer = FeatureTransformer(pipeline_steps=per_item_pipeline)

    # 3. Fit and transform the data
    result_df = grouped_transformer.fit_transform(test_data)

    print("\nOriginal Data:")
    print(test_data)
    print("\nTransformed Data (Per-Item):")
    print(result_df)

    # 4. Assert the expected outcome
    # The running_index should be the cumulative count within each group after sorting by index
    expected_index_A = [1, 0, 2]  # For original rows 10, 12, 14
    expected_index_B = [0, 1]  # For original rows 11, 13

    # print("\nresult_df:")
    # print(result_df.loc[result_df['item_id'] == 'A', 'running_index'].tolist())
    # print(result_df.loc[result_df['item_id'] == 'B', 'running_index'].tolist())
    assert (
        result_df.loc[result_df["item_id"] == "A", "running_index"].tolist()
        == expected_index_A
    )
    assert (
        result_df.loc[result_df["item_id"] == "B", "running_index"].tolist()
        == expected_index_B
    )
    # assert result_df.index.equals(test_data.index), "Original index was not preserved"

    print("\n✅ Test Case 1 Passed!")


def test_global_mode():
    """Tests the global processing with group_by_column=None."""
    print("\n\n--- Running Test Case 2: Global Timestamp Mode ---")

    # 1. Define the pipeline
    global_pipeline_steps = [
        (
            "global_running_index",
            RunningIndexFeatureTransformer(mode="global_timestamp"),
        )
    ]

    # 2. Instantiate the main transformer for global processing
    global_transformer = FeatureTransformer(
        pipeline_steps=global_pipeline_steps,
    )

    # 3. Fit and transform the data
    result_df = global_transformer.fit_transform(test_data)

    print("\nOriginal Data:")
    print(test_data)
    print("\nTransformed Data (Global Timestamp):")
    print(result_df)

    # 4. Assert the expected outcome
    # The global index should correspond to the chronological order of timestamps
    # Original timestamps: 02-Jan, 04-Jan, 01-Jan, 05-Jan, 03-Jan
    # Sorted index      :   1,     3,       0,       4,       2
    expected_global_index = [1, 3, 0, 4, 2]
    assert result_df["running_index"].tolist() == expected_global_index
    # assert result_df.index.equals(test_data.index), "Original index was not preserved"

    print("\n✅ Test Case 2 Passed!")


# --- New Isolated Unit Tests for Transformers ---
@pytest.fixture
def sample_data():
    """Fixture for creating a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "item_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "timestamp": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-02-01",
                    "2023-02-02",
                    "2023-02-03",
                    "2023-02-04",
                ]
            ),
            "target": [10, 12, 15, 11, 20, 22, 25, 21],
        }
    )


@pytest.fixture
def sample_data_with_nan():
    """Fixture for creating a sample prediction DataFrame (target is NaN)."""
    return pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(
                ["2023-01-05", "2023-01-06", "2023-02-05", "2023-02-06"]
            ),
            "target": [np.nan, np.nan, np.nan, np.nan],
        }
    )


# --- Tests for RunningIndexFeatureTransformer ---


def test_running_index_per_item_mode(sample_data, sample_data_with_nan):
    """Tests RunningIndexFeatureTransformer in 'per_item' mode."""
    transformer = RunningIndexFeatureTransformer(mode="per_item")
    train_df = sample_data

    # Fit and transform training data
    transformer.fit(train_df)
    train_transformed = transformer.transform(train_df)

    # Assertions for training data
    pd.testing.assert_series_equal(
        train_transformed.loc[train_transformed["item_id"] == "A", "running_index"],
        pd.Series([0, 1, 2, 3], name="running_index", index=[0, 1, 2, 3]),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        train_transformed.loc[train_transformed["item_id"] == "B", "running_index"],
        pd.Series([0, 1, 2, 3], name="running_index", index=[4, 5, 6, 7]),
        check_names=False,
    )

    # Transform prediction data
    test_df = sample_data_with_nan
    test_transformed = transformer.transform(test_df)

    # Assertions for prediction data (index should continue from training)
    pd.testing.assert_series_equal(
        test_transformed.loc[test_transformed["item_id"] == "A", "running_index"],
        pd.Series([4, 5], name="running_index", index=[0, 1]),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        test_transformed.loc[test_transformed["item_id"] == "B", "running_index"],
        pd.Series([4, 5], name="running_index", index=[2, 3]),
        check_names=False,
    )


def test_running_index_global_mode(sample_data, sample_data_with_nan):
    """Tests RunningIndexFeatureTransformer in 'global_timestamp' mode."""
    transformer = RunningIndexFeatureTransformer(mode="global_timestamp")
    train_df = sample_data

    # Fit and transform training data
    transformer.fit(train_df)
    train_transformed = transformer.transform(train_df)
    expected_train_index = pd.Series(range(8), name="running_index", index=range(8))
    pd.testing.assert_series_equal(
        train_transformed["running_index"], expected_train_index, check_names=False
    )

    # Transform prediction data
    test_df = sample_data_with_nan
    test_transformed = transformer.transform(test_df)
    expected_test_index = pd.Series(range(8, 12), name="running_index", index=range(4))
    pd.testing.assert_series_equal(
        test_transformed["running_index"], expected_test_index, check_names=False
    )


def test_running_index_new_item_raises_error(sample_data):
    """Tests that a new item in transform raises a ValueError in per_item mode."""
    transformer = RunningIndexFeatureTransformer(mode="per_item")
    transformer.fit(sample_data)

    new_item_df = pd.DataFrame(
        {
            "item_id": ["C"],
            "timestamp": [pd.to_datetime("2023-01-01")],
            "target": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="No fitted training data found for item_id"):
        transformer.transform(new_item_df)


# --- Tests for CalendarFeatureTransformer ---


def test_calendar_feature_transformer(sample_data):
    """Tests basic functionality of CalendarFeatureTransformer."""
    transformer = CalendarFeatureTransformer(
        components=["year", "month", "day"],
        seasonal_features={"day_of_week": [7]},
    )
    transformer.fit(sample_data)
    transformed_df = transformer.transform(sample_data)

    # Check for new columns
    expected_cols = ["year", "month", "day", "day_of_week_sin", "day_of_week_cos"]
    for col in expected_cols:
        assert col in transformed_df.columns

    # Check values for a specific row
    first_row = transformed_df.iloc[0]
    timestamp = sample_data.iloc[0]["timestamp"]
    assert first_row["year"] == timestamp.year
    assert first_row["month"] == timestamp.month
    assert first_row["day"] == timestamp.day

    # 2023-01-01 was a Sunday (day_of_week = 6)
    expected_sin = np.sin(2 * np.pi * 6 / 6)
    expected_cos = np.cos(2 * np.pi * 6 / 6)
    assert np.isclose(first_row["day_of_week_sin"], expected_sin)
    assert np.isclose(first_row["day_of_week_cos"], expected_cos)


def test_calendar_feature_empty_df():
    """Tests CalendarFeatureTransformer with an empty DataFrame."""
    transformer = CalendarFeatureTransformer()

    # Corrected way to create the empty DataFrame
    empty_df = pd.DataFrame({"item_id": [], "timestamp": [], "target": []})

    # Now, set the dtypes for the empty columns
    empty_df["timestamp"] = pd.to_datetime(empty_df["timestamp"])
    empty_df["item_id"] = empty_df["item_id"].astype(str)

    transformer.fit(empty_df)
    transformed_df = transformer.transform(empty_df)

    assert transformed_df.empty
    # Check that it has the feature columns, even if there are no rows
    assert "year" in transformed_df.columns


# --- Tests for AutoSeasonalFeatureTransformer ---


@pytest.fixture
def seasonal_data():
    """Fixture for creating data with a clear seasonal pattern."""
    time = np.arange(100)
    # Strong period of 10
    seasonal_signal = np.sin(2 * np.pi * time / 10)
    return pd.DataFrame(
        {
            "item_id": ["S1"] * 100,
            "timestamp": pd.to_datetime(
                pd.to_datetime("2023-01-01") + pd.to_timedelta(time, unit="D")
            ),
            "target": seasonal_signal,
        }
    )


@pytest.fixture
def non_seasonal_data():
    """Fixture for creating data with no clear seasonal pattern."""
    time = np.arange(100)
    random_signal = np.random.rand(100)
    return pd.DataFrame(
        {
            "item_id": ["R1"] * 100,
            "timestamp": pd.to_datetime(
                pd.to_datetime("2023-01-01") + pd.to_timedelta(time, unit="D")
            ),
            "target": random_signal,
        }
    )


def test_autoseasonal_detects_period(seasonal_data):
    """Tests that AutoSeasonalFeatureTransformer correctly detects a known period."""
    transformer = AutoSeasonalFeatureTransformer(max_top_k=1, do_detrend=False)
    transformer.fit(seasonal_data)

    # Check if the correct period was detected for item 'S1'
    detected_params = transformer.fitted_autoseasonal_per_item["S1"]
    detected_periods = detected_params["periods_"]
    assert len(detected_periods) >= 1
    assert np.isclose(detected_periods[0], 10, atol=0.1)


# def test_autoseasonal_handles_non_seasonal(non_seasonal_data):
#     """Tests behavior with non-seasonal (random) data."""
#     transformer = AutoSeasonalFeatureTransformer(max_top_k=1, magnitude_threshold=0.5)
#     transformer.fit(non_seasonal_data)

#     # With a high threshold, no significant period should be found
#     detected_params = transformer.fitted_autoseasonal_per_item["R1"]
#     assert len(detected_params["periods_"]) == 0


def test_autoseasonal_transform_creates_features(seasonal_data):
    """Tests that the transform method adds the correct seasonal features."""
    # FIX: Increase max_top_k to test the padding logic.
    transformer = AutoSeasonalFeatureTransformer(max_top_k=2, do_detrend=False)
    transformed_df = transformer.fit_transform(seasonal_data)

    # Check for the primary detected feature
    assert "sin_#0" in transformed_df.columns
    assert "cos_#0" in transformed_df.columns

    # NOW this assertion is correct, because it should be a placeholder.
    assert "sin_#1" in transformed_df.columns
    assert "cos_#1" in transformed_df.columns

    # Verify the placeholder columns are filled with 0.0
    assert transformed_df["sin_#1"].sum() == 0.0
    assert transformed_df["cos_#1"].sum() == 0.0


def test_autoseasonal_short_series():
    """Tests that short series are handled gracefully."""
    short_df = pd.DataFrame(
        {
            "item_id": ["A"],
            "timestamp": [pd.to_datetime("2023-01-01")],
            "target": [1.0],
        }
    )
    transformer = AutoSeasonalFeatureTransformer(max_top_k=1)
    transformer.fit(short_df)

    # No periods should be found for a series of length 1
    detected_params = transformer.fitted_autoseasonal_per_item["A"]
    assert len(detected_params["periods_"]) == 0

    # Transform should still work and add empty columns
    transformed_df = transformer.transform(short_df)
    assert transformed_df["sin_#0"].iloc[0] == 0.0
    assert transformed_df["cos_#0"].iloc[0] == 0.0


def test_autoseasonal_transform_prediction_offset(seasonal_data):
    """Tests that the time index is correctly offset for prediction dataframes."""
    train_df = seasonal_data
    transformer = AutoSeasonalFeatureTransformer(max_top_k=1, do_detrend=False)
    transformer.fit(train_df)

    # Create a prediction dataframe (target is NaN)
    pred_len = 10
    pred_df = pd.DataFrame(
        {
            "item_id": ["S1"] * pred_len,
            "timestamp": pd.to_datetime(
                pd.to_datetime("2023-01-01")
                + pd.to_timedelta(np.arange(100, 100 + pred_len), unit="D")
            ),
            "target": [np.nan] * pred_len,
        }
    )
    transformed_pred = transformer.transform(pred_df)

    # The time index for the prediction frame should start after the training frame
    train_len = len(train_df)
    time_idx_pred = np.arange(train_len, train_len + pred_len)
    period = 10
    expected_sin = np.sin(2 * np.pi * time_idx_pred / period)

    pd.testing.assert_series_equal(
        transformed_pred["sin_#0"],
        pd.Series(expected_sin, name="sin_#0", index=range(pred_len)),
        check_names=False,
        atol=1e-9,
    )
