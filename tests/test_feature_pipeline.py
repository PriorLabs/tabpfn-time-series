import pytest
import pandas as pd


def is_autogluon_tsdf(df):
    from autogluon.timeseries import TimeSeriesDataFrame

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

    from tabpfn_time_series.features_sklearn.utils_pipeline import (
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
    from tabpfn_time_series.features_sklearn.utils_pipeline import (
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
    from tabpfn_time_series.features_sklearn.utils_pipeline import (
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
    from tabpfn_time_series.features import (
        RunningIndexFeature,
        CalendarFeature,
        AutoSeasonalFeature,
    )
    from tabpfn_time_series.features_sklearn.utils_pipeline import (
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
    from tabpfn_time_series import FeatureTransformer

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

        from tabpfn_time_series.features_sklearn.utils_pipeline import (
            quick_mase_evaluation,
        )

        _, average_mase_original = quick_mase_evaluation(
            train_tsdf, test_tsdf_ground_truth, pred, prediction_length
        )

    # New Feature Transformer
    from tabpfn_time_series.features_sklearn.feature_pipeline import (
        RunningIndexFeatureTransformer,
        AutoSeasonalFeatureTransformer,
        CalendarFeatureTransformer,
        FeaturePipeline,
    )
    from tabpfn_time_series.features_sklearn.utils_pipeline import (
        train_test_split_time_series,
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

    feature_pipeline = FeaturePipeline(pipeline_steps=pipeline)

    # 1. Fit on the training data AND transform it in one step
    train_feat_PDs = feature_pipeline.fit_transform(train_df)

    # 2. ONLY transform the test data using what was learned from train
    test_feat_PDs = feature_pipeline.transform(test_df)

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

        from tabpfn_time_series.features_sklearn.utils_pipeline import (
            quick_mase_evaluation,
        )

        _, average_mase_new = quick_mase_evaluation(
            train_tsdf, test_tsdf_ground_truth, pred, prediction_length
        )

        assert average_mase_original == average_mase_new
