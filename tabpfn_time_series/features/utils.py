import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame
from typing import Tuple, Dict


def train_test_split_time_series(
    df: pd.DataFrame, prediction_length: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into training and testing sets for each time series.

    For each 'item_id', the last 'prediction_length' time steps are held out
    as the test set. The target values in the returned test set are replaced
    with NaN, and the original values are returned in a separate ground_truth DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame, which must contain 'item_id',
                           'timestamp', and 'target' columns.
        prediction_length (int): The number of time steps from the end of
                                 each time series to allocate to the test set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing:
            - train_df (pd.DataFrame): The training data, containing all but the
                                       last 'prediction_length' steps for each item.
            - test_df (pd.DataFrame): The test data, containing the last
                                      'prediction_length' steps for each item,
                                      but with the 'target' column set to NaN.
            - ground_truth (pd.DataFrame): The ground truth data, containing the
                                           last 'prediction_length' steps for
                                           each item with the original 'target' values.
    """
    train_list = []
    test_list = []
    for item_id, group in df.groupby("item_id"):
        group_sorted = group.sort_values("timestamp")
        if len(group_sorted) <= prediction_length:
            # If not enough data, put all in train
            train_list.append(group_sorted)
            continue
        train_list.append(group_sorted.iloc[:-prediction_length])
        test_list.append(group_sorted.iloc[-prediction_length:])
    train_df = pd.concat(train_list, axis=0).reset_index(drop=True)
    test_df = (
        pd.concat(test_list, axis=0).reset_index(drop=True)
        if test_list
        else pd.DataFrame(columns=df.columns)
    )

    # after the train test split, make the "target" column in test_df to be NaN
    ground_truth = test_df.copy()
    test_df["target"] = np.nan

    return train_df, test_df, ground_truth


def from_autogluon_tsdf_to_df(tsdf: TimeSeriesDataFrame) -> pd.DataFrame:
    """
    Converts an AutoGluon TimeSeriesDataFrame to a standard pandas DataFrame.

    Resets the multi-level index ('item_id', 'timestamp') of the
    TimeSeriesDataFrame into regular columns.

    Args:
        tsdf (TimeSeriesDataFrame): The AutoGluon TimeSeriesDataFrame to convert.

    Returns:
        pd.DataFrame: A pandas DataFrame with 'item_id' and 'timestamp' as columns.
    """
    return tsdf.copy().to_data_frame().reset_index()


def from_df_to_autogluon_tsdf(df: pd.DataFrame) -> TimeSeriesDataFrame:
    """
    Converts a pandas DataFrame to an AutoGluon TimeSeriesDataFrame.

    The input DataFrame must have 'item_id' and 'timestamp' columns, which will be
    used to create the multi-level index of the TimeSeriesDataFrame.

    Args:
        df (pd.DataFrame): The pandas DataFrame to convert.

    Returns:
        TimeSeriesDataFrame: An AutoGluon TimeSeriesDataFrame.
    """
    df = df.copy()
    # Drop column "index" if there is any
    if "index" in df.columns:
        df.drop(columns=["index"], inplace=True)
    return TimeSeriesDataFrame.from_data_frame(df)


def quick_mase_evaluation(
    train_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    prediction_length: int,
) -> Tuple[pd.DataFrame, float]:
    """
    Computes the Mean Absolute Scaled Error (MASE) for time series predictions.

    Calculates the MASE score for each item_id and provides an overall average.
    This function internally converts the input DataFrames to AutoGluon's
    TimeSeriesDataFrame format for the calculation.

    Args:
        train_df (pd.DataFrame): The training data.
        ground_truth_df (pd.DataFrame): The ground truth data for the test set.
        pred_df (pd.DataFrame): The predictions made by the model.
        prediction_length (int): The length of the prediction horizon.

    Returns:
        Tuple[pd.DataFrame, float]: A tuple containing:
            - final_results (pd.DataFrame): A DataFrame with 'item_id' and
                                            'mase_score' for each time series,
                                            plus a final row for the 'AVERAGE'.
            - average_mase (float): The average MASE score across all items.
    """
    from autogluon.timeseries.metrics.point import MASE
    from autogluon.timeseries.utils.datetime import get_seasonality
    import pandas as pd

    mase_results = []
    train_tsdf = from_df_to_autogluon_tsdf(train_df)
    test_tsdf_ground_truth = from_df_to_autogluon_tsdf(ground_truth_df)
    pred = from_df_to_autogluon_tsdf(pred_df)
    pred = pred.copy()

    # Loop over each item_id and calculate MASE score
    for item_id, df_item in train_tsdf.groupby(level="item_id"):
        mase_computer = MASE()
        mase_computer.clear_past_metrics()

        pred["mean"] = pred["target"]

        mase_computer.save_past_metrics(
            data_past=train_tsdf.loc[[item_id]],
            seasonal_period=get_seasonality(train_tsdf.freq),
        )

        mase_score = mase_computer.compute_metric(
            data_future=test_tsdf_ground_truth.loc[[item_id]].slice_by_timestep(
                -prediction_length, None
            ),
            predictions=pred.loc[[item_id]],
        )
        print(f"mase_score: {mase_score}")
        mase_results.append({"item_id": item_id, "mase_score": mase_score})

    # Create DataFrame with individual results
    results_df = pd.DataFrame(mase_results)

    # Add average row
    average_mase = results_df["mase_score"].mean()
    average_row = pd.DataFrame({"item_id": ["AVERAGE"], "mase_score": [average_mase]})

    # Combine results
    final_results = pd.concat([results_df, average_row], ignore_index=True)

    return final_results, average_mase


def load_data(
    dataset_choice: str, num_time_series_subset: int, dataset_metadata: Dict
) -> Tuple[
    TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame
]:
    """
    Loads and prepares a time series dataset for forecasting.

    This function performs several key steps:
    1. Loads a specified time series dataset from the "autogluon/chronos_datasets" collection.
    2. Converts the dataset into an AutoGluon TimeSeriesDataFrame.
    3. Selects a specified number of individual time series from the dataset.
    4. Splits the data into training and testing sets for model training and evaluation.

    Args:
        dataset_choice (str): The name of the dataset to load from the
                              Hugging Face hub.
                              Example: "nn5_daily_without_missing"
        num_time_series_subset (int): The number of time series to select from
                                      the dataset.
        dataset_metadata (Dict): A dictionary containing metadata about the
                                 dataset, including the 'prediction_length'.

    Returns:
        Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame, TimeSeriesDataFrame]:
            A tuple containing four TimeSeriesDataFrames:
            - tsdf: The complete dataframe for the selected subset.
            - train_tsdf: The training portion of the data.
            - test_tsdf_ground_truth: The ground truth for the test set.
            - test_tsdf: The test set input, ready for predictions.
    """

    from datasets import load_dataset
    from autogluon.timeseries import TimeSeriesDataFrame

    from tabpfn_time_series.data_preparation import (
        to_gluonts_univariate,
        generate_test_X,
    )

    prediction_length = dataset_metadata[dataset_choice]["prediction_length"]
    dataset = load_dataset("autogluon/chronos_datasets", dataset_choice)

    tsdf = TimeSeriesDataFrame(to_gluonts_univariate(dataset["train"]))
    tsdf = tsdf[
        tsdf.index.get_level_values("item_id").isin(
            tsdf.item_ids[:num_time_series_subset]
        )
    ]
    train_tsdf, test_tsdf_ground_truth = tsdf.train_test_split(
        prediction_length=prediction_length
    )
    test_tsdf = generate_test_X(train_tsdf, prediction_length)

    return tsdf, train_tsdf, test_tsdf_ground_truth, test_tsdf
