import pandas as pd
import numpy as np
from autogluon.timeseries import TimeSeriesDataFrame


def train_test_split_time_series(df: pd.DataFrame, prediction_length: int):
    """
    Splits a DataFrame into train and test sets per item_id using prediction_length.
    Args:
        df (pd.DataFrame): Input DataFrame with 'item_id' and 'timestamp'.
        prediction_length (int): Number of last time steps to use for test per item_id.
    Returns:
        train_df (pd.DataFrame): Training set (all but last prediction_length per item_id).
        test_df (pd.DataFrame): Test set (last prediction_length per item_id).
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


def from_autogluon_tsdf_to_df(tsdf):
    return tsdf.copy().to_data_frame().reset_index()


def from_df_to_autogluon_tsdf(df):
    df = df.copy()
    # Drop column "index" if there is any
    if "index" in df.columns:
        df.drop(columns=["index"], inplace=True)
    return TimeSeriesDataFrame.from_data_frame(df)


def quick_mase_evaluation(train_df, ground_truth_df, pred_df, prediction_length):
    """
    Compute MASE scores for each item_id and overall average.

    Args:
        train_tsdf: TimeSeriesDataFrame, the training data
        test_tsdf_ground_truth: TimeSeriesDataFrame, the ground truth data
        pred: TimeSeriesDataFrame, the predicted data
        prediction_length: int, the prediction length

    Note:
        - The input data is expected to be in the format of TimeSeriesDataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns ['item_id', 'mase_score']
                     Last row contains average with item_id='AVERAGE'
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

        # for debugging
        # print(f'test_tsdf_ground_truth.loc[[item_id]]: {len(test_tsdf_ground_truth.loc[[item_id]].slice_by_timestep(-prediction_length, None))}')
        # print(f'pred.loc[[item_id]]: {len(pred.loc[[item_id]])}')

        mase_score = mase_computer.compute_metric(
            data_future=test_tsdf_ground_truth.loc[[item_id]].slice_by_timestep(
                -prediction_length, None
            ),
            predictions=pred.loc[[item_id]],
        )

        mase_results.append({"item_id": item_id, "mase_score": mase_score})

    # Create DataFrame with individual results
    results_df = pd.DataFrame(mase_results)

    # Add average row
    average_mase = results_df["mase_score"].mean()
    average_row = pd.DataFrame({"item_id": ["AVERAGE"], "mase_score": [average_mase]})

    # Combine results
    final_results = pd.concat([results_df, average_row], ignore_index=True)

    return final_results
