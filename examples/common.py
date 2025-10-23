"""
Common utilities for examples.

This module contains shared functions used across multiple examples.
"""

import pandas as pd
import numpy as np
from tabpfn_time_series import TimeSeriesDataFrame
from tabpfn_time_series.data_preparation import generate_test_X


def create_sample_time_series():
    """Create a simple time series dataset for demonstration.

    Returns
    -------
    train_tsdf : TimeSeriesDataFrame
        Training data with simple random values
    test_tsdf : TimeSeriesDataFrame
        Test timestamps for prediction
    """
    # Create a simple time series dataframe
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    item_ids = [0, 1]

    # Create train data with target
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

    train_tsdf = TimeSeriesDataFrame(
        pd.DataFrame(train_data),
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # Generate test data
    test_tsdf = generate_test_X(train_tsdf, prediction_length=5)

    return train_tsdf, test_tsdf
