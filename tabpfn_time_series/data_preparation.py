import os
import pandas as pd
import numpy as np
from typing import Tuple, List
from joblib import Parallel, delayed

import datasets
from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame


def _extract_max_timestamps(
    train_data: TimeSeriesDataFrame,
) -> Tuple[np.ndarray, pd.Index]:
    """Find the last timestamp for each time series item."""
    item_identifiers = train_data.item_ids

    if train_data.index.is_monotonic_increasing:
        # Fast path: data is sorted, use index pointers
        index_pointers = train_data.get_indptr()
        all_timestamps = train_data.index.get_level_values("timestamp")
        last_timestamps = all_timestamps[index_pointers[1:] - 1]
    else:
        # Slow path: data is unsorted, use groupby
        last_timestamps_by_item = train_data.groupby(level="item_id", sort=False)[
            train_data.columns[0]
        ].apply(lambda group: group.index.get_level_values("timestamp").max())
        last_timestamps = last_timestamps_by_item.values

    return last_timestamps, item_identifiers


def _create_future_timestamps(
    last_timestamps: np.ndarray, frequency: str, num_future_steps: int
) -> List[pd.Timestamp]:
    """Generate future timestamps starting from each item's last timestamp."""
    time_offset = pd.tseries.frequencies.to_offset(frequency)
    future_timestamps = []

    for last_timestamp in last_timestamps:
        forecast_start = last_timestamp + time_offset
        item_future_timestamps = pd.date_range(
            start=forecast_start, periods=num_future_steps, freq=frequency
        )
        future_timestamps.extend(item_future_timestamps)

    return future_timestamps


def _create_future_timestamps_parallel(
    last_timestamps: np.ndarray, frequency: str, num_future_steps: int, num_workers: int
) -> List[pd.Timestamp]:
    """Generate future timestamps using parallel processing for large datasets."""

    def process_chunk(timestamp_chunk, chunk_index):
        time_offset = pd.tseries.frequencies.to_offset(frequency)
        chunk_futures = []

        for last_timestamp in timestamp_chunk:
            forecast_start = last_timestamp + time_offset
            timestamps = pd.date_range(
                start=forecast_start, periods=num_future_steps, freq=frequency
            )
            chunk_futures.extend(timestamps)

        return chunk_futures, chunk_index

    # Split into chunks
    chunk_size = max(1, len(last_timestamps) // num_workers)
    chunks = [
        last_timestamps[i : i + chunk_size]
        for i in range(0, len(last_timestamps), chunk_size)
    ]

    # Process in parallel
    results = Parallel(n_jobs=num_workers, backend="threading")(
        delayed(process_chunk)(chunk, idx) for idx, chunk in enumerate(chunks)
    )

    # Reassemble in correct order
    future_timestamps = []
    for chunk_futures, _ in sorted(results, key=lambda x: x[1]):
        future_timestamps.extend(chunk_futures)

    return future_timestamps


def generate_test_X(
    train_tsdf: TimeSeriesDataFrame,
    prediction_length: int,
    freq: str,
) -> TimeSeriesDataFrame:
    """
    Create test data for forecasting by generating future time periods.

    Automatically adapts to dataset size - uses parallel processing for 1000+ items,
    single-threaded processing for smaller datasets.

    Example:
        If your training data ends on 2023-12-31 and prediction_length=7,
        this creates test data for 2024-01-01 through 2024-01-07.

    Args:
        train_tsdf: Historical time series data used for training
        prediction_length: Number of future time steps to generate per item

    Returns:
        test_tsdf: Test dataset with future timestamps and NaN targets
    """
    # Extract basic information
    last_timestamps, item_ids = _extract_max_timestamps(train_tsdf)
    num_items = len(item_ids)

    # Automatically choose processing method based on dataset size
    if num_items >= 50:
        # Use parallel processing for large datasets
        num_workers = min(os.cpu_count() - 1, num_items)
        future_timestamps = _create_future_timestamps_parallel(
            last_timestamps, freq, prediction_length, num_workers
        )
    else:
        # Use single-threaded processing for smaller datasets
        future_timestamps = _create_future_timestamps(
            last_timestamps, freq, prediction_length
        )

    # Build the test DataFrame
    total_points = num_items * prediction_length
    repeated_item_ids = np.repeat(item_ids.values, prediction_length)
    nan_targets = np.full(total_points, np.nan, dtype=np.float64)

    test_dataframe = pd.DataFrame(
        {
            "target": nan_targets,
            "timestamp": future_timestamps,
            "item_id": repeated_item_ids,
        }
    )

    # Convert and validate
    test_tsdf = TimeSeriesDataFrame.from_data_frame(test_dataframe)

    if not test_tsdf.item_ids.equals(train_tsdf.item_ids):
        raise ValueError("Mismatch between training and test item IDs")

    return test_tsdf


# From pandas._libs.tslibs.dtypes.OFFSET_TO_PERIOD_FREQSTR
offset_alias_to_period_alias = {
    "WEEKDAY": "D",
    "EOM": "M",
    "BME": "M",
    "SME": "M",
    "BQS": "Q",
    "QS": "Q",
    "BQE": "Q",
    "BQE-DEC": "Q",
    "BQE-JAN": "Q",
    "BQE-FEB": "Q",
    "BQE-MAR": "Q",
    "BQE-APR": "Q",
    "BQE-MAY": "Q",
    "BQE-JUN": "Q",
    "BQE-JUL": "Q",
    "BQE-AUG": "Q",
    "BQE-SEP": "Q",
    "BQE-OCT": "Q",
    "BQE-NOV": "Q",
    "MS": "M",
    "D": "D",
    "B": "B",
    "min": "min",
    "s": "s",
    "ms": "ms",
    "us": "us",
    "ns": "ns",
    "h": "h",
    "QE": "Q",
    "QE-DEC": "Q-DEC",
    "QE-JAN": "Q-JAN",
    "QE-FEB": "Q-FEB",
    "QE-MAR": "Q-MAR",
    "QE-APR": "Q-APR",
    "QE-MAY": "Q-MAY",
    "QE-JUN": "Q-JUN",
    "QE-JUL": "Q-JUL",
    "QE-AUG": "Q-AUG",
    "QE-SEP": "Q-SEP",
    "QE-OCT": "Q-OCT",
    "QE-NOV": "Q-NOV",
    "YE": "Y",
    "YE-DEC": "Y-DEC",
    "YE-JAN": "Y-JAN",
    "YE-FEB": "Y-FEB",
    "YE-MAR": "Y-MAR",
    "YE-APR": "Y-APR",
    "YE-MAY": "Y-MAY",
    "YE-JUN": "Y-JUN",
    "YE-JUL": "Y-JUL",
    "YE-AUG": "Y-AUG",
    "YE-SEP": "Y-SEP",
    "YE-OCT": "Y-OCT",
    "YE-NOV": "Y-NOV",
    "W": "W",
    "ME": "M",
    "Y": "Y",
    "BYE": "Y",
    "BYE-DEC": "Y",
    "BYE-JAN": "Y",
    "BYE-FEB": "Y",
    "BYE-MAR": "Y",
    "BYE-APR": "Y",
    "BYE-MAY": "Y",
    "BYE-JUN": "Y",
    "BYE-JUL": "Y",
    "BYE-AUG": "Y",
    "BYE-SEP": "Y",
    "BYE-OCT": "Y",
    "BYE-NOV": "Y",
    "YS": "Y",
    "BYS": "Y",
    "QS-JAN": "Q",
    "QS-FEB": "Q",
    "QS-MAR": "Q",
    "QS-APR": "Q",
    "QS-MAY": "Q",
    "QS-JUN": "Q",
    "QS-JUL": "Q",
    "QS-AUG": "Q",
    "QS-SEP": "Q",
    "QS-OCT": "Q",
    "QS-NOV": "Q",
    "QS-DEC": "Q",
    "BQS-JAN": "Q",
    "BQS-FEB": "Q",
    "BQS-MAR": "Q",
    "BQS-APR": "Q",
    "BQS-MAY": "Q",
    "BQS-JUN": "Q",
    "BQS-JUL": "Q",
    "BQS-AUG": "Q",
    "BQS-SEP": "Q",
    "BQS-OCT": "Q",
    "BQS-NOV": "Q",
    "BQS-DEC": "Q",
    "YS-JAN": "Y",
    "YS-FEB": "Y",
    "YS-MAR": "Y",
    "YS-APR": "Y",
    "YS-MAY": "Y",
    "YS-JUN": "Y",
    "YS-JUL": "Y",
    "YS-AUG": "Y",
    "YS-SEP": "Y",
    "YS-OCT": "Y",
    "YS-NOV": "Y",
    "YS-DEC": "Y",
    "BYS-JAN": "Y",
    "BYS-FEB": "Y",
    "BYS-MAR": "Y",
    "BYS-APR": "Y",
    "BYS-MAY": "Y",
    "BYS-JUN": "Y",
    "BYS-JUL": "Y",
    "BYS-AUG": "Y",
    "BYS-SEP": "Y",
    "BYS-OCT": "Y",
    "BYS-NOV": "Y",
    "BYS-DEC": "Y",
    "Y-JAN": "Y-JAN",
    "Y-FEB": "Y-FEB",
    "Y-MAR": "Y-MAR",
    "Y-APR": "Y-APR",
    "Y-MAY": "Y-MAY",
    "Y-JUN": "Y-JUN",
    "Y-JUL": "Y-JUL",
    "Y-AUG": "Y-AUG",
    "Y-SEP": "Y-SEP",
    "Y-OCT": "Y-OCT",
    "Y-NOV": "Y-NOV",
    "Y-DEC": "Y-DEC",
    "Q-JAN": "Q-JAN",
    "Q-FEB": "Q-FEB",
    "Q-MAR": "Q-MAR",
    "Q-APR": "Q-APR",
    "Q-MAY": "Q-MAY",
    "Q-JUN": "Q-JUN",
    "Q-JUL": "Q-JUL",
    "Q-AUG": "Q-AUG",
    "Q-SEP": "Q-SEP",
    "Q-OCT": "Q-OCT",
    "Q-NOV": "Q-NOV",
    "Q-DEC": "Q-DEC",
    "W-MON": "W-MON",
    "W-TUE": "W-TUE",
    "W-WED": "W-WED",
    "W-THU": "W-THU",
    "W-FRI": "W-FRI",
    "W-SAT": "W-SAT",
    "W-SUN": "W-SUN",
}


# From https://github.com/amazon-science/chronos-forecasting/blob/ad410c9c0ae0d499aeec9a7af09b0636844b6274/scripts/evaluation/evaluate.py#L28
def to_gluonts_univariate(hf_dataset: datasets.Dataset):
    series_fields = [
        col
        for col in hf_dataset.features
        if isinstance(hf_dataset.features[col], datasets.Sequence)
    ]
    series_fields.remove("timestamp")
    dataset_length = hf_dataset.info.splits["train"].num_examples * len(series_fields)
    dataset_freq = pd.infer_freq(hf_dataset[0]["timestamp"])
    dataset_freq = offset_alias_to_period_alias.get(dataset_freq, dataset_freq)

    gts_dataset = []
    for hf_entry in hf_dataset:
        for field in series_fields:
            gts_dataset.append(
                {
                    "start": pd.Period(
                        hf_entry["timestamp"][0],
                        freq=dataset_freq,
                    ),
                    "target": hf_entry[field],
                }
            )
    assert len(gts_dataset) == dataset_length

    return gts_dataset


def split_time_series_to_X_y(df: pd.DataFrame, target_col="target"):
    X = pd.DataFrame(df.drop(columns=[target_col]))
    y = pd.DataFrame(df[target_col])
    return X, y
