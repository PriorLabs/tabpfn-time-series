from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset

from tabpfn import TabPFNRegressor
from tabpfn.preprocessing import RegressorEnsembleConfig
from tabpfn.model.bar_distribution import FullSupportBarDistribution

from tabpfn_time_series.experimental.finetuning.data.raw_time_series_dataset import (
    RawTimeSeriesDataset,
)
from tabpfn_time_series.experimental.features.feature_generator_base import (
    FeatureGenerator,
)


logger = logging.getLogger(__name__)


def make_split_fn(
    split_idx: int,
) -> Callable[
    [np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    """Creates a split function that splits data at a given index."""

    def split_fn(
        X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Splits X and y into training and testing sets at `split_idx`."""
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test

    return split_fn


T_ensemble_list: TypeAlias = list  # Length = # of ensembles


@dataclass
class RegressionDatasetCollectionWithPreprocessing:
    X_train_preprocessed: T_ensemble_list[torch.Tensor]
    X_test_preprocessed: T_ensemble_list[torch.Tensor]
    y_train_standardized: T_ensemble_list[torch.Tensor]
    y_test_standardized: torch.Tensor
    cat_ixs: T_ensemble_list[Optional[list[int]]]
    conf: T_ensemble_list[RegressorEnsembleConfig]
    normalized_bardist: FullSupportBarDistribution
    bardist: FullSupportBarDistribution
    X_train_raw: torch.Tensor
    y_train_raw: torch.Tensor
    X_test_raw: torch.Tensor
    y_test_raw: torch.Tensor


class PreprocessedTimeSeriesDataset(IterableDataset):
    """
    A PyTorch IterableDataset that wraps a RawTimeSeriesDataset to perform
    featurization and preprocessing for each sample on the fly.

    This class creates a processing pipeline where raw samples are drawn from
    the source dataset and immediately transformed, ready for model consumption.
    """

    def __init__(
        self,
        raw_dataset: RawTimeSeriesDataset,
        model: TabPFNRegressor,
        feature_generators: List[FeatureGenerator],
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            raw_dataset: An instance of RawTimeSeriesDataset.
            model: The TabPFNRegressor model instance for preprocessing.
            feature_generators: A list of feature generators to apply to the
                raw time series data.
            torch_dtype: The torch.dtype to use for the output tensors.
        """

        if not feature_generators:
            raise ValueError("feature_generators must not be empty")

        self.raw_dataset = raw_dataset
        self.model = model
        self.feature_generators = feature_generators
        self.torch_dtype = torch_dtype

    def __iter__(self):
        """
        Creates an iterator that yields fully processed samples.
        """
        for sample in self.raw_dataset:
            yield self._process_sample(sample)

    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves, featurizes, and preprocesses a single time series sample.
        """
        # Featurization
        invalid_past_mask = sample["past_is_pad"].astype(bool)
        past_target = sample["past_target"][~invalid_past_mask]
        future_target = sample["future_target"]
        past_len = len(past_target)

        all_timestamps = self._reconstruct_timestamps(
            sample["forecast_start"], past_len, len(future_target)
        )

        # TODO: temporary workaround to avoid leakage
        #   (to be fixed together with AutoSeasonalFeature)
        ts_df = pd.DataFrame(
            data={
                "target": np.concatenate([past_target, future_target]),
                "dummy_col": np.zeros(len(all_timestamps)),
            },
            index=all_timestamps,
        )

        # Filter out missing values
        ts_df = ts_df.dropna()
        features_only_df, all_targets = ts_df.drop(columns=["target"]), ts_df["target"]

        # Generate features using timestamp only (no target)
        for gen in self.feature_generators:
            features_only_df = gen.generate(features_only_df)
        features_only_df = features_only_df.drop(columns=["dummy_col"])

        # Create a split function for the current sample
        split_fn = make_split_fn(past_len)

        # Preprocessing with TabPFN
        preprocessed_collections = list(
            map(
                lambda x: RegressionDatasetCollectionWithPreprocessing(*x),
                self.model.get_preprocessed_datasets(
                    X_raw=features_only_df,
                    y_raw=all_targets,
                    split_fn=split_fn,
                    max_data_size=None,
                ),
            )
        )

        assert len(preprocessed_collections) == 1, "Only one split is expected"
        preprocessed_collection = preprocessed_collections[0]

        return {
            "X_train_preprocessed": [
                x.to(self.torch_dtype)
                for x in preprocessed_collection.X_train_preprocessed
            ],
            "y_train_standardized": [
                y.to(self.torch_dtype)
                for y in preprocessed_collection.y_train_standardized
            ],
            "X_test_preprocessed": [
                x.to(self.torch_dtype)
                for x in preprocessed_collection.X_test_preprocessed
            ],
            "y_test_standardized": preprocessed_collection.y_test_standardized.to(
                self.torch_dtype
            ),
            "cat_ixs": preprocessed_collection.cat_ixs,
            "conf": preprocessed_collection.conf,
            "bardist": preprocessed_collection.bardist,
            "normalized_bardist": preprocessed_collection.normalized_bardist,
            "X_train_raw": preprocessed_collection.X_train_raw.to(self.torch_dtype),
            "y_train_raw": preprocessed_collection.y_train_raw.to(self.torch_dtype),
            "X_test_raw": preprocessed_collection.X_test_raw.to(self.torch_dtype),
            "y_test_raw": preprocessed_collection.y_test_raw.to(self.torch_dtype),
        }

    @staticmethod
    def _reconstruct_timestamps(
        forecast_start: pd.Period, past_len: int, future_len: int
    ) -> pd.DatetimeIndex:
        """
        Reconstructs the full timestamp index for a single time series sample.
        """
        freq = forecast_start.freq
        past_timestamps = pd.date_range(
            end=forecast_start.to_timestamp() - freq,
            periods=past_len,
            freq=freq,
        )
        future_timestamps = pd.date_range(
            start=forecast_start.to_timestamp(), periods=future_len, freq=freq
        )
        return past_timestamps.union(future_timestamps)
