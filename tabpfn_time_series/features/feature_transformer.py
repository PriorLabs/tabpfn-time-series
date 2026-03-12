from typing import List, Tuple

import pandas as pd

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)


class FeatureTransformer:
    def __init__(self, feature_generators: List[FeatureGenerator]):
        self.feature_generators = feature_generators

    def transform(
        self,
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        target_column: str = "target",
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """Transform both train and test data with the configured feature generators"""

        self._validate_input(train_tsdf, test_tsdf, target_column)

        static_features = train_tsdf.static_features

        item_ids = train_tsdf.index.get_level_values("item_id").unique()

        # Tag before concat so we can split correctly after per-series reordering.
        tsdf = pd.concat(
            [
                train_tsdf.assign(_is_train=True),
                test_tsdf.assign(_is_train=False),
            ]
        )

        for generator in self.feature_generators:
            per_series = [
                generator(tsdf.xs(item_id, level="item_id")) for item_id in item_ids
            ]
            tsdf = pd.concat(per_series, keys=item_ids, names=["item_id"])

        # Split by tag (not iloc) since per-series concat reorders rows by item_id
        train_slice = tsdf[tsdf["_is_train"]].drop(columns=["_is_train"])
        test_slice = tsdf[~tsdf["_is_train"]].drop(columns=["_is_train"])

        # Re-cast to TimeSeriesDataFrame and re-attach static_features,
        # since concat/slicing may have stripped the subclass and its metadata.
        train_tsdf = TimeSeriesDataFrame(train_slice, static_features=static_features)
        test_tsdf = TimeSeriesDataFrame(test_slice, static_features=static_features)

        assert not train_tsdf[target_column].isna().any(), (
            "All target values in train_tsdf should be non-NaN"
        )
        assert test_tsdf[target_column].isna().all()

        return train_tsdf, test_tsdf

    @staticmethod
    def _validate_input(
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        target_column: str,
    ):
        if target_column not in train_tsdf.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in training data"
            )

        if not test_tsdf[target_column].isna().all():
            raise ValueError("Test data should not contain target values")
