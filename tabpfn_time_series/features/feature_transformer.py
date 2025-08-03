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
        tsdf = pd.concat([train_tsdf, test_tsdf])

        # Apply all feature generators
        for generator in self.feature_generators:
            tsdf = tsdf.groupby(level="item_id", group_keys=False).apply(generator)

        # Split train and test tsdf
        train_tsdf = tsdf.iloc[: len(train_tsdf)]
        test_tsdf = tsdf.iloc[len(train_tsdf) :]

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
