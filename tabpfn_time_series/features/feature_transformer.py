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

        tsdf = pd.concat([train_tsdf, test_tsdf])

        # Apply all feature generators
        for generator in self.feature_generators:
            tsdf = generator(tsdf)

        # Split train and test tsdf
        train_slice = tsdf.iloc[: len(train_tsdf)]
        test_slice = tsdf.iloc[len(train_tsdf) :]

        # Convert back to TimeSeriesDataFrame and re-attach static features
        # This ensures the metadata remains intact even if generators returned a standard DF
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
