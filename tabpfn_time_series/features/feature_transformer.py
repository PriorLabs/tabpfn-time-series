from typing import List, Tuple

import pandas as pd

from tabpfn_time_series.ts_dataframe import TimeSeriesDataFrame
from tabpfn_time_series.features.feature_generator_base import FeatureGenerator


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

        static_features = self._merge_static_features(
            train_tsdf.static_features, test_tsdf.static_features
        )

        train_plain = pd.DataFrame(train_tsdf).assign(_is_train=True)
        test_plain = pd.DataFrame(test_tsdf).assign(_is_train=False)
        tsdf = pd.concat([train_plain, test_plain])

        item_ids = tsdf.index.get_level_values("item_id").unique()

        processed_series = []
        for item_id in item_ids:
            series_df = tsdf.xs(item_id, level="item_id")
            for generator in self.feature_generators:
                series_df = generator(series_df)
            processed_series.append(series_df)
        tsdf = pd.concat(processed_series, keys=item_ids, names=["item_id"])

        # Split by tag (not iloc) since per-series concat reorders rows by item_id.
        train_slice = tsdf[tsdf["_is_train"]].drop(columns=["_is_train"])
        test_slice = tsdf[~tsdf["_is_train"]].drop(columns=["_is_train"])

        train_tsdf = TimeSeriesDataFrame(
            pd.DataFrame(train_slice), static_features=static_features
        )
        test_tsdf = TimeSeriesDataFrame(
            pd.DataFrame(test_slice), static_features=static_features
        )

        assert not train_tsdf[target_column].isna().any(), (
            "All target values in train_tsdf should be non-NaN"
        )
        assert test_tsdf[target_column].isna().all()

        return train_tsdf, test_tsdf

    @staticmethod
    def _merge_static_features(
        train_static: pd.DataFrame | None,
        test_static: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        """Return static features covering all item_ids from both train and test."""
        if train_static is None:
            return None
        if test_static is None:
            return train_static
        new_items = test_static.index.difference(train_static.index)
        if len(new_items) == 0:
            return train_static
        return pd.concat([train_static, test_static.loc[new_items]])

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
