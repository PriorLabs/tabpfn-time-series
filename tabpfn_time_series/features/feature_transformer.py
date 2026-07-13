from typing import List, Tuple

import numpy as np
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

        # Columns present before featurization (target + covariates). Their dtype is
        # left untouched; only newly *generated* float64 feature columns are downcast
        # to float32 below to roughly halve the peak host-memory footprint of the
        # featurized frame. This matters a lot for many-series datasets (e.g. m5_1D
        # from fev-bench has ~30k series): the full featurized frame is materialized
        # in host RAM before the per-series GPU loop runs.
        input_columns = set(train_tsdf.columns)

        train_plain = pd.DataFrame(train_tsdf).assign(_is_train=True)
        test_plain = pd.DataFrame(test_tsdf).assign(_is_train=False)
        # Convert the train and test to the same data type
        # (or float to support NA, not possible with integer type)
        target_dtype = train_plain[target_column].dtype
        if pd.api.types.is_integer_dtype(target_dtype):
            target_dtype = "float64"
        test_plain[target_column] = test_plain[target_column].astype(target_dtype)
        tsdf = pd.concat([train_plain, test_plain])
        del train_plain, test_plain

        # Each generator runs once on the whole multi-series frame and groups on the
        # item_id index level internally where needed. This is vectorized across
        # series (no per-series Python loop), unlike the previous O(n_series^2)
        # `.xs`-per-item approach.
        for generator in self.feature_generators:
            tsdf = generator(tsdf)

        # Downcast generated features (calendar/seasonal/running-index — all bounded
        # values that are exactly representable in float32) to halve peak host memory.
        generated_float_cols = [
            c
            for c in tsdf.columns
            if c not in input_columns
            and c != "_is_train"
            and tsdf[c].dtype == np.float64
        ]
        if generated_float_cols:
            tsdf = tsdf.astype({c: np.float32 for c in generated_float_cols})

        # Split back into train/test by the tag column added above.
        train_slice = tsdf[tsdf["_is_train"]].drop(columns=["_is_train"])
        test_slice = tsdf[~tsdf["_is_train"]].drop(columns=["_is_train"])
        del tsdf

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
