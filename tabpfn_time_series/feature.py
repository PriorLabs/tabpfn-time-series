import numpy as np
import pandas as pd
from typing import Tuple, List, Callable

import gluonts.time_feature
from autogluon.timeseries import TimeSeriesDataFrame


class DefaultFeatures:
    @staticmethod
    def add_running_index(df: pd.DataFrame) -> pd.Series:
        df["running_index"] = range(len(df))
        return df

    @staticmethod
    def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
        CALENDAR_COMPONENT = [
            "year",
            # "month",
            # "day",
        ]

        CALENDAR_FEATURES = [
            # (feature, natural seasonality)
            ("hour_of_day", 24),
            ("day_of_week", 7),
            ("day_of_month", 30.5),
            ("day_of_year", 365),
            ("week_of_year", 52),
            ("month_of_year", 12),
        ]

        timestamps = df.index.get_level_values("timestamp")

        for component_name in CALENDAR_COMPONENT:
            df[component_name] = getattr(timestamps, component_name)

        for feature_name, seasonality in CALENDAR_FEATURES:
            feature_func = getattr(gluonts.time_feature, f"{feature_name}_index")
            feature = feature_func(timestamps).astype(np.int32)
            if seasonality is not None:
                df[f"{feature_name}_sin"] = np.sin(
                    2 * np.pi * feature / (seasonality - 1)
                )  # seasonality - 1 because the value starts from 0
                df[f"{feature_name}_cos"] = np.cos(
                    2 * np.pi * feature / (seasonality - 1)
                )
            else:
                df[feature_name] = feature

        return df


class FeatureTransformer:
    @staticmethod
    def add_features(
        train_tsdf: TimeSeriesDataFrame,
        test_tsdf: TimeSeriesDataFrame,
        feature_generators: List[Callable[[TimeSeriesDataFrame], TimeSeriesDataFrame]],
        target_column: str = "target",
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        assert target_column in train_tsdf.columns
        assert test_tsdf[target_column].isna().all()

        # Join train and test tsdf
        tsdf = pd.concat([train_tsdf, test_tsdf])

        # Apply feature generators
        for func in feature_generators:
            tsdf = tsdf.groupby(level="item_id", group_keys=False).apply(func)

        # Split train and test tsdf
        train_tsdf = tsdf.iloc[: len(train_tsdf)]
        test_tsdf = tsdf.iloc[len(train_tsdf) :]

        assert test_tsdf[target_column].isna().all()

        return train_tsdf, test_tsdf
