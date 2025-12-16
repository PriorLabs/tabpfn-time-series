from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from tabpfn_time_series import (
    TabPFNMode,
    FeatureTransformer,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series.defaults import (
    DEFAULT_QUANTILE_CONFIG,
    TABPFN_TS_DEFAULT_CONFIG,
)
from tabpfn_time_series.features import (
    AutoSeasonalFeature,
    CalendarFeature,
    RunningIndexFeature,
)
from tabpfn_time_series.predictor import TimeSeriesPredictor

if TYPE_CHECKING:
    import datasets
    import fev
    from tabpfn_time_series.features.feature_generator_base import FeatureGenerator


TABPFN_TS_DEFAULT_FEATURES = [
    RunningIndexFeature(),
    CalendarFeature(),
    AutoSeasonalFeature(),
]


logger = logging.getLogger(__name__)


class TabPFNTSPipeline:
    def __init__(
        self,
        max_context_length: int = 4096,
        temporal_features: list[FeatureGenerator] = TABPFN_TS_DEFAULT_FEATURES,
        tabpfn_mode: TabPFNMode = TabPFNMode.CLIENT,
        tabpfn_output_selection: Literal["mean", "median", "mode"] = "median",
        tabpfn_model_config: dict = TABPFN_TS_DEFAULT_CONFIG,
    ):
        from tabpfn import TabPFNRegressor
        from tabpfn_client import TabPFNRegressor as TabPFNClientRegressor

        self.max_context_length = max_context_length
        self.predictor = TimeSeriesPredictor.from_tabpfn_family(
            tabpfn_class=(
                TabPFNClientRegressor
                if tabpfn_mode == TabPFNMode.CLIENT
                else TabPFNRegressor
            ),
            tabpfn_config=tabpfn_model_config,
            tabpfn_output_selection=tabpfn_output_selection,
        )
        self.feature_transformer = FeatureTransformer(temporal_features)

    def predict(
        self,
        context_tsdf: TimeSeriesDataFrame,
        future_tsdf: TimeSeriesDataFrame,
        quantiles: list[float] = DEFAULT_QUANTILE_CONFIG,
    ) -> TimeSeriesDataFrame:
        # Preprocess
        context_tsdf = self._preprocess_context(context_tsdf)
        future_tsdf = self._preprocess_future(future_tsdf)
        context_tsdf, future_tsdf = self._preprocess_covariates(
            context_tsdf, future_tsdf
        )

        # Featurization
        context_tsdf, future_tsdf = self.feature_transformer.transform(
            context_tsdf, future_tsdf
        )

        # Prediction
        return self.predictor.predict(
            train_tsdf=context_tsdf,
            test_tsdf=future_tsdf,
            quantiles=quantiles,
        )

    def predict_df(
        self,
        context_df: pd.DataFrame,
        future_df: pd.DataFrame | None = None,
        prediction_length: int | None = None,
        quantiles: list[float] = DEFAULT_QUANTILE_CONFIG,
    ) -> pd.DataFrame:
        """
        Predict from pandas DataFrames.

        Args:
            context_df: Historical data with columns 'timestamp', 'target', and optionally 'item_id'.
                If 'item_id' is missing, assumes a single time series.
                Additional columns are treated as covariates.
            future_df: Future timestamps with 'timestamp' and optionally 'item_id'.
                Mutually exclusive with prediction_length.
            prediction_length: Number of steps to forecast. Mutually exclusive with future_df.
            quantiles: Quantiles to predict.

        Returns:
            DataFrame with predictions.

        Note:
            Only covariates present in both context_df and future_df will be used.
        """
        if (future_df is None) == (prediction_length is None):
            raise ValueError("Provide exactly one of future_df or prediction_length")

        # Handle single-series case (no item_id column)
        if "item_id" not in context_df.columns:
            context_df = self._add_dummy_item_id(context_df, "item_id")

        context_tsdf = TimeSeriesDataFrame.from_data_frame(context_df)

        if prediction_length is not None:
            future_tsdf = generate_test_X(
                context_tsdf, prediction_length=prediction_length
            )
        else:
            if "item_id" not in future_df.columns:
                future_df = self._add_dummy_item_id(future_df, "item_id")
            future_tsdf = TimeSeriesDataFrame.from_data_frame(future_df)

        pred = self.predict(context_tsdf, future_tsdf, quantiles=quantiles)
        result = pred.to_data_frame()

        return result

    def predict_fev(
        self,
        task: fev.Task,
        use_covariates: bool = True,
    ) -> tuple[list["datasets.DatasetDict"], float]:
        raise NotImplementedError("predict_fev is not implemented")

    def _preprocess_context(
        self, context_tsdf: TimeSeriesDataFrame
    ) -> TimeSeriesDataFrame:
        # Handle missing target values in context
        context_tsdf = self.handle_missing_values(context_tsdf)
        assert not context_tsdf.target.isnull().any()

        # Slice context to the last max_context_length timesteps
        return context_tsdf.slice_by_timestep(-self.max_context_length, None)

    def _preprocess_future(
        self, future_tsdf: TimeSeriesDataFrame
    ) -> TimeSeriesDataFrame:
        # If "target" column exists, assert all values are NaN; otherwise, add it as all NaN
        # (TimeSeriesPredictor and Featurization assume "target" to be NaN in future_tsdf)
        if "target" in future_tsdf.columns:
            if future_tsdf["target"].notna().any():
                raise ValueError(
                    "future_tsdf: All entries in 'target' must be NaN for the prediction horizon. "
                    "Got at least one non-NaN in 'target'."
                )

        future_tsdf["target"] = np.nan

        return future_tsdf

    def _preprocess_covariates(
        self,
        context_tsdf: TimeSeriesDataFrame,
        future_tsdf: TimeSeriesDataFrame,
    ) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        # Get valid covariates
        # (only use covariates that are present in both context_tsdf and future_tsdf)
        valid_covariates = (
            context_tsdf.columns.intersection(future_tsdf.columns)
            .drop("target")
            .tolist()
        )
        logger.info(f"Valid covariates: {valid_covariates}")

        # Impute missing covariates values with NaN in context
        # This implementation assumes all target values are present in context_tsdf
        assert context_tsdf.target.notna().all()
        context_tsdf = context_tsdf.fill_missing_values(
            method="constant",
            value=np.nan,
        )
        assert not context_tsdf[valid_covariates].isnull().any().any()

        # Assert no missing covariate values in future
        assert not future_tsdf[valid_covariates].isnull().any().any()

        return context_tsdf, future_tsdf

    @staticmethod
    def _add_dummy_item_id(df: pd.DataFrame, item_id_column: str) -> pd.DataFrame:
        df = df.copy()
        df[item_id_column] = 0
        return df

    @staticmethod
    def handle_missing_values(tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """
        Handle missing values in a TimeSeriesDataFrame.

        Strategy:
        - If a series has ≤1 valid values: fill NaNs with 0
        - Otherwise: drop rows with NaN targets

        Args:
            tsdf: TimeSeriesDataFrame with potential NaN values in 'target'

        Returns:
            TimeSeriesDataFrame with NaNs handled
        """

        def process_series(group: pd.DataFrame) -> pd.DataFrame:
            valid_count = group["target"].notna().sum()
            if valid_count <= 1:
                # Using 0 since the time series is not meaningful if it has only one/no valid value
                # (TabPFN would predict 0 for the prediction horizon)
                return group.fillna({"target": 0})
            return group[group["target"].notna()]

        result = tsdf.groupby(level="item_id", group_keys=False).apply(process_series)
        return TimeSeriesDataFrame(result)
