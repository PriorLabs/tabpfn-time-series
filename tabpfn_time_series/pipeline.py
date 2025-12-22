"""
TabPFN-TS Forecasting Pipeline.

This module provides the main user-facing API for TabPFN-TS.

Key Components:
    TabPFNTSPipeline: Main forecasting interface that handles the entire
        prediction pipeline from data preprocessing to forecast generation.
"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import datasets

from tabpfn_time_series import (
    TabPFNMode,
    FeatureTransformer,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series.defaults import (
    DEFAULT_QUANTILE_CONFIG,
    TABPFN_DEFAULT_CONFIG,
)
from tabpfn_time_series.features import (
    AutoSeasonalFeature,
    CalendarFeature,
    RunningIndexFeature,
)
from tabpfn_time_series.predictor import TimeSeriesPredictor

if TYPE_CHECKING:
    import fev
    from tabpfn_time_series.features.feature_generator_base import FeatureGenerator


"""
Default temporal feature generators for TabPFN-TS.

These features are automatically applied to time series data:
- RunningIndexFeature: Timestep index
- CalendarFeature: Temporal patterns (hour-of-the-day, day-of-the-week, etc.)
- AutoSeasonalFeature: Automatic detection and encoding of seasonal patterns
"""
TABPFN_TS_DEFAULT_FEATURES = [
    RunningIndexFeature(),
    CalendarFeature(),
    AutoSeasonalFeature(),
]


logger = logging.getLogger(__name__)


def _handle_missing_values(tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
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


def _preprocess_context(
    context_tsdf: TimeSeriesDataFrame,
    max_context_length: int,
) -> TimeSeriesDataFrame:
    # Handle missing target values in context
    context_tsdf = _handle_missing_values(context_tsdf)
    assert not context_tsdf.target.isnull().any()

    # Slice context to the last max_context_length timesteps
    return context_tsdf.slice_by_timestep(-max_context_length, None)


def _preprocess_future(
    future_tsdf: TimeSeriesDataFrame,
) -> TimeSeriesDataFrame:
    future_tsdf = future_tsdf.copy()

    # If "target" column exists, assert all values are NaN; otherwise, add it as all NaN
    # (TimeSeriesPredictor and Featurization assume "target" to be NaN in future_tsdf)
    if "target" in future_tsdf.columns and future_tsdf["target"].notna().any():
        raise ValueError(
            "future_tsdf: All entries in 'target' must be NaN for the prediction horizon. "
            "Got at least one non-NaN in 'target'."
        )

    future_tsdf["target"] = np.nan

    return future_tsdf


def _preprocess_covariates(
    context_tsdf: TimeSeriesDataFrame,
    future_tsdf: TimeSeriesDataFrame,
) -> tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
    # Get valid covariates
    # (only use covariates that are present in both context_tsdf and future_tsdf)
    valid_covariates = (
        context_tsdf.columns.intersection(future_tsdf.columns).drop("target").tolist()
    )
    logger.info(f"Valid covariates: {valid_covariates}")

    # Impute missing covariates values with NaN in context
    # This implementation assumes all target values are present in context_tsdf
    if not context_tsdf.target.notna().all():
        raise ValueError(
            "All target values in context_tsdf must be present (no missing values)."
        )
    context_tsdf = context_tsdf.fill_missing_values(
        method="constant",
        value=np.nan,
    )

    # Assert no missing covariate values in future
    if future_tsdf[valid_covariates].isnull().any().any():
        raise ValueError(
            "All covariate values in future_tsdf must be present (no missing values)."
        )

    return context_tsdf, future_tsdf


def _add_dummy_item_id(df: pd.DataFrame, item_id_column: str) -> pd.DataFrame:
    """
    Add a dummy item_id column for single time series.

    When users provide a DataFrame without an item_id column, this method
    adds a dummy column with value 0 to enable uniform processing.

    Args:
        df: DataFrame without item_id column
        item_id_column: Name of the item_id column to add

    Returns:
        DataFrame with added item_id column
    """
    df = df.copy()
    df[item_id_column] = 0
    return df


class TabPFNTSPipeline:
    """
    TabPFN-TS forecasting pipeline.

    This is the main interface for TabPFN-TS. The pipeline handles data preprocessing,
    feature engineering, and prediction.

    Key Features:
        - Zero-shot - simply feed in your data and get forecasts
        - Probabilistic predictions with customizable quantiles
        - Support for forecasting with known covariates
        - Flexible inference: local GPU or cloud API (via tabpfn-client)

    Basic Usage:
        >>> from tabpfn_time_series import TabPFNTSPipeline
        >>> import pandas as pd
        >>>
        >>> # Initialize the pipeline
        >>> pipeline = TabPFNTSPipeline()
        >>>
        >>> # Prepare your data (with columns: item_id, timestamp, target)
        >>> context_df = pd.DataFrame({
        ...     'item_id': ['A'] * 100,
        ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
        ...     'target': [1.0, 2.0, 3.0, ...]  # your historical values
        ... })
        >>>
        >>> # Generate forecasts
        >>> predictions = pipeline.predict_df(context_df, prediction_length=24)
    """

    def __init__(
        self,
        max_context_length: int = 4096,
        temporal_features: list[FeatureGenerator] = TABPFN_TS_DEFAULT_FEATURES,
        tabpfn_mode: TabPFNMode = TabPFNMode.CLIENT,
        tabpfn_output_selection: Literal["mean", "median", "mode"] = "median",
        tabpfn_model_config: dict = TABPFN_DEFAULT_CONFIG,
    ):
        """
        Initialize the TabPFN-TS forecasting pipeline.

        Args:
            max_context_length: Maximum number of historical timesteps to use for context.
                The pipeline will automatically slice to the last `max_context_length` timesteps
                if the historical data is longer. Default: 4096.
            temporal_features: List of feature generators to apply to the time series.
                These generate temporal features like calendar features, seasonal patterns, etc.
                Default: [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()].
            tabpfn_mode: Inference mode for TabPFN.
                - TabPFNMode.CLIENT: Use the cloud API (recommended, no GPU needed)
                - TabPFNMode.LOCAL: Run locally on your machine (requires GPU for speed)
                Default: TabPFNMode.CLIENT.
            tabpfn_output_selection: Method to aggregate TabPFN ensemble predictions.
                Options: "mean", "median", "mode". Default: "median".
            tabpfn_model_config: Configuration dictionary for the TabPFN model.
                See TABPFN_DEFAULT_CONFIG for default settings.

        Note:
            - When using TabPFNMode.CLIENT, you'll be prompted to login or create an account
              on the first run.
            - For time series with irregular timestamps, we recommend opting out of AutoSeasonalFeature.
        """
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
        """
        Generate forecasts using TimeSeriesDataFrame objects.

        This is the core prediction method that works with TimeSeriesDataFrame objects.
        For a simpler pandas DataFrame interface, see `predict_df()`.

        Args:
            context_tsdf: Historical time series data used as context for prediction.
                Must contain a 'target' column with historical values.
                May contain additional covariate columns.
            future_tsdf: Future timestamps for which to generate predictions.
                Should contain the same covariate columns as context_tsdf.
                The 'target' column should be NaN (will be filled with predictions).
            quantiles: List of quantiles to predict for probabilistic forecasting.
                Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

        Returns:
            TimeSeriesDataFrame with predictions. Contains:
                - 'target': Point predictions (median by default)
                - Columns for each quantile (e.g., '0.1', '0.9' for prediction intervals)

        Note:
            - Only covariates present in both context_tsdf and future_tsdf will be used
            - Context will be automatically sliced to max_context_length if longer
            - Missing values in context are handled automatically
        """
        # Preprocess
        context_tsdf = _preprocess_context(context_tsdf, self.max_context_length)
        future_tsdf = _preprocess_future(future_tsdf)
        context_tsdf, future_tsdf = _preprocess_covariates(context_tsdf, future_tsdf)

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
        Generate forecasts from pandas DataFrames (recommended method for most users).

        This is the main user-facing API. It accepts standard pandas DataFrames
        and returns predictions as a DataFrame.

        Args:
            context_df: Historical time series data. Required columns:
                - 'timestamp': Timestamps for each observation (datetime)
                - 'target': Historical values to forecast from (numeric)
                - 'item_id' (optional): Identifier for multiple time series. If omitted,
                  assumes a single time series.
                - Additional columns are treated as known covariates (e.g., temperature,
                  holidays, promotional flags). These can improve forecast accuracy when
                  their future values are known. Else, they are treated as unknown covariates
                  and will be ignored.

            future_df: Future timestamps for prediction. Required columns:
                - 'timestamp': Future timestamps to forecast (datetime)
                - 'item_id' (optional): Must match item_ids in context_df if used
                - Covariate columns: Must include all covariates you want to use from context_df

                Mutually exclusive with `prediction_length`. Use this when you have known
                future covariate values or irregular timestamps.

            prediction_length: Number of time steps to forecast into the future.
                Mutually exclusive with `future_df`. Use this for simple forecasting when
                you don't have future covariates.

                Example: If context_df has hourly data, prediction_length=24 forecasts
                the next 24 hours.

            quantiles: List of quantiles to predict for uncertainty estimation.
                Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].

                Common patterns:
                - [0.5]: Point forecast only (median)
                - [0.1, 0.5, 0.9]: 80% prediction interval
                - [0.025, 0.5, 0.975]: 95% prediction interval

        Returns:
            DataFrame with forecasts indexed by (item_id, timestamp). Contains:
                - 'target': Point predictions (median by default)
                - One column per quantile (e.g., '0.1', '0.9') for prediction intervals

        Raises:
            ValueError: If both or neither of future_df and prediction_length are provided.

        Examples:
            Single time series, simple forecast:
                >>> context = pd.DataFrame({
                ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
                ...     'target': np.random.randn(100)
                ... })
                >>> predictions = pipeline.predict_df(context, prediction_length=24)

            Multiple time series with item_id:
                >>> context = pd.DataFrame({
                ...     'item_id': ['A']*50 + ['B']*50,
                ...     'timestamp': pd.date_range('2024-01-01', periods=50, freq='h').tolist() * 2,
                ...     'target': np.random.randn(100)
                ... })
                >>> predictions = pipeline.predict_df(context, prediction_length=24)

            With known future covariates:
                >>> context = pd.DataFrame({
                ...     'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
                ...     'target': np.random.randn(100),
                ...     'temperature': np.random.uniform(15, 25, 100),
                ...     'is_weekend': [0, 0, 0, 0, 0, 1, 1] * 14 + [0, 0]
                ... })
                >>> future = pd.DataFrame({
                ...     'timestamp': pd.date_range('2024-01-01', periods=24, freq='h') + pd.Timedelta(days=100),
                ...     'temperature': np.random.uniform(15, 25, 24),  # known future temp
                ...     'is_weekend': [1, 1, 0, 0, 0, 0, 0] * 3 + [1, 1, 0]
                ... })
                >>> predictions = pipeline.predict_df(context, future_df=future)

        Note:
            - Only covariates present in both context_df and future_df will be used
            - Missing values in context are handled automatically
            - The context is automatically limited to the last max_context_length timesteps
        """
        if (future_df is None) == (prediction_length is None):
            raise ValueError("Provide exactly one of future_df or prediction_length")

        # Handle single-series case (no item_id column)
        if "item_id" not in context_df.columns:
            context_df = _add_dummy_item_id(context_df, "item_id")

        context_tsdf = TimeSeriesDataFrame.from_data_frame(context_df)

        if prediction_length is not None:
            future_tsdf = generate_test_X(
                context_tsdf, prediction_length=prediction_length
            )
        else:
            if "item_id" not in future_df.columns:
                future_df = _add_dummy_item_id(future_df, "item_id")
            future_tsdf = TimeSeriesDataFrame.from_data_frame(future_df)

        pred = self.predict(context_tsdf, future_tsdf, quantiles=quantiles)
        result = pred.to_data_frame()

        return result

    def predict_fev(
        self,
        task: "fev.Task",
        use_covariates: bool = True,
    ) -> tuple[list["datasets.DatasetDict"], float]:
        """Generate predictions for a fev benchmarking task.

        Args:
            task: A fev.Task containing the evaluation windows to predict.
            use_covariates: Whether to use known dynamic covariates. Defaults to True.

        Returns:
            A tuple of (predictions_per_window, inference_time_s) where:
            - predictions_per_window: List of DatasetDict predictions for each window
            - inference_time_s: Total inference time in seconds (excludes data conversion)
        """
        import fev

        quantiles = task.quantile_levels or DEFAULT_QUANTILE_CONFIG
        predictions_per_window = []
        inference_time_s = 0.0

        for window in task.iter_windows():
            past_data, future_data = fev.convert_input_data(
                window, adapter="autogluon", as_univariate=True
            )

            if not use_covariates and task.known_dynamic_columns:
                past_data = past_data.drop(columns=task.known_dynamic_columns)
                future_data = future_data.drop(columns=task.known_dynamic_columns)

            start_time = time.monotonic()

            window_pred = self.predict(past_data, future_data, quantiles=quantiles)
            window_pred_in_dataset_format = datasets.Dataset.from_pandas(
                window_pred.rename(columns={"target": "predictions"})
                .groupby(level=0)
                .agg(list)
                .rename(columns=str),
                preserve_index=False,
            )

            predictions_per_window.append(
                fev.utils.combine_univariate_predictions_to_multivariate(
                    window_pred_in_dataset_format,
                    target_columns=task.target_columns,
                )
            )

            inference_time_s += time.monotonic() - start_time

        return predictions_per_window, inference_time_s
