import logging
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from gluonts.model.forecast import QuantileForecast, Forecast
from gluonts.itertools import batcher
from gluonts.dataset.field_names import FieldName
from autogluon.timeseries import TimeSeriesDataFrame
from torch.cuda import is_available as torch_cuda_is_available

from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series import (
    TabPFNMode,
    TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
)

from tabpfn_time_series.experimental.features import (
    FeatureTransformer,
    RunningIndexFeature,
    CalendarFeature,
    AdditionalCalendarFeature,
    AutoSeasonalFeature,
    PeriodicSinCosineFeature,
)
from tabpfn_time_series.experimental.pipeline.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


FEATURE_MAP = {
    "RunningIndexFeature": RunningIndexFeature,
    "CalendarFeature": CalendarFeature,
    "AdditionalCalendarFeature": AdditionalCalendarFeature,
    "AutoSeasonalFeature": AutoSeasonalFeature,
    "PeriodicSinCosineFeature": PeriodicSinCosineFeature,
}

COVARIATE_FIELD_TYPES = {
    FieldName.PAST_FEAT_DYNAMIC_REAL: float,
    FieldName.PAST_FEAT_DYNAMIC_CAT: str,
}


class TabPFNTSPipeline:
    FALLBACK_FEATURES = [
        RunningIndexFeature(),
        CalendarFeature(),
    ]

    def __init__(
        self,
        config: PipelineConfig,
        ds_prediction_length: int,
        ds_freq: str,
        debug: bool = False,
    ):
        self.ds_prediction_length = ds_prediction_length
        self.ds_freq = ds_freq
        predictor_class = PipelineConfig.get_predictor_class(config.predictor_name)
        self.tabpfn_predictor = predictor_class(
            tabpfn_mode=TabPFNMode.LOCAL
            if torch_cuda_is_available()
            else TabPFNMode.CLIENT,
            config=config.predictor_config,
        )
        self.context_length = config.context_length
        self.slice_before_featurization = config.slice_before_featurization
        self.use_covariates = config.use_covariates
        self.debug = debug

        # Parse feature names from config
        self.selected_features = []
        for feature_name, feature_config in config.features.items():
            if feature_name in FEATURE_MAP:
                self.selected_features.append(
                    FEATURE_MAP[feature_name](**feature_config)
                )
            else:
                logger.warning(f"Feature {feature_name} not found in DefaultFeatures")

        # If no valid features found, use defaults
        if not self.selected_features:
            logger.warning("No valid features found, using defaults")
            self.selected_features = self.FALLBACK_FEATURES

        self.feature_transformer = FeatureTransformer(self.selected_features)

    def predict(self, test_data_input) -> Iterator[Forecast]:
        logger.debug(f"len(test_data_input): {len(test_data_input)}")

        forecasts = []
        for batch in batcher(test_data_input, batch_size=1024):
            forecasts.extend(self._predict_batch(batch))

        return forecasts

    def _predict_batch(self, test_data_input):
        logger.debug(f"Processing batch of size: {len(test_data_input)}")

        # Preprocess the input data
        train_tsdf, test_tsdf = self._preprocess_test_data(test_data_input)

        # Generate predictions
        pred: TimeSeriesDataFrame = self.tabpfn_predictor.predict(train_tsdf, test_tsdf)
        pred = pred.drop(columns=["target"])

        # Pre-allocate forecasts list and get forecast quantile keys
        forecasts = [None] * len(pred.item_ids)
        forecast_keys = list(map(str, TABPFN_TS_DEFAULT_QUANTILE_CONFIG))

        # Generate QuantileForecast objects for each time series
        for i, (_, item_data) in enumerate(pred.groupby(level="item_id")):
            forecast_start_timestamp = item_data.index.get_level_values(1)[0]
            forecasts[i] = QuantileForecast(
                forecast_arrays=item_data.values.T,
                forecast_keys=forecast_keys,
                start_date=forecast_start_timestamp.to_period(self.ds_freq),
            )

        logger.debug(f"Generated {len(forecasts)} forecasts")
        return forecasts

    @staticmethod
    def _process_covariates(item, timestamp):
        """
        Process covariates from input item and return a DataFrame with all features.

        Args:
            item: Dictionary containing time series data and covariates
            timestamp: Timestamp index for the data

        Returns:
            pd.DataFrame: DataFrame containing all covariate features
        """
        covariate_dfs = []

        for key, dtype in COVARIATE_FIELD_TYPES.items():
            if key in item:
                # Get covariate data
                covariate_data = item[key]

                # Create a DataFrame with all features of this type
                feature_df = pd.DataFrame(
                    {
                        f"{key}_{j}": covariate_data[j]
                        for j in range(covariate_data.shape[0])
                    },
                    index=timestamp,
                )

                # Convert types if needed
                feature_df = feature_df.astype(dtype)
                covariate_dfs.append(feature_df)

        # If we have any covariates, combine them
        if covariate_dfs:
            return pd.concat(covariate_dfs, axis=1)
        return pd.DataFrame(index=timestamp)

    @staticmethod
    def convert_to_timeseries_dataframe(test_data_input, use_covariates: bool = False):
        """
        Convert test_data_input to TimeSeriesDataFrame.

        Args:
            test_data_input: List of dictionaries containing time series data
            use_covariates: Whether to include covariates in the output

        Returns:
            TimeSeriesDataFrame: Converted data
        """
        # Pre-allocate list with known size
        time_series = [None] * len(test_data_input)

        for i, item in enumerate(test_data_input):
            target = item["target"]

            # Create timestamp index
            timestamp = pd.date_range(
                start=item["start"].to_timestamp(),
                periods=len(target),
                freq=item["freq"],
            )

            # Create DataFrame with target
            df = pd.DataFrame({"target": target}, index=timestamp)

            # Add covariates if needed
            if use_covariates:
                covariate_df = TabPFNTSPipeline._process_covariates(item, timestamp)
                if not covariate_df.empty:
                    df = pd.concat([df, covariate_df], axis=1)
                else:
                    logger.warning(
                        "Pipeline configured to use covariates,"
                        f"but no covariates found in input data for item {i}"
                    )

            # Create MultiIndex DataFrame
            time_series[i] = df.set_index(
                pd.MultiIndex.from_product(
                    [[i], df.index], names=["item_id", "timestamp"]
                )
            )

        # Concat pre-allocated list
        return TimeSeriesDataFrame(pd.concat(time_series))

    @staticmethod
    def handle_nan_values(tsdf: TimeSeriesDataFrame) -> TimeSeriesDataFrame:
        """
        Handle NaN values in the TimeSeriesDataFrame:
        - If time series has 0 or 1 valid value, fill with 0s
        - Else, drop the NaN values within the time series

        Args:
            tsdf: TimeSeriesDataFrame containing time series data

        Returns:
            TimeSeriesDataFrame: Processed data with NaN values handled
        """
        processed_series = []
        ts_with_0_or_1_valid_value = []
        ts_with_nan = []

        # Process each time series individually
        for item_id, item_data in tsdf.groupby(level="item_id"):
            target = item_data.target.values
            timestamps = item_data.index.get_level_values("timestamp")

            # If there are 0 or 1 valid values, fill NaNs with 0
            valid_value_count = np.count_nonzero(~np.isnan(target))
            if valid_value_count <= 1:
                ts_with_0_or_1_valid_value.append(item_id)
                target = np.where(np.isnan(target), 0, target)
                processed_df = pd.DataFrame(
                    {"target": target},
                    index=pd.MultiIndex.from_product(
                        [[item_id], timestamps], names=["item_id", "timestamp"]
                    ),
                )
                processed_series.append(processed_df)

            # Else drop NaN values
            elif np.isnan(target).any():
                ts_with_nan.append(item_id)
                valid_indices = ~np.isnan(target)
                processed_df = pd.DataFrame(
                    {"target": target[valid_indices]},
                    index=pd.MultiIndex.from_product(
                        [[item_id], timestamps[valid_indices]],
                        names=["item_id", "timestamp"],
                    ),
                )
                processed_series.append(processed_df)

            # No NaNs, keep as is
            else:
                processed_series.append(item_data)

        # Log warnings about NaN handling
        if ts_with_0_or_1_valid_value:
            logger.warning(
                f"Found time-series with 0 or 1 valid values, item_ids: {ts_with_0_or_1_valid_value}"
            )

        if ts_with_nan:
            logger.warning(
                f"Found time-series with NaN targets, item_ids: {ts_with_nan}"
            )

        # Combine processed series
        return TimeSeriesDataFrame(pd.concat(processed_series))

    def _preprocess_test_data(
        self, test_data_input
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """
        Preprocess includes:
        - Turn the test_data_input into a TimeSeriesDataFrame
        - Handle NaN values in "target" column
        - If context_length is set, slice the train_tsdf to the last context_length timesteps
        - Generate test data and apply feature transformations
        """
        # Convert input to TimeSeriesDataFrame
        train_tsdf = self.convert_to_timeseries_dataframe(
            test_data_input, self.use_covariates
        )

        # Handle NaN values
        train_tsdf = self.handle_nan_values(train_tsdf)

        # Assert no more NaN in train_tsdf target
        assert not train_tsdf.target.isnull().any()

        # Apply context length slicing
        if self.context_length > 0 and self.slice_before_featurization:
            logger.info(
                f"Slicing train_tsdf to last {self.context_length} timesteps before featurization"
            )
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)

        # Generate test data and features
        test_tsdf = generate_test_X(
            train_tsdf, prediction_length=self.ds_prediction_length
        )
        train_tsdf, test_tsdf = self.feature_transformer.transform(
            train_tsdf, test_tsdf
        )

        # Apply context length slicing after featurization if configured
        if self.context_length > 0 and not self.slice_before_featurization:
            logger.info(
                f"Slicing train_tsdf to last {self.context_length} timesteps after featurization"
            )
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)

        return train_tsdf, test_tsdf
