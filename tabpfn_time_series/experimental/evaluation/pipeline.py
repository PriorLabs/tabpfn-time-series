import logging
import json
from typing import List
from typing import Iterator, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from gluonts.model.forecast import QuantileForecast, Forecast
from gluonts.itertools import batcher
from autogluon.timeseries import TimeSeriesDataFrame
from torch.cuda import is_available as torch_cuda_is_available

from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series import (
    TabPFNTimeSeriesPredictor,
    FeatureTransformer,
    DefaultFeatures,
    TabPFNMode,
    TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
)
from tabpfn_time_series.experimental.noisy_transform.tabpfn_noisy_transform_predictor import (
    TabPFNNoisyTranformPredictor,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    predictor_name: str
    predictor_config: dict
    feature_names: List[str]
    context_length: int

    _PREDICTOR_NAME_TO_CLASS = {
        "TabPFNTimeSeriesPredictor": TabPFNTimeSeriesPredictor,
        "TabPFNNoisyTranformPredictor": TabPFNNoisyTranformPredictor,
    }

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            config = json.load(f)
        return cls(**config)

    @staticmethod
    def get_predictor_class(predictor_name: str):
        return PipelineConfig._PREDICTOR_NAME_TO_CLASS[predictor_name]


class TabPFNTSPipeline:
    FALLBACK_FEATURES = [
        DefaultFeatures.add_running_index,
        DefaultFeatures.add_calendar_features,
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
            **config.predictor_config,
        )
        self.context_length = config.context_length
        self.debug = debug

        # Parse feature names from config
        self.selected_features = []
        for feature_name in config.feature_names:
            if hasattr(DefaultFeatures, feature_name):
                self.selected_features.append(getattr(DefaultFeatures, feature_name))
            else:
                logger.warning(f"Feature {feature_name} not found in DefaultFeatures")

        # If no valid features found, use defaults
        if not self.selected_features:
            logger.warning("No valid features found, using defaults")
            self.selected_features = self.FALLBACK_FEATURES

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

    def _preprocess_test_data(
        self, test_data_input
    ) -> Tuple[TimeSeriesDataFrame, TimeSeriesDataFrame]:
        """
        Preprocess includes:
        - Turn the test_data_input into a TimeSeriesDataFrame
        - Drop rows with NaN values in "target" column
            - If time series has all NaN or only 1 valid value, fill with 0s
            - Else, drop the NaN values within the time series
        - If context_length is set, slice the train_tsdf to the last context_length timesteps
        """

        # Pre-allocate list with known size
        time_series = [None] * len(test_data_input)
        ts_with_0_or_1_valid_value = []
        ts_with_nan = []

        for i, item in enumerate(test_data_input):
            target = item["target"]

            # If there are 0 or 1 valid values, consider this an "all NaN" time series
            # and replace NaN with 0
            valid_value_count = np.count_nonzero(~np.isnan(target))
            if valid_value_count <= 1:
                ts_with_0_or_1_valid_value.append(i)
                target = np.where(np.isnan(target), 0, target)

            # Else (i.e. there are more than 1 valid values),
            # drop NaN values within the time series
            elif np.isnan(target).any():
                ts_with_nan.append(i)
                target = target[~np.isnan(target)]

            # Create timestamp index once
            timestamp = pd.date_range(
                start=item["start"].to_timestamp(),
                periods=len(target),
                freq=item["freq"],
            )

            # Create DataFrame directly with final structure
            time_series[i] = pd.DataFrame(
                {
                    "target": target,
                },
                index=pd.MultiIndex.from_product(
                    [[i], timestamp], names=["item_id", "timestamp"]
                ),
            )

        if ts_with_0_or_1_valid_value:
            logger.warning(
                f"Found time-series with 0 or 1 valid values, item_ids: {ts_with_0_or_1_valid_value}"
            )

        if ts_with_nan:
            logger.warning(
                f"Found time-series with NaN targets, item_ids: {ts_with_nan}"
            )

        # Concat pre-allocated list
        train_tsdf = TimeSeriesDataFrame(pd.concat(time_series))

        # assert no more NaN in train_tsdf target
        assert not train_tsdf.target.isnull().any()

        # Slice if needed
        if self.context_length > 0:
            logger.info(
                f"Slicing train_tsdf to {self.context_length} timesteps for each time series"
            )
            train_tsdf = train_tsdf.slice_by_timestep(-self.context_length, None)

        # Generate test data and features
        test_tsdf = generate_test_X(
            train_tsdf, prediction_length=self.ds_prediction_length
        )
        train_tsdf, test_tsdf = FeatureTransformer.add_features(
            train_tsdf, test_tsdf, self.selected_features
        )

        return train_tsdf, test_tsdf
