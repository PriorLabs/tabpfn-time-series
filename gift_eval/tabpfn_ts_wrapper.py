from typing import Iterator
import logging

import pandas as pd
from gluonts.model.forecast import QuantileForecast, Forecast
from autogluon.timeseries import TimeSeriesDataFrame

from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series import (
    TabPFNTimeSeriesPredictor,
    FeatureTransformer,
    DefaultFeatures,
    TabPFNMode,
    TABPFN_TS_DEFAULT_QUANTILE_CONFIG,
)

logger = logging.getLogger(__name__)


class TabPFNTSPredictor:
    SELECTED_FEATURES = [
        DefaultFeatures.add_running_index,
        DefaultFeatures.add_calendar_features,
    ]

    def __init__(
        self,
        ds_prediction_length: int,
        ds_freq: str,
        tabpfn_mode: TabPFNMode = TabPFNMode.CLIENT,
        context_length: int = -1,
    ):
        self.ds_prediction_length = ds_prediction_length
        self.ds_freq = ds_freq
        self.tabpfn_predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=tabpfn_mode,
        )
        self.context_length = context_length

    def predict(self, test_data_input) -> Iterator[Forecast]:
        time_series = []
        for i, item in enumerate(test_data_input):
            time_series.append(
                pd.DataFrame(
                    {
                        "item_id": i,  # Use index i as monotonic item_id
                        # 'item_id': item['item_id'],
                        "target": item["target"],
                        "timestamp": pd.date_range(
                            start=item["start"].to_timestamp(),
                            periods=len(item["target"]),
                            freq=item["freq"],
                        ),
                    }
                ).set_index(["item_id", "timestamp"])
            )

        assert len(time_series) == len(test_data_input)

        time_series = pd.concat(time_series)
        train_tsdf = TimeSeriesDataFrame(time_series)

        if self.context_length > 0:
            logger.info(
                f"Slicing train_tsdf to {self.context_length} timesteps for each time series"
            )
            train_tsdf = train_tsdf.slice_by_timestep(
                -self.context_length, None
            )

        test_tsdf = generate_test_X(
            train_tsdf, prediction_length=self.ds_prediction_length
        )

        train_tsdf, test_tsdf = FeatureTransformer.add_features(
            train_tsdf, test_tsdf, self.SELECTED_FEATURES
        )

        pred: TimeSeriesDataFrame = self.tabpfn_predictor.predict(train_tsdf, test_tsdf)
        pred = pred.drop(columns=["target"])

        forecasts = []
        for item_id in pred.item_ids:
            forecast_start_date = pred.loc[item_id].index[0].to_period(self.ds_freq)
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=pred.loc[item_id].values.T,
                    forecast_keys=list(map(str, TABPFN_TS_DEFAULT_QUANTILE_CONFIG)),
                    start_date=forecast_start_date,
                )
            )

        return forecasts
