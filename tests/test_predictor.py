import os

import numpy as np
import pandas as pd
import pytest
import tabpfn
import tabpfn_client
from sklearn.ensemble import RandomForestRegressor

from tabpfn_time_series import (
    FeatureTransformer,
    TabPFNMode,
    TabPFNTimeSeriesPredictor,
    TimeSeriesDataFrame,
)
from tabpfn_time_series.data_preparation import generate_test_X
from tabpfn_time_series.features import (
    AutoSeasonalFeature,
    CalendarFeature,
    RunningIndexFeature,
)
from tabpfn_time_series.predictor import TimeSeriesPredictor


def create_test_data():
    # Create a simple time series dataframe for testing
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    item_ids = [0, 1]

    # Create train data with target
    train_data = []
    for item in item_ids:
        for date in dates:
            train_data.append(
                {
                    "item_id": item,
                    "timestamp": date,
                    "target": np.random.rand(),
                }
            )

    train_tsdf = TimeSeriesDataFrame(
        pd.DataFrame(train_data),
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # Generate test data
    test_tsdf = generate_test_X(train_tsdf, prediction_length=5)

    # Create feature transformer with multiple feature generators
    feature_transformer = FeatureTransformer(
        [
            RunningIndexFeature(),
            CalendarFeature(),
            AutoSeasonalFeature(),
        ]
    )

    # Apply feature transformation
    train_tsdf, test_tsdf = feature_transformer.transform(train_tsdf, test_tsdf)

    return train_tsdf, test_tsdf


def maybe_setup_tabpfn_client_on_github_actions() -> None:
    if not os.getenv("GITHUB_ACTIONS"):
        return

    access_token = os.getenv("TABPFN_CLIENT_API_KEY")
    assert access_token is not None, "TABPFN_CLIENT_API_KEY is not set"

    tabpfn_client.set_access_token(access_token)


class TestTabPFNTimeSeriesPredictor:
    @pytest.mark.uses_tabpfn_client
    def test_client_mode(self):
        """Test that predict method calls the worker's predict method"""
        maybe_setup_tabpfn_client_on_github_actions()
        train_tsdf, test_tsdf = create_test_data()

        predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.CLIENT)
        result = predictor.predict(train_tsdf, test_tsdf)

        assert result is not None

    def test_local_mode(self):
        """Test that predict method calls the worker's predict method"""
        train_tsdf, test_tsdf = create_test_data()
        predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)
        result = predictor.predict(train_tsdf, test_tsdf)
        assert result is not None


class TestTimeSeriesPredictor:
    @pytest.mark.parametrize(
        "tabpfn_class",
        [
            pytest.param(tabpfn.TabPFNRegressor, marks=pytest.mark.uses_tabpfn_local),
            pytest.param(
                tabpfn_client.TabPFNRegressor, marks=pytest.mark.uses_tabpfn_client
            ),
        ],
    )
    def test_from_tabpfn_family(
        self, tabpfn_class: tabpfn.TabPFNRegressor | tabpfn_client.TabPFNRegressor
    ):
        if tabpfn_class == tabpfn_client.TabPFNRegressor:
            maybe_setup_tabpfn_client_on_github_actions()
        train_tsdf, test_tsdf = create_test_data()

        predictor = TimeSeriesPredictor.from_tabpfn_family(
            tabpfn_class=tabpfn_class,
            tabpfn_config={"n_estimators": 1},
            tabpfn_output_selection="median",
        )
        result = predictor.predict(train_tsdf, test_tsdf)
        assert result is not None

    def test_from_point_prediction_regressor(self):
        train_tsdf, test_tsdf = create_test_data()
        predictor = TimeSeriesPredictor.from_point_prediction_regressor(
            regressor_class=RandomForestRegressor,
            regressor_config={"n_estimators": 1},
            regressor_fit_config={
                # "...": "...",
            },
            regressor_predict_config={
                # "...": "...",
            },
        )
        result = predictor.predict(train_tsdf, test_tsdf)
        assert result is not None
