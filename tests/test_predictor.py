import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

from tabpfn_time_series import (
    TimeSeriesDataFrame,
    TabPFNTimeSeriesPredictor,
    TabPFNMode,
    FeatureTransformer,
)
from tabpfn_time_series.predictor import TimeSeriesPredictor
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)
from tabpfn_time_series.data_preparation import generate_test_X


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


def setup_github_actions_tabpfn_client():
    from tabpfn_client import set_access_token

    access_token = os.getenv("TABPFN_CLIENT_API_KEY")
    assert access_token is not None, "TABPFN_CLIENT_API_KEY is not set"

    set_access_token(access_token)


class TestTabPFNTimeSeriesPredictor(unittest.TestCase):
    def setUp(self):
        self.train_tsdf, self.test_tsdf = create_test_data()

        if os.getenv("GITHUB_ACTIONS"):
            setup_github_actions_tabpfn_client()

    def test_client_mode(self):
        """Test that predict method calls the worker's predict method"""
        # Create predictor and call predict
        predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.CLIENT)
        result = predictor.predict(self.train_tsdf, self.test_tsdf)

        assert result is not None

    @patch("torch.cuda.is_available", return_value=True)
    def test_local_mode(self, mock_is_available):
        """Test that predict method calls the worker's predict method"""
        # Create predictor and call predict
        predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)

        with self.assertRaises(ValueError):
            _ = predictor.predict(self.train_tsdf, self.test_tsdf)


class TestTimeSeriesPredictor(unittest.TestCase):
    def setUp(self):
        self.train_tsdf, self.test_tsdf = create_test_data()

        if os.getenv("GITHUB_ACTIONS"):
            setup_github_actions_tabpfn_client()

    def test_from_tabpfn_family(self):
        from tabpfn_client import TabPFNRegressor as TabPFNClientRegressor

        predictor = TimeSeriesPredictor.from_tabpfn_family(
            tabpfn_class=TabPFNClientRegressor,
            tabpfn_config={"n_estimators": 1},
            tabpfn_output_selection="median",
        )
        result = predictor.predict(self.train_tsdf, self.test_tsdf)
        assert result is not None

    def test_from_point_prediction_regressor(self):
        from sklearn.ensemble import RandomForestRegressor

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
        result = predictor.predict(self.train_tsdf, self.test_tsdf)
        assert result is not None


if __name__ == "__main__":
    unittest.main()
