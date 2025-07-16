import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from tabpfn_time_series import (
    TimeSeriesDataFrame,
    TabPFNTimeSeriesPredictor,
    TabPFNMode,
    FeatureTransformer,
)
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    AutoSeasonalFeature,
)
from tabpfn_time_series.data_preparation import generate_test_X


class TestTabPFNTimeSeriesPredictor(unittest.TestCase):
    def setUp(self):
        self.train_tsdf, self.test_tsdf = self._create_test_data()

        if os.getenv("GITHUB_ACTIONS"):
            self._setup_github_actions_tabpfn_client()

    def _setup_github_actions_tabpfn_client(self):
        from tabpfn_client import set_access_token

        access_token = os.getenv("TABPFN_CLIENT_API_KEY")
        assert access_token is not None, "TABPFN_CLIENT_API_KEY is not set"

        set_access_token(access_token)

    def _create_test_data(self):
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

    def test_init_with_default_mode(self):
        """Test that the predictor initializes with default mode (CLIENT)"""
        predictor = TabPFNTimeSeriesPredictor()
        self.assertIsNotNone(predictor.tabpfn_worker)

    def test_init_with_local_mode_without_gpu(self):
        """Test that the predictor initializes with LOCAL mode"""
        with self.assertRaises(ValueError):
            _ = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)

    @patch("torch.cuda.is_available", return_value=True)
    def test_init_with_local_mode_with_gpu(self, mock_is_available):
        """Test that the predictor initializes with LOCAL mode"""
        predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)
        self.assertIsNotNone(predictor.tabpfn_worker)

    def test_init_with_mock_mode(self):
        """Test that the predictor initializes with MOCK mode"""
        predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.MOCK)
        self.assertIsNotNone(predictor.tabpfn_worker)

    def test_predict_calls_worker_predict(self):
        """Test that predict method calls the worker's predict method"""
        # Create predictor and call predict
        predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.CLIENT)
        result = predictor.predict(self.train_tsdf, self.test_tsdf)

        assert result is not None

    @patch("tabpfn_time_series.predictor.MockTabPFN")
    def test_predict_with_mock_mode(self, mock_tabpfn):
        """Test prediction with mock mode"""
        # Setup mock
        mock_instance = MagicMock()
        mock_tabpfn.return_value = mock_instance
        mock_instance.predict.return_value = self.test_tsdf.copy()

        # Create predictor and call predict
        predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.MOCK)
        result = predictor.predict(self.train_tsdf, self.test_tsdf)

        # Assert
        mock_instance.predict.assert_called_once()
        self.assertIsInstance(result, TimeSeriesDataFrame)


if __name__ == "__main__":
    unittest.main()
