import numpy as np
from sklearn.linear_model import LinearRegression

from tabpfn_time_series.worker.model_adapter import BaseModelAdapter
from tabpfn_time_series.defaults import DEFAULT_QUANTILE_CONFIG


class LinearRegressionModelAdapter(BaseModelAdapter):
    """Linear Regression model adapter for time series forecasting."""

    _DEFAULT_MODEL_CONFIG = {
        "fit_intercept": True,
        "copy_X": True,
        "n_jobs": None,
        "positive": False,
    }

    def __init__(
        self,
        model_config: dict = None,
        inference_config: dict = None,
    ):
        if model_config is None:
            model_config = self._DEFAULT_MODEL_CONFIG.copy()

        super().__init__(
            model_class=LinearRegression,
            model_config=model_config,
            inference_config=inference_config or {},
        )

    def predict(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        quantiles: list[float | str] = DEFAULT_QUANTILE_CONFIG,
    ):
        """
        Train linear regression model and make predictions.

        Args:
            train_X: Training features
            train_y: Training targets
            test_X: Test features
            quantiles: List of quantiles to predict (ignored for basic linear regression)

        Returns:
            Dictionary with 'target' and quantile predictions
        """
        # Get point predictions from base class
        pred_output = super().predict(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
        )

        # Linear regression provides point estimates, so we return the same
        # prediction for all quantiles (similar to TabDPT)
        result = {"target": pred_output}
        result.update({q: pred_output for q in quantiles})

        return result
