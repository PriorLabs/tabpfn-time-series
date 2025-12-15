from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeAlias, Union

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from tabpfn_time_series.defaults import DEFAULT_QUANTILE_CONFIG


InferenceConfig: TypeAlias = Dict[str, Any]
"""
Configuration dictionary for model adapters.

Expected structure:
{
    "fit": {
        "param1": value1,
        "param2": value2,
        ...
    },
    "predict": {
        "param1": value1,
        "param2": value2,
        ...
    }
}
"""


PredictionOutput: TypeAlias = Dict[Union[str, float], np.ndarray]
"""
Structure for model prediction outputs.

Required key:
- "target": Array of predictions (must always be present)

Optional keys: 
- float values (e.g., 0.1, 0.5, 0.9): Quantile predictions when requested

Type contract:
- result["target"] -> np.ndarray (always present)
- result[0.1] -> np.ndarray (present only if 0.1 quantile was requested)

Example usage:
- Simple predictions: {"target": np.array([1, 2, 3])}
- Probabilistic: {
        "target": np.array([1, 2, 3]),
        0.1: np.array([0.5, 1.5, 2.5]),
        0.9: np.array([1.5, 2.5, 3.5]),
    }
"""


class BaseModelAdapter(ABC):
    """Base model adapter for scikit-learn compatible models."""

    def __init__(
        self,
        model_class: Type[RegressorMixin],
        model_config: Dict[str, Any],
        inference_config: InferenceConfig = None,
    ) -> None:
        """
        Initialize the base model adapter.

        Args:
            model_class: Scikit-learn compatible regressor class
            model_config: Configuration parameters for model initialization
            inference_config: Configuration for fit and predict methods
        """
        self.model_class = model_class
        self.model_config = deepcopy(model_config)
        self.inference_config = deepcopy(inference_config or {})

    def predict(
        self,
        train_X: Union[np.ndarray, pd.DataFrame],
        train_y: Union[np.ndarray, pd.Series],
        test_X: Union[np.ndarray, pd.DataFrame],
        quantiles: list[float] = DEFAULT_QUANTILE_CONFIG,
    ) -> PredictionOutput:
        """
        Train model and make predictions.

        Args:
            train_X: Training features
            train_y: Training targets
            test_X: Test features
            quantiles: List of quantiles to use for probabilistic output

        Returns:
            Predictions as PredictionOutput
        """
        model = self.model_class(**self.model_config)

        fit_kwargs = self.inference_config.get("fit", {})
        predict_kwargs = self.inference_config.get("predict", {})
        predict_kwargs = {**predict_kwargs, "quantiles": quantiles}

        # Convert dataframe to numpy array
        if isinstance(train_X, pd.DataFrame):
            train_X = train_X.values
        if isinstance(train_y, pd.Series):
            train_y = train_y.values
        if isinstance(test_X, pd.DataFrame):
            test_X = test_X.values

        raw_pred_output = self._fit_and_predict(
            model=model,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
        )

        return self.postprocess_pred_output(
            raw_pred_output=raw_pred_output,
            quantiles=quantiles,
        )

    def _fit_and_predict(
        self,
        model: RegressorMixin,
        train_X: Union[np.ndarray, pd.DataFrame],
        train_y: Union[np.ndarray, pd.Series],
        test_X: Union[np.ndarray, pd.DataFrame],
        fit_kwargs: dict,
        predict_kwargs: dict,
    ) -> Any:
        """Fit and predict the model."""
        model.fit(train_X, train_y, **fit_kwargs)
        return model.predict(test_X, **predict_kwargs)

    @abstractmethod
    def postprocess_pred_output(
        self,
        raw_pred_output: Any,
        quantiles: list[float],
    ) -> PredictionOutput:
        """
        Postprocess the raw prediction output to match the PredictionOutput structure.
        This method is expected to be implemented by subclasses.

        Args:
            raw_pred_output: Raw prediction output from the model
            quantiles: List of quantiles to use for probabilistic output

        Returns:
            PredictionOutput
        """
        pass


class PointPredictionModelAdapter(BaseModelAdapter):
    """
    Adapter for models that only produce point predictions.

    Converts point predictions to the standard PredictionOutput format.
    For quantile predictions, uses the same point value for all quantiles.

    Note: This adapter DOES NOT compute/approximate the quantiles, but simply
    mocks the output from point prediction to probabilistic output.
    """

    def postprocess_pred_output(
        self,
        raw_pred_output: Any,
        quantiles: list[float],
    ) -> PredictionOutput:
        return PointPredictionModelAdapter._mock_probabilistic_output(
            raw_pred_output,
            quantiles,
        )

    def _fit_and_predict(
        self,
        model: RegressorMixin,
        train_X: Union[np.ndarray, pd.DataFrame],
        train_y: Union[np.ndarray, pd.Series],
        test_X: Union[np.ndarray, pd.DataFrame],
        fit_kwargs: dict,
        predict_kwargs: dict,
    ) -> Any:
        """Fit and predict the model."""
        # Ignore the quantiles parameter for point prediction models
        predict_kwargs = predict_kwargs.copy()
        predict_kwargs.pop("quantiles", None)

        return super()._fit_and_predict(
            model=model,
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            fit_kwargs=fit_kwargs,
            predict_kwargs=predict_kwargs,
        )

    @staticmethod
    def _mock_probabilistic_output(
        raw_pred_output: Any,
        quantiles: list[float],
    ) -> PredictionOutput:
        result: PredictionOutput = {"target": raw_pred_output}
        result.update({q: raw_pred_output for q in quantiles})
        return result
