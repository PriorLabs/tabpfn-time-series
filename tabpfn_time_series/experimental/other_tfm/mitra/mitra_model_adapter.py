import os
import numpy as np

from dotenv import load_dotenv

from tabpfn_time_series.worker.model_adapter import (
    PointPredictionModelAdapter,
    PredictionOutput,
)
from tabpfn_time_series.defaults import DEFAULT_QUANTILE_CONFIG

from autogluon.tabular.models.mitra.sklearn_interface import MitraRegressor
from autogluon.core.data import LabelCleaner
from autogluon.features.generators import AutoMLPipelineFeatureGenerator


load_dotenv()


class MitraModelAdapter(PointPredictionModelAdapter):
    """Model adapter for AutoGluon TabularPredictor with time series focus."""

    def __init__(
        self,
        model_config: dict = {},
        inference_config: dict = {},
    ):
        model_config["state_dict"] = os.getenv("MITRA_MODEL_PATH")
        super().__init__(
            model_class=MitraRegressor,
            model_config=model_config,
            inference_config=inference_config,
        )

    def predict(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        quantiles: list[float | str] = DEFAULT_QUANTILE_CONFIG,
    ) -> PredictionOutput:
        # ---- Perform feature engineering and label cleaning
        feature_generator, label_cleaner = (
            AutoMLPipelineFeatureGenerator(),
            LabelCleaner.construct(problem_type="regression", y=train_y),
        )
        train_X, train_y = (
            feature_generator.fit_transform(train_X),
            label_cleaner.transform(train_y),
        )
        test_X = feature_generator.transform(test_X)

        return super().predict(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
            quantiles=quantiles,
        )
