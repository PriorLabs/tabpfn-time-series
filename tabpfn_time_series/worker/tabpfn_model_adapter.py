import numpy as np

from sklearn.base import RegressorMixin
from tabpfn import TabPFNRegressor
from tabpfn_client import (
    init as tabpfn_client_init,
    TabPFNRegressor as TabPFNClientRegressor,
)

from tabpfn_time_series.defaults import DEFAULT_QUANTILE_CONFIG
from tabpfn_time_series.worker.model_adapter import ModelAdapter


def process_tabpfn_pred_output(
    pred_output: dict,
    output_selection: str,
    quantiles: list[float | str],
) -> dict[str, np.ndarray]:
    """Translates raw TabPFN output to the standardized dictionary format."""
    result = {"target": pred_output[output_selection]}
    result.update({q: q_pred for q, q_pred in zip(quantiles, pred_output["quantiles"])})

    return result


class BaseTabPFNModelAdapter(ModelAdapter):
    def __init__(
        self,
        model_class: RegressorMixin,
        model_config: dict,
        tabpfn_output_selection: str,
    ):
        super().__init__(
            model_class,
            model_config,
            inference_config={
                "predict": {
                    "output_type": "main",
                }
            },
        )

        self.tabpfn_output_selection = tabpfn_output_selection

    def predict(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        quantiles: list[float | str] = DEFAULT_QUANTILE_CONFIG,
    ):
        tabpfn_pred_output = super().predict(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
        )

        return process_tabpfn_pred_output(
            tabpfn_pred_output,
            self.tabpfn_output_selection,
            quantiles,
        )


class TabPFNClientModelAdapter(BaseTabPFNModelAdapter):
    def __init__(
        self,
        tabpfn_config: dict,
        tabpfn_output_selection: str,
    ):
        super().__init__(
            model_class=TabPFNClientRegressor,
            model_config=tabpfn_config,
            tabpfn_output_selection=tabpfn_output_selection,
        )

        # Perform initialization of the TabPFN client (authentication)
        tabpfn_client_init()

        # Parse the model name to get the correct model path
        # that is supported by the TabPFN client
        self.model_config["model_path"] = self._parse_model_name(
            self.model_config["model_path"]
        )

    def _parse_model_name(self, model_name: str) -> str:
        available_models = TabPFNClientRegressor.list_available_models()

        for m in available_models:
            if m in model_name:
                return m
        raise ValueError(
            f"Model {model_name} not found. Available models: {available_models}."
        )


class LocalTabPFNModelAdapter(BaseTabPFNModelAdapter):
    def __init__(
        self,
        tabpfn_config: dict,
        tabpfn_output_selection: str,
    ):
        super().__init__(
            model_class=TabPFNRegressor,
            model_config=tabpfn_config,
            tabpfn_output_selection=tabpfn_output_selection,
        )

        # Download the model if needed (for once)
        self.model_config["model_path"] = self._download_model(
            self.model_config["model_path"]
        )

    @staticmethod
    def _download_model(model_name: str):
        from tabpfn.model.loading import resolve_model_path, download_model

        # Resolve the model path
        # If the model path is not specified, this resolves to the default model path
        model_path, _, model_name, which = resolve_model_path(
            model_name,
            which="regressor",
        )

        if not model_path.exists():
            download_model(
                to=model_path,
                which=which,
                version="v2",
                model_name=model_name,
            )

        return model_path
