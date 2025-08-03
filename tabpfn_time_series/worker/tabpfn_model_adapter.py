import numpy as np
from typing import Dict, Type

from sklearn.base import RegressorMixin
from tabpfn import TabPFNRegressor
from tabpfn_client import (
    init as tabpfn_client_init,
    TabPFNRegressor as TabPFNClientRegressor,
)

from tabpfn_time_series.defaults import DEFAULT_QUANTILE_CONFIG
from tabpfn_time_series.worker.model_adapter import BaseModelAdapter, PredictionOutput


def parse_tabpfn_client_model_name(model_name: str) -> str:
    available_models = TabPFNClientRegressor.list_available_models()
    for m in available_models:
        if m in model_name:
            return m

    raise ValueError(
        f"Model {model_name} not found. Available models: {available_models}."
    )


class TabPFNModelAdapter(BaseModelAdapter):
    def __init__(
        self,
        model_class: Type[RegressorMixin],
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

        if model_class == TabPFNClientRegressor:
            self._init_tabpfn_client_regressor(self.model_config)
        elif model_class == TabPFNRegressor:
            self._init_local_tabpfn_regressor(self.model_config)
        else:
            raise ValueError(
                f"Expected TabPFN-family regressor, got {self.model_class}"
            )

    def postprocess_pred_output(
        self,
        raw_pred_output: Dict[str, np.ndarray],
        quantiles: list[float],
    ) -> PredictionOutput:
        # Translate TabPFN output to the standardized dictionary format
        assert quantiles == DEFAULT_QUANTILE_CONFIG, (
            "Quantiles must be the default quantiles for TabPFN"
        )
        result: PredictionOutput = {
            "target": raw_pred_output[self.tabpfn_output_selection]
        }
        result.update(
            {q: raw_pred_output["quantiles"][i] for i, q in enumerate(quantiles)}
        )

        return result

    @staticmethod
    def _init_tabpfn_client_regressor(tabpfn_config: dict):
        # Initialize the TabPFN client (authentication)
        tabpfn_client_init()

        # Parse the model name to get the correct model path that is
        # supported by the TabPFN client (slightly different naming convention)
        if "model_path" in tabpfn_config:
            model_name = parse_tabpfn_client_model_name(tabpfn_config["model_path"])
            tabpfn_config["model_path"] = model_name

    @staticmethod
    def _init_local_tabpfn_regressor(tabpfn_config: dict):
        from tabpfn.model.loading import resolve_model_path, download_model

        model_path, _, model_name, which = resolve_model_path(
            tabpfn_config["model_path"] if "model_path" in tabpfn_config else None,
            which="regressor",
        )

        if not model_path.exists():
            download_model(
                to=model_path,
                which=which,
                version="v2",
                model_name=model_name,
            )

        tabpfn_config["model_path"] = model_path
