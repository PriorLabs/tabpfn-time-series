import numpy as np
from typing import Dict, Type

from sklearn.base import RegressorMixin
from tabpfn import TabPFNRegressor
from tabpfn_client import (
    init as tabpfn_client_init,
    TabPFNRegressor as TabPFNClientRegressor,
)

from tabpfn_time_series.worker.model_adapters.base import (
    BaseModelAdapter,
    PredictionOutput,
)


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
        from tabpfn.model_loading import (
            download_model,
            resolve_model_path,
            resolve_model_version,
        )

        model_path = tabpfn_config.get("model_path")
        model_version = resolve_model_version(model_path)
        resolved_model_paths, _, model_names, which = resolve_model_path(
            model_path,
            which="regressor",
        )
        assert len(resolved_model_paths) == 1
        resolved_model_path = resolved_model_paths[0]
        assert len(model_names) == 1
        model_name = model_names[0]

        if not resolved_model_path.exists():
            download_model(
                to=resolved_model_path,
                which=which,
                version=model_version,
                model_name=model_name,
            )

        tabpfn_config["model_path"] = resolved_model_path
