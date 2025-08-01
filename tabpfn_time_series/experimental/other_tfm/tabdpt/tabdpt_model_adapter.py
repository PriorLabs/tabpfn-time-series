import os
import numpy as np
from pathlib import Path
import logging

from dotenv import load_dotenv

from huggingface_hub import hf_hub_download
from tabdpt import TabDPTRegressor

from tabpfn_time_series.worker.model_adapter import ModelAdapter
from tabpfn_time_series.defaults import DEFAULT_QUANTILE_CONFIG

logger = logging.getLogger(__name__)

load_dotenv()


class TabDPTModelAdapter(ModelAdapter):
    _DEFAULT_MODEL_CONFIG = {
        "use_flash": False,
        "compile": False,
        "device": "cuda",
    }

    _MODEL_NAME = "tabdpt1_1.safetensors"
    _MODEL_PATH = Path.cwd() / "tabdpt_model"

    def __init__(
        self,
        model_config: dict = _DEFAULT_MODEL_CONFIG,
        inference_config: dict = {},
    ):
        # self._download_model(self._MODEL_NAME, self._MODEL_PATH)
        # model_config["model_path"] = str(self._MODEL_PATH / self._MODEL_NAME)

        model_config["model_path"] = os.getenv("TABDPT_MODEL_PATH")
        super().__init__(
            model_class=TabDPTRegressor,
            model_config=model_config,
            inference_config=inference_config,
        )

    def predict(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        quantiles: list[float | str] = DEFAULT_QUANTILE_CONFIG,
    ):
        pred_output = super().predict(
            train_X=train_X,
            train_y=train_y,
            test_X=test_X,
        )

        # TabDPT doesn't return uncertainty estimates
        # so workaround, we will return the estimated target instead
        # Therefore, we ignore the uncertainty estimates.
        result = {"target": pred_output}
        result.update({q: pred_output for q in quantiles})

        return result

    @staticmethod
    def _download_model(model_name: str, model_path: str) -> str:
        # Download model and save to model_path
        logger.info(f"Downloading model {model_name} to {model_path}")

        return hf_hub_download(
            repo_id="Layer6/TabDPT",
            filename=model_name,
            local_dir=model_path,
        )
