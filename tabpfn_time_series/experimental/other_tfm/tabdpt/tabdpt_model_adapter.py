import os
from pathlib import Path
import logging

from dotenv import load_dotenv

from huggingface_hub import hf_hub_download
from tabdpt import TabDPTRegressor

from tabpfn_time_series.worker.model_adapter import PointPredictionModelAdapter

logger = logging.getLogger(__name__)

load_dotenv()


class TabDPTModelAdapter(PointPredictionModelAdapter):
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
        model_config["model_path"] = os.getenv("TABDPT_MODEL_PATH")
        assert model_config["model_path"] is not None, "TABDPT_MODEL_PATH is not set"
        assert model_config["model_path"].endswith(self._MODEL_NAME), (
            f"Model path must end with '{self._MODEL_NAME}', "
            f"but got: {model_config['model_path']}"
        )

        # ---- Download the model if needed (for once)
        if not Path(model_config["model_path"]).exists():
            self._download_model(self._MODEL_NAME, self._MODEL_PATH)

        super().__init__(
            model_class=TabDPTRegressor,
            model_config=model_config,
            inference_config=inference_config,
        )

    @staticmethod
    def _download_model(model_name: str, model_path: str) -> str:
        # Download model and save to model_path
        logger.info(f"Downloading model {model_name} to {model_path}")

        return hf_hub_download(
            repo_id="Layer6/TabDPT",
            filename=model_name,
            local_dir=model_path,
        )
