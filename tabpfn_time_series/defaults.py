import logging

logger = logging.getLogger(__name__)

DEFAULT_QUANTILE_CONFIG = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# TabPFN-TS-3 ship config from the TabPFN-3 paper.
# Hosted at https://huggingface.co/Prior-Labs/tabpfn_3
_HF_REPO = "Prior-Labs/tabpfn_3"
TABPFN_V3_TS_CHECKPOINT = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"

TABPFN_DEFAULT_CONFIG = {"model_path": TABPFN_V3_TS_CHECKPOINT}


def _resolve_v3_ts_checkpoint() -> str:
    """Fetch the TabPFN-TS-3 checkpoint from Hugging Face (cached after first call)."""
    try:
        from huggingface_hub import hf_hub_download

        logger.info(
            "Resolving TabPFN-TS-3 checkpoint via Hugging Face Hub "
            "(first run downloads ~hundreds of MB; cached under ~/.cache/huggingface/)."
        )
        return hf_hub_download(repo_id=_HF_REPO, filename=TABPFN_V3_TS_CHECKPOINT)
    except Exception as e:
        raise RuntimeError(
            f"Could not resolve TabPFN-TS-3 checkpoint ({type(e).__name__}: {e}). "
            f"Download it manually from https://huggingface.co/{_HF_REPO}/resolve/main/"
            f"{TABPFN_V3_TS_CHECKPOINT} and pass via "
            "tabpfn_model_config={'model_path': '/path/to/ckpt'}."
        ) from e


def resolve_default_ckpt(tabpfn_config: dict) -> dict:
    """Auto-resolve the default v3 checkpoint when `model_path` is missing or
    matches the bare v3 filename. User-supplied paths pass through unchanged."""
    path = tabpfn_config.get("model_path")
    if path is None or path == TABPFN_V3_TS_CHECKPOINT:
        return {**tabpfn_config, "model_path": _resolve_v3_ts_checkpoint()}
    return tabpfn_config
