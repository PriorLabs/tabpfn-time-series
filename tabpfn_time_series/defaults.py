DEFAULT_QUANTILE_CONFIG = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# TabPFN-TS-3 ship config from the TabPFN-3 paper.
# `tabpfn` downloads this ckpt automatically on first init (after license
# acceptance at https://ux.priorlabs.ai).
TABPFN_V3_TS_CHECKPOINT = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"

# Empty by default: LOCAL mode's `resolve_default_ckpt` fills in the v3 ckpt;
# CLIENT mode lets the cloud server pick whichever ts model it currently hosts
# (the cloud client doesn't ship v3 yet).
TABPFN_DEFAULT_CONFIG: dict = {}


def resolve_default_ckpt(tabpfn_config: dict) -> dict:
    """Set `model_path` to the v3 ts ckpt when not provided. tabpfn handles the
    actual download on first call. User-supplied paths pass through unchanged."""
    return {"model_path": TABPFN_V3_TS_CHECKPOINT, **tabpfn_config}
