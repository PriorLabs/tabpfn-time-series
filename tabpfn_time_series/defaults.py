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
    """Default `model_path` to the v3 TS ckpt (in tabpfn's cache dir) when
    absent or None; a user-supplied path passes through unchanged."""
    config = {**tabpfn_config}
    if config.get("model_path") is None:
        # Route the bare filename through tabpfn's cache resolver: otherwise
        # `resolve_model_path` treats it as a literal path relative to the cwd,
        # bypassing the model cache dir (re-downloads per working directory).
        from tabpfn.model_loading import prepend_cache_path

        config["model_path"] = prepend_cache_path(TABPFN_V3_TS_CHECKPOINT)
    return config
