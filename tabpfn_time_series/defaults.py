DEFAULT_QUANTILE_CONFIG = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# TabPFN-TS-3 ship config from the TabPFN-3 paper.
# `tabpfn` downloads this ckpt automatically on first init (after license
# acceptance at https://ux.priorlabs.ai).
TABPFN_V3_TS_CHECKPOINT = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"

# Empty by default: LOCAL mode's `resolve_default_ckpt` fills in the v3 ckpt;
# CLIENT mode lets the cloud server pick whichever ts model it currently hosts
# (the cloud client doesn't ship v3 yet).
TABPFN_DEFAULT_CONFIG: dict = {}


def _default_ckpt_path() -> str:
    """Absolute path to the v3 TS ckpt inside tabpfn's model cache dir.

    `tabpfn`'s `resolve_model_path` treats a non-None `model_path` as a literal
    path: a bare filename resolves relative to the *cwd*, bypassing the cache
    dir entirely (re-downloads per working directory). Routing the filename
    through tabpfn's own cache resolver instead makes the ckpt download into
    (and load from) the standard cache dir -- honoring `model_cache_dir` /
    `XDG_CACHE_HOME`, shared with plain `tabpfn`, cwd-independent."""
    from tabpfn.model_loading import prepend_cache_path

    return prepend_cache_path(TABPFN_V3_TS_CHECKPOINT)


def resolve_default_ckpt(tabpfn_config: dict) -> dict:
    """Default `model_path` to the v3 TS ckpt (in tabpfn's cache dir) when
    absent or None; a user-supplied path passes through unchanged."""
    config = {**tabpfn_config}
    if config.get("model_path") is None:
        config["model_path"] = _default_ckpt_path()
    return config
