DEFAULT_QUANTILE_CONFIG = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Pinned in both modes, so a new tabpfn or cloud default cannot change forecasts
# under a released tabpfn-time-series version.
TABPFN_MODEL_VERSION = "v3.5"

# Empty by default: `resolve_default_model_path` fills in the pinned version.
TABPFN_DEFAULT_CONFIG: dict = {}


def resolve_default_model_path(tabpfn_config: dict, *, client: bool) -> dict:
    """Default `model_path` to `TABPFN_MODEL_VERSION` when absent or None.

    LOCAL mode pins that version's default checkpoint in tabpfn's cache dir;
    CLIENT mode pins the server-side `<version>_default` alias. A user-supplied
    path passes through unchanged.
    """
    config = {**tabpfn_config}
    if config.get("model_path") is not None:
        return config
    if client:
        config["model_path"] = f"{TABPFN_MODEL_VERSION}_default"
    else:
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion

        config["model_path"] = TabPFNRegressor.create_default_for_version(
            ModelVersion(TABPFN_MODEL_VERSION)
        ).model_path
    return config
