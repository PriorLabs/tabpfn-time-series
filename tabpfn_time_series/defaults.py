DEFAULT_QUANTILE_CONFIG = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Empty by default: LOCAL mode loads tabpfn's own default model (TabPFN-3.5 in
# tabpfn 9, see `tabpfn.settings.model_version`); CLIENT mode lets the cloud
# server pick whichever ts model it currently hosts.
TABPFN_DEFAULT_CONFIG: dict = {}
