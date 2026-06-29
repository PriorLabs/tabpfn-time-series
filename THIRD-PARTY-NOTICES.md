# Third-Party Notices

This file documents third-party code copied or adapted into this repository,
with upstream attribution preserved. Transitive dependencies installed via
`pip` are governed by their own licenses (see `pyproject.toml` for the
canonical list).

---

## Summary

| Upstream | Local path | Upstream license |
|---|---|---|
| AutoGluon — `TimeSeriesDataFrame` | `tabpfn_time_series/ts_dataframe.py` | Apache-2.0 |
| Salesforce GIFT-Eval — `data.py` | `gift_eval/data.py` | Apache-2.0 |
| Amazon Chronos — `to_gluonts_univariate` | `tabpfn_time_series/data_preparation.py` (function) | Apache-2.0 |

---

## Per-upstream notices

### AutoGluon — TimeSeriesDataFrame

**Upstream:** https://github.com/autogluon/autogluon (path: `timeseries/src/autogluon/timeseries/dataset/ts_dataframe.py`)
**Local path:** `tabpfn_time_series/ts_dataframe.py`
**License:** Apache-2.0
**Copyright:** Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved. (per the AutoGluon `NOTICE` file)
**Modifications:** Adapted from upstream; removed the AutoGluon dependency and the
local-file-loading features. Upstream attribution preserved at the top of the file. Upstream
does not ship a per-file copyright header, so attribution is carried in this NOTICE plus the
in-file pointer to the upstream path.

### Salesforce GIFT-Eval — `data.py`

**Upstream:** https://github.com/SalesforceAIResearch/gift-eval
**Local path:** `gift_eval/data.py`
**License:** Apache-2.0
**Copyright:** Copyright (c) 2023, Salesforce, Inc.
**Modifications:** Per-file copyright and Apache-2.0 license header preserved
verbatim at the top of the file.

> Note: the rest of `gift_eval/` contains TabPFN integration code that targets
> the Salesforce-published GIFT-EVAL benchmark but is not directly derived
> from Salesforce source.

### Amazon Chronos — `to_gluonts_univariate`

**Upstream:** https://github.com/amazon-science/chronos-forecasting (path: `scripts/evaluation/evaluate.py`, line 28 at the pinned commit)
**Local path:** `tabpfn_time_series/data_preparation.py` (function `to_gluonts_univariate`)
**License:** Apache-2.0
**Copyright:** Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
**Modifications:** Adapted as a single-function vendoring from the upstream Chronos evaluation script. Upstream pins to a specific commit (`ad410c9`) referenced in the in-file attribution block.

---

## Adding new entries

When vendoring or adapting third-party code:

1. Preserve any upstream per-file copyright and license header verbatim. If the upstream does not ship a per-file header, add an attribution block citing the upstream URL, copyright holder, and SPDX license identifier (as in `tabpfn_time_series/ts_dataframe.py`).
2. When vendoring a whole directory of upstream code, also vendor the upstream `LICENSE` / `NOTICE` file alongside it. For single-file adaptations, the in-file attribution plus the entry in this NOTICE file is sufficient.
3. Add a row to the summary table and a per-upstream notice to this file, including the upstream copyright line when one is published.
