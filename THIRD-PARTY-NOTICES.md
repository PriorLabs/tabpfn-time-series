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

---

## Per-upstream notices

### AutoGluon — TimeSeriesDataFrame

**Upstream:** https://github.com/autogluon/autogluon/blob/main/autogluon/timeseries/ts_dataframe.py
**Local path:** `tabpfn_time_series/ts_dataframe.py`
**License:** Apache-2.0
**Modifications:** File copied verbatim; removed AutoGluon dependency and the
local-file-loading features. Per-file header preserved at the top of the file.

### Salesforce — GIFT-Eval data loading

**Upstream:** https://github.com/SalesforceAIResearch/gift-eval
**Local path:** `gift_eval/data.py`
**License:** Apache-2.0
**Copyright:** Copyright (c) 2023, Salesforce, Inc.
**Modifications:** Per-file copyright and Apache-2.0 license header preserved
verbatim at the top of the file.

> Note: the rest of `gift_eval/` contains TabPFN integration code that targets
> the Salesforce-published GIFT-EVAL benchmark but is not directly derived
> from Salesforce source.

---

## Adding new entries

When vendoring or adapting third-party code:

1. Preserve the upstream copyright and license header verbatim at the top of the affected source file.
2. If the upstream ships a `LICENSE` / `NOTICE` file, vendor that file alongside the code.
3. Add a row to the summary table and a per-upstream notice to this file.
