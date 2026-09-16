# Changelog

## [Unreleased]

## [1.3.0] - 2026-09-16

### Breaking Changes

- Default model is now **TabPFN-3.5** in both `LOCAL` and `CLIENT` mode. `LOCAL` mode previously pinned the finetuned **TabPFN-TS-3** checkpoint; `CLIENT` mode previously let the cloud API pick its own default. Requires `tabpfn>=9.0.0` and `tabpfn-client>=0.5.3`. ([#176](https://github.com/PriorLabs/tabpfn-time-series/pull/176))

    To keep TabPFN-TS-3 in `LOCAL` mode (`tabpfn` still ships the checkpoint):

    ```python
    TabPFNTSPipeline(
        tabpfn_mode=TabPFNMode.LOCAL,
        tabpfn_model_config={"model_path": "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt"},
    )
    ```

### Changed

- Releases are now assembled from changelog fragments: add a `changelog/<PR>.<category>.md` file describing your change, instead of editing `CHANGELOG.md` directly. See `changelog/README.md`. ([#151](https://github.com/PriorLabs/tabpfn-time-series/pull/151))

### Fixed

- Require `datasets>=4`, `fev>=0.8.0` and `pandas>=2.2`. The `how-it-works` notebook failed to load the Chronos datasets from the Hub with older `datasets` releases (`Feature type 'List' not found`); `fev` 0.8.0 is the first release that allows `datasets>=4`; and the package maps the `ME` month-end alias that pandas only accepts from 2.2 on. ([#177](https://github.com/PriorLabs/tabpfn-time-series/pull/177))


## [1.1.0] — 2026-05-12

Default config now ships the finetuned **TabPFN-TS-3** checkpoint from the
[TabPFN-3 report](https://priorlabs.ai/reports/tabpfn-3).

### Changed

- Default checkpoint → **TabPFN-TS-3** in `LOCAL` mode (auto-downloaded by
  `tabpfn` on first init).
- `max_context_length` default: 4096 → **32768**.
- `AutoSeasonalFeature.max_top_k` default: 5 → **12**.
- `tabpfn>=8.0.0` (first PyPI release shipping the `tabpfn_v3` architecture).

### Docs

- README: new "Covariate model" section.

### Migration

To keep v1.0.x behaviour, pass:

```python
TabPFNTSPipeline(
    max_context_length=4096,
    tabpfn_model_config={"model_path": "tabpfn-v2-regressor-2noar4o2.ckpt"},
)
```
