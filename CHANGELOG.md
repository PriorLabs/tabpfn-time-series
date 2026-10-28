# Changelog

## [Unreleased]

## [1.4.0] - 2026-10-28

### Added

- Support in-memory TabPFN ModelSpecs for local forecasts without copying weights or loading checkpoints. ([#185](https://github.com/PriorLabs/tabpfn-time-series/pull/185))
- Add `KnownCovariateLagFeature` (lags {1, season} and a trailing mean of every numeric known covariate, over context + horizon) as the first default TabPFN-TS feature. ([#195](https://github.com/PriorLabs/tabpfn-time-series/pull/195))
- Add `num_workers_per_gpu` to `TabPFNTSPipeline` to run several worker processes per GPU, each capped at its share of GPU memory and kept alive across `predict` calls. ([#197](https://github.com/PriorLabs/tabpfn-time-series/pull/197))

### Changed

- Require TabPFN 9.1.0 or newer from PyPI instead of a fixed Git commit. ([#186](https://github.com/PriorLabs/tabpfn-time-series/pull/186))
- `TabPFNTSPipeline` now compiles TabPFN with `torch.compile` on CUDA by default (`compile=True`), which requires TabPFN 9.2.0 or newer. ([#197](https://github.com/PriorLabs/tabpfn-time-series/pull/197))

### Fixed

- Development and release tooling no longer trusts the compromised eMudhra (emSign) CAs: pinned certifi, uv, pip and pypa/gh-action-pypi-publish are bumped to their fixed releases. ([#193](https://github.com/PriorLabs/tabpfn-time-series/pull/193))
- Custom quantiles are no longer replaced by the defaults when predicting with several GPU workers. ([#197](https://github.com/PriorLabs/tabpfn-time-series/pull/197))


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

- Require `datasets>=4`, `fev>=0.8.0` and `pandas>=2.2`. ([#177](https://github.com/PriorLabs/tabpfn-time-series/pull/177))


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
