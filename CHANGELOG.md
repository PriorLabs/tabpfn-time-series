# Changelog

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
