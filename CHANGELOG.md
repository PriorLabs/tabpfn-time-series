# Changelog

## [1.1.0] — 2026-05-12

Default config now reproduces the **TabPFN-TS-3** paper's fev-bench numbers
(SQL 43.1, rank-2) out of the box.

### Changed

- Default checkpoint → **TabPFN-TS-3** in `LOCAL` mode (auto-downloaded by
  `tabpfn` on first init).
- `max_context_length` default: 4096 → **32768**.
- `AutoSeasonalFeature.max_top_k` default: 5 → **12**.
- `tabpfn>=8.0.0` (first PyPI release shipping the `tabpfn_v3` architecture).

### Docs

- README: new "Covariate model" section and checkpoint download notes.

### Migration

To keep v1.0.x behaviour, pass:

```python
TabPFNTSPipeline(
    max_context_length=4096,
    tabpfn_model_config={"model_path": "tabpfn-v2-regressor-2noar4o2.ckpt"},
)
```
