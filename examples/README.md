# TabPFN Time Series Examples

This folder contains usage examples demonstrating core functionalities of the `tabpfn-time-series` library.

## Examples

- `tabpfn_family_model_as_backbone.py`: showcases how to use any TabPFN-family model as the inference backbone.
- `sklearn_model_as_backbone.py`: showcases how to use any standard Sklearn regressors as the inference backbone.
- `explainability_electricity.py`: explains a forecast with `TabPFNTSExplainer` (partial dependence, window SHAP, forecast decomposition) on the electricity-price dataset. Saves figures to `explainability_outputs/`.

### Explainability (`explainability_electricity.py`)

`TabPFNTSExplainer` wraps a fitted `TabPFNTSPipeline` and offers three lightweight explanations. The two model-based ones (PDP, window SHAP) keep cost down by fitting the model once per window and reusing that fit across every perturbation (only the forecast horizon changes), so the number of TabPFN fits equals the number of windows, not the number of perturbations.

- **Partial dependence** of the point forecast over calendar concepts (in the original feature space, e.g. sweep hour 0..23 and re-encode to sin/cos) and over known covariates.
- **Window SHAP**: grouped Shapley attributions of the point forecast across rolling forecast windows (the "window" framing follows [WindowSHAP](https://arxiv.org/abs/2211.06507)), plotted as a feature × time "spectrogram". Shapley values are estimated with [`shapiq`](https://github.com/mmschlk/shapiq)'s `KernelSHAP`. You can set the window size and forecasting horizon with keyword arguments.
- **Series decomposition**: a classic model-free additive decomposition of the target signal itself (not the forecast) into trend + per-time-feature seasonal components + residual (`observed = trend + hour_of_day + day_of_week + residual`).

Window SHAP attributes to interpretable **feature groups**, not raw featurizer columns, so each calendar concept reads as a single row rather than an opaque sin/cos pair:

- one group per calendar concept (`hour_of_day`, `day_of_week`, ...), owning its `{concept}_sin` / `{concept}_cos` columns;
- `trend`: `running_index` and `year` (the non-periodic drift terms);
- `auto_seasonal`: the auto-detected Fourier columns (`sin_#.../cos_#...`), by running a Fast Fourier Transform (FFT) over the series and keeping the most important periods;
- one group per remaining column, treated as a user covariate.

#### Interpreting the results

TabPFN-Time Series does not use lagged features, so you will not see these in the SHAP plots and PDP. This auto-regressive signal is captured implicitly through the seasonal features (hour of day, day of week, etc...). The model-free decomposition is the most direct view of the target's own structure.

Install the optional dependencies (`shapiq`, `matplotlib`), then run the example to generate the figures (written to `explainability_outputs/`, which is git-ignored):

```bash
pip install 'tabpfn-time-series[explainability]'
python examples/explainability_electricity.py
```
