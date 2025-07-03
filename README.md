# TabPFN-TS

> Zero-Shot Time Series Forecasting with TabPFNv2

[![PyPI version](https://badge.fury.io/py/tabpfn-time-series.svg)](https://badge.fury.io/py/tabpfn-time-series)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/tabpfn-time-series/blob/main/demo.ipynb)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![arXiv](https://img.shields.io/badge/arXiv-2501.02945-<COLOR>.svg)](https://arxiv.org/abs/2501.02945v3)

## 📌 News
- **27-05-2025**: 📝 New **[paper](https://arxiv.org/abs/2501.02945v3)** version and **v1.0.0** release! Strong [GIFT-EVAL](https://huggingface.co/spaces/Salesforce/GIFT-Eval) results, new AutoSeasonalFeatures, improved CalendarFeatures.
- **27-01-2025**: 🚀 Ranked _**1st**_ on [GIFT-EVAL](https://huggingface.co/spaces/Salesforce/GIFT-Eval) benchmark<sup>[1]</sup>!
- **10-10-2024**: 🚀 TabPFN-TS [paper](https://arxiv.org/abs/2501.02945v2) accepted to NeurIPS 2024 [TRL](https://table-representation-learning.github.io/NeurIPS2024/) and [TSALM](https://neurips-time-series-workshop.github.io/) workshops!

_[1] Last checked on: 10/03/2025_

## ✨ Introduction
We demonstrate that the tabular foundation model **[TabPFNv2](https://github.com/PriorLabs/TabPFN)**, combined with lightweight feature engineering, enables zero-shot time series forecasting for both point and probabilistic tasks. On the **[GIFT-EVAL](https://huggingface.co/spaces/Salesforce/GIFT-Eval)** benchmark, our method achieves performance on par with top-tier models across both evaluation metrics.

## 📖 How does it work?

Our work proposes to frame **univariate time series forecasting** as a **tabular regression problem**.

![How it works](docs/tabpfn-ts-method-overview.png)

Concretely, we:
1. Transform a time series into a table
2. Extract features and add them to the table
3. Perform regression on the table using TabPFNv2
4. Use regression results as time series forecasting outputs

For more details, please refer to our [paper](https://arxiv.org/abs/2501.02945v3).
 <!-- and our [poster](docs/tabpfn-ts-neurips-poster.pdf) (presented at NeurIPS 2024 TRL and TSALM workshops). -->

## 👉 **Why give us a try?**
- **Zero-shot forecasting**: this method is extremely fast and requires no training, making it highly accessible for experimenting with your own problems.
- **Point and probabilistic forecasting**: it provides accurate point forecasts as well as probabilistic forecasts.
- **Support for exogenous variables**: if you have exogenous variables, this method can seemlessly incorporate them into the forecasting model.

On top of that, thanks to **[tabpfn-client](https://github.com/automl/tabpfn-client)** from **[Prior Labs](https://priorlabs.ai)**, you won't even need your own GPU to run fast inference with TabPFNv2. 😉 We have included `tabpfn-client` as the default engine in our implementation.

## ⚙️ Installation

You can install the package via pip:

```bash
pip install tabpfn-time-series
```

### For Developers

To install the package in editable mode with all development dependencies, run the following command in your terminal:

```bash
pip install -e ".[dev]"
# or with uv
uv pip install -e ".[dev]"
```

## How to use it?

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PriorLabs/tabpfn-time-series/blob/main/demo.ipynb)

The demo should explain it all. 😉
