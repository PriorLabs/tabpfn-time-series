# Time Series Forecasting with TabPFN

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liam-sbhoo/tabpfn-time-series/blob/main/demo.ipynb)
[![Discord](https://img.shields.io/discord/1285598202732482621?color=7289da&label=Discord&logo=discord&logoColor=ffffff)](https://discord.com/channels/1285598202732482621/)
[![arXiv](https://img.shields.io/badge/arXiv-<INDEX>-<COLOR>.svg)](https://arxiv.org/abs/<INDEX>)


We demonstrate that the tabular foundation model TabPFN, when paired with minimal featurization, can perform zero-shot time series forecasting. Its performance on point forecasting matches or even slightly outperforms state-of-the-art methods.

## 📖 How does it work?

Our work proposes to perform **univariate time series forecasting** by frame it as a **tabular regression problem**.

![How it works](docs/tabpfn-ts-method-overview.png)

Concretely, we:
1. transform a time series into a table
2. extract features from timestamp and add them to the table
3. perform regression on the table with TabPFN
4. regression results = time series forecasting results



For details, please refer to our [paper](TODO link to paper) and our [poster](docs/tabpfn-ts-poster.pdf) (presented at NeurIPS 2024 workshops).
## 👉 **Why gives us a try?**
- **Zero-shot forecasting**: this method is extremely fast and requires no training, making it highly accessible for experimenting with your own problems.
- **Point and probabilistic forecasting**: this method provides accurate both point and probabilistic forecasts.
- **Support exogenous variables**: know your exogenous variables? this method can easily incorporate exogenous variables into the forecasting model.

On top of that, thanks to [tabpfn-client](https://github.com/automl/tabpfn-client) from [Prior Labs](https://priorlabs.ai), you won’t even need your own GPU to run fast inference with TabPFN. 😉 We have included `tabpfn-client` as the default engine in our implementation.

## How to use it?

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liam-sbhoo/tabpfn-time-series/blob/main/demo.ipynb)

The demo should explain it all. 😉
