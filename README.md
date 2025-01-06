# Time Series Forecasting with TabPFN

We demonstrate that the tabular foundation model TabPFN, when paired with minimal featurization, can perform zero-shot time series forecasting. Its performance on point forecasting matches or even slightly outperforms state-of-the-art methods.

## 📖 How does it work?

Our work proposes to perform **univariate time series forecasting** by frame it as a **tabular regression problem**.

![How it works](docs/tabpfn-ts-method-overview.png)

Concretely, we:
1. transform a time series into a table
2. extract features from timestamp and add them to the table
3. perform regression on the table with TabPFN
4. regression results = time series forecasting results


For details, please refer to our [paper](TODO link to paper) and [poster](TODO link to poster).

## 👉 **Why gives us a try?**
- **Zero-shot forecasting**: this method is extremely fast and requires no training, making it highly accessible for experimenting with your own problems.
- **Point and probabilistic forecasting**: this method provides accurate both point and probabilistic forecasts.
- **Support exogenous variables**: know your exogenous variables? this method can easily incorporate exogenous variables into the forecasting model.

On top of that, thanks to [tabpfn-client](https://github.com/automl/tabpfn-client) from [Prior Labs](https://priorlabs.ai), you won’t even need your own GPU to run fast inference with TabPFN. 😉 We have included `tabpfn-client` as the default engine in our implementation.

## How to use it?

This demo should explain it all. 😉
