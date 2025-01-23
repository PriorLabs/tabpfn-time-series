# Evaluating TabPFN-TS on GIFT-EVAL

This repository provides a structured workflow for evaluating **TabPFN-TS** on [**GIFT-EVAL**](https://github.com/SalesforceAIResearch/gift-eval).

Follow the steps below to set up your environment, run evaluations, and aggregate results!  

## ⚙️ Model Configurations

The model used for the evaluation builds upon our [paper](https://arxiv.org/abs/2501.02945), with some adjustments in the preprocessing steps:
1. **Handling Missing Values**: We drop the data points containing NaN values.
2. **Context Length**: We limit the context length to a maximum of 4096 data points.

## 📌 Setup

Getting started is easy! We’ve included a setup script that will:

✅ Install all required dependencies\
✅ Automatically download the GIFT-EVAL datasets

Run the following command to set up your environment: 

```bash
cd gift_eval
./setup.sh
```

## 📊 Running the Evaluation

Once you’re set up, evaluating a dataset is as simple as running:

```bash
python evaluate.py --dataset <dataset_name> --output_dir <output_dir>
```

> [!TIP]
> **It is highly recommended to run the evaluation on a GPU or a multi-GPU machine.**
> Since TabPFN-TS is limited by its inference speed, our implementation supports multi-GPU inference to optimize the evaluation process.


## 📈 Aggregating Results

Since evaluation results for each dataset are stored separately, we’ve included a utility to merge all results into a single file for easy comparison:

```bash
python aggregate_results.py --result_root_dir <result_root_dir>
```

where `result_root_dir` is the same as the `--output_dir` used in the evaluation command.
