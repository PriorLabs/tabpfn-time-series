# Synthetic Covariate Study Tools

This directory contains tools for studying the effect of different types of synthetic covariates on time series prediction using TabPFN.

***Note**: This README was updated with assistance from Google's Gemini. Might contain slight misalignment.* 

## Files

- `main.py` - Main script for running covariate studies.
- `eval.py` - Contains the core logic for running a single experiment.
- `visualization.py` - Generates output plots and summary reports.
- `covariate_generators/` - Directory containing definitions for various synthetic covariate types.
- `covariate_study_on_synthetic.ipynb` - Original notebook with exploratory analysis.
- `playground.ipynb` - Additional experimentation notebook.

## Usage

The main script `main.py` allows you to study one or more covariate types systematically.

```bash
python main.py --covariate_type ramps,steps --n_samples 20
```

### Available Covariate Types

1.  **`linear_trend`**: A simple linear trend with a random positive or negative slope.
2.  **`ar1_trend`**: A stationary or near-unit-root AR(1) process, creating a smooth, slowly varying trend.
3.  **`logistic_growth`**: A logistic (sigmoid) growth curve, modeling saturating phenomena.
4.  **`random_walk`**: A random walk process, with configurable step sizes and intervals.
5.  **`pulses`**: Generates sparse pulse events at random timesteps.
6.  **`steps`**: Creates multiple step-wise changes (on/off) at random intervals.
7.  **`ramps`**: Generates multiple ramp-up or ramp-down events over random intervals.

### Command Line Arguments

- `--covariate_type` (required) - Comma-separated list of covariate types to study (e.g., "ramps,steps").
- `--n_samples` (default: 10) - Number of experiments to run for each covariate type.
- `--n_timesteps` (default: 1000) - Number of timesteps in each time series.
- `--train_ratio` (default: 0.6) - Ratio of data used for training.
- `--weight` (default: 1.0) - Weight for covariate effect.
- `--weight_sampling_concentration` (default: 5.0) - Concentration parameter for Beta distribution used for sampling covariate weights. Higher values lead to weights closer to the mean.
- `--relation` (default: additive) - Relationship between time series and covariate (`additive` or `multiplicative`).
- `--output_root_dir` (default: `./covariate_study_results`) - The root directory where study results will be saved.
- `--seed` (default: 42) - Random seed for reproducibility.
- `--verbose` or `-v` - Enable verbose (debug) logging.
- `--n_jobs` (default: -1) - Number of parallel jobs for predictions (-1 to use all available cores).

### Example Commands

```bash
# Study a combination of ramp and step covariates with 20 samples
python main.py --covariate_type ramps,steps --n_samples 20

# Study a logistic growth covariate with a multiplicative relationship
python main.py --covariate_type logistic_growth --relation multiplicative --weight 0.8

# Study a random walk with a longer time series
python main.py --covariate_type random_walk --n_timesteps 2000 --n_samples 15

# Run with verbose logging for debugging
python main.py --covariate_type pulses --verbose

# Run with 4 parallel jobs
python main.py --covariate_type ar1_trend --n_jobs 4 --n_samples 20
```

## Output

The script generates a new directory inside `--output_root_dir` for each run, named with a timestamp and the covariate types studied. This directory contains:

1.  **`results.pkl`**: A pickle file containing a list of `CovariateStudyResult` objects, with detailed metrics and data for each experimental run.
2.  **`summary.pdf`**: A PDF report with summary statistics and plots:
    - MSE and MAE comparison boxplots (with vs. without covariates).
    - Distribution of performance improvements.
    - Summary statistics table.
3.  **`predictions.pdf`**: A PDF report showing detailed visualizations for a subset of the experiments, including:
    - Prediction plots (with and without covariates).
    - Covariate signal visualization.
    - Raw vs. transformed data comparison.

## Requirements

- `tabpfn_client` - For TabPFN regression
- `numpy` - For numerical operations
- `matplotlib` - For plotting
- Standard Python libraries: `argparse`, `logging`, `pathlib`, `pickle`, `typing`.

## How It Works

1.  **Data Generation**: Creates synthetic time series with seasonality patterns.
2.  **Covariate Sampling**: Generates multiple instances of the specified covariate type(s), with randomized parameters for each sample.
3.  **Feature Engineering**: Creates time-based features and incorporates covariates.
4.  **Model Training**: Trains TabPFN models with and without covariates.
5.  **Evaluation**: Compares prediction performance using MSE and MAE metrics.
6.  **Visualization**: Generates comprehensive PDF reports with plots and statistics.

Each experiment uses a different random seed to ensure variety in the generated data and covariates, providing a robust evaluation of covariate effectiveness.

## Logging

The script uses Python's logging module to provide informative output:

-   **INFO level** (default): Shows experiment progress, results, and major milestones.
-   **DEBUG level** (`--verbose` flag): Additionally shows detailed covariate generation parameters and internal steps.

Example with verbose logging:
```bash
python main.py --covariate_type pulses --n_samples 5 --verbose
```

This will show detailed information about each covariate being generated, including periods, amplitudes, and other parameters.

## Parallelization

The script supports parallel execution to speed up the covariate study:

-   **Experiment-level parallelization**: Multiple experiments are distributed across available CPU cores using `joblib`.
-   **TabPFN calls**: Each TabPFN prediction runs independently, making it suitable for parallelization.

### Performance Benefits

Using parallelization can significantly reduce execution time:

```bash
# Sequential execution
python main.py --covariate_type ramps --n_samples 20 --n_jobs 1

# Parallel execution with all available cores
python main.py --covariate_type ramps --n_samples 20 --n_jobs -1
```

**Note**: Use `--n_jobs 1` for debugging or when memory is constrained.
