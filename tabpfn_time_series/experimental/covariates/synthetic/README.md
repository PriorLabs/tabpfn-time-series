# Synthetic Covariate Study

Tools for studying the effect of synthetic covariates on time series prediction using TabPFN.

## Quick Start

```bash
python main.py --covariate_type ramps,steps --n_samples 20
```

## Available Covariate Types

- **`linear_trend`** - Linear trend with random slope
- **`ar1`** - Stationary/near-unit-root AR(1) process  
- **`logistic_growth`** - Logistic (sigmoid) growth curve
- **`random_walk`** - Random walk with configurable steps
- **`pulses`** - Sparse pulse events at random timesteps
- **`steps`** - Step-wise changes (on/off) over intervals
- **`ramps`** - Ramp-up/down events over time intervals

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--covariate_type` | *required* | Comma-separated covariate types (e.g., "ramps,steps") |
| `--n_samples` | 10 | Number of experiments per covariate type |
| `--n_timesteps` | 1000 | Time series length |
| `--train_ratio` | 0.6 | Training data ratio |
| `--relation` | additive | Covariate relationship (`additive`/`multiplicative`) |
| `--use_random_delay` | False | Add random delay to covariate impact |
| `--n_jobs` | -1 | Parallel jobs (-1 for all cores) |
| `--verbose` | False | Enable debug logging |

## Examples

```bash
# Multiple covariate types
python main.py --covariate_type ramps,steps --n_samples 20

# Multiplicative relationship
python main.py --covariate_type logistic_growth --relation multiplicative

# Longer series with random delay
python main.py --covariate_type random_walk --n_timesteps 2000 --use_random_delay

# Debug mode
python main.py --covariate_type pulses --verbose
```

## Output

Results are saved to timestamped directories in `./covariate_study_results/`:

- **`results.pkl`** - Raw experiment data
- **`summary.pdf`** - Performance comparison plots and statistics  
- **`predictions.pdf`** - Detailed prediction visualizations

## Files

- `main.py` - Main experiment runner
- `eval.py` - Core evaluation logic
- `visualization.py` - Report generation
- `covariate_generators/` - Covariate type definitions
- `covariate_study_on_synthetic.ipynb` - Exploratory analysis notebook

## How It Works

1. **Generate** synthetic time series with seasonality
2. **Create** specified covariate types with random parameters
3. **Train** TabPFN models with/without covariates
4. **Compare** prediction performance (MSE, MAE, MASE, SQL)
5. **Visualize** results in comprehensive PDF reports

Each experiment uses different random seeds for robust evaluation across varied synthetic data patterns.


