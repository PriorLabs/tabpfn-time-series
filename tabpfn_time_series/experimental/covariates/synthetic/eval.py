import os
from dataclasses import dataclass
from typing import Dict, Any, List
import logging
from joblib import Parallel, delayed
from dotenv import load_dotenv

import numpy as np
from tabpfn_client import TabPFNRegressor, set_access_token

from tabpfn_time_series.experimental.utils.metrics import compute_mase, compute_sql

from .covariate_generators import CovariateGenerator, get_covariate_generator


load_dotenv()
set_access_token(os.getenv("TABPFN_ACCESS_TOKEN"))

RANDOM_DELAY_MIN_DELAY = 20
RANDOM_DELAY_MAX_DELAY = 100  # in timesteps

logger = logging.getLogger(__name__)


def generate_raw_data(
    n_timesteps: int,
    seasonality_periods: List[int] = [24, 48],
    noise_level: float = 0,
    seed: int = 0,
) -> np.ndarray:
    """Generate raw time series data with seasonality."""
    rng = np.random.RandomState(seed)

    series = np.zeros(n_timesteps)
    for period in seasonality_periods:
        amplitude = rng.uniform(0.5, 1.5)
        phase = rng.uniform(0, 2 * np.pi)
        series += amplitude * np.sin(
            2 * np.pi * np.arange(n_timesteps) / period + phase
        )

    if noise_level > 0:
        noise = rng.normal(0, noise_level, n_timesteps)
        series += noise

    return series


def transform_with_covariate(
    x: np.ndarray,
    covariate: np.ndarray,
    relation: str = "additive",
    weight: float = 1.0,
    delay: int = 0,
) -> np.ndarray:
    """Transform time series with covariate.

    Args:
        x: Time series data
        covariate: Covariate data
        relation: Relation between time series and covariate
        delay: Delay of impact in timesteps
        weight: Weight of the covariate
    """

    # Delay the impact of the covariate
    if delay > 0:
        covariate = np.concatenate([np.zeros(delay), covariate[:-delay]])
    elif delay < 0:
        raise ValueError(f"Negative delay is not supported: {delay}")

    # Apply the relation between the time series and the covariate
    if relation == "additive":
        return x + weight * covariate
    elif relation == "multiplicative":
        return x * (1 + weight * covariate)
    else:
        raise ValueError(f"Invalid relation: {relation}")


def create_features(n_timesteps: int, sinusoids_period: int = 96) -> np.ndarray:
    """Create feature matrix for TabPFN."""
    x = np.arange(n_timesteps)
    X = np.array(
        [
            np.arange(-1, 1, 2 / n_timesteps),
            np.sin(x * 2 * np.pi / sinusoids_period),
            np.cos(x * 2 * np.pi / sinusoids_period),
        ]
    )
    return X.T


@dataclass
class PredictionResult:
    predictions: np.ndarray
    quantiles: np.ndarray
    mase: float
    sql: float


@dataclass
class SingleExperimentResult:
    experiment_id: int
    covariate_types: List[str]
    covariate_configs: List[Dict]
    covariate_weights: List[float]
    raw_data: np.ndarray
    individual_covariates: List[np.ndarray]
    data_with_covariate: np.ndarray
    n_train_timesteps: int
    results_without_covariate: PredictionResult
    results_with_covariate: PredictionResult


@dataclass
class StudyResult:
    covariate_types: List[str]
    n_samples: int
    n_timesteps: int
    exp_results: List[SingleExperimentResult]
    parameters: Dict[str, Any]


def evaluate_prediction_job(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> PredictionResult:
    """Evaluate TabPFN prediction (parallelizable job)."""

    tabpfn = TabPFNRegressor()
    tabpfn.fit(X_train, y_train)
    full_preds = tabpfn.predict(X_test, output_type="main")

    preds = full_preds["median"]
    quantiles = full_preds["quantiles"]

    # Calculate scale-invariant metrics
    mase = compute_mase(y_test, preds, y_train, seasonality=1)
    sql = compute_sql(y_test, quantiles, y_train, seasonality=1)

    return PredictionResult(predictions=preds, quantiles=quantiles, mase=mase, sql=sql)


def run_single_experiment(
    experiment_id: int,
    covariate_gens: List[CovariateGenerator],
    base_features: np.ndarray,
    n_timesteps: int,
    n_train_timesteps: int,
    weight: float,
    relation: str,
    seed: int,
    use_random_delay: bool = False,
    n_jobs: int = 1,
    weight_sampling_concentration: float = 5.0,
) -> SingleExperimentResult:
    """Run a single experiment (parallelizable job)."""
    logger.info(f"Experiment {experiment_id + 1}: Starting")

    # Generate raw data
    experiment_seed = seed + experiment_id  # Different seed for each experiment
    rng = np.random.RandomState(experiment_seed)
    raw_data = generate_raw_data(n_timesteps, seed=experiment_seed)

    # Generate random weights for covariates
    n_covariates = len(covariate_gens)
    covariate_weights = rng.dirichlet(
        np.ones(n_covariates) * weight_sampling_concentration
    ).tolist()

    # Generate individual covariates
    individual_covariates: List[np.ndarray] = []
    covariate_configs: List[Dict] = []
    for i, gen in enumerate(covariate_gens):
        random_params = gen.generate_random_parameters(
            seed=seed + experiment_id + i
        )  # different seed for each covariate
        covariate = gen.generate(n_timesteps=n_timesteps, **random_params)
        individual_covariates.append(covariate)
        covariate_configs.append(random_params)

    # Iteratively transform data with each covariate
    data_with_covariate = raw_data.copy()
    for i, cov in enumerate(individual_covariates):
        # The final effect strength is a product of the global weight and the individual random weight
        effective_weight = weight * covariate_weights[i]
        data_with_covariate = transform_with_covariate(
            x=data_with_covariate,
            covariate=cov,
            relation=relation,
            weight=effective_weight,
            delay=rng.randint(RANDOM_DELAY_MIN_DELAY, RANDOM_DELAY_MAX_DELAY)
            if use_random_delay
            else 0,
        )

    # Prepare features by concatenating all individual covariates
    individual_covariates_features = np.stack(individual_covariates, axis=1)
    X_with_covariate = np.concatenate(
        [base_features, individual_covariates_features], axis=1
    )

    # Split data
    X_train = base_features[:n_train_timesteps]
    X_train_with_cov = X_with_covariate[:n_train_timesteps]
    y_train = data_with_covariate[:n_train_timesteps]

    X_test = base_features[n_train_timesteps:]
    X_test_with_cov = X_with_covariate[n_train_timesteps:]
    y_test = data_with_covariate[n_train_timesteps:]

    # Create prediction jobs
    prediction_jobs: List[PredictionResult] = [
        delayed(evaluate_prediction_job)(
            X_train,
            y_train,
            X_test,
            y_test,
        ),
        delayed(evaluate_prediction_job)(
            X_train_with_cov,
            y_train,
            X_test_with_cov,
            y_test,
        ),
    ]

    # Execute predictions in parallel
    prediction_results = Parallel(n_jobs=n_jobs)(prediction_jobs)
    results_no_cov, results_with_cov = prediction_results

    # Store results
    experiment_result = SingleExperimentResult(
        experiment_id=experiment_id,
        covariate_types=[str(cg) for cg in covariate_gens],
        covariate_configs=covariate_configs,
        covariate_weights=covariate_weights,
        raw_data=raw_data,
        individual_covariates=individual_covariates,
        data_with_covariate=data_with_covariate,
        n_train_timesteps=n_train_timesteps,
        results_without_covariate=results_no_cov,
        results_with_covariate=results_with_cov,
    )

    mase_improvement = (
        (results_no_cov.mase - results_with_cov.mase) / results_no_cov.mase * 100
    )
    sql_improvement = (
        (results_no_cov.sql - results_with_cov.sql) / results_no_cov.sql * 100
    )
    logger.info(
        f"Experiment {experiment_id + 1}: "
        f"Weights {np.array(covariate_weights).round(2)}, "
        f"MASE {results_no_cov.mase:.4f} → {results_with_cov.mase:.4f} ({mase_improvement:+.2f}%), "
        f"SQL {results_no_cov.sql:.4f} → {results_with_cov.sql:.4f} ({sql_improvement:+.2f}%)"
    )

    return experiment_result


def run_covariate_study(
    covariate_types: List[str],
    n_samples: int = 10,
    n_timesteps: int = 1000,
    train_ratio: float = 0.6,
    weight: float = 1.0,
    relation: str = "additive",
    use_random_delay: bool = False,
    seed: int = 42,
    n_jobs: int = -1,
    weight_sampling_concentration: float = 5.0,
) -> StudyResult:
    """Run the complete covariate study."""

    covariate_gens = [get_covariate_generator(ct) for ct in covariate_types]

    logger.info(f"Studying covariate types: {[str(cg) for cg in covariate_gens]}")
    logger.info("Covariate weights will be randomized for each experiment.")
    logger.info(f"Parallelization: Using {n_jobs} job(s)")

    n_train_timesteps = int(n_timesteps * train_ratio)

    # Create base features
    base_features = create_features(n_timesteps)

    logger.info(f"Running {n_samples} experiments...")

    # Create experiment jobs
    experiment_jobs = [
        delayed(run_single_experiment)(
            i,
            covariate_gens,
            base_features,
            n_timesteps,
            n_train_timesteps,
            weight,
            relation,
            seed,
            use_random_delay,
            n_jobs,
            weight_sampling_concentration,
        )
        for i in range(n_samples)
    ]

    # Execute experiments in parallel
    logger.info("Starting parallel execution of experiments...")
    results: List[SingleExperimentResult] = Parallel(n_jobs=n_jobs)(experiment_jobs)
    logger.info("All experiments completed.")

    return StudyResult(
        covariate_types=covariate_types,
        n_samples=n_samples,
        n_timesteps=n_timesteps,
        exp_results=results,
        parameters={
            "weight": weight,
            "relation": relation,
            "train_ratio": train_ratio,
        },
    )
