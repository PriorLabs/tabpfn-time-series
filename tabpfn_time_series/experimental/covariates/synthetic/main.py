#!/usr/bin/env python3
"""
Covariate Study Script

This script studies the effect of different types of covariates on time series prediction.
It samples multiple instances of a specified covariate type and evaluates prediction
performance with and without covariates using TabPFN.

Usage:
    python covariate_study.py --covariate_type sinusoidal --n_samples 10 --output results.pdf
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import pickle
from typing import List


from tabpfn_time_series.experimental.covariates.synthetic.covariate_generators import (
    get_available_covariate_types,
)
from tabpfn_time_series.experimental.covariates.synthetic.visualization import (
    create_summary_report,
    create_prediction_visualizations,
)
from tabpfn_time_series.experimental.covariates.synthetic.eval import (
    run_covariate_study,
    RANDOM_DELAY_MIN_DELAY,
    RANDOM_DELAY_MAX_DELAY,
)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_covariate_types(value: str) -> List[str]:
    """Parse comma-separated covariate types and validate them."""
    types = [v.strip() for v in value.split(",")]
    available_types = get_available_covariate_types()
    for t in types:
        if t not in available_types:
            raise argparse.ArgumentTypeError(
                f"Invalid choice: '{t}'. Choose from {available_types}"
            )
    return types


def get_output_dir(output_root_dir: Path, args: argparse.Namespace) -> Path:
    """Get the output directory for the study."""
    output_dir_name = "_".join(args.covariate_type) + f"_{args.relation}"
    if args.use_random_delay:
        output_dir_name += "_random_delay"
    return (
        output_root_dir
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{output_dir_name}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Study the effect of covariates on time series prediction"
    )
    parser.add_argument(
        "--covariate_type",
        type=parse_covariate_types,
        required=True,
        help='Comma-separated list of covariate types to study (e.g., "ramps,steps").',
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of experiments to run (default: 10)",
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=1000,
        help="Number of timesteps in each time series (default: 1000)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.6,
        help="Ratio of data used for training (default: 0.6)",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Weight for covariate effect (default: 1.0)",
    )
    parser.add_argument(
        "--weight_sampling_concentration",
        type=float,
        default=5.0,
        help="Concentration parameter for weight sampling (default: 5.0)",
    )
    parser.add_argument(
        "--relation",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="Relationship between time series and covariate (default: additive)",
    )
    parser.add_argument(
        "--use_random_delay",
        action="store_true",
        help="Use random delay for covariate impact (default: False). "
        f"If True, the delay is sampled uniformly from {RANDOM_DELAY_MIN_DELAY} to {RANDOM_DELAY_MAX_DELAY} timesteps.",
    )
    parser.add_argument(
        "--output_root_dir",
        type=str,
        default=Path(__file__).parent / "covariate_study_results",
        help="Output root directory (default: ./covariate_study_results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose (debug) logging"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs for predictions (default: -1, -1 for all cores)",
    )

    args = parser.parse_args()

    # Update logging level if verbose mode is requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    output_dir = get_output_dir(args.output_root_dir, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Covariate Study, output directory: {output_dir}")

    # Log all args programmatically
    for arg, value in vars(args).items():
        logger.info(f"{arg.replace('_', ' ').title()}: {value}")

    # Run the study
    results = run_covariate_study(
        covariate_types=args.covariate_type,
        n_samples=args.n_samples,
        n_timesteps=args.n_timesteps,
        train_ratio=args.train_ratio,
        weight=args.weight,
        relation=args.relation,
        use_random_delay=args.use_random_delay,
        seed=args.seed,
        n_jobs=args.n_jobs,
        weight_sampling_concentration=args.weight_sampling_concentration,
    )

    # Save results to output directory
    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)

    # Create separate reports
    create_summary_report(results, output_dir / "summary.pdf")
    create_prediction_visualizations(results, output_dir / "predictions.pdf")

    logger.info("=" * 60)
    logger.info("STUDY COMPLETED SUCCESSFULLY!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
