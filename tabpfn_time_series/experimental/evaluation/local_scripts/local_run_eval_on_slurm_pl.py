#!/usr/bin/env python
"""
Script to submit evaluation jobs on SLURM using submitit.
"""

import os
import argparse
import submitit
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from tabpfn_time_series.experimental.evaluation.dataset_definition import (
    ALL_DATASETS,
    DATASETS_WITH_COVARIATES,
    SHORT_DATASETS,
    MED_LONG_DATASETS,
)

load_dotenv()

EVALUATION_SCRIPT_PATH = Path(__file__).parent.parent / "evaluate_pipeline.py"

HUGE_DATASETS = [
    "bitbrains_rnd/5T",
    "bitbrains_fast_storage/5T",
    "electricity/15T",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "temperature_rain_with_missing",
    # "bitbrains_fast_storage/H",
    # "bitbrains_rnd/H",
    # "LOOP_SEATTLE/D",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate and run sbatch script for time series benchmark"
    )
    parser.add_argument(
        "--cluster_partition",
        default=os.getenv("DEFAULT_CLUSTER_PARTITION"),
        help="Cluster partition to use",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset to run, either a single dataset or 'all'",
    )
    parser.add_argument(
        "--evaluate_covariates",
        action="store_true",
        help="Evaluate covariates",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file to use",
    )
    parser.add_argument(
        "--freq",
        default=None,
        help="Evaluate datasets with this frequency only",
    )
    parser.add_argument(
        "--exclude_freq",
        default=None,
        help="Exclude datasets with this frequency only",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        required=True,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--terms",
        type=str,
        help="Comma-separated list of terms to evaluate",
        default="short,medium,long",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast evaluation, skip big datasets",
    )
    parser.add_argument(
        "--debug_slurm",
        action="store_true",
        help="Debug SLURM jobs",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run, don't submit any jobs",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        help="Comma-separated list of tags to distinguish the experiment/study",
        required=True,
    )
    # parser.add_argument(
    #     "--mem",
    #     type=int,
    #     default=128,
    #     help="Memory allocation in GB",
    # )
    # parser.add_argument(
    #     "--cpus",
    #     type=int,
    #     default=16,
    #     help="Number of CPUs to request",
    # )

    return parser.parse_args()


def get_datasets_to_evaluate(args, terms):
    # If a single task is specified, check if it is valid
    DATASET_COLLECTION = (
        ALL_DATASETS if not args.evaluate_covariates else DATASETS_WITH_COVARIATES
    )

    if args.dataset != "all":
        if args.dataset not in DATASET_COLLECTION:
            raise ValueError(f"Dataset {args.dataset} not found in dataset definition")
        datasets = [args.dataset]
    else:
        datasets = DATASET_COLLECTION

    if args.debug_slurm:
        print("Debugging SLURM jobs")
        datasets = datasets[:1]

    if args.fast:
        print("Running fast evaluation, skipping huge datasets")
        datasets = [ds for ds in datasets if ds not in HUGE_DATASETS]

    if args.freq:
        assert is_valid_frequency(
            args.freq
        ), f"Frequency {args.freq} not supported. Must be a valid pandas frequency string."
        assert (
            args.dataset == "all"
        ), "Cannot specify dataset when frequency is specified"
        datasets = [ds for ds in datasets if ds.endswith(args.freq)]

    if args.exclude_freq:
        assert is_valid_frequency(
            args.exclude_freq
        ), f"Frequency {args.exclude_freq} not supported. Must be a valid pandas frequency string."
        datasets = [ds for ds in datasets if not ds.endswith(args.exclude_freq)]

    datasets_and_terms = []
    for d in datasets:
        for t in terms:
            if t == ["short"] and d not in SHORT_DATASETS:
                continue

            if t in ["medium", "long"] and d not in MED_LONG_DATASETS:
                continue

            datasets_and_terms.append((d, t))

    return datasets, datasets_and_terms


def get_dataset_storage_path(args):
    if args.evaluate_covariates:
        return os.getenv("COVARIATE_DATASET_STORAGE_PATH")
    else:
        return os.getenv("DATASET_STORAGE_PATH")


def is_valid_frequency(freq):
    import pandas as pd

    try:
        pd.to_timedelta("1" + freq)
        return True
    except ValueError:
        return False


class EvaluationJob:
    def __init__(
        self,
        dataset,
        dataset_storage_path,
        term,
        args,
    ):
        self.dataset = dataset
        self.dataset_storage_path = dataset_storage_path
        self.term = term
        self.args = args

    def __call__(self):
        import sys
        import os
        import subprocess

        # Print some debug information
        print(f"Job running on node: {os.uname().nodename}")
        print(f"CUDA devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

        # Construct command to run evaluate_pipeline.py
        cmd = [sys.executable, "-u", str(EVALUATION_SCRIPT_PATH)]

        # Add all required arguments
        cmd.extend(
            [
                "--config_path",
                self.args.config_path,
                "--dataset",
                self.dataset,
                "--dataset_storage_path",
                self.dataset_storage_path,
                "--output_dir",
                f"slurm/{self.args.job_name}",
                "--enable_wandb",
                "--wandb_tags",
                self.args.experiment_tag,
                "--terms",
                self.term,
            ]
        )

        if self.args.evaluate_covariates:
            cmd.append("--evaluate_covariates")

        # Execute the command
        print(f"Running command: {' '.join(cmd)}")
        process = subprocess.run(cmd, check=True)
        return process.returncode


def main():
    args = parse_arguments()
    args.job_name = f"time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    terms = args.terms.split(",")
    datasets, datasets_and_terms = get_datasets_to_evaluate(args, terms)
    dataset_storage_path = get_dataset_storage_path(args)
    num_datasets = len(datasets)
    total_jobs = len(datasets_and_terms)

    # Report the benchmark parameters
    print("\nRunning evaluation with the following parameters:")
    print(f" . CLUSTER_PARTITION: {args.cluster_partition}")
    print(f" . # GPUS: {args.ngpus}")
    print(f" . # DATASETS: {num_datasets}")
    print(f" . DATASETS: {datasets}")
    print(f" . TERMS: {terms}")
    print(f" . TOTAL JOBS: {total_jobs}")

    if args.dry_run:
        print("Dry run, not submitting any jobs")
        return

    # Create logs directory if it doesn't exist
    log_dir = Path("slurm_logs") / args.job_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize the executor
    executor = submitit.AutoExecutor(folder=str(log_dir))

    # Set slurm parameters
    executor.update_parameters(
        name=args.job_name,
        nodes=1,
        tasks_per_node=1,
        # cpus_per_task=args.cpus,
        # slurm_mem=f"{args.mem}GB",
        slurm_gres=f"gpu:{args.ngpus}",
        slurm_partition=args.cluster_partition,
        slurm_array_parallelism=total_jobs,
        timeout_min=48 * 60,  # 48 hours in minutes
        slurm_additional_parameters={"comment": "TabPFN Time Series Evaluation"},
    )

    jobs = []
    # Submit all jobs at once using batch context
    with executor.batch():
        for dataset, term in datasets_and_terms:
            # Create the job
            job = EvaluationJob(
                dataset,
                dataset_storage_path,
                term,
                args,
            )

            # Submit the job
            submitted_job = executor.submit(job)
            jobs.append(submitted_job)

    print(f"Submitted {len(jobs)} jobs")
    print(f"Logs will be saved to: {log_dir}")


if __name__ == "__main__":
    main()
