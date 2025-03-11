#!/usr/bin/env python3
import os
import argparse
import submitit
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

from tabpfn_time_series.experimental.evaluation.dataset_definition import ALL_DATASETS


load_dotenv()

EVALUATION_SCRIPT_PATH = Path(__file__).parent.parent / "evaluate_pipeline.py"


HUGE_DATASETS = [
    "bitbrains_fast_storage/5T",
    "bitbrains_fast_storage/H",
    "bitbrains_rnd/5T",
    "bitbrains_rnd/H",
    "LOOP_SEATTLE/5T",
    "LOOP_SEATTLE/H",
    "LOOP_SEATTLE/D",
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
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--terms",
        type=str,
        help="Comma-separated list of terms to evaluate",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run fast evaluation, skip big datasets"
    )
    parser.add_argument("--debug_slurm", action="store_true", help="Debug SLURM jobs")
    parser.add_argument(
        "--dry_run", action="store_true", help="Dry run, don't submit any jobs"
    )

    return parser.parse_args()


def get_datasets_to_evaluate(args):
    # If a single task is specified, check if it is valid
    if args.dataset != "all":
        if args.dataset not in ALL_DATASETS:
            raise ValueError(f"Dataset {args.dataset} not found in dataset definition")
        datasets = [args.dataset]
    else:
        datasets = ALL_DATASETS

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

    return datasets


def is_valid_frequency(freq):
    import pandas as pd

    try:
        pd.to_timedelta("1" + freq)
        return True
    except ValueError:
        return False


def main():
    args = parse_arguments()
    job_name = f"time_series_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    memory = 128
    num_cpus = 16
    num_gpus = args.ngpus

    datasets = get_datasets_to_evaluate(args)
    num_datasets = len(datasets)

    # Report the benchmark parameters
    print("\nRunning evaluation with the following parameters:")
    print(f" . CLUSTER_PARTITION: {args.cluster_partition}")
    print(f" . # GPUS: {num_gpus}")
    print(f" . # DATASETS: {num_datasets}")
    print(f" . DATASETS: {datasets}")

    if args.dry_run:
        print("Dry run, not submitting any jobs")
        return

    # Setup submitit executor
    executor = submitit.AutoExecutor(folder=f"slurm_logs/{job_name}")
    executor.update_parameters(
        name=job_name,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=num_cpus,
        mem_gb=memory,
        slurm_gres=f"gpu:{num_gpus}",
        slurm_partition=args.cluster_partition,
        slurm_array_parallelism=num_datasets,
        slurm_setup=[f"source {os.getenv('ENVIRONMENT_BASHRC_PATH')}"],
        timeout_min=1439,  # 23 hours and 59 minutes
        slurm_additional_parameters=f"exclude={os.getenv('EXCLUDE_CLUSTER_NODES')}",
    )

    jobs = []
    with executor.batch():
        for dataset in datasets:
            cmd = ["python", str(EVALUATION_SCRIPT_PATH)]
            script_args = [
                "--config_path",
                args.config_path,
                "--dataset",
                dataset,
                "--dataset_storage_path",
                os.getenv("DATASET_STORAGE_PATH"),
                "--output_dir",
                f"slurm/{job_name}",
                "--enable_wandb",
            ]

            if args.terms:
                script_args.append("--terms")
                script_args.append(args.terms)

            job = executor.submit(submitit.helpers.CommandFunction(cmd), *script_args)
            jobs.append(job)

    print(f"Submitted {len(jobs)} jobs")


if __name__ == "__main__":
    main()
