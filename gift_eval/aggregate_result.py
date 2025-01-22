import pandas as pd
import glob
import argparse
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent))

from gift_eval.dataset_definition import (
    MED_LONG_DATASETS,
    ALL_DATASETS,
    DATASET_PROPERTIES_MAP,
)


def get_all_datasets_full_name():
    pretty_names = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }
    terms = ["short", "medium", "long"]
    datasets_full_names = []
    for name in ALL_DATASETS:
        for term in terms:
            if term in ["medium", "long"] and name not in MED_LONG_DATASETS:
                continue

            if "/" in name:
                ds_key = name.split("/")[0]
                ds_freq = name.split("/")[1]
                ds_key = ds_key.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
            else:
                ds_key = name.lower()
                ds_key = pretty_names.get(ds_key, ds_key)
                ds_freq = DATASET_PROPERTIES_MAP[ds_key]["frequency"]
            datasets_full_names.append(f"{ds_key}/{ds_freq}/{term}")

    return datasets_full_names


def main(args):
    # Find all CSV result files
    result_files = glob.glob(
        f"{args.result_root_dir}/**/{args.model_name}/**/results.csv", recursive=True
    )

    # print("Result files:")
    # for i, file in enumerate(result_files):
    #     print(f"{i}: {file}")

    # Initialize empty list to store dataframes
    dfs = []

    # Read and combine all CSV files
    for file in result_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {file}")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")

    if dfs:
        # Combine all dataframes and sort by dataset
        dfs = pd.concat(dfs, ignore_index=True).sort_values("dataset")

        # If there are duplicate datasets, raise an error
        if len(dfs) != len(set(dfs.dataset)):
            duplicate_datasets = dfs.dataset[dfs.dataset.duplicated()]
            raise ValueError(f"Duplicate datasets found: {duplicate_datasets}")
    else:
        print("No valid CSV files found to combine")
        raise ValueError("No valid CSV files found to combine")

    all_datasets_full_name = get_all_datasets_full_name()
    all_experiments = dfs.dataset.to_list()
    completed_experiments = []
    for i, dataset_full_name in enumerate(all_datasets_full_name):
        if dataset_full_name in all_experiments:
            completed_experiments.append(dataset_full_name)

    missing_or_failed_experiments = [
        x for x in all_datasets_full_name if x not in completed_experiments
    ]

    print("Completed experiments:")
    for i, exp in enumerate(completed_experiments):
        print(f"  {i}: {exp}")
    print("Missing or failed experiments:")
    for i, exp in enumerate(missing_or_failed_experiments):
        print(f"  {i}: {exp}")

    # Save combined results
    output_file = Path(args.result_root_dir) / f"{args.model_name}_results.csv"
    dfs.to_csv(output_file, index=False)
    print(f"Combined results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="tabpfn-ts-paper")
    args = parser.parse_args()

    args.result_root_dir = Path(args.result_root_dir)

    main(args)
