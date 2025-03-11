from typing import List, Tuple
from pathlib import Path
import csv
import wandb

from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)

from .data import Dataset
from .dataset_definition import (
    MED_LONG_DATASETS,
    DATASET_PROPERTIES_MAP,
)


# Instantiate the metrics
METRICS = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]


pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


def construct_evaluation_data(
    dataset_name: str,
    dataset_storage_path: Path,
    terms: List[str] = ["short", "medium", "long"],
) -> List[Tuple[Dataset, dict]]:
    sub_datasets = []

    # Construct evaluation data
    ds_key = dataset_name.split("/")[0]
    for term in terms:
        if (
            term == "medium" or term == "long"
        ) and dataset_name not in MED_LONG_DATASETS:
            continue

        if "/" in dataset_name:
            ds_key = dataset_name.split("/")[0]
            ds_freq = dataset_name.split("/")[1]
            ds_key = ds_key.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
        else:
            ds_key = dataset_name.lower()
            ds_key = pretty_names.get(ds_key, ds_key)
            ds_freq = DATASET_PROPERTIES_MAP[ds_key]["frequency"]

        # Initialize the dataset
        to_univariate = (
            False
            if Dataset(
                name=dataset_name,
                term=term,
                to_univariate=False,
                storage_path=dataset_storage_path,
            ).target_dim
            == 1
            else True
        )
        dataset = Dataset(
            name=dataset_name,
            term=term,
            to_univariate=to_univariate,
            storage_path=dataset_storage_path,
        )
        season_length = get_seasonality(dataset.freq)

        dataset_metadata = {
            "full_name": f"{ds_key}/{ds_freq}/{term}",
            "key": ds_key,
            "freq": ds_freq,
            "term": term,
            "season_length": season_length,
        }
        sub_datasets.append((dataset, dataset_metadata))

    return sub_datasets


def create_csv_file(csv_file_path):
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write the header
        writer.writerow(
            [
                "dataset",
                "model",
                "eval_metrics/MSE[mean]",
                "eval_metrics/MSE[0.5]",
                "eval_metrics/MAE[0.5]",
                "eval_metrics/MASE[0.5]",
                "eval_metrics/MAPE[0.5]",
                "eval_metrics/sMAPE[0.5]",
                "eval_metrics/MSIS",
                "eval_metrics/RMSE[mean]",
                "eval_metrics/NRMSE[mean]",
                "eval_metrics/ND[0.5]",
                "eval_metrics/mean_weighted_sum_quantile_loss",
                "domain",
                "num_variates",
            ]
        )


def append_results_to_csv(
    res,
    csv_file_path,
    dataset_metadata,
    model_name,
):
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                dataset_metadata["full_name"],
                model_name,
                res["MSE[mean]"][0],
                res["MSE[0.5]"][0],
                res["MAE[0.5]"][0],
                res["MASE[0.5]"][0],
                res["MAPE[0.5]"][0],
                res["sMAPE[0.5]"][0],
                res["MSIS"][0],
                res["RMSE[mean]"][0],
                res["NRMSE[mean]"][0],
                res["ND[0.5]"][0],
                res["mean_weighted_sum_quantile_loss"][0],
                DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["domain"],
                DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["num_variates"],
            ]
        )

    print(f"Results for {dataset_metadata['key']} have been written to {csv_file_path}")


def log_results_to_wandb(
    model_name,
    res,
    dataset_metadata,
):
    wandb_log_data = {
        "model": model_name,
        "dataset": dataset_metadata["full_name"],
        "MSE_mean": res["MSE[mean]"][0],
        "MSE_0.5": res["MSE[0.5]"][0],
        "MAE_0.5": res["MAE[0.5]"][0],
        "MASE_0.5": res["MASE[0.5]"][0],
        "MAPE_0.5": res["MAPE[0.5]"][0],
        "sMAPE_0.5": res["sMAPE[0.5]"][0],
        "MSIS": res["MSIS"][0],
        "RMSE_mean": res["RMSE[mean]"][0],
        "NRMSE_mean": res["NRMSE[mean]"][0],
        "ND_0.5": res["ND[0.5]"][0],
        "wSQL_mean": res["mean_weighted_sum_quantile_loss"][0],
        "domain": DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["domain"],
        "num_variates": DATASET_PROPERTIES_MAP[dataset_metadata["key"]]["num_variates"],
        "term": dataset_metadata["term"],
    }
    wandb.log(wandb_log_data)
