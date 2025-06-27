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

from .data import GiftEvalDataset
from .dataset_definition import (
    MED_LONG_DATASETS,
    DATASET_PROPERTIES_MAP,
    DATASETS_WITH_COVARIATES,
    DATASET_W_COVARIATES_PROPERTIES_MAP,
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


def get_gift_eval_dataset(
    dataset_name: str,
    dataset_storage_path: Path | str,
    terms: List[str] = ["short", "medium", "long"],
) -> List[Tuple[GiftEvalDataset, dict]]:
    if isinstance(dataset_storage_path, str):
        dataset_storage_path = Path(dataset_storage_path)

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
            if GiftEvalDataset(
                name=dataset_name,
                term=term,
                to_univariate=False,
                storage_path=dataset_storage_path,
            ).target_dim
            == 1
            else True
        )
        dataset = GiftEvalDataset(
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
            "domain": DATASET_PROPERTIES_MAP[ds_key]["domain"],
            "num_variates": DATASET_PROPERTIES_MAP[ds_key]["num_variates"],
        }
        sub_datasets.append((dataset, dataset_metadata))

    return sub_datasets


def get_gift_eval_dataset_with_covariates(
    dataset_name: str,
    dataset_storage_path: Path,
    terms: List[str] = ["short", "medium", "long"],
) -> Tuple[GiftEvalDataset, dict]:
    if dataset_name not in DATASETS_WITH_COVARIATES:
        raise ValueError(
            f"Dataset {dataset_name} not found in DATASETS_WITH_COVARIATES"
        )

    ds_freq = DATASET_W_COVARIATES_PROPERTIES_MAP[dataset_name]["frequency"]

    term_datasets = []
    for term in terms:
        dataset = GiftEvalDataset(
            name=dataset_name,
            term=term,
            to_univariate=False,
            storage_path=dataset_storage_path,
        )

        if dataset.target_dim != 1:
            raise NotImplementedError(
                f"Dataset {dataset_name} has {dataset.target_dim} target variables,"
                "but only 1 is supported"
            )

        # Sanity check
        if (
            dataset.past_feat_dynamic_real_dim
            != DATASET_W_COVARIATES_PROPERTIES_MAP[dataset_name]["num_covariates"]
        ):
            raise ValueError(
                f"Dataset {dataset_name} has {dataset.past_feat_dynamic_real_dim} past_feat_dynamic_real_dim,"
                f"but {DATASET_W_COVARIATES_PROPERTIES_MAP[dataset_name]['num_covariates']} are expected"
            )
        season_length = get_seasonality(dataset.freq)

        dataset_metadata = {
            "full_name": f"{dataset_name}/{ds_freq}/{term}",
            "key": dataset_name,
            "freq": ds_freq,
            "term": term,
            "season_length": season_length,
            "domain": DATASET_W_COVARIATES_PROPERTIES_MAP[dataset_name]["domain"],
            "num_variates": DATASET_W_COVARIATES_PROPERTIES_MAP[dataset_name][
                "num_variates"
            ],
        }
        term_datasets.append((dataset, dataset_metadata))

    return term_datasets


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
        res = res.iloc[0]
        writer.writerow(
            [
                dataset_metadata["full_name"],
                model_name,
                res["MSE[mean]"],
                res["MSE[0.5]"],
                res["MAE[0.5]"],
                res["MASE[0.5]"],
                res["MAPE[0.5]"],
                res["sMAPE[0.5]"],
                res["MSIS"],
                res["RMSE[mean]"],
                res["NRMSE[mean]"],
                res["ND[0.5]"],
                res["mean_weighted_sum_quantile_loss"],
                dataset_metadata["domain"],
                dataset_metadata["num_variates"],
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
        "MSE_mean": res["MSE[mean]"].iloc[0],
        "MSE_0.5": res["MSE[0.5]"].iloc[0],
        "MAE_0.5": res["MAE[0.5]"].iloc[0],
        "MASE_0.5": res["MASE[0.5]"].iloc[0],
        "MAPE_0.5": res["MAPE[0.5]"].iloc[0],
        "sMAPE_0.5": res["sMAPE[0.5]"].iloc[0],
        "MSIS": res["MSIS"].iloc[0],
        "RMSE_mean": res["RMSE[mean]"].iloc[0],
        "NRMSE_mean": res["NRMSE[mean]"].iloc[0],
        "ND_0.5": res["ND[0.5]"].iloc[0],
        "wSQL_mean": res["mean_weighted_sum_quantile_loss"].iloc[0],
        "domain": dataset_metadata["domain"],
        "num_variates": dataset_metadata["num_variates"],
        "term": dataset_metadata["term"],
    }
    wandb.log(wandb_log_data)
