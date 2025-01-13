import csv
import logging
import argparse
from pathlib import Path
from typing import Tuple, List

from gluonts.model import evaluate_model
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

from gift_eval.data import Dataset
from gift_eval.dataset_definition import (
    MED_LONG_DATASETS,
    ALL_DATASETS,
    DATASET_PROPERTIES_MAP,
)
from gift_eval.tabpfn_ts_wrapper import TabPFNTSPredictor, TabPFNMode

# Instantiate the metrics
metrics = [
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


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def construct_evaluation_data(dataset_name: str) -> List[Tuple[Dataset, dict]]:
    sub_datasets = []

    # Construct evaluation data
    ds_key = dataset_name.split("/")[0]
    terms = ["short", "medium", "long"]
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
            if Dataset(name=dataset_name, term=term, to_univariate=False).target_dim
            == 1
            else True
        )
        dataset = Dataset(name=dataset_name, term=term, to_univariate=to_univariate)
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
                DATASET_PROPERTIES_MAP[dataset_metadata["full_name"]]["domain"],
                DATASET_PROPERTIES_MAP[dataset_metadata["full_name"]]["num_variates"],
            ]
        )

    print(
        f"Results for {dataset_metadata['full_name']} have been written to {csv_file_path}"
    )


def main(args):
    if args.dataset not in ALL_DATASETS:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    logger.info(f"Evaluating dataset {args.dataset}")

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "results.csv"

    # Construct evaluation data (i.e. sub-datasets) for this dataset
    # (some datasets contain different forecasting terms, e.g. short, medium, long)
    sub_datasets = construct_evaluation_data(args.dataset)

    # Evaluate model
    for sub_dataset, dataset_metadata in sub_datasets:
        tabpfn_predictor = TabPFNTSPredictor(
            ds_prediction_length=sub_dataset.prediction_length,
            ds_freq=sub_dataset.freq,
            tabpfn_mode=TabPFNMode.LOCAL,
        )

        res = evaluate_model(
            tabpfn_predictor,
            test_data=sub_dataset.test_data,
            metrics=metrics,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=dataset_metadata["season_length"],
        )

        # Write results to csv
        append_results_to_csv(
            res,
            output_csv_path,
            dataset_metadata,
            "tabpfn-ts-paper",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    main(args)
