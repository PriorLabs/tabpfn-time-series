import argparse
from pathlib import Path
import logging

import wandb
from gluonts.model import evaluate_model

from tabpfn_time_series.experimental.pipeline import PipelineConfig

from tabpfn_time_series.experimental.evaluation.dataset_definition import (
    ALL_DATASETS,
    DATASETS_WITH_COVARIATES,
)
from tabpfn_time_series.experimental.evaluation.evaluate_utils import (
    construct_evaluation_data,
    construct_evaluation_data_with_covariates,
    create_csv_file,
    append_results_to_csv,
    log_results_to_wandb,
    METRICS,
)


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

logger = logging.getLogger(__name__)


def main(args):
    # Assert dataset exists
    DATASET_COLLECTION = (
        DATASETS_WITH_COVARIATES if args.evaluate_covariates else ALL_DATASETS
    )
    if args.dataset not in DATASET_COLLECTION:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    logger.info(f"Evaluating dataset {args.dataset}")

    # Check if the dataset storage path exists
    if not Path(args.dataset_storage_path).exists():
        raise ValueError(
            f"Dataset storage path {args.dataset_storage_path} does not exist"
        )

    # Load pipeline config
    pipeline_config = PipelineConfig.from_json(args.config_path)
    logger.info(f"Pipeline config: {pipeline_config}")
    model_name = pipeline_config.predictor_name

    # Construct evaluation data (i.e. sub-datasets) for this dataset
    # (some datasets contain different forecasting terms, e.g. short, medium, long)
    if args.evaluate_covariates:
        sub_datasets = construct_evaluation_data_with_covariates(
            args.dataset, args.dataset_storage_path, args.terms
        )
        if not isinstance(sub_datasets, list):
            sub_datasets = [sub_datasets]
    else:
        sub_datasets = construct_evaluation_data(
            args.dataset, args.dataset_storage_path, args.terms
        )

    # Create output directory
    output_dir = args.output_dir / pipeline_config.predictor_name / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "results.csv"

    # Create CSV file
    create_csv_file(output_csv_path)

    # Evaluate model
    for i, (sub_dataset, dataset_metadata) in enumerate(sub_datasets):
        logger.info(
            f"Evaluating {i + 1}/{len(sub_datasets)} dataset {sub_dataset.name}"
        )
        logger.info(f"Dataset size: {len(sub_dataset.test_data)}")
        logger.info(f"Dataset freq: {sub_dataset.freq}")
        logger.info(f"Dataset term: {dataset_metadata['term']}")
        logger.info(f"Dataset prediction length: {sub_dataset.prediction_length}")
        logger.info(f"Dataset target dim: {sub_dataset.target_dim}")

        # Initialize wandb
        if args.enable_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"{model_name}/{dataset_metadata['full_name']}",
                config=vars(args),
                tags=[model_name] + args.wandb_tags.split(",")
                if args.wandb_tags
                else [],
            )

            wandb.summary.update(
                {
                    "model": model_name,
                    "dataset_full_name": dataset_metadata["full_name"],
                }
            )

        # Initialize pipeline
        pipeline_class = PipelineConfig.get_pipeline_class(
            pipeline_config.pipeline_name
        )
        logger.info(f"Pipeline class: {pipeline_class}")
        pipeline = pipeline_class(
            config=pipeline_config,
            ds_prediction_length=sub_dataset.prediction_length,
            ds_freq=sub_dataset.freq,
            debug=args.debug,
            **pipeline_config.additional_pipeline_config,
        )

        res = evaluate_model(
            pipeline,
            test_data=sub_dataset.test_data,
            metrics=METRICS,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=dataset_metadata["season_length"],
        )

        # Write results to csv
        append_results_to_csv(
            res=res,
            csv_file_path=output_csv_path,
            dataset_metadata=dataset_metadata,
            model_name=model_name,
        )

        # Finish wandb run
        if args.enable_wandb:
            if args.enable_wandb:
                log_results_to_wandb(
                    model_name=model_name,
                    res=res,
                    dataset_metadata=dataset_metadata,
                )

            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--output_dir", type=str, default=str(Path(__file__).parent / "results")
    )
    parser.add_argument(
        "--terms",
        type=str,
        default="short,medium,long",
        help="Comma-separated list of terms to evaluate",
    )
    parser.add_argument(
        "--dataset_storage_path",
        type=str,
        default=str(Path(__file__).parent / "data"),
    )
    parser.add_argument(
        "--evaluate_covariates",
        action="store_true",
        help="Whether to evaluate covariates",
    )
    parser.add_argument("--debug", action="store_true")

    # Wandb settings
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="tabpfn-ts-experiments")
    parser.add_argument(
        "--wandb_tags", type=str, default=""
    )  # model_name will be added later anyway

    args = parser.parse_args()
    args.dataset_storage_path = Path(args.dataset_storage_path)
    args.output_dir = Path(args.output_dir)
    args.terms = args.terms.split(",")

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug("Debug mode enabled")
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Command Line Arguments: {vars(args)}")

    main(args)
