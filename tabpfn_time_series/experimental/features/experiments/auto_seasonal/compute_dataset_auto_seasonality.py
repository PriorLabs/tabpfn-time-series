import logging
import json
from pathlib import Path
from argparse import ArgumentParser
from typing import List, Tuple
from joblib import Parallel, delayed
import pickle

from tqdm import tqdm
import pandas as pd
from gluonts.itertools import batcher

from tabpfn_time_series.experimental.evaluation.data import Dataset
from tabpfn_time_series.experimental.evaluation.dataset_definition import ALL_DATASETS
from tabpfn_time_series.experimental.evaluation.evaluate_utils import (
    construct_evaluation_data,
)
from tabpfn_time_series.experimental.features.auto_features import AutoSeasonalFeature
from tabpfn_time_series.experimental.pipeline import TimeSeriesEvalPipeline
from tabpfn_time_series.experimental.utils.general import find_repo_root

handle_nan_values = TimeSeriesEvalPipeline.handle_nan_values
convert_to_timeseries_dataframe = TimeSeriesEvalPipeline.convert_to_timeseries_dataframe


logger = logging.getLogger(__name__)


REPO_ROOT = find_repo_root()
MAX_CONTEXT_LENGTH = 4096


def find_seasonality_per_time_series(
    df: pd.DataFrame,
    config: dict,
) -> List[Tuple[float, float]]:
    """Find seasonality per time series."""
    return AutoSeasonalFeature.find_seasonal_periods(df.target, **config)


def find_seasonality_per_dataset(
    dataset: Dataset,
    config: dict,
) -> pd.DataFrame:
    """Find seasonality per dataset."""

    eval_data = dataset.test_data
    for batch in batcher(eval_data, batch_size=1024):
        train_data = [x[0] for x in batch]
        tsdf = convert_to_timeseries_dataframe(train_data)

        # print("DEBUG, before slicing", tsdf.num_timesteps_per_item())
        # tsdf = tsdf.slice_by_timestep(-MAX_CONTEXT_LENGTH, None)
        # print("DEBUG, after slicing", tsdf.num_timesteps_per_item())
        tsdf = handle_nan_values(tsdf)

        detected_seasonalities = Parallel(
            n_jobs=16,
            backend="loky",
        )(
            delayed(find_seasonality_per_time_series)(tsdf.loc[item_id], config)
            for item_id in tqdm(tsdf.item_ids, desc="Finding seasonality")
        )

    return detected_seasonalities


def main(args):
    # Assert dataset exists
    if args.dataset not in ALL_DATASETS:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    logger.info(f"Evaluating dataset {args.dataset}")

    # Check if the dataset storage path exists
    if not Path(args.dataset_storage_path).exists():
        raise ValueError(
            f"Dataset storage path {args.dataset_storage_path} does not exist"
        )

    # Load auto seasonality detection config
    with open(args.config_path, "r") as f:
        auto_season_config = json.load(f)

    # Load dataset
    dataset, dataset_metadata = construct_evaluation_data(
        dataset_name=args.dataset,
        dataset_storage_path=args.dataset_storage_path,
        terms=args.terms,
    )[0]

    detected_seasonalities = find_seasonality_per_dataset(
        dataset,
        auto_season_config,
    )

    # Create output directory
    config_name = Path(args.config_path).stem
    output_dir = args.output_dir / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detected seasonalities into pickle file
    output_path = output_dir / f"{args.dataset.replace('/', '_')}.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(detected_seasonalities, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--terms",
        type=str,
        default="short,medium,long",
        help="Comma-separated list of terms to evaluate",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(Path(__file__).parent / "results")
    )
    parser.add_argument(
        "--dataset_storage_path",
        type=str,
        default=str(REPO_ROOT / "gift_eval" / "data"),
    )

    args = parser.parse_args()

    args.dataset_storage_path = Path(args.dataset_storage_path)
    args.config_path = Path(args.config_path)
    args.output_dir = Path(args.output_dir)
    args.terms = args.terms.split(",")

    main(args)
