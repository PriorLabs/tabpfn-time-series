from typing import Optional
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from tabpfn.regressor import TabPFNRegressor
from tabpfn_time_series.experimental.evaluation.data import GiftEvalDataset
from tabpfn_time_series.experimental.evaluation.evaluate_utils import (
    get_gift_eval_dataset,
)
from tabpfn_time_series.features import (
    RunningIndexFeature,
    CalendarFeature,
    # AutoSeasonalFeature,
)

from .preprocessed_dataset import PreprocessedTimeSeriesDataset
from .raw_time_series_dataset import RawTimeSeriesDataset, DatasetMode


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for preparing a single time series dataset for fine-tuning.

    This module handles the setup of training, validation, and test datasets
    using the RawTimeSeriesDataset. It takes a single dataset configuration,
    instantiates it using gift_eval.data.Dataset, and wraps it for use
    in a fine-tuning pipeline.
    """

    # Fixed to evaluate on short term
    #   (the training data is the same for all terms and is already included anyway)
    _EVALUATION_TERM: str = "short"

    def __init__(
        self,
        dataset_name: str,
        dataset_storage_path: Path,
        model: TabPFNRegressor,
        batch_size: int = 1,
        num_workers: int = 4,
        past_length: int = 10_000,
        future_length: int = 1_024,
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            dataset_name: The name of the dataset to load.
            dataset_storage_path: The path to the dataset storage.
            model: The TabPFNRegressor model instance for preprocessing.
            batch_size: The batch size for the dataloaders. Must be 1.
            num_workers: The number of workers for the dataloaders.
            past_length: The context length for the time series samples.
            future_length: The prediction length for the time series samples.
            torch_dtype: The torch.dtype to use for the dataset.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_storage_path = dataset_storage_path
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.past_length = past_length
        self.future_length = future_length
        self.save_hyperparameters(ignore=["model"])

        self.feature_generators = [
            RunningIndexFeature(),
            CalendarFeature(),
            # AutoSeasonalFeature(),    # TODO: current implementation might cause leakage
        ]

        # These will be populated in setup()
        self.train_dataset: Optional[PreprocessedTimeSeriesDataset] = None
        self.val_dataset: Optional[PreprocessedTimeSeriesDataset] = None
        self.test_dataset: Optional[PreprocessedTimeSeriesDataset] = None
        self.torch_dtype = torch_dtype
        self.collate_fn = self.collate_fn

    def setup(self, stage: Optional[str] = None):
        """
        Prepares the datasets for the given stage.

        This method instantiates the gift_eval dataset and then wraps it
        with RawTimeSeriesDataset, which handles the creation of training,
        validation, and test instances on the fly using GluonTS samplers.
        """

        ge_dataset: GiftEvalDataset
        ge_dataset, _ = get_gift_eval_dataset(
            self.dataset_name,
            self.dataset_storage_path,
            [self._EVALUATION_TERM],
        )[0]

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            raw_train_dataset = RawTimeSeriesDataset(
                datasets=[ge_dataset.training_dataset],
                probabilities=[1.0],
                past_length=self.past_length,
                future_length=self.future_length,
                mode=DatasetMode.TRAINING,
                min_future=ge_dataset.prediction_length,
            )
            self.train_dataset = PreprocessedTimeSeriesDataset(
                raw_dataset=raw_train_dataset,
                model=self.model,
                feature_generators=self.feature_generators,
                torch_dtype=self.torch_dtype,
            )

        if stage in ("fit", "validate", None):
            raw_val_dataset = RawTimeSeriesDataset(
                datasets=[ge_dataset.validation_dataset],
                probabilities=[1.0],
                past_length=self.past_length,
                future_length=ge_dataset.prediction_length,
                mode=DatasetMode.VALIDATION,
                min_future=ge_dataset.prediction_length,
            )
            self.val_dataset = PreprocessedTimeSeriesDataset(
                raw_dataset=raw_val_dataset,
                model=self.model,
                feature_generators=self.feature_generators,
                torch_dtype=self.torch_dtype,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            raise NotImplementedError("Test dataset is not supported yet")

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=1,  # Single worker avoids duplicates
            persistent_workers=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        raise NotImplementedError("Test dataset is not supported yet")

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        A collate function for batching pre-processed time series data from a
        PreprocessedTimeSeriesDataset.

        This function is designed to work with a batch size of 1. It takes a
        list containing a single sample dictionary and processes it to be
        compatible with the model's `fit_from_preprocessed` method, which
        expects batched tensors.

        It mirrors the logic of `meta_dataset_collator`, but operates on and
        returns a dictionary. It processes each value from the sample:
        - `torch.Tensor`: Adds a leading batch dimension of size 1.
        - `list`: Assumes it's for ensemble members. It processes each item
          in the list, adding a batch dimension to tensors and wrapping non-
          tensors in a list.
        - Other types: Wraps the item in a list for consistency

        Args:
            batch: A list containing a single sample dictionary.

        Returns:
            A dictionary where each key maps to a processed, "batched" value.
        """
        if len(batch) != 1:
            raise ValueError(
                f"This collate_fn only supports a batch size of 1, but got {len(batch)}."
            )

        sample = batch[0]  # Get the dictionary from the single-item batch list
        processed_sample = {}

        for key, item in sample.items():
            if isinstance(item, list):
                # This is for per-ensemble-member data
                estim_list = []
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        # Add a batch dimension
                        estim_list.append(sub_item.unsqueeze(0))
                    else:
                        # For non-tensor items (like configs), wrap in a list for consistency
                        estim_list.append([sub_item])
                processed_sample[key] = estim_list
            elif isinstance(item, torch.Tensor):
                # For tensors that are not per-ensemble, just add a batch dimension
                processed_sample[key] = item.unsqueeze(0)
            else:
                # For other data types, wrap in a list for consistency
                processed_sample[key] = [item]

        return processed_sample
