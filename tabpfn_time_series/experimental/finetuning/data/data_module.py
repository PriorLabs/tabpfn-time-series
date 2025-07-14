from typing import Optional
from pathlib import Path
import logging

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
from .synthetic import SyntheticDataset, LinearTrendGenerator

logger = logging.getLogger(__name__)


class BaseTimeSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        model: TabPFNRegressor,
        batch_size: int = 1,
        num_workers: int = 4,
        past_length: int = 10_000,
        future_length: int = 1_024,
        torch_dtype: torch.dtype = torch.float32,
        enable_sampling: bool = True,
    ):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.past_length = past_length
        self.future_length = future_length
        self.torch_dtype = torch_dtype
        self.enable_sampling = enable_sampling
        # self.save_hyperparameters(ignore=["model"]) # This will be called by child classes

        self.feature_generators = [
            RunningIndexFeature(),
            CalendarFeature(),
            # AutoSeasonalFeature(),    # TODO: current implementation might cause leakage
        ]

        # These will be populated in setup()
        self.train_dataset: Optional[PreprocessedTimeSeriesDataset] = None
        self.val_dataset: Optional[PreprocessedTimeSeriesDataset] = None
        self.test_dataset: Optional[PreprocessedTimeSeriesDataset] = None

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 1,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 1,
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


class TimeSeriesDataModule(BaseTimeSeriesDataModule):
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
        enable_sampling: bool = True,
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
            enable_sampling: Whether to enable random sampling during training. When
                False, training mode will use validation-like sampling (single
                window from the end of each series) instead of random sampling.
                (default: True)
        """
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            past_length=past_length,
            future_length=future_length,
            torch_dtype=torch_dtype,
            enable_sampling=enable_sampling,
        )
        self.dataset_name = dataset_name
        self.dataset_storage_path = dataset_storage_path
        self.save_hyperparameters(ignore=["model"])

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
                enable_sampling=self.enable_sampling,
            ).shuffle()
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


class SyntheticDataModule(BaseTimeSeriesDataModule):
    """
    A DataModule for generating and using synthetic time series data for fine-tuning.

    This module is designed for debugging and testing purposes, allowing for full
    control over the data generation process.
    """

    def __init__(
        self,
        model: TabPFNRegressor,
        num_train_series: int,
        num_val_series: int,
        series_length: int,
        slope: float = 0.1,
        intercept: float = 0.0,
        noise_std: float = 0.1,
        use_train_as_val: bool = False,
        batch_size: int = 1,
        num_workers: int = 4,
        past_length: int = 10_000,
        future_length: int = 1_024,
        torch_dtype: torch.dtype = torch.float32,
        enable_sampling: bool = True,
    ):
        """
        Args:
            model: The TabPFNRegressor model instance for preprocessing.
            num_train_series: Number of series for the training set.
            num_val_series: Number of series for the validation set.
            series_length: The total length of each generated time series.
            slope: The slope for the linear trend generator.
            intercept: The intercept for the linear trend generator.
            noise_std: The noise standard deviation for the linear trend generator.
            use_train_as_val: If True, use the training set for validation.
            batch_size: The batch size for the dataloaders. Must be 1.
            num_workers: The number of workers for the dataloaders.
            past_length: The context length for the time series samples.
            future_length: The prediction length for the time series samples.
            torch_dtype: The torch.dtype to use for the dataset.
            enable_sampling: Whether to enable random sampling during training.
        """
        super().__init__(
            model=model,
            batch_size=batch_size,
            num_workers=num_workers,
            past_length=past_length,
            future_length=future_length,
            torch_dtype=torch_dtype,
            enable_sampling=enable_sampling,
        )
        self.save_hyperparameters(ignore=["model"])

    def setup(self, stage: Optional[str] = None):
        """Prepares the synthetic datasets for the given stage."""

        # This generator is used for training, and for validation if use_train_as_val is True
        train_generator = LinearTrendGenerator(
            length=self.hparams.series_length,
            slope=self.hparams.slope,
            intercept=self.hparams.intercept,
            noise_std=self.hparams.noise_std,
        )
        train_gluonts_dataset = SyntheticDataset(
            num_series=self.hparams.num_train_series, generator=train_generator
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            raw_train_dataset = RawTimeSeriesDataset(
                datasets=[train_gluonts_dataset],
                probabilities=[1.0],
                past_length=self.past_length,
                future_length=self.future_length,
                mode=DatasetMode.TRAINING,
                enable_sampling=self.enable_sampling,
            ).shuffle()
            self.train_dataset = PreprocessedTimeSeriesDataset(
                raw_dataset=raw_train_dataset,
                model=self.model,
                feature_generators=self.feature_generators,
                torch_dtype=self.torch_dtype,
            )

        if stage in ("fit", "validate", None):
            if self.hparams.use_train_as_val:
                val_gluonts_dataset = train_gluonts_dataset
                if self.hparams.num_train_series != self.hparams.num_val_series:
                    logger.info(
                        f"Overfit test mode: Using {self.hparams.num_train_series} training series for validation "
                        f"(ignoring num_val_series={self.hparams.num_val_series})."
                    )
            else:
                val_generator = LinearTrendGenerator(
                    length=self.hparams.series_length,
                    slope=self.hparams.slope,
                    intercept=self.hparams.intercept,
                    noise_std=self.hparams.noise_std,
                )
                val_gluonts_dataset = SyntheticDataset(
                    num_series=self.hparams.num_val_series, generator=val_generator
                )

            raw_val_dataset = RawTimeSeriesDataset(
                datasets=[val_gluonts_dataset],
                probabilities=[1.0],
                past_length=self.past_length,
                future_length=self.future_length,
                mode=DatasetMode.VALIDATION,
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


class OverfitTestDataModule(TimeSeriesDataModule):
    """
    A specialized DataModule for running overfitting tests on a small subset of data.

    This module inherits from TimeSeriesDataModule and modifies the setup process
    to use the same small subset of the training data for both training and
    validation. This is useful for quickly checking if the model can overfit
    on a small amount of data.
    """

    def __init__(self, n_series: int, use_train_as_val: bool = True, **kwargs):
        """
        Args:
            n_series: The number of time series to use for the sanity check.
            **kwargs: Additional arguments to pass to the TimeSeriesDataModule.
        """
        super().__init__(**kwargs)
        self.n_series = n_series
        self.use_train_as_val = use_train_as_val

        if "enable_sampling" in kwargs and kwargs["enable_sampling"]:
            logger.warning(
                "Overfit test mode is on, but enable_sampling is True. "
                "Disabling sampling to ensure consistent batches for memorization."
            )

    def setup(self, stage: Optional[str] = None):
        """
        Prepares the datasets for the sanity check.

        This method takes a small subset of the training data and uses it for
        both the training and validation sets.
        """

        ge_dataset: GiftEvalDataset
        ge_dataset, _ = get_gift_eval_dataset(
            self.dataset_name,
            self.dataset_storage_path,
            [self._EVALUATION_TERM],
        )[0]

        overfit_train_dataset = list(ge_dataset.training_dataset)[: self.n_series]
        if self.use_train_as_val:
            overfit_val_dataset = overfit_train_dataset
        else:
            overfit_val_dataset = list(ge_dataset.validation_dataset)[: self.n_series]

        logger.debug(f"Overfit train dataset: \n{overfit_train_dataset}")
        logger.debug(f"Overfit val dataset: \n{overfit_val_dataset}")

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            raw_train_dataset = RawTimeSeriesDataset(
                datasets=[overfit_train_dataset],
                probabilities=[1.0],
                past_length=self.past_length,
                future_length=self.future_length,
                mode=DatasetMode.TRAINING,
                min_future=ge_dataset.prediction_length,
                enable_sampling=self.enable_sampling,
            ).shuffle()
            self.train_dataset = PreprocessedTimeSeriesDataset(
                raw_dataset=raw_train_dataset,
                model=self.model,
                feature_generators=self.feature_generators,
                torch_dtype=self.torch_dtype,
            )

        if stage in ("fit", "validate", None):
            raw_val_dataset = RawTimeSeriesDataset(
                datasets=[overfit_val_dataset],
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
