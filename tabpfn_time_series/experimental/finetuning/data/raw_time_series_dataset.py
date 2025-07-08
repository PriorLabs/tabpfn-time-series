from enum import Enum
from typing import Literal, List, Iterator, Iterable

import numpy as np
import torch
from torch.utils.data import IterableDataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset import DataEntry
from gluonts.itertools import Cyclic
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    TestSplitSampler,
    ValidationSplitSampler,
    InstanceSplitter,
    FilterTransformation,
    Transformation,
    Chain,
)


class DatasetMode(Enum):
    """An enum to represent the mode of the dataset (training, validation, or test)."""

    TRAINING = "training"
    VALIDATION = "validation"  # Single window from the end of each series
    TEST = "test"  # Single window from the end of each series


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffles entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    This is useful for adding stochasticity to an IterableDataset, which
    cannot be shuffled by conventional means.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer used to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset: IterableDataset, shuffle_buffer_length: int = 100):
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self) -> Iterator:
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                ).item()
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(
                len(shuffle_buffer), size=(), generator=self.generator
            ).item()
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    A mix-in class that datasets can inherit from to get shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        """Returns a shuffled version of the dataset."""
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class RawTimeSeriesDataset(IterableDataset, ShuffleMixin):
    """
    An IterableDataset that generates windowed time series samples from one or
    more source datasets.

    This class is designed to be the first step in a data pipeline. Its sole
    responsibility is to load raw time series and use GluonTS's InstanceSplitter
    to create training/validation/test samples.

    Each sample yielded by this dataset is a dictionary containing raw data,
    including `past_target`, `future_target`, and the crucial `forecast_start`
    timestamp object.

    **Usage with a DataLoader:**
    This dataset should be used with a `torch.utils.data.DataLoader`.
    To add model-specific features (e.g., time-based features like 'hour',
    'day_of_week'), a custom `collate_fn` should be provided to the DataLoader.
    This decouples featurization from data loading, allowing for flexible and
    performant on-the-fly feature generation in parallel worker processes.
    """

    def __init__(
        self,
        datasets: List[Iterable[DataEntry]],
        probabilities: List[float],
        past_length: int = 10_000,
        future_length: int = 1_024,
        mode: Literal["training", "validation", "test"]
        | DatasetMode = DatasetMode.TRAINING,
        min_past: int = 128,
        min_future: int = 64,
        enable_sampling: bool = True,
    ):
        """
        Args:
            datasets: A list of iterables of DataEntry objects.
            probabilities: A list of probabilities for each dataset.
            past_length: The exact length of the historical context ("past") fed to
                the model. Shorter series are padded, longer series are truncated
                to this length. (default: 10_000)
            future_length: The exact length of the prediction horizon ("future") that
                the model is trained to predict. (default: 1_024)
            mode: The mode of the dataset (training, validation, or test). This
                determines how windows are sampled. (default: DatasetMode.TRAINING)
            min_past: A filter for training. This ensures that each sampled window
                has at least this many valid (non-padded) data points in its
                `past_target`, preventing training on nearly-empty samples.
                (default: 128)
            min_future: A filter for training and validation/test. For training, it
                ensures `future_target` has at least this many valid points. For
                validation/test, it ensures the series is long enough to create a
                meaningful final window. (default: 64)
            enable_sampling: Whether to enable random sampling during training. When
                False, training mode will use validation-like sampling (single
                window from the end of each series) instead of random sampling.
                (default: True)
        """

        super().__init__()

        assert len(probabilities) == len(datasets), (
            "Probabilities and datasets must have the same length."
        )
        assert sum(probabilities) > 0.999 and sum(probabilities) < 1.001, (
            "Probabilities must sum to 1."
        )

        self.datasets = datasets
        self.probabilities = probabilities
        self.past_length = past_length
        self.future_length = future_length
        self.min_past = min_past
        self.min_future = min_future
        self.mode = DatasetMode(mode) if isinstance(mode, str) else mode
        self.enable_sampling = enable_sampling

    @staticmethod
    def _is_past_target_not_constant(entry: DataEntry) -> bool:
        """
        Checks if 'past_target' is not a constant series.

        This check is designed to be run after filtering out all-NaN series.
        It identifies series with a standard deviation of zero.
        We use a small epsilon for floating-point comparisons.
        """
        return np.nanstd(entry["past_target"]) > 1e-8

    def _create_transformation_chain(self) -> Transformation:
        """Creates the appropriate transformation chain based on the dataset mode."""
        if self.mode == DatasetMode.TRAINING and self.enable_sampling:
            sampler = ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.min_future,
            )
        elif self.mode == DatasetMode.TRAINING and not self.enable_sampling:
            # Use validation-like sampling when sampling is disabled during training
            sampler = ValidationSplitSampler(min_future=self.min_future)
        elif self.mode == DatasetMode.VALIDATION:
            sampler = ValidationSplitSampler(min_future=self.min_future)
        else:  # TEST mode
            sampler = TestSplitSampler()

        splitter = InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=sampler,
            past_length=self.past_length,
            future_length=self.future_length,
            dummy_value=np.nan,
        )

        # Only filter for training mode to remove all-NaN samples
        if self.mode is not DatasetMode.TRAINING:
            return splitter

        return Chain(
            [
                splitter,
                # 1. Filter out series that are all-NaN
                FilterTransformation(
                    condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
                ),
                # 2. Filter out constant series
                FilterTransformation(condition=self._is_past_target_not_constant),
            ]
        )

    def __iter__(self) -> Iterator:
        """
        Creates an iterator that interleaves samples from the source datasets.
        """
        transform = self._create_transformation_chain()
        is_train = self.mode == DatasetMode.TRAINING
        # Use cyclic iteration only when in training mode with sampling enabled
        use_cyclic = is_train and self.enable_sampling

        # Create a list of iterators over transformed datasets
        transformed_iterators = []
        for ds in self.datasets:
            source = Cyclic(ds) if use_cyclic else ds
            transformed = transform.apply(source, is_train=is_train)
            transformed_iterators.append(iter(transformed))

        # The main sampling loop
        while True:
            # Choose a source dataset iterator based on probabilities
            dataset_idx = np.random.choice(len(self.datasets), p=self.probabilities)

            try:
                # Yield the next available sample from the chosen iterator
                yield next(transformed_iterators[dataset_idx])

            except StopIteration:
                # This will only be reached in non-training modes when an
                # iterator is exhausted. We stop the entire iteration.
                break
