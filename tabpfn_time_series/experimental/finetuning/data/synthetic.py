from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np
import pandas as pd
from gluonts.dataset import DataEntry


class SyntheticGenerator(ABC):
    """Abstract base class for synthetic time series generators."""

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generates a single time series."""
        pass


class LinearTrendGenerator(SyntheticGenerator):
    """
    Generates a time series with a linear trend and Gaussian noise.

    Attributes:
        length: The length of the generated time series.
        slope: The slope of the linear trend.
        intercept: The intercept of the linear trend.
        noise_std: The standard deviation of the Gaussian noise.
    """

    def __init__(
        self,
        length: int,
        slope: float = 0.1,
        intercept: float = 0.0,
        noise_std: float = 0.1,
    ):
        self.length = length
        self.slope = slope
        self.intercept = intercept
        self.noise_std = noise_std

    def generate(self) -> np.ndarray:
        """Generates a single time series with a linear trend."""
        time = np.arange(self.length)
        trend = self.slope * time + self.intercept
        noise = np.random.normal(0, self.noise_std, size=self.length)
        return (trend + noise).astype(np.float32)


class SyntheticDataset:
    """
    An iterable dataset that generates synthetic time series using a provided generator.

    This dataset is compatible with the GluonTS framework and can be used as a drop-in
    replacement for file-based datasets.
    """

    def __init__(
        self,
        num_series: int,
        generator: SyntheticGenerator,
        freq: str = "D",
    ):
        """
        Args:
            num_series: The total number of time series to generate in this dataset.
            generator: An instance of a SyntheticGenerator subclass.
            freq: The frequency of the time series (e.g., 'D' for daily).
        """
        self.num_series = num_series
        self.generator = generator
        self.freq = freq

    def __iter__(self) -> Iterator[DataEntry]:
        """Yields synthetic time series as DataEntry objects."""
        for i in range(self.num_series):
            start_date = pd.Timestamp("2000-01-01", freq=self.freq)
            target = self.generator.generate()

            yield {
                "start": start_date,
                "target": target,
                "item_id": f"synthetic_series_{i}",
            }

    def __len__(self) -> int:
        """Returns the number of series in the dataset."""
        return self.num_series
