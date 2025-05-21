import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Literal

import logging

from scipy import fft
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

from tabpfn_time_series.features.feature_generator_base import (
    FeatureGenerator,
)
from tabpfn_time_series.features.basic_features import (
    PeriodicSinCosineFeature,
)


logger = logging.getLogger(__name__)


class AutoSeasonalFeature(FeatureGenerator):
    class Config:
        max_top_k: int = 5
        do_detrend: bool = True
        detrend_type: Literal["first_diff", "loess", "linear", "constant"] = "linear"
        use_peaks_only: bool = True
        apply_hann_window: bool = True
        zero_padding_factor: int = 2
        round_to_closest_integer: bool = True
        validate_with_acf: bool = False
        sampling_interval: float = 1.0
        magnitude_threshold: Optional[float] = 0.05
        relative_threshold: bool = True
        exclude_zero: bool = True

    def __init__(self, config: Optional[dict] = None):
        # Create default config from Config class
        default_config = {
            k: v for k, v in vars(self.Config).items() if not k.startswith("__")
        }

        # Initialize config with defaults
        self.config = default_config.copy()

        # Update with user-provided config if any
        if config is not None:
            self.config.update(config)

        # Validate config parameters
        self._validate_config()

        logger.debug(f"Initialized AutoSeasonalFeature with config: {self.config}")

    def _validate_config(self):
        """Validate configuration parameters"""
        if self.config["max_top_k"] < 1:
            logger.warning("max_top_k must be at least 1, setting to 1")
            self.config["max_top_k"] = 1

        if self.config["zero_padding_factor"] < 1:
            logger.warning("zero_padding_factor must be at least 1, setting to 1")
            self.config["zero_padding_factor"] = 1

        if self.config["detrend_type"] not in [
            "first_diff",
            "loess",
            "linear",
            "constant",
        ]:
            logger.warning(
                f"Invalid detrend_type: {self.config['detrend_type']}, using 'linear'"
            )
            self.config["detrend_type"] = "linear"

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Detect seasonal periods from target data
        detected_periods_and_magnitudes = self.find_seasonal_periods(
            df.target, **self.config
        )
        logger.debug(
            f"Found {len(detected_periods_and_magnitudes)} seasonal periods: {detected_periods_and_magnitudes}"
        )

        # Extract just the periods (without magnitudes)
        periods = [period for period, _ in detected_periods_and_magnitudes]

        # Generate features for detected periods using PeriodicSinCosineFeature
        if periods:
            feature_generator = PeriodicSinCosineFeature(periods=periods)
            df = feature_generator.generate(df)

        # Standardize column names for consistency across time series
        renamed_columns = {}
        for i, period in enumerate(periods):
            renamed_columns[f"sin_{period}"] = f"sin_#{i}"
            renamed_columns[f"cos_{period}"] = f"cos_#{i}"

        df = df.rename(columns=renamed_columns)

        # Add placeholder zero columns for missing periods up to max_top_k
        for i in range(len(periods), self.config["max_top_k"]):
            df[f"sin_#{i}"] = 0.0
            df[f"cos_#{i}"] = 0.0

        return df

    @staticmethod
    def find_seasonal_periods(
        target_values: pd.Series,
        max_top_k: int = 10,
        do_detrend: bool = True,
        detrend_type: Literal[
            "first_diff", "loess", "linear", "constant"
        ] = "first_diff",
        use_peaks_only: bool = True,
        apply_hann_window: bool = True,
        zero_padding_factor: int = 2,
        round_to_closest_integer: bool = True,
        validate_with_acf: bool = False,
        sampling_interval: float = 1.0,
        magnitude_threshold: Optional[
            float
        ] = 0.05,  # Default relative threshold (5% of max)
        relative_threshold: bool = True,  # Interpret threshold as a fraction of max FFT magnitude
        exclude_zero: bool = False,
    ) -> List[Tuple[float, float]]:
        """
        Identify dominant seasonal periods in a time series using FFT.

        Parameters:
        - target_values: pd.Series
            Input time series data.
        - max_top_k: int
            Maximum number of dominant periods to return.
        - do_detrend: bool
            If True, remove the linear trend from the signal.
        - use_peaks_only: bool
            If True, consider only local peaks in the FFT magnitude spectrum.
        - apply_hann_window: bool
            If True, apply a Hann window to reduce spectral leakage.
        - zero_padding_factor: int
            Factor by which to zero-pad the signal for finer frequency resolution.
        - round_to_closest_integer: bool
            If True, round the detected periods to the nearest integer.
        - validate_with_acf: bool
            If True, validate detected periods against the autocorrelation function.
        - sampling_interval: float
            Time interval between consecutive samples.
        - magnitude_threshold: Optional[float]
            Threshold to filter out less significant frequency components.
            Default is 0.05, interpreted as 5% of the maximum FFT magnitude if relative_threshold is True.
        - relative_threshold: bool
            If True, the `magnitude_threshold` is interpreted as a fraction of the maximum FFT magnitude.
            Otherwise, it is treated as an absolute threshold value.
        - exclude_zero: bool
            If True, exclude periods of 0 from the results.

        Returns:
        - List[Tuple[float, float]]:
            A list of (period, magnitude) tuples, sorted in descending order by magnitude.
        """
        # Convert the Pandas Series to a NumPy array
        values = np.array(target_values, dtype=float)

        # Quick hack to ignore the test_X
        #   (Assuming train_X target is not NaN, and test_X target is NaN)
        #   Dropping all the NaN values
        values = values[~np.isnan(values)]

        N_original = len(values)

        # Detrend the signal using a linear detrend method if requested
        if do_detrend:
            values = detrend(values, detrend_type)

        # Apply a Hann window to reduce spectral leakage
        if apply_hann_window:
            window = np.hanning(N_original)
            values = values * window

        # Zero-pad the signal for improved frequency resolution
        if zero_padding_factor > 1:
            padded_length = int(N_original * zero_padding_factor)
            padded_values = np.zeros(padded_length)
            padded_values[:N_original] = values
            values = padded_values
            N = padded_length
        else:
            N = N_original

        # Compute the FFT (using rfft) and obtain frequency bins
        fft_values = fft.rfft(values)
        fft_magnitudes = np.abs(fft_values)
        freqs = np.fft.rfftfreq(N, d=sampling_interval)

        # Exclude the DC component (0 Hz) to avoid bias from the signal's mean
        fft_magnitudes[0] = 0.0

        # Determine the threshold (absolute value)
        if magnitude_threshold is not None and relative_threshold:
            threshold_value = magnitude_threshold * np.max(fft_magnitudes)
        else:
            threshold_value = magnitude_threshold

        # Identify dominant frequencies
        if use_peaks_only:
            if threshold_value is not None:
                peak_indices, _ = find_peaks(fft_magnitudes, height=threshold_value)
            else:
                peak_indices, _ = find_peaks(fft_magnitudes)
            if len(peak_indices) == 0:
                # Fallback to considering all frequency bins if no peaks are found
                peak_indices = np.arange(len(fft_magnitudes))
            # Sort the peak indices by magnitude in descending order
            sorted_peak_indices = peak_indices[
                np.argsort(fft_magnitudes[peak_indices])[::-1]
            ]
            top_indices = sorted_peak_indices[:max_top_k]
        else:
            sorted_indices = np.argsort(fft_magnitudes)[::-1]
            if threshold_value is not None:
                sorted_indices = [
                    i for i in sorted_indices if fft_magnitudes[i] >= threshold_value
                ]
            top_indices = sorted_indices[:max_top_k]

        # Convert frequencies to periods (avoiding division by zero)
        periods = np.zeros_like(freqs)
        non_zero = freqs > 0
        periods[non_zero] = 1.0 / freqs[non_zero]
        top_periods = periods[top_indices]

        logger.debug(f"Top periods: {top_periods}")

        # Optionally round the periods to the nearest integer
        if round_to_closest_integer:
            top_periods = np.round(top_periods)

        # Filter out zero periods if requested
        if exclude_zero:
            non_zero_mask = top_periods != 0
            top_periods = top_periods[non_zero_mask]
            top_indices = top_indices[non_zero_mask]

        # Keep unique periods only
        if len(top_periods) > 0:
            unique_period_indices = np.unique(top_periods, return_index=True)[1]
            top_periods = top_periods[unique_period_indices]
            top_indices = top_indices[unique_period_indices]

        # Pair each period with its corresponding magnitude
        results = [
            (top_periods[i], fft_magnitudes[top_indices[i]])
            for i in range(len(top_indices))
        ]

        # Validate with ACF if requested and filter the results accordingly
        if validate_with_acf:
            # Compute ACF on the original (non-padded) detrended signal
            acf_values = acf(
                np.array(target_values, dtype=float)[:N_original],
                nlags=N_original,
                fft=True,
            )
            acf_peak_indices, _ = find_peaks(
                acf_values, height=1.96 / np.sqrt(N_original)
            )
            validated_results = []
            for period, mag in results:
                period_int = int(round(period))
                if period_int < len(acf_values) and any(
                    abs(period_int - peak) <= 1 for peak in acf_peak_indices
                ):
                    validated_results.append((period, mag))
            if validated_results:
                results = validated_results

        # Ensure the final results are sorted in descending order by magnitude
        results.sort(key=lambda x: x[1], reverse=True)

        return results


def detrend(
    x: np.ndarray, detrend_type: Literal["first_diff", "loess", "linear"]
) -> np.ndarray:
    if detrend_type == "first_diff":
        return np.diff(x, prepend=x[0])

    elif detrend_type == "loess":
        from statsmodels.api import nonparametric

        indices = np.arange(len(x))
        lowess = nonparametric.lowess(x, indices, frac=0.1)
        trend = lowess[:, 1]
        return x - trend

    elif detrend_type in ["linear", "constant"]:
        from scipy.signal import detrend as scipy_detrend

        return scipy_detrend(x, type=detrend_type)

    else:
        raise ValueError(f"Invalid detrend method: {detrend_type}")
