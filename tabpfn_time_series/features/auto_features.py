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


logger = logging.getLogger(__name__)


class AutoSeasonalFeature(FeatureGenerator):
    # Safe on the whole frame: period detection is per-series (an FFT over each
    # series' target), but the sin/cos feature columns are built for all rows in a
    # single vectorized pass. Running on the whole frame avoids per-series DataFrame
    # concat + groupby reassembly. Depends only on the target and the per-series
    # row position (computed internally), not on other generators' outputs.
    PER_SERIES = False

    class Config:
        max_top_k: int = 12
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
        """Detect seasonal periods per series (an FFT over each series' target) and
        build the sin_#i / cos_#i columns for every row in a single vectorized pass.

        Runs on the whole multi-series frame (PER_SERIES=False): only the period
        detection loops over series (unavoidable FFT); the feature columns are then
        computed for all rows at once, avoiding per-series DataFrame concat/reassembly.
        """
        n = len(df)
        max_top_k = self.config["max_top_k"]

        if "item_id" in (df.index.names or []):
            # Within-series row position (0..len-1), matching the per-series
            # ``np.arange(len)`` used previously.
            pos = df.groupby(level="item_id", sort=False).cumcount().to_numpy()
            # Integer code per row identifying its series, in first-appearance order
            # (matches the enumeration order of groupby(sort=False) below).
            codes, _ = pd.factorize(
                df.index.get_level_values("item_id"), sort=False
            )
            n_series = int(codes.max()) + 1 if n else 0
            # Per-series detected periods (padded with NaN for empty slots).
            periods_by_series = np.full((n_series, max_top_k), np.nan)
            for code, (_item_id, target) in enumerate(
                df["target"].groupby(level="item_id", sort=False)
            ):
                periods = [
                    p for p, _ in self.find_seasonal_periods(target, **self.config)
                ]
                for i, period in enumerate(periods[:max_top_k]):
                    periods_by_series[code, i] = period
            period_per_row = periods_by_series[codes]
        else:
            # Single-series frame (item_id already dropped).
            pos = np.arange(n)
            periods = [
                p for p, _ in self.find_seasonal_periods(df["target"], **self.config)
            ]
            period_per_row = np.full((n, max_top_k), np.nan)
            for i, period in enumerate(periods[:max_top_k]):
                period_per_row[:, i] = period

        # Vectorized sin/cos for all rows and slots at once. Empty slots (NaN period)
        # become zeros, matching the previous zero-padding of missing periods.
        with np.errstate(divide="ignore", invalid="ignore"):
            angle = 2 * np.pi * pos[:, None] / period_per_row
        valid = ~np.isnan(period_per_row)
        sin = np.where(valid, np.sin(angle), 0.0)
        cos = np.where(valid, np.cos(angle), 0.0)

        feature_columns = {}
        for i in range(max_top_k):
            feature_columns[f"sin_#{i}"] = sin[:, i]
            feature_columns[f"cos_#{i}"] = cos[:, i]
        feature_df = pd.DataFrame(feature_columns, index=df.index)
        return pd.concat([df, feature_df], axis=1)

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
    x: np.ndarray, detrend_type: Literal["first_diff", "loess", "linear", "constant"]
) -> np.ndarray:
    if detrend_type == "first_diff":
        return np.diff(x, prepend=x[0])

    elif detrend_type == "loess":
        from statsmodels.api import nonparametric

        indices = np.arange(len(x))
        lowess = nonparametric.lowess(x, indices, frac=0.1)
        trend = lowess[:, 1]
        return x - trend

    elif detrend_type == "linear":
        # Use numpy polyfit instead of scipy.signal.detrend for numerical stability
        # (scipy's implementation can cause overflow/divide-by-zero on Apple Silicon)
        indices = np.arange(len(x))
        coeffs = np.polyfit(indices, x, 1, rcond=None)
        trend = np.polyval(coeffs, indices)
        return x - trend

    elif detrend_type == "constant":
        return x - np.mean(x)

    else:
        raise ValueError(f"Invalid detrend method: {detrend_type}")
