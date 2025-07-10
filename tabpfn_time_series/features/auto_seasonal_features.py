import pandas as pd
import numpy as np
import logging
from joblib import Parallel, delayed
from typing import List, Literal, Optional, Tuple

from scipy import fft
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf


from .pipeline_configs import ColumnConfig, DefaultColumnConfig
from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class AutoSeasonalFeatureTransformer(BaseFeatureTransformer):
    """
    A scikit-learn compatible transformer that automatically detects and creates
    seasonal features from a time series.

    This transformer identifies dominant seasonal periods from the target variable
    during the `fit` phase using Fast Fourier Transform (FFT). It then generates
    sine and cosine features for these detected periods in the `transform` phase.

    This transformer supports multi-item time series, fitting and transforming
    each item independently based on its `item_id_col_name`.

    Notes
    -----
    - This transformer currently only supports regularly-sampled time series.
      It will not work as expected with time series that have irregular intervals
      between observations.
    - This transformer distinguishes between training and testing (prediction) data by checking the target column.
      The testing data's target values are always NaNs.
    """

    def __init__(
        self,
        max_top_k: int = 5,
        do_detrend: bool = True,
        detrend_type: Literal["first_diff", "loess", "linear", "constant"] = "linear",
        use_peaks_only: bool = True,
        apply_hann_window: bool = True,
        zero_padding_factor: int = 2,
        round_to_closest_integer: bool = True,
        validate_with_acf: bool = False,
        sampling_interval: float = 1.0,
        magnitude_threshold: Optional[float] = 0.05,
        relative_threshold: bool = True,
        exclude_zero: bool = True,
        column_config: ColumnConfig = DefaultColumnConfig(),
        n_jobs: int = -1,
    ):
        """
        Initializes the AutoSeasonalFeatureTransformer.

        Parameters
        ----------
        max_top_k : int, optional
            The maximum number of dominant periods to identify, by default 5.
        do_detrend : bool, optional
            Whether to detrend the series before FFT, by default True.
        detrend_type : Literal["first_diff", "loess", "linear", "constant"], optional
            The detrending method to use, by default "linear".
        use_peaks_only : bool, optional
            If True, considers only local peaks in the FFT spectrum, by default True.
        apply_hann_window : bool, optional
            If True, applies a Hann window to reduce spectral leakage, by default True.
        zero_padding_factor : int, optional
            Factor for zero-padding to improve frequency resolution, by default 2.
        round_to_closest_integer : bool, optional
            If True, rounds detected periods to the nearest integer, by default True.
        validate_with_acf : bool, optional
            If True, validates periods with the Autocorrelation Function, by default False.
        sampling_interval : float, optional
            Time interval between samples, by default 1.0.
        magnitude_threshold : Optional[float], optional
            Threshold for filtering frequency components, by default 0.05.
        relative_threshold : bool, optional
            If True, `magnitude_threshold` is a fraction of the max FFT magnitude, by default True.
        exclude_zero : bool, optional
            If True, excludes periods of 0 from the results, by default True.
        column_config : ColumnConfig, optional
            Configuration object specifying column names for timestamp, target, and item ID.
            Defaults to `DefaultColumnConfig()`.
        n_jobs : int, optional
            The number of jobs to run in parallel for "per_item" mode.
            -1 means using all available processors. By default -1.
        """
        super().__init__(column_config)
        self.max_top_k = max_top_k
        self.do_detrend = do_detrend
        self.detrend_type = detrend_type
        self.use_peaks_only = use_peaks_only
        self.apply_hann_window = apply_hann_window
        self.zero_padding_factor = zero_padding_factor
        self.round_to_closest_integer = round_to_closest_integer
        self.validate_with_acf = validate_with_acf
        self.sampling_interval = sampling_interval
        self.magnitude_threshold = magnitude_threshold
        self.relative_threshold = relative_threshold
        self.exclude_zero = exclude_zero
        self.train_df = None
        self.n_jobs = n_jobs
        self._required_columns = [
            "timestamp_col_name",
            "target_col_name",
            "item_id_col_name",
        ]

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "AutoSeasonalFeatureTransformer":
        """
        Fits the transformer to the data by detecting seasonal periods for each item.

        This method iterates through each time series item specified by `item_id_col_name`,
        detects its dominant seasonal periods, and stores them internally.

        Parameters
        ----------
        X : pd.DataFrame
            The input data, containing columns for timestamp, target, and item ID.
        y : pd.Series, optional
            Ignored. This parameter exists for compatibility with scikit-learn pipelines.

        Returns
        -------
        AutoSeasonalFeatureTransformer
            The fitted transformer instance.
        """
        super().fit(X, y)  # validate the data

        # --- Parallelized version of fitting per item---
        grouped = X.groupby(self.item_id_col_name)
        group_names = [name for name, _ in grouped]

        # Run fit_item for each group in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.fit_item)(group_data) for _, group_data in grouped
        )

        # Combine the group names and results into the final dictionary
        self.fitted_autoseasonal_per_item = dict(zip(group_names, results))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data by adding seasonal features.

        For each time series item, this method adds sine and cosine features based
        on the seasonal periods detected during the `fit` stage.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with added seasonal features.

        Raises
        ------
        ValueError
            If an item ID is present in `X` that was not seen during `fit`.
        """
        # Fail fast if any item in X was not seen during fit
        seen_items = set(self.fitted_autoseasonal_per_item.keys())
        transform_items = set(X[self.item_id_col_name].unique())
        missing_items = transform_items - seen_items
        if missing_items:
            raise ValueError(
                f"No fitted data found for item_id(s) {missing_items}. "
                "Cannot create seasonal features for new items."
            )

        # --- Parallelized version ---
        grouped = X.groupby(self.item_id_col_name)

        all_transformed_items = Parallel(n_jobs=self.n_jobs)(
            delayed(self.transform_item)(
                group_data,
                periods_=self.fitted_autoseasonal_per_item[group_name]["periods_"],
                train_df=self.fitted_autoseasonal_per_item[group_name]["train_df"],
            )
            for group_name, group_data in grouped
        )

        if not all_transformed_items:
            X_transformed = X.copy()
            for i in range(self.max_top_k):
                X_transformed[f"sin_#{i}"] = 0.0
                X_transformed[f"cos_#{i}"] = 0.0
            return X_transformed
        transformed_df = pd.concat(all_transformed_items)
        return transformed_df.reindex(X.index)

    def fit_item(self, X: pd.DataFrame) -> dict:
        """
        Detects seasonal periods for a single time series item.

        This method analyzes the target series of a single item to find the most
        significant seasonal periods using FFT and stores them.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for a single item, containing a target column.

        Returns
        -------
        dict
            A dictionary containing the detected `periods_` and the original
            `train_df` for the item.
        """

        # save the train_df
        train_df = X

        target_values = X[self.target_col_name]

        detected_periods_and_magnitudes = self._find_seasonal_periods(
            target_values=target_values,
            max_top_k=self.max_top_k,
            do_detrend=self.do_detrend,
            detrend_type=self.detrend_type,
            use_peaks_only=self.use_peaks_only,
            apply_hann_window=self.apply_hann_window,
            zero_padding_factor=self.zero_padding_factor,
            round_to_closest_integer=self.round_to_closest_integer,
            validate_with_acf=self.validate_with_acf,
            sampling_interval=self.sampling_interval,
            magnitude_threshold=self.magnitude_threshold,
            relative_threshold=self.relative_threshold,
            exclude_zero=self.exclude_zero,
        )

        periods_ = [period for period, _ in detected_periods_and_magnitudes]

        return {"periods_": periods_, "train_df": train_df}

    def transform_item(
        self, X: pd.DataFrame, periods_: List[float], train_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds seasonal features (sine and cosine) to a single item's DataFrame.

        This method uses the periods detected during fitting to generate and
        append seasonal features. It correctly handles time indices for both
        training and forecasting horizons.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform for a single item.
        periods_ : List[float]
            The list of seasonal periods detected for this item.
        train_df : pd.DataFrame
            The training DataFrame for this item, used to establish the correct
            time index for forecasting.

        Returns
        -------
        pd.DataFrame
            The DataFrame for the item with added seasonal features.
        """
        X_transformed = X.copy()
        if not X["target"].isnull().all():
            time_idx = np.arange(len(X_transformed))
        else:
            time_idx = np.arange(len(X_transformed))
            time_idx += len(train_df)

        # Generate features for detected periods
        for i, period in enumerate(periods_):
            if period > 1:  # Avoid creating features for non-periodic signals
                angle = 2 * np.pi * time_idx / period
                X_transformed[f"sin_#{i}"] = np.sin(angle)
                X_transformed[f"cos_#{i}"] = np.cos(angle)

        # Add placeholder columns for missing periods up to max_top_k
        for i in range(len(periods_), self.max_top_k):
            X_transformed[f"sin_#{i}"] = 0.0
            X_transformed[f"cos_#{i}"] = 0.0

        return X_transformed

    @staticmethod
    def _find_seasonal_periods(
        target_values: pd.Series, **kwargs
    ) -> List[Tuple[float, float]]:
        """
        Identifies dominant seasonal periods in a time series using FFT.

        This is a static helper method that contains the core logic for period
        detection based on the spectral density of the signal.
        """
        # Extract parameters from kwargs
        max_top_k = kwargs.get("max_top_k", 5)
        do_detrend = kwargs.get("do_detrend", True)
        detrend_type = kwargs.get("detrend_type", "linear")
        use_peaks_only = kwargs.get("use_peaks_only", True)
        apply_hann_window = kwargs.get("apply_hann_window", True)
        zero_padding_factor = kwargs.get("zero_padding_factor", 2)
        round_to_closest_integer = kwargs.get("round_to_closest_integer", True)
        validate_with_acf = kwargs.get("validate_with_acf", False)
        sampling_interval = kwargs.get("sampling_interval", 1.0)
        magnitude_threshold = kwargs.get("magnitude_threshold", 0.05)
        relative_threshold = kwargs.get("relative_threshold", True)
        exclude_zero = kwargs.get("exclude_zero", True)

        values = np.array(target_values, dtype=float)
        # Drop NaN values, assuming they correspond to the test set
        values = values[~np.isnan(values)]

        if len(values) < 2:
            return []

        n_original = len(values)

        if do_detrend:
            values = detrend(values, detrend_type)

        if apply_hann_window:
            values = values * np.hanning(n_original)

        if zero_padding_factor > 1:
            padded_length = int(n_original * zero_padding_factor)
            values = np.pad(values, (0, padded_length - n_original), "constant")

        n = len(values)
        fft_values = fft.rfft(values)
        fft_magnitudes = np.abs(fft_values)
        freqs = np.fft.rfftfreq(n, d=sampling_interval)
        fft_magnitudes[0] = 0.0  # Exclude DC component

        threshold_value = (
            magnitude_threshold * np.max(fft_magnitudes)
            if magnitude_threshold is not None and relative_threshold
            else magnitude_threshold
        )

        if use_peaks_only:
            peak_indices, _ = find_peaks(fft_magnitudes, height=threshold_value)
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

        non_zero_freqs = freqs[top_indices] > 0
        top_indices = np.array(top_indices)[non_zero_freqs]
        top_periods = 1.0 / freqs[top_indices]

        if round_to_closest_integer:
            top_periods = np.round(top_periods)

        if exclude_zero:
            non_zero_mask = top_periods != 0
            top_periods = top_periods[non_zero_mask]
            top_indices = top_indices[non_zero_mask]

        if len(top_periods) > 0:
            _, unique_indices = np.unique(top_periods, return_index=True)
            top_periods = top_periods[unique_indices]
            top_indices = top_indices[unique_indices]

        results = [
            (period, fft_magnitudes[index])
            for period, index in zip(top_periods, top_indices)
        ]

        if validate_with_acf:
            acf_values = acf(
                np.array(target_values, dtype=float)[:n_original],
                nlags=n_original - 1,
                fft=True,
            )
            acf_peak_indices, _ = find_peaks(
                acf_values, height=1.96 / np.sqrt(n_original)
            )
            validated_results = [
                (period, mag)
                for period, mag in results
                if any(abs(int(round(period)) - peak) <= 1 for peak in acf_peak_indices)
            ]
            if validated_results:
                results = validated_results

        results.sort(key=lambda x: x[1], reverse=True)
        return results


def detrend(
    x: np.ndarray,
    detrend_type: Literal["first_diff", "loess", "linear", "constant"],
) -> np.ndarray:
    """
    Remove the trend from a time series.

    Parameters
    ----------
    x : np.ndarray
        The input time series.
    detrend_type : Literal["first_diff", "loess", "linear", "constant"]
        The detrending method to use.

    Returns
    -------
    np.ndarray
        The detrended time series.

    Raises
    ------
    ValueError
        If an invalid detrend method is specified.
    """
    if detrend_type == "first_diff":
        return np.diff(x, prepend=x[0])
    if detrend_type == "loess":
        from statsmodels.api import nonparametric

        indices = np.arange(len(x))
        lowess = nonparametric.lowess(x, indices, frac=0.1)
        trend = lowess[:, 1]
        return x - trend
    if detrend_type in ["linear", "constant"]:
        from scipy.signal import detrend as scipy_detrend

        return scipy_detrend(x, type=detrend_type)

    raise ValueError(f"Invalid detrend method: {detrend_type}")
