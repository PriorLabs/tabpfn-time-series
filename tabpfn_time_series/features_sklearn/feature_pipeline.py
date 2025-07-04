import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import copy

from typing import List, Literal, Optional, Tuple

from scipy import fft
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf

import gluonts.time_feature


class FeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline_steps):
        """
        This transformer takes the steps for a pipeline as input.

        Args:
            pipeline_steps (list): A list of tuples, where each tuple contains the
                                   name and the transformer instance, e.g.,
                                   [('step_name', TransformerObject())].
        """
        self.pipeline_steps = pipeline_steps
        # Create an unfitted template pipeline from the provided steps.
        # We will clone this template for each group.
        self._template_pipeline = Pipeline(steps=self.pipeline_steps)
        self.fitted_pipelines_ = {}  # Dictionary to store a fitted pipeline for each group

    def fit(self, X, y=None):
        """
        Fit one separate pipeline for each item_id.

        This method iterates through each 'item_id' in the input DataFrame X,
        creates a deep copy (clone) of the template pipeline, and fits that
        copy on the data for that specific item. This ensures that each item_id
        gets its own independently fitted pipeline without any state leaking
        from other items.
        """
        # Reset fitted pipelines on each call to fit
        self.fitted_pipelines_ = {}

        grouped = X.groupby("item_id")
        for item_id, group in grouped:
            # This prevents the state of transformers from one group affecting another.
            pipeline_for_item = copy.deepcopy(self._template_pipeline)

            pipeline_for_item.fit(group, y)
            self.fitted_pipelines_[item_id] = pipeline_for_item
        return self

    def transform(self, X):
        """
        Transform each group in X using its corresponding fitted pipeline.

        This method applies the correct, pre-fitted pipeline to each 'item_id'
        group in the input DataFrame X. It handles cases where an item_id might
        be in the transformation set but not in the training set (it will be skipped).
        Finally, it concatenates the results and reorders them to match the
        original index of X, preventing row shuffling.
        """
        all_transformed_groups = []
        grouped = X.groupby("item_id")

        for item_id, group in grouped:
            if item_id in self.fitted_pipelines_:
                # Use the pipeline that was fitted specifically for this item_id
                transformed_group = self.fitted_pipelines_[item_id].transform(group)
                all_transformed_groups.append(transformed_group)
            else:
                # Optional: Handle cases where an item_id appears in the transform set
                # but was not seen during fit. Here, we'll just print a warning.
                print(
                    f"Warning: No fitted pipeline found for item_id '{item_id}'. Skipping."
                )

        # Check if any groups were transformed to avoid errors on empty lists
        if not all_transformed_groups:
            return pd.DataFrame(
                columns=X.columns
            )  # Return empty DataFrame with original columns

        # Concatenate all the transformed group DataFrames
        transformed_df = pd.concat(all_transformed_groups, axis=0)

        # IMPORTANT: Reorder the final DataFrame to match the original index.
        # This prevents the rows from being shuffled due to the groupby-concat process.
        return transformed_df.reindex(X.index)


class RunningIndexFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Adds a running index feature to the DataFrame.

    Parameters
    ----------
    mode : str, default="per_item"
        - "per_item": running index resets for each item_id.
        - "global_timestamp": running index is assigned based on unique sorted timestamps across all item_ids.

    Attributes
    ----------
    train_df : pd.DataFrame
        The training data.
    """

    def __init__(self):
        self.train_df = None

    def fit(self, X, y=None):
        """
        Fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain columns: "item_id", "timestamp", "target" as the training data
        y : Ignored

        Returns
        -------
        self
        """
        self.train_df = X
        return self

    def transform(self, X):
        """
        Transform the DataFrame by adding the running index feature.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        X_out : pd.DataFrame
            With new column:
                - "running_index" (per_item mode)
                - "timestamp_index" (global_timestamp mode)
        """
        X = X.copy()
        if self.train_df is None:
            raise ValueError("Must call fit before transform")

        if not X["target"].isnull().all():
            X["running_index"] = range(len(X))
        else:
            X["running_index"] = range(len(X))
            X["running_index"] += len(self.train_df)
        return X


class CalendarFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper for CalendarFeature to provide sklearn-style transform interface.

    Parameters
    ----------
    components : list of str, optional
        Calendar components to extract.
    seasonal_features : dict, optional
        Seasonal features to extract.

    Notes
    -----
    Stateless; fit does nothing.
    """

    def __init__(self, components=None, seasonal_features=None):
        self.components = components or ["year"]
        self.seasonal_features = seasonal_features or {
            # (feature, natural seasonality)
            "second_of_minute": [60],
            "minute_of_hour": [60],
            "hour_of_day": [24],
            "day_of_week": [7],
            "day_of_month": [30.5],
            "day_of_year": [365],
            "week_of_year": [52],
            "month_of_year": [12],
        }

    def fit(self, X, y=None):
        """
        Fit the transformer on the training data.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain columns: "item_id", "timestamp", "target" as the training data
        y : Ignored

        Returns
        -------
        self
        """
        return self

    def transform(self, X):
        X_copy = X.copy()

        # LET'S JUST SET LEVEL
        # X_copy = X_copy.set_index(["item_id", "timestamp"])

        # Ensure the index is a DatetimeIndex
        timestamps = pd.DatetimeIndex(pd.to_datetime(X_copy["timestamp"]))

        # Add basic calendar components
        for component in self.components:
            X_copy[component] = getattr(timestamps, component)

        # Add seasonal features
        for feature_name, periods in self.seasonal_features.items():
            feature_func = getattr(gluonts.time_feature, f"{feature_name}_index")
            feature = feature_func(timestamps).astype(np.int32)

            if periods is not None:
                for period in periods:
                    period = period - 1  # Adjust for 0-based indexing
                    X_copy[f"{feature_name}_sin"] = np.sin(2 * np.pi * feature / period)
                    X_copy[f"{feature_name}_cos"] = np.cos(2 * np.pi * feature / period)
            else:
                X_copy[feature_name] = feature

        return X_copy


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


class AutoSeasonalFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn compatible transformer that automatically detects and creates
    seasonal features from a time series.

    This transformer identifies dominant seasonal periods from the target variable
    during the `fit` phase using Fast Fourier Transform (FFT). It then generates
    sine and cosine features for these detected periods in the `transform` phase.

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

    Notes
    -----
    This transformer currently only supports regularly-sampled time series.
    It will not work as expected with time series that have irregular intervals
    between observations.
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
    ):
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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Detects seasonal periods from the target time series.

        This method analyzes the provided time series (`y` or `X['target']`)
        to find the most significant seasonal periods using FFT and stores
        them for the `transform` step.

        Parameters
        ----------
        X : pd.DataFrame
            The input data. If `y` is None, `X` must contain a 'target' column.
        y : Optional[pd.Series], optional
            The target time series, by default None.

        Returns
        -------
        self
            The fitted estimator.
        """
        # save the train_df
        self.train_df = X

        if y is None:
            if "target" not in X.columns:
                raise ValueError(
                    "Target column 'target' not found in X. "
                    "Please provide the target series via the y argument."
                )
            target_values = X["target"]
        else:
            target_values = y

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

        self.periods_ = [period for period, _ in detected_periods_and_magnitudes]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds seasonal features (sine and cosine) to the DataFrame.

        This method uses the periods detected during the `fit` phase to
        generate and append seasonal features to the input DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the added seasonal features.
        """
        if not hasattr(self, "periods_"):
            raise RuntimeError(
                "The transformer has not been fitted yet. "
                "Please call 'fit' before 'transform'."
            )

        X_transformed = X.copy()
        if not X["target"].isnull().all():
            time_idx = np.arange(len(X_transformed))
        else:
            time_idx = np.arange(len(X_transformed))
            time_idx += len(self.train_df)

        # time_idx = np.arange(len(X_transformed))

        # Generate features for detected periods
        for i, period in enumerate(self.periods_):
            if period > 1:  # Avoid creating features for non-periodic signals
                angle = 2 * np.pi * time_idx / period
                X_transformed[f"sin_#{i}"] = np.sin(angle)
                X_transformed[f"cos_#{i}"] = np.cos(angle)

        # Add placeholder columns for missing periods up to max_top_k
        for i in range(len(self.periods_), self.max_top_k):
            X_transformed[f"sin_#{i}"] = 0.0
            X_transformed[f"cos_#{i}"] = 0.0

        return X_transformed

    @staticmethod
    def _find_seasonal_periods(
        target_values: pd.Series, **kwargs
    ) -> List[Tuple[float, float]]:
        """
        Identifies dominant seasonal periods in a time series using FFT.

        This is a helper method that contains the core logic for period detection.
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
                peak_indices = np.arange(len(fft_magnitudes))
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
