import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, List, Optional

import gluonts.time_feature

from .pipeline_configs import ColumnConfig, DefaultColumnConfig
from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class CalendarFeatureTransformer(BaseFeatureTransformer):
    """
    Wrapper for CalendarFeature to provide sklearn-style transform interface.
    """

    def __init__(
        self,
        components: Optional[List[str]] = None,
        seasonal_features: Optional[Dict[str, List[float]]] = None,
        column_config: ColumnConfig = DefaultColumnConfig(),
    ):
        """
        Initializes the CalendarFeatureTransformer.

        Parameters
        ----------
        components : Optional[List[str]], optional
            A list of basic calendar components to extract from the timestamp.
            These correspond to pandas.DatetimeIndex attributes (e.g., 'year', 'month').
            Defaults to ["year"].
        seasonal_features : Optional[Dict[str, List[float]]], optional
            A dictionary mapping seasonal features to their corresponding periods for
            sine/cosine transformation. The keys are feature names from gluonts.time_feature
            (e.g., 'day_of_week'), and the values are lists of periods.
            Defaults to a standard set of time-based seasonal features.
        column_config : ColumnConfig, optional
            Configuration object specifying the names of timestamp, target, and item ID columns.
            Defaults to DefaultColumnConfig().
        """
        super().__init__(column_config)
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
        self._required_columns = ["timestamp_col_name"]

    def fit(
        self, X: pd.DataFrame, y: Optional[Any] = None
    ) -> "CalendarFeatureTransformer":
        """
        Fits the transformer by validating the input DataFrame.

        This method checks for the presence of the required timestamp column.
        As this transformer is stateless, it performs no actual fitting.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame, which must contain the timestamp column
            specified in `column_config`.
        y : Any, optional
            Ignored. This parameter exists for scikit-learn compatibility.

        Returns
        -------
        CalendarFeatureTransformer
            The fitted transformer instance.
        """
        super().fit(X, y)  # validate the data
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Adds calendar and seasonal features to the DataFrame.

        This method takes the input DataFrame, extracts date and time features from
        the timestamp column, and appends them as new columns.

        Parameters
        ----------
        X : pd.DataFrame
            The DataFrame to transform. It must contain the timestamp column.

        Returns
        -------
        pd.DataFrame
            A new DataFrame with the added calendar and seasonal feature columns.
        """
        X_copy = X.copy()

        # Ensure the index is a DatetimeIndex
        timestamps = pd.DatetimeIndex(pd.to_datetime(X_copy[self.timestamp_col_name]))

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
