import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging


import gluonts.time_feature

from .pipeline import ColumnConfig, DefaultColumnConfig

logger = logging.getLogger(__name__)


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

    def __init__(
        self,
        components=None,
        seasonal_features=None,
        column_config: ColumnConfig = DefaultColumnConfig(),
    ):
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
        self.timestamp_col_name = column_config.timestamp_col_name
        self.target_col_name = column_config.target_col_name
        self.item_id_col_name = column_config.item_id_col_name

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
        assert self.timestamp_col_name is not None, (
            "timestamp_col_name must be provided"
        )
        assert self.timestamp_col_name in X.columns, (
            f"timestamp_col_name {self.timestamp_col_name} not in X.columns"
        )

        return self

    def transform(self, X):
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
