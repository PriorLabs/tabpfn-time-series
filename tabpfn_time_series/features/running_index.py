import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import logging

from typing import Literal


from .pipeline import ColumnConfig, DefaultColumnConfig

logger = logging.getLogger(__name__)


class RunningIndexFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Adds a running index feature to the DataFrame.

    The index can be calculated globally across all time stamps or on a
    per-item basis.

    Attributes
    ----------
    mode : str
        The mode of operation, either "per_item" or "global_timestamp".
    train_data : pd.DataFrame or Dict[str, pd.DataFrame]
        The stored training data. For "global_timestamp" mode, this is a
        single DataFrame. For "per_item" mode, this is a dictionary mapping
        item IDs to their corresponding training DataFrames.
    """

    def __init__(
        self,
        column_config: ColumnConfig = DefaultColumnConfig(),
        mode: Literal["per_item", "global_timestamp"] = "per_item",
    ):
        """
        Initializes the transformer.

        Parameters
        ----------
        column_config : ColumnConfig, optional
            Configuration object for column names, by default DefaultColumnConfig()
        mode : {"per_item", "global_timestamp"}, optional
            Determines how the running index is calculated.
            - "per_item": Index restarts for each item.
            - "global_timestamp": A single index across all items.
            By default "per_item".
        """
        self.mode = mode
        self.train_data = None
        self.timestamp_col_name = column_config.timestamp_col_name
        self.target_col_name = column_config.target_col_name
        self.item_id_col_name = column_config.item_id_col_name

    def fit(self, X, y=None):
        """
        Fit the transformer on the training data.

        Based on the mode, it either stores the entire DataFrame or a
        dictionary of DataFrames split by item_id.

        Parameters
        ----------
        X : pd.DataFrame
            The training data, containing columns specified in column_config.
        y : Ignored

        Returns
        -------
        self
        """
        # --- Assertions to ensure data quality ---
        for col in [
            self.timestamp_col_name,
            self.target_col_name,
            self.item_id_col_name,
        ]:
            assert col is not None, f"{col} must be provided in column_config"
            assert col in X.columns, f"Column '{col}' not found in the DataFrame."

        if self.mode == "per_item":
            self.train_data = {
                group_name: group_data
                for group_name, group_data in X.groupby(self.item_id_col_name)
            }
        elif self.mode == "global_timestamp":
            self.train_data = X.copy()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        return self

    def transform(self, X):
        """
        Transform the DataFrame by adding the running index feature.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the added "running_index" column.
        """
        X = X.copy()

        if self.mode == "per_item":
            all_item_X_out = []
            for group_name, group_data in X.groupby(self.item_id_col_name):
                transformed_group = self._add_running_index(
                    group_data, item_id=group_name
                )
                all_item_X_out.append(transformed_group)
            return pd.concat(all_item_X_out)

        elif self.mode == "global_timestamp":
            return self._add_running_index(X)

        # This case should be caught in __init__, but as a safeguard:
        raise ValueError(f"Invalid mode specified: {self.mode}")

    def _add_running_index(self, X: pd.DataFrame, item_id=None) -> pd.DataFrame:
        """
        Helper function to calculate and add the running index.
        """
        X = X.copy()

        # --- Create a base index starting from 0 for the current data chunk ---
        ts_index = (
            X[[self.timestamp_col_name]]
            .sort_values(by=self.timestamp_col_name)
            .assign(running_index=range(len(X)))
        )
        X = X.join(ts_index["running_index"])

        # --- If data is for prediction (no target), add an offset ---
        # This logic assumes that if all target values are NaN, it's a forecast horizon.
        if X[self.target_col_name].isnull().all():
            offset = 0
            if self.mode == "global_timestamp":
                offset = len(self.train_data)
            elif self.mode == "per_item" and self.train_data is not None:
                # When predicting, an item must exist in the training data to calculate
                # the correct running index offset.
                if item_id not in self.train_data:
                    raise ValueError(
                        f"No fitted training data found for item_id '{item_id}'. "
                        "Cannot create running_index for new items."
                    )
                offset = len(self.train_data[item_id])

            X["running_index"] += offset

        return X
