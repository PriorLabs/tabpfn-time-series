import pandas as pd
import logging
from joblib import Parallel, delayed
from typing import Literal, Optional, Any


from .pipeline_configs import ColumnConfig, DefaultColumnConfig
from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class RunningIndexFeatureTransformer(BaseFeatureTransformer):
    """
    A transformer that adds a running index feature to a DataFrame.

    This feature can be calculated in two ways: globally across all timestamps
    or on a per-item basis, restarting the index for each unique item. The
    transformer is designed to work within a scikit-learn pipeline and
    differentiates between training and prediction data to apply correct offsets.

    Attributes
    ----------
    mode : Literal["per_item", "global_timestamp"]
        The mode of operation. "per_item" calculates the index per unique
        item, while "global_timestamp" calculates a single index across all
        data.
    train_data : Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
        The stored training data, used to calculate the index offset for
        prediction sets. For "global_timestamp" mode, this is a single
        DataFrame. For "per_item" mode, this is a dictionary mapping
        item IDs to their corresponding training DataFrames.
    timestamp_col_name : str
        The name of the timestamp column.
    target_col_name : str
        The name of the target variable column.
    item_id_col_name : str
        The name of the item identifier column.

    Note
    ----------
    - This transformer distinguishes between training and testing (prediction) data by checking the target column.
      The testing data's target values are always NaNs.
    """

    def __init__(
        self,
        column_config: ColumnConfig = DefaultColumnConfig(),
        mode: Literal["per_item", "global_timestamp"] = "per_item",
        n_jobs: int = -1,
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
        n_jobs : int, optional
            The number of jobs to run in parallel for "per_item" mode.
            -1 means using all available processors. By default -1.
        """
        super().__init__(column_config)
        self.mode = mode
        self.n_jobs = n_jobs
        self.train_data = None
        self._required_columns = [
            "timestamp_col_name",
            "target_col_name",
            "item_id_col_name",
        ]

    def fit(
        self, X: pd.DataFrame, y: Optional[Any] = None
    ) -> "RunningIndexFeatureTransformer":
        """
        Fit the transformer on the training data.

        Based on the mode, it either stores the entire DataFrame or a
        dictionary of DataFrames split by item_id.

        Parameters
        ----------
        X : pd.DataFrame
            The training data, containing columns specified in column_config.
        y : Any, optional
            Ignored. This parameter exists for scikit-learn compatibility.
            By default None.

        Returns
        -------
        self : RunningIndexFeatureTransformer
            The fitted transformer instance.
        """
        super().fit(X, y)  # validate the data

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

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
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
            # The list of processed DataFrames is generated in parallel
            all_item_X_out = Parallel(n_jobs=self.n_jobs)(
                delayed(self._add_running_index)(group_data, item_id=group_name)
                for group_name, group_data in X.groupby(self.item_id_col_name)
            )
            return pd.concat(all_item_X_out).reindex(X.index)

        elif self.mode == "global_timestamp":
            return self._add_running_index(X)

        # This case should be caught in __init__, but as a safeguard:
        raise ValueError(f"Invalid mode specified: {self.mode}")

    def _add_running_index(
        self, X: pd.DataFrame, item_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Helper function to calculate and add the running index to a DataFrame.

        This function sorts the data by timestamp, computes a sequential index,
        and applies an offset if the data is identified as a prediction set
        (i.e., target column is all NaN).

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame for a single item or the entire dataset.
        item_id : str, optional
            The item identifier, required when `mode` is "per_item" to
            calculate the correct offset for prediction data. By default None.

        Returns
        -------
        pd.DataFrame
            The DataFrame with the "running_index" column added.
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
        # Differentiate train/test sets by checking the target column.
        # This logic assumes the test set's target values are always NaN.
        if X[self.target_col_name].isnull().all():
            # This block is for the testing set, whose target values are always NaNs.
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
