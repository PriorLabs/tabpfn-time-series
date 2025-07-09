import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Any

from .pipeline_configs import ColumnConfig, DefaultColumnConfig


class BaseFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for feature transformers in the project.

    This class provides a common structure for transformers, including:
    - Initialization with column configuration.
    - A data validation method to ensure required columns are present.
    """

    def __init__(
        self,
        column_config: ColumnConfig = DefaultColumnConfig(),
    ):
        """
        Initializes the base transformer.

        Parameters
        ----------
        column_config : ColumnConfig, optional
            Configuration object specifying column names.
            Defaults to DefaultColumnConfig().
        """
        self.timestamp_col_name = column_config.timestamp_col_name
        self.target_col_name = column_config.target_col_name
        self.item_id_col_name = column_config.item_id_col_name
        self._required_columns: List[str] = []

    def _validate_data(self, X: pd.DataFrame) -> None:
        """
        Validates the input DataFrame `X` against `self._required_columns`.

        Raises
        ------
        ValueError
            If a required column name attribute is not set on the instance,
            or if the column itself is not found in the DataFrame.
        """
        for col_attr in self._required_columns:
            col_name = getattr(self, col_attr, None)
            if col_name is None:
                raise ValueError(
                    f"Attribute '{col_attr}' is not set. "
                    "It should be provided via `column_config`."
                )
            if col_name not in X.columns:
                raise ValueError(
                    f"Required column '{col_name}' not found in DataFrame."
                )

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "BaseFeatureTransformer":
        """
        Fits the transformer by validating the input data.

        Subclasses should call `super().fit(X, y)` before their own
        fitting logic.

        Parameters
        ----------
        X : pd.DataFrame
            The input training data.
        y : Optional[Any], optional
            Ignored. For scikit-learn compatibility. By default None.

        Returns
        -------
        BaseFeatureTransformer
            The fitted transformer instance.
        """
        self._validate_data(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data.

        This method must be implemented by subclasses.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the transform method.")
