from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import copy
import logging


logger = logging.getLogger(__name__)

from dataclasses import dataclass


@dataclass
class ColumnConfig:
    """
    Base class for column configuration.
    """

    timestamp_col_name: str = None
    target_col_name: str = None
    item_id_col_name: str = None


@dataclass
class DefaultColumnConfig(ColumnConfig):
    """
    Default column configuration.
    """

    timestamp_col_name: str = "timestamp"
    target_col_name: str = "target"
    item_id_col_name: str = "item_id"


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer that fits and applies a separate pipeline
    to different groups of data within a pandas DataFrame.

    Parameters
    ----------
    pipeline_steps : list of tuples
        A list of (name, transformer) tuples that define the steps of the
        pipeline to be applied to each group. For example:
        `[('scaler', StandardScaler()), ('poly', PolynomialFeatures())]`.

    group_by_column : str, default="item_id"
        The name of the column in the input DataFrame `X` to group by. A
        separate pipeline will be fitted for each unique value in this column.
        If `None`, a single global pipeline is fitted on the entire dataset.

    Attributes
    ----------
    fitted_pipelines_ : dict
        A dictionary to store the fitted pipeline for each group. The keys are
        the unique group names, and the values are the corresponding fitted
        `Pipeline` objects. If `group_by_column` is `None`, the dictionary
        contains a single entry with the key "__global__".

    timestamp_col_name : str
        The name of the column in the input DataFrame `X` that contains the timestamps.
    target_col_name : str
        The name of the column in the input DataFrame `X` that contains the target values.
    item_id_col_name : str
        The name of the column in the input DataFrame `X` that contains the item IDs.
    """

    def __init__(self, pipeline_steps):
        """
        Initializes the FeatureTransformer.

        Args:
            pipeline_steps (list): A list of tuples, where each tuple contains the
                                   name of the step and the transformer instance,
                                   e.g., [('step_name', TransformerObject())].
            group_by_column (str): The column name to group the data by. If None,
                                   a single pipeline is used for all data.
        """
        # Create an unfitted template pipeline from the provided steps.
        # We will clone this template for each group.
        self.pipeline_steps = pipeline_steps
        # self.group_by_column = group_by_column
        self._template_pipeline = Pipeline(steps=self.pipeline_steps)
        self.fitted_pipelines_ = {}  # Dictionary to store a fitted pipeline for each group

    def fit(self, X, y=None):
        """
        Fits a separate pipeline for each group defined by `group_by_column`.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the pipelines on. Must contain the column
            specified in `group_by_column` if it is not None.

        y : pd.Series or np.array, optional (default=None)
            The target values, passed to the fit method of the underlying
            pipelines.

        Returns
        -------
        self : FeatureTransformer
            The fitted transformer instance.
        """
        # Reset fitted pipelines on each call to fit
        self.fitted_pipelines_ = {}

        pipeline_for_item = copy.deepcopy(self._template_pipeline)
        pipeline_for_item.fit(X, y)
        self.fitted_pipelines_["__global__"] = pipeline_for_item

        return self

    def transform(self, X):
        """
        Transforms the data using the appropriate fitted pipeline for each group.

        Parameters
        ----------
        X : pd.DataFrame
            The data to transform. Must contain the `group_by_column`.

        Returns
        -------
        pd.DataFrame
            The transformed data, with the same index as the input `X`.

        Raises
        ------
        RuntimeError
            If `transform` is called before `fit` when no grouping is used.
        """

        global_pipeline = self.fitted_pipelines_.get("__global__")
        if global_pipeline:
            return global_pipeline.transform(X)
        else:
            raise RuntimeError("Transformer has not been fitted yet.")
