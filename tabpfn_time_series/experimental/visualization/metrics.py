import numpy as np
import pandas as pd
import scipy.stats as st


def gmean_and_ci(x: pd.Series, confidence: float = 0.95) -> pd.Series:
    """
    Calculate geometric mean and confidence intervals for a series of values.

    Args:
        x (pd.Series): Input series of positive values
        confidence (float): Confidence level for intervals (default: 0.95)

    Returns:
        pd.Series: Series containing mean, ci_lower, and ci_upper values

    Raises:
        ValueError: If input contains negative values
    """
    if np.any(x < 0):
        raise ValueError(
            "Input contains negative values which are not valid for geometric mean"
        )

    # If all values are the same, return the value
    if x.nunique() == 1:
        return pd.Series(
            {"mean": x.values[0], "ci_lower": x.values[0], "ci_upper": x.values[0]}
        )

    gmean = st.gmean(x)

    # Calculate confidence intervals
    log_x = np.log(x[x > 0])
    se = st.sem(log_x)
    ci = np.exp(st.t.interval(confidence, len(log_x) - 1, loc=np.mean(log_x), scale=se))

    return pd.Series({"mean": gmean, "ci_lower": ci[0], "ci_upper": ci[1]})


def amean_and_ci(x: pd.Series, confidence: float = 0.95) -> pd.Series:
    """
    Calculate arithmetic mean and confidence intervals for a series of values.

    Args:
        x (pd.Series): Input series of values
        confidence (float): Confidence level for intervals (default: 0.95)

    Returns:
        pd.Series: Series containing mean, ci_lower, and ci_upper values
    """
    # If all values are the same, return the value
    if x.nunique() == 1:
        return pd.Series(
            {"mean": x.values[0], "ci_lower": x.values[0], "ci_upper": x.values[0]}
        )

    amean = np.mean(x)

    # Calculate confidence intervals
    se = st.sem(x)
    ci = st.t.interval(confidence, len(x) - 1, loc=amean, scale=se)

    return pd.Series({"mean": amean, "ci_lower": ci[0], "ci_upper": ci[1]})


def normalize_metric_by_baseline(
    df: pd.DataFrame,
    metric: str = "MASE_0.5",
    baseline_model: str = "Seasonal_Naive",
    model_col: str = "config_name",
    dataset_col: str = "dataset_full_name",
) -> pd.DataFrame:
    """
    Normalize a metric by dividing each model's value by the baseline model's value for each dataset.

    Args:
        df (pd.DataFrame): The dataframe containing the results
        metric (str): The metric column to normalize (default: 'MASE_0.5')
        baseline_model (str): The model to use as baseline (default: 'Seasonal_Naive')
        model_col (str): Name of the column containing model names (default: 'config_name')
        dataset_col (str): Name of the column containing dataset names (default: 'dataset_full_name')

    Returns:
        pd.DataFrame: A dataframe with normalized metric values
    """
    # Create pivot table with datasets as rows and models as columns
    df_pivot = df.pivot_table(
        index=dataset_col,
        columns=model_col,
        values=metric,
    )

    # Divide each model's metric by the baseline model's metric
    normalized_pivot = df_pivot.div(df_pivot[baseline_model], axis=0)

    # Convert back to long format
    normalized_df = normalized_pivot.reset_index().melt(
        id_vars=dataset_col, var_name=model_col, value_name=metric
    )

    return normalized_df


def compute_ranking(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """
    Compute ranking of models based on a metric.

    Args:
        df (pd.DataFrame): Input dataframe
        metric_column (str): Name of the column to rank by

    Returns:
        pd.DataFrame: DataFrame with additional 'rank' column
    """

    def compute_single_dataset_ranking(
        df: pd.DataFrame, metric_column: str
    ) -> pd.DataFrame:
        """
        Compute ranking of models based on a metric for a single dataset.
        """
        df_copy = df.copy()
        df_copy["rank"] = df_copy[metric_column].rank()
        return df_copy

    df_copy = df.copy()

    # Compute rankings for each individual dataset based on metric_column
    rankings = (
        df_copy.groupby("dataset_full_name")
        .apply(lambda x: compute_single_dataset_ranking(x, metric_column))[
            ["config_name", "dataset_full_name", "rank"]
        ]
        .reset_index(drop=True)
    )

    return rankings
