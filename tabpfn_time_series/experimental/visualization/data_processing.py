from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
from tabpfn_time_series.experimental.utils.wandb import download_wandb_runs


def load_wandb_runs(
    entity: str, project: str, tags: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load experiment runs from Weights & Biases and do preprocessing.

    """

    # Load raw data from Weights & Biases
    raw_df = download_wandb_runs(entity=entity, project=project, tags=tags)

    # Define columns of interest
    interested_columns = [
        "MASE_0.5",
        "wSQL_mean",
        "dataset_full_name",
        "config/config_path",
        "config/dataset",
        "term",
        "state",
        "_timestamp",
        "_runtime",
    ]

    # Filter and process DataFrame
    df = raw_df[interested_columns].copy()
    df = df.dropna()

    # Extract config name from path
    df["config/config_path"] = (
        df["config/config_path"].str.split("/").str[-1].str.replace(".json", "")
    )
    df = df.rename(columns={"config/config_path": "config_name"})

    return df


def rename_config_names(
    df: pd.DataFrame, rename_dict: Dict[str, str], config_col: str = "config_name"
) -> pd.DataFrame:
    """
    Rename configuration names in a DataFrame using a mapping dictionary.

    Args:
        df (pd.DataFrame): Input DataFrame
        rename_dict (Dict[str, str]): Mapping of old names to new names
        config_col (str): Name of the configuration column

    Returns:
        pd.DataFrame: DataFrame with renamed configurations
    """
    df = df.copy()
    df[config_col] = df[config_col].map(rename_dict)
    return df


def load_external_results(
    root_dir: Optional[str] = Path(__file__).parent / "gift-eval-ext-results",
    interested_models: Optional[List[str]] = None,
    use_column_mapping: bool = True,
) -> pd.DataFrame:
    """
    Load external evaluation results from CSV files.

    Args:
        root_dir (str): Root directory containing result files
        interested_models (List[str], optional): List of model names to filter

    Returns:
        pd.DataFrame: Combined results from all CSV files
    """
    root_path = Path(root_dir)
    all_results_files = list(root_path.glob("**/all_results.csv"))

    all_results_dfs = []
    for file_path in all_results_files:
        single_df = pd.read_csv(file_path)
        all_results_dfs.append(single_df)

    if not all_results_dfs:
        return pd.DataFrame()

    combined_results_df = pd.concat(all_results_dfs, ignore_index=True)

    if interested_models:
        combined_results_df = combined_results_df[
            combined_results_df["model"].isin(interested_models)
        ]

    if use_column_mapping:
        combined_results_df = standardize_column_names(combined_results_df)

    return combined_results_df


def standardize_column_names(
    df: pd.DataFrame, column_mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Standardize column names in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        column_mapping (Dict[str, str], optional): Custom column name mapping

    Returns:
        pd.DataFrame: DataFrame with standardized column names
    """
    default_mapping = {
        "dataset": "dataset_full_name",
        "model": "config_name",
        "eval_metrics/MASE[0.5]": "MASE_0.5",
        "eval_metrics/mean_weighted_sum_quantile_loss": "wSQL_mean",
    }

    mapping = column_mapping if column_mapping else default_mapping

    # Keep only the columns that are keys in the mapping
    columns_to_keep = list(mapping.keys())
    df = df[columns_to_keep].copy()

    # Rename the columns according to the mapping
    df = df.rename(columns=mapping)

    # Add term and state columns if they don't exist
    if "term" not in df.columns and "dataset_full_name" in df.columns:
        df["term"] = df["dataset_full_name"].str.split("/").str[-1]
    if "state" not in df.columns:
        df["state"] = "finished"

    return df


def get_common_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the minimum common set of datasets that have completed experiments for all configurations.

    Args:
        df (pd.DataFrame): DataFrame containing experiment results

    Returns:
        pd.DataFrame: Filtered DataFrame with only datasets that have all configurations completed
    """
    # Get completed experiments
    completed_df = df[df["state"] == "finished"]

    # Get unique configurations
    unique_configs = df["config_name"].unique()

    # Count completed configurations per dataset
    dataset_completion = completed_df.groupby("dataset_full_name")[
        "config_name"
    ].nunique()

    # Find datasets with all configurations completed
    complete_datasets = dataset_completion[
        dataset_completion == len(unique_configs)
    ].index.tolist()

    print(
        f"Number of datasets with all configurations completed: {len(complete_datasets)}"
    )

    # Filter to only include complete datasets
    common_set_df = df[df["dataset_full_name"].isin(complete_datasets)]

    # Verify configuration counts
    config_counts = (
        common_set_df[common_set_df["state"] == "finished"]
        .groupby("config_name")
        .size()
    )
    print("\nConfigurations per dataset:")
    print(config_counts)

    return common_set_df
