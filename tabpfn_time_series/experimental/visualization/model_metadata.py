"""
Functions for visualizing model comparisons with categorization and styling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple

# Model categorization
MODEL_CATEGORIES = {
    "tabular_foundation": {
        "name": "Tabular Foundation Model",
        "color": "dodgerblue",
        "models": ["TabPFN-TS"],
    },
    "tabular_classical": {
        "name": "Tabular Classical Model",
        "color": "red",
        "models": ["CatBoost-TS"],
    },
    "time_series_foundation": {
        "name": "Time Series Foundation Model",
        "color": "skyblue",
        "models": [
            "timesfm_2_0_500m",
            "chronos-bolt-small",
            "chronos_bolt_base",
            "chronos-bolt-tiny",
        ],
    },
    "deep_learning": {
        "name": "Deep Learning Time Series Model",
        "color": "lightgreen",
        "models": ["DeepAR", "TFT", "PatchTST"],
    },
    "statistical": {
        "name": "Statistical Time Series Model",
        "color": "violet",
        "models": ["Seasonal_Naive", "Auto_Arima", "Auto_Theta"],
    },
}

# Pretty names mapping
MODEL_PRETTY_NAMES = {
    "TabPFN-TS": "TabPFN-TS",
    "timesfm_2_0_500m": "TimesFM-2.0-500m",
    "chronos-bolt-small": "Chronos-Bolt-Small",
    "chronos_bolt_base": "Chronos-Bolt-Base",
    "chronos-bolt-tiny": "Chronos-Bolt-Tiny",
    "DeepAR": "DeepAR",
    "TFT": "TFT",
    "PatchTST": "PatchTST",
    "Seasonal_Naive": "Seasonal-Naive",
    "Auto_Arima": "Auto-Arima",
    "Auto_Theta": "Auto-Theta",
    "CatBoost-TS": "CatBoost-TS",
}


def get_model_category(model_name: str) -> str:
    """
    Get the category name for a given model.

    Args:
        model_name (str): Name of the model

    Returns:
        str: Category name or 'Unknown'
    """
    for category, info in MODEL_CATEGORIES.items():
        if model_name in info["models"]:
            return info["name"]
    return "Unknown"


def get_category_color(category_name: str) -> str:
    """
    Get the color for a given category.

    Args:
        category_name (str): Name of the category

    Returns:
        str: Color code
    """
    for info in MODEL_CATEGORIES.values():
        if info["name"] == category_name:
            return info["color"]
    return "gray"


def add_model_metadata(
    df: pd.DataFrame, index_col: str = "config_name", use_pretty_names: bool = True
) -> pd.DataFrame:
    """
    Add model type and pretty name information to a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame
        index_col (str): Name of the column containing model names
        use_pretty_names (bool): Whether to use pretty names for models

    Returns:
        pd.DataFrame: DataFrame with added metadata
    """
    df = df.copy()

    # Add model type information
    model_types = {idx: get_model_category(idx) for idx in df.index}
    df["model_type"] = pd.Series(model_types)

    # Create pretty names version if requested
    if use_pretty_names:
        df_pretty = df.copy()
        df_pretty.index = [MODEL_PRETTY_NAMES.get(idx, idx) for idx in df.index]
        return df_pretty

    return df


def plot_model_comparison(
    data: pd.DataFrame,
    metric_name: str,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 7),
    font_size: int = 18,
    font_weight: str = "semibold",
    y_offset: float = 0.25,
    spine_line_width: float = 0.8,
    sort_by: str = "mean",
    ascending: bool = True,
    use_pretty_names: bool = True,
    show_legend: bool = True,
    legend_kwargs: Optional[Dict] = None,
) -> plt.Axes:
    """
    Create a bar plot comparing models with error bars and styling.

    Args:
        data (pd.DataFrame): DataFrame with model comparison data
        metric_name (str): Name of the metric being plotted
        ax (plt.Axes, optional): Existing axes to plot on
        figsize (Tuple[float, float]): Figure size if creating new axes
        font_size (int): Base font size
        font_weight (str): Font weight for text
        y_offset (float): Vertical offset for error bars
        spine_line_width (float): Line width for plot spines
        sort_by (str): Column to sort by
        ascending (bool): Sort order
        use_pretty_names (bool): Whether to use pretty names for models
        show_legend (bool): Whether to show the legend
        legend_kwargs (Dict, optional): Additional legend parameters

    Returns:
        plt.Axes: The axes object with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.set_style("whitegrid")

    # Sort and add metadata
    data_sorted = data.sort_values(sort_by, ascending=ascending)
    data_with_meta = add_model_metadata(data_sorted, use_pretty_names=use_pretty_names)

    # Get colors for each category
    colors = {cat["name"]: cat["color"] for cat in MODEL_CATEGORIES.values()}

    # Create bar plot
    sns.barplot(
        x="mean",
        y=data_with_meta.index,
        data=data_with_meta,
        ax=ax,
        palette=colors,
        hue="model_type",
        dodge=False,
    )

    # Add error bars
    ax.errorbar(
        x=data_with_meta["mean"],
        y=np.arange(len(data_with_meta)) + y_offset,
        xerr=[
            data_with_meta["mean"] - data_with_meta["ci_lower"],
            data_with_meta["ci_upper"] - data_with_meta["mean"],
        ],
        fmt="none",
        ecolor="black",
        capsize=2,
        linewidth=1,
    )

    # Add value annotations
    for i, v in enumerate(data_with_meta["mean"]):
        ax.text(
            v + 0.02,
            i,
            f"{v:.3f}",
            va="center",
            fontsize=font_size - 2,
            weight=font_weight,
        )

    # Customize appearance
    ax.set_xlabel(
        f"Relative {metric_name} (lower is better)",
        fontsize=font_size,
        weight=font_weight,
    )
    ax.set_ylabel("", fontsize=font_size, weight=font_weight)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(spine_line_width)
        spine.set_color("black")

    # Customize tick labels
    ax.tick_params(axis="both", which="major", labelsize=font_size - 2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight(font_weight)

    # Remove grid
    ax.grid(False)

    if show_legend:
        # Create legend with ordered categories
        handles, labels = ax.get_legend_handles_labels()
        ordered_types = [cat["name"] for cat in MODEL_CATEGORIES.values()]

        # Filter to only include categories present in the data
        present_types = []
        present_handles = []
        for model_type, handle in zip(labels, handles):
            if model_type in ordered_types:
                present_types.append(model_type)
                present_handles.append(handle)

        # Set default legend kwargs
        default_legend_kwargs = {
            "loc": "upper right",
            "ncol": 1,
            "fontsize": font_size - 6.5,
            "frameon": True,
            "borderaxespad": 0.3,
            "handletextpad": 0.3,
        }
        if legend_kwargs:
            default_legend_kwargs.update(legend_kwargs)

        ax.legend(present_handles, present_types, **default_legend_kwargs)
    else:
        ax.get_legend().remove()

    return ax
