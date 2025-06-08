import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from typing import List, Optional, Tuple, Dict, Union
from tabpfn_time_series.experimental.visualization.model_metadata import (
    add_model_metadata,
    MODEL_CATEGORIES,
)


def plot_experiment_status(
    df: pd.DataFrame,
    all_datasets_names: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """
    Create a comprehensive experiment status visualization.

    Args:
        df (pd.DataFrame): DataFrame with columns: dataset_full_name, config_name, state
            Optional columns: _timestamp, _runtime
        all_datasets_names (List[str], optional): List of all dataset names to include
        figsize (Tuple[float, float], optional): Custom figure size (width, height)

    Returns:
        plt.Figure: The generated figure object
    """
    # Get all unique datasets and configurations
    if all_datasets_names is None:
        all_datasets = df["dataset_full_name"].unique()
    else:
        all_datasets = all_datasets_names
    all_configs = df["config_name"].unique()

    # Create a multi-index for all possible combinations
    index = pd.MultiIndex.from_product(
        [all_datasets, all_configs], names=["dataset", "config"]
    )

    # Create an empty DataFrame with the multi-index
    status_df = pd.DataFrame(index=index)
    status_df["state"] = "missing"  # Default state
    status_df["runtime"] = np.nan  # Default runtime

    # Handle duplicates by keeping the latest result
    if "_timestamp" in df.columns:
        df_sorted = df.sort_values("_timestamp", ascending=False)
    elif "_runtime" in df.columns:
        df_sorted = df.sort_values("_runtime", ascending=False)
    else:
        df_sorted = df
    df_deduped = df_sorted.drop_duplicates(subset=["dataset_full_name", "config_name"])

    # Update states and runtimes
    for _, row in df_deduped.iterrows():
        if pd.notna(row["dataset_full_name"]) and pd.notna(row["config_name"]):
            idx = (row["dataset_full_name"], row["config_name"])
            if idx in status_df.index:
                status_df.loc[idx, "state"] = row["state"]
                if "_runtime" in df.columns and pd.notna(row["_runtime"]):
                    status_df.loc[idx, "runtime"] = row["_runtime"]

    # Reshape for visualization
    heatmap_df = status_df.reset_index().pivot_table(
        index="dataset", columns="config", values="state", aggfunc="first"
    )

    # Create runtime heatmap for finished experiments
    runtime_df = (
        status_df[status_df.state == "finished"]
        .reset_index()
        .pivot_table(
            index="dataset", columns="config", values="runtime", aggfunc="first"
        )
    )

    # Create numeric heatmap
    state_mapping = {"missing": 0, "failed": 1, "running": 2, "finished": 3}
    numeric_heatmap = heatmap_df.map(lambda x: state_mapping.get(x, -1))

    # Define colors and create colormap
    colors = ["#d9d9d9", "#ffcccc", "#aed6f1", "#a8e6cf"]
    cmap = ListedColormap(colors)

    # Calculate figure size if not provided
    if figsize is None:
        width = max(10, len(all_configs) * 1.5)
        height = max(8, len(all_datasets) * 0.2)
        figsize = (width, height)

    # Create plot
    fig = plt.figure(figsize=figsize)
    ax = sns.heatmap(
        numeric_heatmap,
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        vmin=0,
        vmax=3,
    )

    # Add runtime annotations
    for i, dataset in enumerate(heatmap_df.index):
        for j, config in enumerate(heatmap_df.columns):
            runtime = (
                runtime_df.loc[dataset, config]
                if dataset in runtime_df.index and config in runtime_df.columns
                else np.nan
            )
            if pd.notna(runtime):
                runtime_text = (
                    f"{runtime/60:.1f}m" if runtime >= 60 else f"{runtime:.0f}s"
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    runtime_text,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    # Add legend
    legend_elements = [
        Patch(facecolor=colors[0], label="Missing"),
        Patch(facecolor=colors[1], label="Failed"),
        Patch(facecolor=colors[2], label="Running"),
        Patch(facecolor=colors[3], label="Finished"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.title("Experiment Status by Dataset and Configuration", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def plot_metric_comparison(
    mean_ci_data: pd.DataFrame,
    xaxis_label: str,
    feature_groups: List[str],
    ax: Optional[plt.Axes] = None,
    font_size: int = 18,
    font_weight: str = "semibold",
    y_offset: float = 0.25,
    spine_line_width: float = 0.8,
    xlim: Tuple[float, float] = (0, 1.2),
    palette: Optional[Union[str, List[str]]] = None,
) -> plt.Axes:
    """
    Create a bar plot comparing metric values across different feature groups.

    Args:
        mean_ci_data (pd.DataFrame): DataFrame with mean and CI values
        xaxis_label (str): Label for the x-axis
        feature_groups (List[str]): List of feature groups to plot
        ax (plt.Axes, optional): Existing axes to plot on
        font_size (int): Base font size for the plot
        font_weight (str): Font weight for text elements
        y_offset (float): Vertical offset for error bars
        spine_line_width (float): Line width for plot spines
        xlim (Tuple[float, float]): X-axis limits
        palette (Union[str, List[str]], optional): Color palette for the bars

    Returns:
        plt.Axes: The axes object with the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5))

    sns.set_style("whitegrid")

    # Define color palette
    if palette is None:
        palette = sns.color_palette("hls", len(feature_groups))
    elif isinstance(palette, str):
        palette = sns.color_palette(palette, len(feature_groups))
    color_mapping = dict(zip(feature_groups, palette))

    # Reorder the dataframe according to feature_groups
    mean_ci_data = mean_ci_data.reindex(feature_groups)

    # Create bar plot
    sns.barplot(
        x="mean",
        y=mean_ci_data.index,
        data=mean_ci_data,
        ax=ax,
        palette=color_mapping,
        hue=mean_ci_data.index,
        legend=False,
    )

    # Add error bars
    ax.errorbar(
        x=mean_ci_data["mean"],
        y=np.arange(len(mean_ci_data)) + y_offset,
        xerr=[
            mean_ci_data["mean"] - mean_ci_data["ci_lower"],
            mean_ci_data["ci_upper"] - mean_ci_data["mean"],
        ],
        fmt="none",
        ecolor="black",
        capsize=2,
        linewidth=1,
    )

    # Add value annotations
    for i, v in enumerate(mean_ci_data["mean"]):
        ax.text(
            v + 0.02,
            i,
            f"{v:.3f}",
            va="center",
            fontsize=font_size - 2,
            weight=font_weight,
        )

    # Customize appearance
    ax.set_xlabel(xaxis_label, fontsize=font_size, weight=font_weight)
    ax.set_ylabel("", fontsize=font_size, weight=font_weight)
    ax.set_xlim(xlim)

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

    return ax


def plot_metric_comparison_with_model_metadata(
    data: pd.DataFrame,
    xaxis_label: str,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (10, 7),
    font_size: int = 18,
    font_weight: str = "semibold",
    y_offset: float = 0.25,
    spine_line_width: float = 0.8,
    order: List[str] = None,
    use_pretty_names: bool = True,
    show_legend: bool = True,
    legend_kwargs: Optional[Dict] = None,
    xlim: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Axes, Optional[Tuple[List, List]]]:
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
        order (List[str]): List of models to order by
        use_pretty_names (bool): Whether to use pretty names for models
        show_legend (bool): Whether to show the legend
        legend_kwargs (Dict, optional): Additional legend parameters
        xlim (Tuple[float, float], optional): x-axis limits (min, max)

    Returns:
        Tuple[plt.Axes, Optional[Tuple[List, List]]]: The axes object with the plot and optionally legend handles and labels
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.set_style("whitegrid")

    # Sort and add metadata
    if order:
        data = data.reindex(order)
    data_with_meta = add_model_metadata(data, use_pretty_names=use_pretty_names)

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
    ax.set_xlabel(xaxis_label, fontsize=font_size, weight=font_weight)
    ax.set_ylabel("", fontsize=font_size, weight=font_weight)

    # Set x-axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)

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

    # Get legend handles and labels
    handles, labels = ax.get_legend_handles_labels()
    ordered_types = [cat["name"] for cat in MODEL_CATEGORIES.values()]

    # Filter to only include categories present in the data
    present_types = []
    present_handles = []
    for model_type, handle in zip(labels, handles):
        if model_type in ordered_types:
            present_types.append(model_type)
            present_handles.append(handle)

    if show_legend:
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

    return ax, (present_handles, present_types) if present_handles else None


def save_plot_to_pdf(
    fig: plt.Figure,
    filename: str,
    output_dir: Optional[str] = Path(__file__).parent / "output",
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    """
    Save a matplotlib figure to a PDF file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches=bbox_inches)
