import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autogluon.timeseries import TimeSeriesDataFrame


def is_subset(tsdf_A: TimeSeriesDataFrame, tsdf_B: TimeSeriesDataFrame) -> bool:
    tsdf_index_set_A, tsdf_index_set_B = set(tsdf_A.index), set(tsdf_B.index)
    return tsdf_index_set_A.issubset(tsdf_index_set_B)


def plot_time_series(
    df: TimeSeriesDataFrame,
    item_ids: list[int] | None = None,
    in_single_plot: bool = False,
    y_limit: tuple[float, float] | None = None,
    show_points: bool = False,
    target_col: str = "target",
):
    if item_ids is None:
        item_ids = df.index.get_level_values("item_id").unique()
    elif not set(item_ids).issubset(df.index.get_level_values("item_id").unique()):
        raise ValueError(f"Item IDs {item_ids} not found in the dataframe")

    if not in_single_plot:
        # create subplots
        fig, axes = plt.subplots(
            len(item_ids), 1, figsize=(10, 3 * len(item_ids)), sharex=True
        )

        if len(item_ids) == 1:
            axes = [axes]

        for ax, item_id in zip(axes, item_ids):
            df_item = df.xs(item_id, level="item_id")
            ax.plot(df_item.index, df_item[target_col])
            if show_points:
                ax.scatter(
                    df_item.index,
                    df_item[target_col],
                    color="lightcoral",
                    s=8,
                    alpha=0.8,
                )
            ax.set_title(f"Item ID: {item_id}")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Target")
            if y_limit is not None:
                ax.set_ylim(*y_limit)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        for item_id in item_ids:
            df_item = df.xs(item_id, level="item_id")
            ax.plot(df_item.index, df_item[target_col], label=f"Item ID: {item_id}")
            if show_points:
                ax.scatter(
                    df_item.index,
                    df_item[target_col],
                    color="lightcoral",
                    s=8,
                    alpha=0.8,
                )
        ax.legend()
        if y_limit is not None:
            ax.set_ylim(*y_limit)

    plt.tight_layout()
    plt.show()


def plot_actual_ts(
    train: TimeSeriesDataFrame,
    test: TimeSeriesDataFrame,
    item_ids: list[int] | None = None,
    show_points: bool = False,
):
    if item_ids is None:
        item_ids = train.index.get_level_values("item_id").unique()
    elif not set(item_ids).issubset(train.index.get_level_values("item_id").unique()):
        raise ValueError(f"Item IDs {item_ids} not found in the dataframe")

    _, ax = plt.subplots(len(item_ids), 1, figsize=(10, 3 * len(item_ids)))
    ax = [ax] if not isinstance(ax, np.ndarray) else ax

    def plot_single_item(ax, item_id):
        train_item = train.xs(item_id, level="item_id")
        test_item = test.xs(item_id, level="item_id")

        if is_subset(train_item, test_item):
            ground_truth = test_item["target"]
        else:
            ground_truth = pd.concat([train_item[["target"]], test_item[["target"]]])
        ax.plot(ground_truth.index, ground_truth, label="Ground Truth")
        if show_points:
            ax.scatter(
                ground_truth.index, ground_truth, color="lightblue", s=8, alpha=0.8
            )

        train_item_length = train.xs(item_id, level="item_id").iloc[-1].name
        ax.axvline(
            x=train_item_length, color="r", linestyle="--", label="Train/Test Split"
        )

        ax.set_title(f"Item ID: {item_id}")
        ax.legend()

    for i, item_id in enumerate(item_ids):
        plot_single_item(ax[i], item_id)

    plt.tight_layout()
    plt.show()


def plot_pred_and_actual_ts(
    pred: TimeSeriesDataFrame,
    train: TimeSeriesDataFrame,
    test: TimeSeriesDataFrame,
    item_ids: list[int] | None = None,
    show_quantiles: bool = True,
    show_points: bool = False,
    save_path: str | None = None,
    plot_config: dict | None = None,
):
    """
    Plot predictions and actual time series data with configurable styling.

    Args:
        pred: Prediction TimeSeriesDataFrame
        train: Training TimeSeriesDataFrame
        test: Test TimeSeriesDataFrame
        item_ids: List of item IDs to plot, defaults to all items
        show_quantiles: Whether to show prediction quantiles
        show_points: Whether to show data points
        save_path: Path to save the figure
        plot_config: Dictionary of plot configuration parameters
    """
    # Default configuration
    default_config = {
        "figsize": (20, 6),  # Base figure size per item
        "linewidth": 2,
        "fontsize": 12,
        "title_fontsize": 14,
        "title_fontweight": "bold",
        "legend_fontsize": 12,
        "fontweight": "normal",
        "grid": True,
        "colors": {
            "ground_truth": "blue",
            "prediction": "green",
            "quantile_fill": "gray",
            "split_line": "red",
            "points": "lightblue",
        },
        "alpha": {"quantile_fill": 0.2, "points": 0.8},
        "point_size": 8,
    }

    # Update with user configuration if provided
    if plot_config:
        for category, values in plot_config.items():
            if (
                isinstance(values, dict)
                and category in default_config
                and isinstance(default_config[category], dict)
            ):
                default_config[category].update(values)
            else:
                default_config[category] = values

    config = default_config

    if item_ids is None:
        item_ids = train.index.get_level_values("item_id").unique()
    elif not set(item_ids).issubset(train.index.get_level_values("item_id").unique()):
        raise ValueError(f"Item IDs {item_ids} not found in the dataframe")

    if pred.shape[0] != test.shape[0]:
        if not is_subset(pred, test):
            raise ValueError(
                "Pred and Test have different number of items and Pred is not a subset of Test"
            )

        filled_pred = test.copy()
        filled_pred["target"] = np.nan
        for col in pred.columns:
            filled_pred.loc[pred.index, col] = pred[col]
        pred = filled_pred

    assert pred.shape[0] == test.shape[0]

    # Calculate figure size based on number of items
    figsize = (config["figsize"][0], config["figsize"][1] * len(item_ids))
    _, ax = plt.subplots(len(item_ids), 1, figsize=figsize)
    ax = [ax] if not isinstance(ax, np.ndarray) else ax

    def plot_single_item(ax, item_id):
        pred_item = pred.xs(item_id, level="item_id")
        train_item = train.xs(item_id, level="item_id")
        test_item = test.xs(item_id, level="item_id")

        if is_subset(train_item, test_item):
            ground_truth = test_item["target"]
        else:
            ground_truth = pd.concat([train_item[["target"]], test_item[["target"]]])

        # Plot ground truth
        ax.plot(
            ground_truth.index,
            ground_truth,
            label="Ground Truth",
            linewidth=config["linewidth"],
            color=config["colors"]["ground_truth"],
        )

        # Plot prediction
        ax.plot(
            pred_item.index,
            pred_item["target"],
            label="Prediction",
            linewidth=config["linewidth"],
            color=config["colors"]["prediction"],
        )

        # Plot points if requested
        if show_points:
            ax.scatter(
                ground_truth.index,
                ground_truth,
                color=config["colors"]["points"],
                s=config["point_size"],
                alpha=config["alpha"]["points"],
            )

        # Plot quantiles if requested
        if show_quantiles:
            quantile_config = sorted(
                pred_item.columns.drop(["target"]).tolist(), key=lambda x: float(x)
            )
            if quantile_config:  # Only if quantiles exist
                lower_quantile = quantile_config[0]
                upper_quantile = quantile_config[-1]
                ax.fill_between(
                    pred_item.index,
                    pred_item[lower_quantile],
                    pred_item[upper_quantile],
                    color=config["colors"]["quantile_fill"],
                    alpha=config["alpha"]["quantile_fill"],
                    label=f"{lower_quantile}-{upper_quantile} Quantile Range",
                )

        # Plot train/test split line
        train_item_length = train.xs(item_id, level="item_id").iloc[-1].name
        ax.axvline(
            x=train_item_length,
            color=config["colors"]["split_line"],
            linestyle="--",
            label="Train/Test Split",
            linewidth=config["linewidth"],
        )

        # Set title and legend
        ax.set_title(
            f"Item ID: {item_id}",
            fontsize=config["title_fontsize"],
            fontweight=config["title_fontweight"],
        )
        ax.legend(
            loc="upper left", bbox_to_anchor=(0, 1), fontsize=config["legend_fontsize"]
        )

        # Set font size and weight for tick labels
        ax.tick_params(axis="both", which="major", labelsize=config["fontsize"])
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight(config["fontweight"])

        # Add grid if configured
        if config["grid"]:
            ax.grid(True, linestyle="--", alpha=0.7)

    for i, item_id in enumerate(item_ids):
        plot_single_item(ax[i], item_id)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()
