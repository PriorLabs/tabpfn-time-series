import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_learning_curves(
    train_losses, val_epochs, val_maes, val_q_losses, output_dir: Path
):
    """
    Plots the learning curves for training loss and validation metrics, using three separate y-axes.
    """
    train_epochs = range(len(train_losses))

    fig, ax1 = plt.subplots(figsize=(18, 7))
    fig.subplots_adjust(right=0.75)

    # --- Axis 1: Training Loss (Left) ---
    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg Batch-Train Loss", color=color)
    ax1.plot(train_epochs, train_losses, "r-", label="Avg Batch-Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Axis 2: Validation MAE (Right) ---
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Validation MAE", color=color)
    ax2.plot(val_epochs, val_maes, "b-", label="Validation MAE")
    # ax2.scatter(
    #     val_epochs[0],
    #     val_maes[0],
    #     marker="o",
    #     color="blue",
    #     s=100,
    #     zorder=5,
    #     label="Baseline Val MAE",
    # )
    ax2.tick_params(axis="y", labelcolor=color)

    # --- Axis 3: Validation Q-Loss (Right, offset) ---
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 80))
    color = "tab:green"
    ax3.set_ylabel("Validation Q-Loss", color=color)
    ax3.plot(val_epochs, val_q_losses, "g-", label="Validation Q-Loss")
    # ax3.scatter(
    #     val_epochs[0],
    #     val_q_losses[0],
    #     marker="o",
    #     color="green",
    #     s=100,
    #     zorder=5,
    #     label="Baseline Val Q-Loss",
    # )
    ax3.tick_params(axis="y", labelcolor=color)

    plt.title("Fine-tuning Learning Curves")
    # Combine legends from all axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax3.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper right")

    plot_path = output_dir / "learning_curves.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    print(f"\n--- Learning curves saved to: {plot_path} ---")
    plt.close()


def visualize_prediction(
    context_X: np.ndarray,
    context_y: np.ndarray,
    target_X: np.ndarray,
    target_y: np.ndarray,
    predicted_X: np.ndarray,
    predicted_y: np.ndarray,
    predicted_quantiles: np.ndarray,
    epoch: int,
    val_mae: float,
    val_q_loss: float,
    output_path: Path,
):
    """
    Visualizes the results of the interpolation task for a single validation sample.

    Args:
        context_X: The x-coordinates of the training context.
        context_y: The y-values of the training context.
        target_X: The x-coordinates of the points to be interpolated/extrapolated.
        target_y: The y-values of the points to be interpolated/extrapolated.
        predicted_X: The combined and sorted x-coordinates for the predictions.
        predicted_y: The combined and sorted median predictions.
        predicted_quantiles: The combined and sorted quantile predictions.
        epoch: The current training epoch.
        val_mae: The MAE on this specific validation sample.
        val_q_loss: The quantile loss on this specific validation sample.
        output_path: The full path where the plot will be saved.
    """
    plt.figure(figsize=(15, 7))

    # --- Add shaded region for test samples ---
    if len(target_X) > 0:
        # Sort test indices to find contiguous blocks
        sorted_x_test = np.sort(target_X.flatten())
        diffs = np.diff(sorted_x_test)
        # A jump in time indices indicates a new block
        block_starts = np.concatenate(
            ([sorted_x_test[0]], sorted_x_test[1:][diffs > 1])
        )
        block_ends = np.concatenate(
            (sorted_x_test[:-1][diffs > 1], [sorted_x_test[-1]])
        )

        # Draw a shaded region for each block
        for i, (start, end) in enumerate(zip(block_starts, block_ends)):
            label = "Target Region" if i == 0 else None
            plt.axvspan(
                start - 0.5,
                end + 0.5,
                color="gray",
                alpha=0.15,
                zorder=0,
                label=label,
            )

    # 1. Plot the ground truth data as two separate scatter plots
    plt.scatter(
        context_X,
        context_y,
        label="Ground Truth (Context)",
        color="blue",
        s=20,
        zorder=3,
    )
    plt.scatter(
        target_X,
        target_y,
        label="Ground Truth (Target)",
        color="green",
        s=20,
        zorder=3,
    )

    # 2. Plot the unified prediction as a single line
    plt.plot(
        predicted_X,
        predicted_y,
        label="Prediction (Median)",
        color="red",
        linestyle="--",
        linewidth=2,
        zorder=5,
    )

    # 3. Add the unified confidence interval
    if predicted_quantiles is not None and len(predicted_quantiles) >= 9:
        lower_bound = predicted_quantiles[0]  # 0.1 quantile
        upper_bound = predicted_quantiles[-1]  # 0.9 quantile
        plt.fill_between(
            predicted_X.flatten(),
            lower_bound,
            upper_bound,
            alpha=0.2,
            color="red",
            label="80% Prediction Interval",
        )

    title = (
        f"Epoch {epoch} | Sample MAE: {val_mae:.4f} | Sample Q-Loss: {val_q_loss:.4f}"
    )
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(False)

    # Save the plot instead of showing it
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def visualize_meta_training_samples(
    meta_X_train: np.ndarray,
    meta_y_train: np.ndarray,
    num_samples_to_visualize: int,
    output_dir: Path,
    splitter: callable,
):
    """
    Visualizes a subset of the meta-training samples, one plot per sample,
    including the train-test split.

    Args:
        meta_X_train: The x-coordinates of the meta-training samples.
        meta_y_train: The y-values of the meta-training samples.
        num_samples_to_visualize: The number of samples to plot.
        output_dir: The directory where the plots will be saved.
        splitter: The function used to split the data into train and test sets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    num_samples = min(num_samples_to_visualize, len(meta_X_train))

    for i in range(num_samples):
        X_train, X_test, y_train, y_test = splitter(meta_X_train[i], meta_y_train[i])

        plt.figure(figsize=(15, 7))

        plt.scatter(X_train, y_train, label="Train Data (Context)", color="blue", s=20)
        plt.scatter(X_test, y_test, label="Test Data (Target)", color="green", s=20)

        plt.title(f"Meta-Training Sample {i + 1} with Train/Test Split")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plot_path = output_dir / f"meta_training_sample_{i + 1}.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()


def visualize_batch_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epoch: int,
    batch_idx: int,
    output_dir: Path,
):
    """
    Visualizes the train/test split for a single batch from the dataloader.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(15, 7))

    plt.scatter(x_train, y_train, label="Train Data (Context)", color="blue", s=20)
    plt.scatter(x_test, y_test, label="Test Data (Target)", color="green", s=20)

    plt.title(f"Epoch {epoch + 1}, Batch {batch_idx + 1} Train/Test Split")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plot_path = output_dir / f"epoch_{epoch + 1}_batch_{batch_idx + 1}_split.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
