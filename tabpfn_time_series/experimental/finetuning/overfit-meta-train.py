from typing import Tuple, Dict
from pathlib import Path
from datetime import datetime
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from schedulefree import AdamWScheduleFree
from tqdm import tqdm
import argparse

from tabpfn import TabPFNRegressor
from tabpfn.utils import meta_dataset_collator
from tabpfn.finetune_utils import clone_model_for_evaluation

from tabpfn_time_series.experimental.finetuning.logits_smoothie import (
    LogitSmoothieMaker,
)


def generate_sinusoid(
    num_points: int = 200,
    amplitude: float = 1.0,
    num_periods: float = 2.0,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a time series with a sinusoidal pattern and Gaussian noise.

    Args:
        num_points: The total number of points in the generated time series.
        amplitude: The amplitude of the sinusoid.
        num_periods: The number of full cycles over the series length.
        noise_std: The standard deviation of the Gaussian noise.

    Returns:
        A tuple of (time_indices, series_values).
    """
    time = np.arange(num_points)
    sinusoid = amplitude * np.sin(2 * np.pi * num_periods * time / num_points)
    noise = np.random.normal(0, noise_std, size=num_points)
    series = (sinusoid + noise).astype(np.float32)
    return time.reshape(-1, 1), series


def random_chunk_splitter(
    X: np.ndarray,
    y: np.ndarray,
    num_chunks: int = 5,
    chunk_len: int = 10,
    random_state=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data by masking a number of non-overlapping, randomly placed chunks.

    Args:
        X: Feature array (time indices).
        y: Target array (time series values).
        num_chunks: The number of chunks to mask.
        chunk_len: The length of each individual chunk.
        random_state: Seed for the random number generator.

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    n_total = len(y)
    mask = np.zeros(n_total, dtype=bool)
    masked_count = 0
    target_masked = num_chunks * chunk_len

    rng = np.random.default_rng(random_state)
    possible_starts = np.arange(n_total - chunk_len + 1)
    rng.shuffle(possible_starts)

    for start in possible_starts:
        if masked_count >= target_masked:
            break
        chunk_indices = np.arange(start, start + chunk_len)
        if not np.any(mask[chunk_indices]):
            mask[chunk_indices] = True
            masked_count += chunk_len

    X_train, y_train = X[~mask], y[~mask]
    X_test, y_test = X[mask], y[mask]

    return X_train, X_test, y_train, y_test


def random_sample_splitter(
    X: np.ndarray,
    y: np.ndarray,
    mask_frac: float = 0.2,
    random_state=None,  # to match sklearn's splitter signature
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A custom splitter that splits data by masking a random fraction of samples.

    Args:
        X: Feature array (expected to be time indices).
        y: Target array (time series values).
        mask_frac: The fraction of samples to mask for the test set.
        random_state: Seed for the random number generator for reproducibility.

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    n_total = len(y)
    n_masked = int(n_total * mask_frac)

    # Use a local random state to not affect the global one
    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(n_total)
    test_indices = shuffled_indices[:n_masked]

    # Create a boolean mask from these indices
    mask = np.zeros(n_total, dtype=bool)
    mask[test_indices] = True

    X_train, y_train = X[~mask], y[~mask]
    X_test, y_test = X[mask], y[mask]

    return X_train, X_test, y_train, y_test


def interpolation_splitter(
    X: np.ndarray,
    y: np.ndarray,
    mask_start_frac: float = 0.4,
    mask_len_frac: float = 0.2,
    random_state=None,  # to match sklearn's splitter signature
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A custom splitter that splits data for an interpolation task.
    It masks a middle segment of the data, using the outer parts for training
    and the middle part for testing.

    Args:
        X: Feature array (expected to be time indices).
        y: Target array (time series values).
        mask_start_frac: The fraction of the series where the mask begins.
        mask_len_frac: The fraction of the series to mask.

    Returns:
        A tuple (X_train, X_test, y_train, y_test).
    """
    n_total = len(y)
    mask_start = int(n_total * mask_start_frac)
    mask_len = int(n_total * mask_len_frac)
    mask_end = mask_start + mask_len

    mask = np.zeros(n_total, dtype=bool)
    mask[mask_start:mask_end] = True

    X_train, y_train = X[~mask], y[~mask]
    X_test, y_test = X[mask], y[mask]

    return X_train, X_test, y_train, y_test


def visualize_interpolation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    predicted_X: np.ndarray,
    predicted_y: np.ndarray,
    predicted_quantiles: np.ndarray,
    epoch: int,
    loss: float,
    mae: float,
    q_loss: float,
    tying_loss: float,
    output_dir: Path,
):
    """
    Visualizes the results of the interpolation task with a unified plot.

    Args:
        X_train: The x-coordinates of the training context.
        y_train: The y-values of the training context.
        X_test: The x-coordinates of the points to be interpolated.
        y_test: The y-values of the points to be interpolated.
        predicted_X: The combined and sorted x-coordinates for the predictions.
        predicted_y: The combined and sorted median predictions.
        predicted_quantiles: The combined and sorted quantile predictions.
        epoch: The current training epoch.
        loss: The average loss for the epoch.
        mae: The average MAE for the epoch.
        q_loss: The average quantile loss for the epoch.
        tying_loss: The average weight tying loss for the epoch.
        output_dir: The directory where the plot will be saved.
    """
    plt.figure(figsize=(15, 7))

    # --- Add shaded region for test samples ---
    if len(X_test) > 0:
        # Sort test indices to find contiguous blocks
        sorted_x_test = np.sort(X_test.flatten())
        diffs = np.diff(sorted_x_test)
        block_starts = np.concatenate(
            ([sorted_x_test[0]], sorted_x_test[1:][diffs > 1])
        )
        block_ends = np.concatenate(
            (sorted_x_test[:-1][diffs > 1], [sorted_x_test[-1]])
        )

        # Draw a shaded region for each block
        for i, (start, end) in enumerate(zip(block_starts, block_ends)):
            label = "Test Region" if i == 0 else None
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
        X_train,
        y_train,
        label="Ground Truth (Context)",
        color="blue",
        s=20,
        zorder=3,
    )
    plt.scatter(
        X_test,
        y_test,
        label="Ground Truth (Interpolation)",
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

    plt.title(
        f"TabPFN Overfit on Meta-Training Sample (Epoch {epoch}) | "
        f"Loss: {loss:.4f}, Tying Loss: {tying_loss:.4f}, MAE: {mae:.4f}, Q-Loss: {q_loss:.4f}"
    )
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(False)

    # Save the plot instead of showing it
    plot_path = output_dir / f"epoch_{epoch}.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()


def pinball_loss(y_true, y_pred, quantile):
    """
    Calculates the pinball loss for a given quantile.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        quantile: The quantile to evaluate (e.g., 0.5 for MAE).

    Returns:
        The pinball loss.
    """
    error = y_true - y_pred
    return torch.mean(torch.max((quantile * error), (quantile - 1) * error))


def _weight_tying_loss(
    current_model: torch.nn.Module,
    original_params: Dict[str, torch.Tensor],
    l2_sp_lambda: float,
    device: str,
) -> torch.Tensor:
    """
    Calculate the weight-tying loss for the model.
    This computes an L2 penalty between the current model parameters and the
    original parameters to regularize finetuning and prevent catastrophic
    forgetting.
    """
    tying_loss = torch.tensor(0.0).to(device)
    for name, param in current_model.named_parameters():
        if param.requires_grad:
            original_param = original_params[name].to(device)
            tying_loss += torch.sum((param - original_param) ** 2) * 0.5

    return l2_sp_lambda * tying_loss


def main():
    """Main function to run the interpolation proof of concept."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Run a proof-of-concept for fine-tuning TabPFN on a time series interpolation task."
    )
    parser.add_argument(
        "--masking_strategy",
        type=str,
        default="chunk",
        choices=["block", "random", "chunk"],
        help="The strategy to use for masking the time series ('block', 'random', or 'chunk').",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=200,
        help="The total number of points in the generated time series.",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=8,
        help="The number of chunks to use for the chunk strategy.",
    )
    parser.add_argument(
        "--chunk_len",
        type=int,
        default=10,
        help="The length of the chunks to use for the chunk strategy.",
    )
    parser.add_argument(
        "--chunk_len_ratio",
        type=float,
        default=None,
        help="Ratio of num_points to determine chunk_len. Overrides --chunk_len if set.",
    )
    parser.add_argument(
        "--random_mask_frac",
        type=float,
        default=0.3,
        help="The fraction of the series to mask for the random strategy.",
    )
    parser.add_argument(
        "--disable_target_preprocessing",
        action="store_true",
        help="Whether to disable the model's target preprocessing.",
    )
    parser.add_argument(
        "--use_logits_smoothie",
        action="store_true",
        help="Whether to use the logits smoothie.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epochs to run.",
    )
    args = parser.parse_args()

    # --- Configuration ---
    NUM_POINTS = args.num_points
    NUM_PERIODS = 3.0
    N_ENSEMBLE_CONFIGURATIONS = 1
    RANDOM_SEED = 42

    # --- Masking Configuration ---
    MASKING_STRATEGY = args.masking_strategy
    # Block strategy
    MASK_START_FRAC = 0.4
    MASK_LEN_FRAC = 0.2
    # Random strategy
    RANDOM_MASK_FRAC = args.random_mask_frac
    # Chunk strategy
    NUM_CHUNKS = args.num_chunks
    CHUNK_LEN = args.chunk_len
    if args.chunk_len_ratio is not None:
        CHUNK_LEN = int(NUM_POINTS * args.chunk_len_ratio)

    # --- Fine-tuning Hyperparameters ---
    LEARNING_RATE = 1e-6
    EPOCHS = args.epochs
    L2_SP_LAMBDA = 1000.0  # Weight tying scaling
    L2_SP_WEIGHT = 0.1
    USE_LOGITS_SMOOTHIE = args.use_logits_smoothie
    LOGITS_SMOOTHIE_KERNEL_SIZE = 101
    LOGITS_SMOOTHIE_SIGMA = 15.0

    # The context size for the model. For interpolation, this is the number of
    # unmasked points it sees.
    # FINETUNING_CONTEXT_SIZE = int(NUM_POINTS * 0.8)
    CHECK_CONSISTENCY_EVERY_N_EPOCHS = 20

    print(f"--- Running with masking strategy: {MASKING_STRATEGY} ---")

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # --- Check for GPU ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Logits Smoothie Configuration ---
    logits_smoothie_maker = None
    if USE_LOGITS_SMOOTHIE:
        logits_smoothie_maker = LogitSmoothieMaker(
            kernel_size=LOGITS_SMOOTHIE_KERNEL_SIZE,
            sigma=LOGITS_SMOOTHIE_SIGMA,
        ).to(device)

    # --- Create Output Directory ---
    output_root = Path("output/finetune_interpolation")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"--- Plots will be saved to: {run_dir} ---\n")

    # --- 1. Generate a Single Synthetic Series for Finetuning ---
    print("--- 1. Generating a single synthetic series for fine-tuning ---")
    source_X, source_y = generate_sinusoid(
        num_points=NUM_POINTS, num_periods=NUM_PERIODS, amplitude=1.0
    )

    # Create a "meta-dataset" with just one series
    all_X_raw = [source_X]
    all_y_raw = [source_y]
    print("Series generated and wrapped in a single-item meta-dataset.\n")

    # --- 2. Setup Model and Preprocessing ---
    print("--- 2. Setting up Model and Preprocessing ---")
    regressor_config = {
        "device": device,
        "n_estimators": N_ENSEMBLE_CONFIGURATIONS,
        "fit_mode": "batched",
        "differentiable_input": False,  # Important for this workflow
        "inference_config": {
            "REGRESSION_Y_PREPROCESS_TRANSFORMS": (None, None)
            if args.disable_target_preprocessing
            else (None, "safepower")
        },
    }
    regressor = TabPFNRegressor(**regressor_config)
    regressor.initialize_model()

    # Store original model parameters for weight tying
    original_params = {
        name: p.clone().detach()
        for name, p in regressor.model_.named_parameters()  # type: ignore
    }
    print("--- Stored original model parameters for weight tying loss ---\n")

    # --- Create a single, fixed splitter based on the strategy ---
    # For an overfitting test, we use the same fixed mask for both training and validation.
    if MASKING_STRATEGY == "block":
        splitter = partial(
            interpolation_splitter,
            mask_start_frac=MASK_START_FRAC,
            mask_len_frac=MASK_LEN_FRAC,
        )
    elif MASKING_STRATEGY == "random":
        splitter = partial(
            random_sample_splitter,
            mask_frac=RANDOM_MASK_FRAC,
            random_state=RANDOM_SEED,
        )
    elif MASKING_STRATEGY == "chunk":
        splitter = partial(
            random_chunk_splitter,
            num_chunks=NUM_CHUNKS,
            chunk_len=CHUNK_LEN,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown MASKING_STRATEGY: {MASKING_STRATEGY}")

    # --- Generate a reference split to ensure consistency ---
    print("--- Generating reference data split for consistency check ---")
    X_train_ref, X_val_ref, y_train_ref, y_val_ref = splitter(source_X, source_y)
    print("Reference split generated.\n")

    training_datasets = regressor.get_preprocessed_datasets(
        all_X_raw,
        all_y_raw,
        split_fn=splitter,
    )
    # The dataloader will yield the same (but differently masked) series repeatedly
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=1,  # Meta-batch-size is always 1
        collate_fn=meta_dataset_collator,
        shuffle=True,  # Shuffle to re-trigger the splitter
    )
    print("Length of finetuning_dataloader: ", len(finetuning_dataloader))

    # --- 3. Setup Optimizer ---
    # optimizer = torch.optim.Adam(regressor.model_.parameters(), lr=LEARNING_RATE)  # type: ignore
    optimizer = AdamWScheduleFree(
        regressor.model_.parameters(),
        lr=LEARNING_RATE,
    )
    print(f"--- 3. Optimizer Initialized: Adam, LR: {LEARNING_RATE} ---\n")

    # Debug (HACK)
    cache_data_batch = None

    # --- 4. Fine-tuning Loop ---
    print(f"--- 4. Starting Fine-tuning for {EPOCHS} epochs ---")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_mae = 0.0
        total_q_loss = 0.0
        total_tying_loss = 0.0

        if isinstance(optimizer, AdamWScheduleFree):
            optimizer.train()

        # The dataloader is set to shuffle, so each epoch will get a new random mask
        pbar = tqdm(
            finetuning_dataloader, desc=f"Finetuning Epoch {epoch + 1}/{EPOCHS}"
        )

        for data_batch in pbar:
            # Debug (HACK)
            if cache_data_batch is None:
                cache_data_batch = data_batch
            else:
                data_batch = cache_data_batch

            optimizer.zero_grad()

            (
                X_trains_p,
                X_tests_p,
                y_trains_p,
                y_test_std,
                cat_ixs,
                confs,
                norm_bardist,
                bardist,
                x_train_raw,
                y_train_raw,
                x_test_raw,
                y_test_raw,
            ) = data_batch

            # print("hash(X_trains_p): ", hash(X_trains_p[0]))
            # print("confs: ", confs)

            loss_fn = norm_bardist[0]
            y_target = y_test_std[0]

            regressor.fit_from_preprocessed(
                [X_trains_p[0]], [y_trains_p[0]], cat_ixs, confs
            )
            logits, _, _ = regressor.forward([X_tests_p[0]])
            logits = logits.squeeze(0)
            if logits_smoothie_maker is not None:
                print("Using logits smoothie")
                logits = logits_smoothie_maker(logits)

            pred_loss = loss_fn(logits, y_target.to(device)).mean()

            tying_loss = _weight_tying_loss(
                current_model=regressor.model_,  # type: ignore
                original_params=original_params,
                l2_sp_lambda=L2_SP_LAMBDA,
                device=device,
            )

            loss = pred_loss * (1 - L2_SP_WEIGHT) + tying_loss * L2_SP_WEIGHT
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_tying_loss += tying_loss.item()

            # --- Calculate additional metrics ---
            with torch.no_grad():
                # The returned y_test_raw is a list containing one tensor
                y_true_raw = y_test_raw[0].to(device)

                # --- Get predictions on original scale ---
                # `norm_bardist` is the criterion that knows about un-normalization
                criterion = norm_bardist[0]
                y_pred_mean = criterion.mean(logits).detach()

                # --- MAE ---
                mae = torch.nn.functional.l1_loss(y_pred_mean, y_true_raw)

                # --- Quantile Loss (Pinball Loss) ---
                quantiles = [0.1, 0.5, 0.9]
                quantile_loss = torch.tensor(0.0, device=device)
                for q in quantiles:
                    y_pred_q = criterion.icdf(logits, q).detach()
                    quantile_loss += pinball_loss(y_true_raw, y_pred_q, q)
                quantile_loss /= len(quantiles)

                total_mae += mae.item()
                total_q_loss += quantile_loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                pred_loss=f"{pred_loss.item():.4f}",
                tying_loss=f"{tying_loss.item():.4f}",
                mae=f"{mae.item():.4f}",
                q_loss=f"{quantile_loss.item():.4f}",
            )

        if isinstance(optimizer, AdamWScheduleFree):
            optimizer.eval()

        avg_loss = total_loss / len(finetuning_dataloader)
        avg_mae = total_mae / len(finetuning_dataloader)
        avg_q_loss = total_q_loss / len(finetuning_dataloader)
        avg_tying_loss = total_tying_loss / len(finetuning_dataloader)

        # --- 5. Periodic Visualization on a Fixed Validation Mask ---

        # For evaluation, it's best practice to clone the model to not interfere
        # with the training model's state (e.g., batch norm stats)
        eval_config = regressor_config.copy()
        eval_config.pop("fit_mode")
        eval_regressor = clone_model_for_evaluation(
            regressor, eval_config, TabPFNRegressor
        )

        # Since this is an overfitting test, we validate on the exact same data split
        # used for training.
        X_val_train, X_val_test, y_val_train, y_val_test = splitter(source_X, source_y)

        # --- Periodically check for data consistency ---
        if (epoch + 1) % CHECK_CONSISTENCY_EVERY_N_EPOCHS == 0:
            print(f"\n--- Performing consistency check at epoch {epoch + 1} ---")
            assert np.array_equal(X_val_train, X_train_ref), (
                "Validation X_train does not match reference."
            )
            assert np.array_equal(y_val_train, y_train_ref), (
                "Validation y_train does not match reference."
            )
            assert np.array_equal(X_val_test, X_val_ref), (
                "Validation X_test does not match reference."
            )
            assert np.array_equal(y_val_test, y_val_ref), (
                "Validation y_test does not match reference."
            )
            print(
                "Consistency check passed: Training and validation data are identical."
            )

        with torch.no_grad():
            # The canonical way to evaluate is to first fit on the context (the
            # unmasked points) and then predict the test points (the masked points).
            eval_regressor.fit(X_val_train, y_val_train)
            pred_dict_on_train = eval_regressor.predict(
                X_val_train,
                output_type="full",  # type: ignore
            )
            pred_dict_on_test = eval_regressor.predict(X_val_test, output_type="full")  # type: ignore

            # Combine and sort predictions for a unified plot
            combined_X = np.concatenate((X_val_train, X_val_test))
            combined_y = np.concatenate(
                (pred_dict_on_train["median"], pred_dict_on_test["median"])
            )
            combined_quantiles = np.concatenate(
                (
                    pred_dict_on_train["quantiles"],
                    pred_dict_on_test["quantiles"],
                ),
                axis=1,
            )

            sort_indices = np.argsort(combined_X.flatten())
            sorted_X = combined_X[sort_indices]
            sorted_y = combined_y[sort_indices]
            sorted_quantiles = combined_quantiles[:, sort_indices]

        visualize_interpolation(
            X_val_train,
            y_val_train,
            X_val_test,
            y_val_test,
            sorted_X,
            sorted_y,
            sorted_quantiles,
            epoch=epoch + 1,
            loss=avg_loss,
            mae=avg_mae,
            q_loss=avg_q_loss,
            tying_loss=avg_tying_loss,
            output_dir=run_dir,
        )

        # Note: We don't need to call model.train() because the original
        # regressor was never put in eval mode. The cloned one is discarded.

    print("\n--- ✅ Fine-tuning Finished ---")


if __name__ == "__main__":
    main()
