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
    phase_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a time series with a sinusoidal pattern and Gaussian noise.

    Args:
        num_points: The total number of points in the generated time series.
        amplitude: The amplitude of the sinusoid.
        num_periods: The number of full cycles over the series length.
        noise_std: The standard deviation of the Gaussian noise.
        phase_shift: The phase shift in radians.

    Returns:
        A tuple of (time_indices, series_values).
    """
    time = np.arange(num_points)
    sinusoid = amplitude * np.sin(
        2 * np.pi * num_periods * time / num_points + phase_shift
    )
    noise = np.random.normal(0, noise_std, size=num_points)
    series = (sinusoid + noise).astype(np.float32)
    return time.reshape(-1, 1), series


def generate_ood_sinusoid(
    num_points: int = 200,
    amplitude: float = 1.0,
    num_periods: float = 2.0,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a time series with a square wave pattern and Gaussian noise.
    The time series is multiplied by the time index to create an OOD signal.
    """
    time = np.arange(num_points)
    # Generate square wave
    sinusoid = generate_sinusoid(num_points, amplitude, num_periods, noise_std)[1]
    sinusoid[sinusoid > 0] += 1
    square_wave = amplitude * np.sign(sinusoid)

    # Add noise
    noise = np.random.normal(0, noise_std, size=num_points)
    series = (square_wave + noise).astype(np.float32)

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


def time_series_splitter(
    X: np.ndarray,
    y: np.ndarray,
    mask_frac: float = 0.2,
    random_state=None,  # to match sklearn's splitter signature
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    A custom splitter that splits data by masking the future (according to chronological order).
    The mask is applied to the last n_masked points.
    """
    n_total = len(y)
    n_masked = int(n_total * mask_frac)
    X_train, y_train = X[:-n_masked], y[:-n_masked]
    X_test, y_test = X[-n_masked:], y[-n_masked:]
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
    train_mae: float,
    val_mae: float,
    train_q_loss: float,
    val_q_loss: float,
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
        train_mae: The MAE on the training set.
        val_mae: The MAE on the validation set.
        train_q_loss: The quantile loss on the training set.
        val_q_loss: The quantile loss on the validation set.
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

    title = (
        f"Epoch {epoch} | Tying Loss: {tying_loss:.4f} | Avg Batch-Train Loss: {loss:.4f}\n"
        f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f} | "
        f"Train Q-Loss: {train_q_loss:.4f}, Val Q-Loss: {val_q_loss:.4f}"
    )
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(False)

    # Save the plot instead of showing it
    plot_path = output_dir / f"epoch_{epoch}.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()


def validate_and_visualize(
    regressor: TabPFNRegressor,
    regressor_config: Dict,
    meta_X_train_raw: np.ndarray,
    meta_y_train_raw: np.ndarray,
    meta_X_val_raw: np.ndarray,
    meta_y_val_raw: np.ndarray,
    epoch: int,
    avg_loss: float,
    avg_tying_loss: float,
    output_dir: Path,
    device: str,
):
    """
    Runs validation, calculates metrics, and visualizes the results.
    """
    print(f"--- Running validation for Epoch {epoch} ---")
    eval_config = regressor_config.copy()
    if "fit_mode" in eval_config:
        eval_config.pop("fit_mode")

    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    with torch.no_grad():
        eval_regressor.fit(meta_X_train_raw, meta_y_train_raw)
        pred_dict_on_train = eval_regressor.predict(
            meta_X_train_raw, output_type="full"
        )
        pred_dict_on_test = eval_regressor.predict(meta_X_val_raw, output_type="full")

        # --- Validation Metrics ---
        train_mae = np.mean(
            np.abs(pred_dict_on_train["median"] - meta_y_train_raw)
        ).item()
        val_mae = np.mean(np.abs(pred_dict_on_test["median"] - meta_y_val_raw)).item()

        # --- Quantile Loss ---
        quantiles_to_eval = [0.1, 0.5, 0.9]
        # The quantiles from predict are [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        # We want to calculate for [0.1, 0.5, 0.9], which are at indices 2, 4, 6
        quantile_indices = [2, 4, 6]

        y_true_train_tensor = torch.from_numpy(meta_y_train_raw).to(device)
        y_true_val_tensor = torch.from_numpy(meta_y_val_raw).to(device)
        train_q_loss_tensor = torch.tensor(0.0, device=device)
        val_q_loss_tensor = torch.tensor(0.0, device=device)
        for i, q in zip(quantile_indices, quantiles_to_eval):
            # Train q-loss
            y_pred_q_train = torch.from_numpy(pred_dict_on_train["quantiles"][i]).to(
                device
            )
            train_q_loss_tensor += pinball_loss(y_true_train_tensor, y_pred_q_train, q)
            # Val q-loss
            y_pred_q_val = torch.from_numpy(pred_dict_on_test["quantiles"][i]).to(
                device
            )
            val_q_loss_tensor += pinball_loss(y_true_val_tensor, y_pred_q_val, q)

        train_q_loss = (train_q_loss_tensor / len(quantiles_to_eval)).item()
        val_q_loss = (val_q_loss_tensor / len(quantiles_to_eval)).item()

        # Combine and sort predictions for a unified plot
        combined_X = np.concatenate((meta_X_train_raw, meta_X_val_raw))
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
        meta_X_train_raw,
        meta_y_train_raw,
        meta_X_val_raw,
        meta_y_val_raw,
        sorted_X,
        sorted_y,
        sorted_quantiles,
        epoch=epoch,
        loss=avg_loss,
        train_mae=train_mae,
        val_mae=val_mae,
        train_q_loss=train_q_loss,
        val_q_loss=val_q_loss,
        tying_loss=avg_tying_loss,
        output_dir=output_dir,
    )


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
        choices=["block", "random", "chunk", "future"],
        help="The strategy to use for masking the time series ('block', 'random', 'chunk', or 'future').",
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
        "--future_mask_frac",
        type=float,
        default=0.2,
        help="The fraction of the series to mask for the future strategy.",
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
    parser.add_argument(
        "--l2_sp_weight",
        type=float,
        default=0.1,
        help="The weight of the L2 regularization on the weight tying loss.",
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
    # Future strategy
    FUTURE_MASK_FRAC = args.future_mask_frac
    # Chunk strategy
    NUM_CHUNKS = args.num_chunks
    CHUNK_LEN = args.chunk_len
    if args.chunk_len_ratio is not None:
        CHUNK_LEN = int(NUM_POINTS * args.chunk_len_ratio)

    # --- Fine-tuning Hyperparameters ---
    LEARNING_RATE = 1e-6
    EPOCHS = args.epochs
    L2_SP_LAMBDA = 1000.0  # Weight tying scaling
    L2_SP_WEIGHT = args.l2_sp_weight
    USE_LOGITS_SMOOTHIE = args.use_logits_smoothie
    LOGITS_SMOOTHIE_KERNEL_SIZE = 101
    LOGITS_SMOOTHIE_SIGMA = 15.0

    # The context size for the model. For interpolation, this is the number of
    # unmasked points it sees.
    # FINETUNING_CONTEXT_SIZE = int(NUM_POINTS * 0.8)
    # CHECK_CONSISTENCY_EVERY_N_EPOCHS = 20

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
    all_X, all_y = generate_sinusoid(
        num_points=NUM_POINTS,
        num_periods=NUM_PERIODS,
        amplitude=1.0,
        noise_std=0.0,
        phase_shift=0.0,
    )

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
    elif MASKING_STRATEGY == "future":
        splitter = partial(
            time_series_splitter,
            mask_frac=FUTURE_MASK_FRAC,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown MASKING_STRATEGY: {MASKING_STRATEGY}")

    # --- Create meta-split for finetuning  ---
    print("--- Creating meta-split for finetuning ---")
    meta_X_train_raw, meta_X_val_raw, meta_y_train_raw, meta_y_val_raw = splitter(
        all_X, all_y
    )
    print(
        f"Meta-split created: meta_train length {len(meta_y_train_raw)}, meta_val length {len(meta_y_val_raw)}"
    )

    # Generate bunch of different sinusoids as meta-train
    # NUM_META_TRAIN_SAMPLES = 100
    NUM_META_TRAIN_SAMPLES = 500
    NUM_META_TRAIN_LENGTH = NUM_POINTS
    # NUM_META_TRAIN_LENGTH = len(meta_y_train_raw)
    meta_X_train = np.zeros(shape=(NUM_META_TRAIN_SAMPLES, NUM_POINTS, 1))
    meta_y_train = np.zeros(shape=(NUM_META_TRAIN_SAMPLES, NUM_META_TRAIN_LENGTH))
    for i in range(NUM_META_TRAIN_SAMPLES):
        # Randomly shift the sinusoid by a random number of points
        random_x_start = np.random.randint(0, 500)
        meta_X_train[i] = np.arange(
            random_x_start, random_x_start + NUM_META_TRAIN_LENGTH
        ).reshape(-1, 1)

        # Generate a sinusoid with a random frequency and phase shift
        meta_y_train[i] = generate_sinusoid(
            num_points=NUM_META_TRAIN_LENGTH,
            # num_periods=NUM_PERIODS,
            num_periods=np.random.uniform(2, 4),
            amplitude=1.0,
            phase_shift=np.random.uniform(0, 2 * np.pi),
        )[1]
    # meta_X_train = np.zeros(shape=(NUM_META_TRAIN_SAMPLES, NUM_META_TRAIN_LENGTH, 1))
    # meta_X_train[:] = np.arange(NUM_META_TRAIN_LENGTH).reshape(-1, 1)

    print("meta_X_train.shape: ", meta_X_train.shape)
    print("meta_y_train.shape: ", meta_y_train.shape)

    # Convert first dimension to list
    meta_X_train = meta_X_train.tolist()
    meta_y_train = meta_y_train.tolist()

    training_datasets = regressor.get_preprocessed_datasets(
        meta_X_train,
        meta_y_train,
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

    # --- Pre-Training Validation ---
    validate_and_visualize(
        regressor=regressor,
        regressor_config=regressor_config,
        meta_X_train_raw=meta_X_train_raw,
        meta_y_train_raw=meta_y_train_raw,
        meta_X_val_raw=meta_X_val_raw,
        meta_y_val_raw=meta_y_val_raw,
        epoch=0,
        avg_loss=0.0,
        avg_tying_loss=0.0,
        output_dir=run_dir,
        device=device,
    )

    # --- 4. Fine-tuning Loop ---
    print(f"--- 4. Starting Fine-tuning for {EPOCHS} epochs ---")
    for epoch in range(EPOCHS):
        total_loss = 0.0
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

            print("DEBUG: hash(X_trains_p): ", hash(X_trains_p[0]))
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

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                pred_loss=f"{pred_loss.item():.4f}",
                tying_loss=f"{tying_loss.item():.4f}",
            )

        if isinstance(optimizer, AdamWScheduleFree):
            optimizer.eval()

        avg_loss = total_loss / len(finetuning_dataloader)
        avg_tying_loss = total_tying_loss / len(finetuning_dataloader)

        # --- 5. Periodic Visualization on a Fixed Validation Mask ---
        validate_and_visualize(
            regressor=regressor,
            regressor_config=regressor_config,
            meta_X_train_raw=meta_X_train_raw,
            meta_y_train_raw=meta_y_train_raw,
            meta_X_val_raw=meta_X_val_raw,
            meta_y_val_raw=meta_y_val_raw,
            epoch=epoch + 1,
            avg_loss=avg_loss,
            avg_tying_loss=avg_tying_loss,
            output_dir=run_dir,
            device=device,
        )

        # Note: We don't need to call model.train() because the original
        # regressor was never put in eval mode. The cloned one is discarded.

    print("\n--- ✅ Fine-tuning Finished ---")


if __name__ == "__main__":
    main()
