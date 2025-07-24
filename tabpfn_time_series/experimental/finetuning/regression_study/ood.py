from typing import Dict
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from schedulefree import AdamWScheduleFree
from tqdm import tqdm
import argparse
import json

from tabpfn import TabPFNRegressor
from tabpfn.preprocessing import (
    PreprocessorConfig,
    default_regressor_preprocessor_configs,
)
from tabpfn.utils import meta_dataset_collator
from tabpfn.finetune_utils import clone_model_for_evaluation

from tabpfn_time_series.experimental.finetuning.logits_smoothie import (
    LogitSmoothieMaker,
)
from tabpfn_time_series.experimental.finetuning.regression_study.plot_utils import (
    plot_learning_curves,
    visualize_prediction,
    visualize_meta_training_samples,
)
from tabpfn_time_series.experimental.finetuning.regression_study.dataset_generators import (
    generate_sinusoid_dataset,
)
from tabpfn_time_series.experimental.finetuning.regression_study.splitters import (
    get_splitter,
)


def run_validation(
    regressor: TabPFNRegressor,
    regressor_config: Dict,
    meta_X_val: np.ndarray,
    meta_y_val: np.ndarray,
    splitter: callable,
    epoch: int,
    output_dir: Path,
    device: str,
):
    """
    Runs validation on a hold-out set, calculates metrics, and visualizes results.
    """
    print(f"--- Running validation for Epoch {epoch} ---")
    eval_config = regressor_config.copy()
    if "fit_mode" in eval_config:
        eval_config.pop("fit_mode")

    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)

    all_val_maes = []
    all_val_q_losses = []

    pbar = tqdm(range(len(meta_X_val)), desc=f"Validating on {len(meta_X_val)} samples")
    for i in pbar:
        X_sample, y_sample = meta_X_val[i], meta_y_val[i]
        X_context, X_target, y_context, y_target = splitter(X_sample, y_sample)

        with torch.no_grad():
            eval_regressor.fit(X_context, y_context)
            pred_on_context = eval_regressor.predict(X_context, output_type="full")
            pred_on_target = eval_regressor.predict(X_target, output_type="full")

            # --- Validation Metrics for this sample ---
            val_mae = np.mean(np.abs(pred_on_target["median"] - y_target)).item()
            all_val_maes.append(val_mae)

            # --- Quantile Loss for this sample ---
            quantiles_to_eval = [0.1, 0.5, 0.9]
            quantile_indices = [2, 4, 6]  # Corresponds to 0.1, 0.5, 0.9

            y_true_target_tensor = torch.from_numpy(y_target).to(device)
            val_q_loss_tensor = torch.tensor(0.0, device=device)
            for j, q in zip(quantile_indices, quantiles_to_eval):
                y_pred_q_val = torch.from_numpy(pred_on_target["quantiles"][j]).to(
                    device
                )
                val_q_loss_tensor += pinball_loss(y_true_target_tensor, y_pred_q_val, q)

            val_q_loss = (val_q_loss_tensor / len(quantiles_to_eval)).item()
            all_val_q_losses.append(val_q_loss)

            pbar.set_postfix(
                avg_mae=f"{np.mean(all_val_maes):.4f}",
                avg_q_loss=f"{np.mean(all_val_q_losses):.4f}",
            )

            # --- Visualization for this sample ---
            # Combine and sort predictions for a unified plot
            combined_X = np.concatenate((X_context, X_target))
            combined_y_pred = np.concatenate(
                (pred_on_context["median"], pred_on_target["median"])
            )
            combined_quantiles = np.concatenate(
                (
                    pred_on_context["quantiles"],
                    pred_on_target["quantiles"],
                ),
                axis=1,
            )
            sort_indices = np.argsort(combined_X.flatten())
            sorted_X = combined_X[sort_indices]
            sorted_y_pred = combined_y_pred[sort_indices]
            sorted_quantiles = combined_quantiles[:, sort_indices]

            sample_plot_dir = output_dir / "validation_samples" / f"sample_{i}"
            plot_path = sample_plot_dir / f"epoch_{epoch}.png"

            visualize_prediction(
                context_X=X_context,
                context_y=y_context,
                target_X=X_target,
                target_y=y_target,
                predicted_X=sorted_X,
                predicted_y=sorted_y_pred,
                predicted_quantiles=sorted_quantiles,
                epoch=epoch,
                val_mae=val_mae,
                val_q_loss=val_q_loss,
                output_path=plot_path,
            )

    avg_val_mae = np.mean(all_val_maes)
    avg_val_q_loss = np.mean(all_val_q_losses)

    print(
        f"\n--- Epoch {epoch} Validation Summary ---\n"
        f"  Avg Validation MAE: {avg_val_mae:.4f}\n"
        f"  Avg Validation Q-Loss: {avg_val_q_loss:.4f}\n"
        f"------------------------------------"
    )

    return {"val_mae": avg_val_mae, "val_q_loss": avg_val_q_loss}


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
        "--validate_every_n_epochs",
        type=int,
        default=1,
        help="The number of epochs to validate after.",
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
        default=120,
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
        default=0.33,
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
        "--lr",
        type=float,
        default=1e-6,
        help="The learning rate to use for the optimizer.",
    )
    parser.add_argument(
        "--l2_sp_weight",
        type=float,
        default=0.1,
        help="The weight of the L2 regularization on the weight tying loss.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before performing an optimizer step.",
    )
    parser.add_argument(
        "--use_val_as_train",
        action="store_true",
        help="Whether to use the validation set as the training set.",
    )
    parser.add_argument(
        "--disable_feature_preprocessing",
        action="store_true",
        help="Whether to disable the model's feature preprocessing.",
    )
    args = parser.parse_args()

    # --- Configuration ---
    NUM_POINTS = args.num_points
    N_ENSEMBLE_CONFIGURATIONS = 1
    RANDOM_SEED = 42

    # --- Dataset Configuration ---
    NUM_META_TRAIN_SAMPLES = 500  # Large set for fine-tuning
    NUM_META_VAL_SAMPLES = 20  # Hold-out set for validation
    NUM_MIN_PERIODS = 2
    NUM_MAX_PERIODS = 8

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
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    L2_SP_LAMBDA = 1000.0  # Weight tying scaling
    L2_SP_WEIGHT = args.l2_sp_weight
    USE_LOGITS_SMOOTHIE = args.use_logits_smoothie
    LOGITS_SMOOTHIE_KERNEL_SIZE = 101
    LOGITS_SMOOTHIE_SIGMA = 15.0
    GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps

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

    # --- Save args to log file ---
    args_path = run_dir / "args.json"
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"--- Arguments saved to: {args_path} ---\n")

    # --- 1. Generate Datasets ---
    print("--- 1. Generating datasets for finetuning and validation ---")
    meta_X_train, meta_y_train = generate_sinusoid_dataset(
        num_samples=NUM_META_TRAIN_SAMPLES,
        num_points=NUM_POINTS,
        min_periods=NUM_MIN_PERIODS,
        max_periods=NUM_MAX_PERIODS,
    )
    meta_X_val, meta_y_val = generate_sinusoid_dataset(
        num_samples=NUM_META_VAL_SAMPLES,
        num_points=NUM_POINTS,
        min_periods=NUM_MIN_PERIODS,
        max_periods=NUM_MAX_PERIODS,
    )

    if args.use_val_as_train:
        # Overwrite meta_X_train and meta_y_train with meta_X_val and meta_y_val
        #   (putting it here since we want the validation set generated
        #   over the previous run to be the same)
        print("--- Using validation set as training set ---")
        # Create duplicates of meta_X_val to make up to NUM_META_TRAIN_SAMPLES
        num_repeats = (
            NUM_META_TRAIN_SAMPLES + NUM_META_VAL_SAMPLES - 1
        ) // NUM_META_VAL_SAMPLES
        meta_X_train = np.tile(meta_X_val, (num_repeats, 1, 1))
        meta_y_train = np.tile(meta_y_val, (num_repeats, 1))

    print("META_TRAIN SHAPE: ", meta_X_train.shape)
    print("META_Y_TRAIN SHAPE: ", meta_y_train.shape)
    print("META_VAL SHAPE: ", meta_X_val.shape)
    print("META_Y_VAL SHAPE: ", meta_y_val.shape)

    print(
        f"Meta-datasets created:\n"
        f"  - Fine-tuning set: {len(meta_y_train)} samples\n"
        f"  - Validation set:  {len(meta_y_val)} samples"
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
            else (None, "safepower"),
            "FEATURE_SHIFT_METHOD": None,
            "FINGERPRINT_FEATURE": False,
            "PREPROCESS_TRANSFORMS": (
                [
                    PreprocessorConfig(
                        "none",
                        categorical_name="ordinal_very_common_categories_shuffled",
                    ),
                ]
                if args.disable_feature_preprocessing
                else default_regressor_preprocessor_configs()
            ),
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
    splitter = get_splitter(
        masking_strategy=MASKING_STRATEGY,
        mask_start_frac=MASK_START_FRAC,
        mask_len_frac=MASK_LEN_FRAC,
        random_mask_frac=RANDOM_MASK_FRAC,
        future_mask_frac=FUTURE_MASK_FRAC,
        num_chunks=NUM_CHUNKS,
        chunk_len=CHUNK_LEN,
        random_state=RANDOM_SEED,
    )

    # --- Visualize a few meta-training samples ---
    print("--- Visualizing a few meta-training samples ---")
    visualize_meta_training_samples(
        meta_X_train=meta_X_train,
        meta_y_train=meta_y_train,
        num_samples_to_visualize=min(5, NUM_META_TRAIN_SAMPLES),
        output_dir=run_dir / "meta_training_samples",
        splitter=splitter,
    )

    # Convert first dimension to list for the dataloader
    meta_X_train_list = meta_X_train.tolist()
    meta_y_train_list = meta_y_train.tolist()

    training_datasets = regressor.get_preprocessed_datasets(
        meta_X_train_list,
        meta_y_train_list,
        split_fn=splitter,
    )
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=1,  # Meta-batch-size is always 1
        collate_fn=meta_dataset_collator,
        shuffle=True,  # Shuffle to re-trigger the splitter
    )
    print("Length of finetuning_dataloader: ", len(finetuning_dataloader))

    # --- 3. Setup Optimizer ---
    optimizer = AdamWScheduleFree(
        regressor.model_.parameters(),
        lr=LEARNING_RATE,
    )
    print(f"--- 3. Optimizer Initialized: Adam, LR: {LEARNING_RATE} ---\n")

    # --- Pre-Training Validation (Baseline) ---
    baseline_metrics = run_validation(
        regressor=regressor,
        regressor_config=regressor_config,
        meta_X_val=meta_X_val,
        meta_y_val=meta_y_val,
        splitter=splitter,
        epoch=0,
        output_dir=run_dir,
        device=device,
    )

    # --- 4. Fine-tuning Loop ---
    print(f"--- 4. Starting Fine-tuning for {EPOCHS} epochs ---")
    # Track metrics for learning curves
    train_losses = [0]  # No training loss for epoch 0
    val_epochs = [0]
    val_maes = [baseline_metrics["val_mae"]]
    val_q_losses = [baseline_metrics["val_q_loss"]]

    for epoch in range(EPOCHS):
        total_loss = 0.0
        total_tying_loss = 0.0

        if isinstance(optimizer, AdamWScheduleFree):
            optimizer.train()

        pbar = tqdm(
            finetuning_dataloader, desc=f"Finetuning Epoch {epoch + 1}/{EPOCHS}"
        )

        optimizer.zero_grad()
        for i, data_batch in enumerate(pbar):
            (
                X_trains_p,
                X_tests_p,
                y_trains_p,
                y_test_std,
                cat_ixs,
                confs,
                norm_bardist,  # -> this is actually in raw space
                bardist,  # -> this is actually in normalized space
                x_train_raw,
                y_train_raw,
                x_test_raw,
                y_test_raw,
            ) = data_batch

            # loss_fn = norm_bardist[0]
            loss_fn = bardist[0]  # Calculate the loss in Z-norm space
            y_target = y_test_std[0]

            regressor.fit_from_preprocessed(
                [X_trains_p[0]], [y_trains_p[0]], cat_ixs, confs
            )
            logits, _, _ = regressor.forward([X_tests_p[0]])
            logits = logits.squeeze(0)
            if logits_smoothie_maker is not None:
                logits = logits_smoothie_maker(logits)

            pred_loss = loss_fn(logits, y_target.to(device)).mean()

            tying_loss = _weight_tying_loss(
                current_model=regressor.model_,  # type: ignore
                original_params=original_params,
                l2_sp_lambda=L2_SP_LAMBDA,
                device=device,
            )

            unnormalized_loss = (
                pred_loss * (1 - L2_SP_WEIGHT) + tying_loss * L2_SP_WEIGHT
            )

            loss = unnormalized_loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(
                finetuning_dataloader
            ):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += unnormalized_loss.item()
            total_tying_loss += tying_loss.item()

            pbar.set_postfix(
                loss=f"{unnormalized_loss.item():.4f}",
                pred_loss=f"{pred_loss.item() * (1 - L2_SP_WEIGHT):.4f}",
                tying_loss=f"{tying_loss.item() * L2_SP_WEIGHT:.4f}",
            )

        if isinstance(optimizer, AdamWScheduleFree):
            optimizer.eval()

        avg_loss = total_loss / len(finetuning_dataloader)
        train_losses.append(avg_loss)

        # --- 5. Periodic Validation on Hold-out Set ---
        if (epoch + 1) % args.validate_every_n_epochs == 0:
            val_metrics = run_validation(
                regressor=regressor,
                regressor_config=regressor_config,
                meta_X_val=meta_X_val,
                meta_y_val=meta_y_val,
                splitter=splitter,
                epoch=epoch + 1,
                output_dir=run_dir,
                device=device,
            )
            val_epochs.append(epoch + 1)
            val_maes.append(val_metrics["val_mae"])
            val_q_losses.append(val_metrics["val_q_loss"])

            plot_learning_curves(
                train_losses=train_losses,
                val_epochs=val_epochs,
                val_maes=val_maes,
                val_q_losses=val_q_losses,
                output_dir=run_dir,
            )

    print("\n--- ✅ Fine-tuning Finished ---")

    # --- 6. Plot Final Learning Curves ---
    # The plot is now updated after each epoch inside the training loop.


if __name__ == "__main__":
    main()
