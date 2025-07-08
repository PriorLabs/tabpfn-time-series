import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from dotenv import load_dotenv

from tabpfn import TabPFNRegressor
from tabpfn_time_series.experimental.finetuning.data.data_module import (
    TimeSeriesDataModule,
)
from tabpfn_time_series.experimental.finetuning.lightning_model import (
    FinetuneTabPFNModule,
)
from tabpfn_time_series.experimental.utils.general import OUTPUT_ROOT

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize validation samples and model predictions."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to use."
    )
    parser.add_argument(
        "--tabpfn_model_path",
        type=str,
        default="tabpfn-v2-regressor-2noar4o2.ckpt",
        help="Path to the base TabPFN model checkpoint.",
    )
    parser.add_argument(
        "--finetuned_checkpoint_path",
        type=str,
        default=None,
        help="Path to a fine-tuned model checkpoint (.ckpt or .pt). If none, the base model will be used.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of validation samples to visualize.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_ROOT / "visualizations"),
        help="Directory to save the plots.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--past_length", type=int, default=2048, help="Context length for the model."
    )
    parser.add_argument(
        "--future_length",
        type=int,
        default=512,
        help="Prediction length for training samples.",
    )
    return parser.parse_args()


def plot_and_save_samples(
    loader, model_for_vis, num_samples, dataset_name, output_dir, sample_type
):
    # Sanitize dataset name for use in filenames
    sanitized_dataset_name = dataset_name.replace("/", "_")

    for i, batch in enumerate(loader):
        if i >= num_samples:
            break

        logger.info(
            f"Processing and plotting {sample_type} sample {i + 1}/{num_samples}..."
        )

        x_train_raw = batch["X_train_raw"][0].cpu().numpy()
        y_train_raw = batch["y_train_raw"][0].cpu().numpy()
        x_test_raw = batch["X_test_raw"][0].cpu().numpy()
        y_test_raw = batch["y_test_raw"][0].cpu().numpy()

        # Get model prediction
        model_for_vis.fit(x_train_raw, y_train_raw)
        prediction = model_for_vis.predict(x_test_raw)

        plt.figure(figsize=(15, 7))

        train_indices = range(len(y_train_raw))
        plt.plot(train_indices, y_train_raw, label="Training Context (Input)")

        future_indices = range(len(y_train_raw), len(y_train_raw) + len(y_test_raw))
        plt.plot(
            future_indices, y_test_raw, label="Ground Truth (Future)", color="green"
        )
        plt.plot(
            future_indices,
            prediction,
            label="Model Prediction",
            color="red",
            linestyle="--",
        )

        plt.title(
            f"{sample_type.capitalize()} Sample {i + 1} - Dataset: {dataset_name}"
        )
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.5)

        plot_path = (
            output_dir / f"{sanitized_dataset_name}_{sample_type}_sample_{i + 1}.png"
        )
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved plot to {plot_path}")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(args.seed, workers=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # --- Load Model ---
    model_for_vis = None
    if args.finetuned_checkpoint_path:
        logger.info(f"Loading fine-tuned model from {args.finetuned_checkpoint_path}")
        if args.finetuned_checkpoint_path.endswith(".pt"):
            # Load raw state dict
            tabpfn_model_config = {
                "device": device,
                "model_path": args.tabpfn_model_path,
            }
            regressor = TabPFNRegressor(
                **tabpfn_model_config,
                **{"ignore_pretraining_limits": True, "fit_mode": "batched"},
                random_state=args.seed,
            )
            regressor.initialize_model()
            regressor.model_.load_state_dict(
                torch.load(args.finetuned_checkpoint_path, map_location=device)
            )
            model_for_vis = regressor
        else:  # Assumes .ckpt
            # Load from Lightning checkpoint
            lightning_model = FinetuneTabPFNModule.load_from_checkpoint(
                args.finetuned_checkpoint_path, map_location=torch.device(device)
            )
            model_for_vis = lightning_model.regressor
    else:
        logger.info(f"Loading base TabPFN model from {args.tabpfn_model_path}")
        model_for_vis = TabPFNRegressor(
            device=device,
            model_path=args.tabpfn_model_path,
            random_state=args.seed,
        )
        model_for_vis.initialize_model()

    # --- Load Data ---
    load_dotenv()
    storage_path_str = os.getenv("DATASET_STORAGE_PATH")
    if not storage_path_str or not Path(storage_path_str).exists():
        logger.error(
            f"DATASET_STORAGE_PATH ('{storage_path_str}') not found. "
            "Please run `cd gift_eval && ./setup.sh`."
        )
        return

    # DataModule requires a CPU regressor for its internal preprocessing
    cpu_regressor = TabPFNRegressor(
        device="cpu",
        model_path=args.tabpfn_model_path,
        ignore_pretraining_limits=True,
        random_state=args.seed,
    )
    cpu_regressor.initialize_model()

    data_module = TimeSeriesDataModule(
        dataset_name=args.dataset,
        dataset_storage_path=Path(storage_path_str),
        model=cpu_regressor,
        batch_size=1,
        num_workers=0,
        past_length=args.past_length,
        future_length=args.future_length,
    )
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # --- Visualize Samples ---
    logger.info("--- Visualizing Validation Samples ---")
    plot_and_save_samples(
        val_loader,
        model_for_vis,
        args.num_samples,
        args.dataset,
        output_dir,
        "validation",
    )

    logger.info("--- Visualizing Training Samples ---")
    plot_and_save_samples(
        train_loader,
        model_for_vis,
        args.num_samples,
        args.dataset,
        output_dir,
        "training",
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    arguments = parse_args()
    main(arguments)
