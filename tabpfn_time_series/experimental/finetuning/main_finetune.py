import os
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tabpfn_time_series.experimental.utils.general import OUTPUT_ROOT
from tabpfn_time_series.experimental.finetuning.data.data_module import (
    TimeSeriesDataModule,
)
from tabpfn_time_series.experimental.finetuning.lightning_model import (
    FinetuneTabPFNModule,
)
from tabpfn_time_series.experimental.finetuning.callbacks import (
    MetricWorseningVisualizationCallback,
)


logger = logging.getLogger(__name__)

load_dotenv()


class TeeStream:
    """A stream that writes to both stdout and a file."""

    def __init__(self, file_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if hasattr(self.log_file, "close"):
            self.log_file.close()


def setup_file_logging(output_dir: Path, run_name: str) -> None:
    """Set up file logging to save logs and stdout/stderr in the output directory."""
    log_file = output_dir / f"{run_name}.log"

    # Set up file handler for logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    # Redirect both stdout and stderr to also write to the log file
    tee_stream = TeeStream(log_file)
    sys.stdout = tee_stream
    sys.stderr = tee_stream

    logger.info(f"Logging to file: {log_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TabPFN for Time Series.")

    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode.")
    parser.add_argument("--no_wandb", action="store_true", help="Do not use wandb.")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the training process with PyTorch Profiler.",
    )
    parser.add_argument(
        "--overfit_test",
        action="store_true",
        help="Run an overfit test on a single batch to check for bugs.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Run in mini mode with limited validation batches (10 samples).",
    )
    parser.add_argument(
        "--checkpoint_model",
        action="store_true",
        help="Checkpoint the model based on validation metrics.",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable early stopping when val/sql stops improving (patience=15).",
    )

    # Data args
    parser.add_argument(
        "--dataset", type=str, required=True, help="Name of the dataset to use."
    )

    # Model args
    parser.add_argument(
        "--tabpfn_model_path",
        type=str,
        default="tabpfn-v2-regressor-2noar4o2.ckpt",
        help="Path to pretrained TabPFN model checkpoint.",
    )

    # Training args
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_ROOT / "finetuning"),
        help="Directory to save results.",
    )
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument(
        "--l2_sp_lambda",
        type=float,
        default=0.1,
        help="Lambda for weight tying loss.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1024,
        help="Total number of training optimizer steps (see accumulate_grad_batches).",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=128,
        help="Run validation every N training steps (should be a multiple of accumulate_grad_batches)",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=16,
        help="Number of batches for gradient accumulation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Currently, only batch size 1 is supported.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloading."
    )
    parser.add_argument("--past_length", type=int, default=2048, help="Context length.")
    parser.add_argument(
        "--future_length",
        type=int,
        default=512,
        help="Prediction length for training samples.",
    )
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="PyTorch Lightning accelerator."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Floating point precision. Default '16-mixed' enables Automated Mixed Precision for faster training and lower memory usage. Use '32-true' for full precision.",
    )
    parser.add_argument(
        "--disable_sampling",
        action="store_true",
        help="Disable random sampling during training. When enabled, training mode will use validation-like sampling (single window from the end of each series) instead of random sampling.",
    )

    # Visualization args
    parser.add_argument(
        "-vis",
        "--enable_metric_visualization",
        action="store_true",
        help="Enable automatic visualization when validation metrics worsen.",
    )

    # Wandb args
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tabpfn-ts-finetuning",
        help="Wandb project name.",
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity.")

    # Schedule-free optimizer args
    parser.add_argument(
        "--no_schedulefree",
        action="store_false",
        dest="use_schedulefree",
        help="Disable schedule-free optimizer and use regular Adam instead. "
        "By default, schedule-free optimizer (AdamWScheduleFree) is used with "
        "sensible defaults (10 warmup steps, beta1=0.9).",
    )

    args = parser.parse_args()

    return args


def main(args):
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Define configs
    tabpfn_model_config = {
        "device": args.accelerator,
        "model_path": args.tabpfn_model_path,
        "inference_precision": args.precision,
    }
    if args.debug:
        tabpfn_model_config["n_estimators"] = 2

    training_config = {
        "lr": args.lr,
        "l2_sp_lambda": args.l2_sp_lambda,
        "use_schedulefree": args.use_schedulefree,
    }

    # Set up directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer_suffix = "sf" if args.use_schedulefree else "adam"
    run_name = (
        f"{timestamp}_{args.dataset.replace('/', '_')}"
        f"_past{args.past_length}_future{args.future_length}"
        f"_lr{args.lr}_lambda{args.l2_sp_lambda}"
        f"_{optimizer_suffix}_seed{args.seed}"
    )
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_file_logging(output_dir, run_name)

    logger.info("--- Starting Fine-tuning ---")
    logger.info("Arguments:")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"  {arg_name}: {arg_value}")

    # The storage path for gift_eval datasets is typically at the repo root + /gift_eval/data
    storage_path = Path(os.getenv("DATASET_STORAGE_PATH"))
    if not storage_path.exists():
        logger.error(f"Dataset storage path not found: {storage_path}")
        logger.error("Please run `cd gift_eval && ./setup.sh` to download the data.")
        exit(1)

    model = FinetuneTabPFNModule(
        training_config=training_config,
        tabpfn_model_config=tabpfn_model_config,
        seed=args.seed,
    )

    data_module = TimeSeriesDataModule(
        dataset_name=args.dataset,
        dataset_storage_path=storage_path,
        model=model.get_cpu_regressor_for_preprocessing(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        past_length=args.past_length if not args.debug else 30,
        future_length=args.future_length if not args.debug else 20,
        enable_sampling=not args.disable_sampling,
    )

    # Set up trainer
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            save_dir=str(output_dir),
            tags=[args.dataset],
        )
    else:
        wandb_logger = None

    # Set up callbacks
    callbacks = []

    # Add early stopping callback
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val/sql",
            mode="min",
            patience=20,
            min_delta=0.0,
            verbose=True,
        )
        callbacks.append(early_stop_callback)
        logger.info(
            "Early stopping enabled: monitoring 'val/sql' (mode: min, patience: 20)"
        )

    # Add model checkpointing callbacks
    if args.checkpoint_model:
        for metric in ["val/mase", "val/sql"]:
            # Create a safe directory name (replace '/' with '_')
            safe_metric_name = metric.replace("/", "_")
            checkpoint_callback = ModelCheckpoint(
                monitor=metric,
                mode="min",
                save_top_k=2,  # Save top 2 best models for each metric
                save_last=False,  # Only save last once
                dirpath=output_dir / "checkpoints" / safe_metric_name,
                filename=f"best-{safe_metric_name}-step{{step:04d}}-{{{metric}:.4f}}",
                auto_insert_metric_name=False,
            )
            callbacks.append(checkpoint_callback)

            logger.info(
                f"Model checkpointing enabled: monitoring '{metric}' (mode: min)"
            )

    if args.enable_metric_visualization:
        visualization_callback = MetricWorseningVisualizationCallback(
            monitor="val/mase",
            mode="min",
            output_dir=output_dir / "visualization",
        )
        callbacks.append(visualization_callback)

    trainer_kwargs = {
        "max_steps": args.max_steps,
        "val_check_interval": args.val_check_interval,
        "accelerator": args.accelerator,
        "precision": args.precision,
        "logger": wandb_logger,
        "callbacks": callbacks,
        "log_every_n_steps": 1,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "profiler": "simple" if args.profile else None,
    }

    if args.mini:
        logger.info("--- Running in mini mode ---")
        trainer_kwargs["limit_val_batches"] = 10

    if args.overfit_test:
        logger.info("--- Running in overfit test mode ---")
        trainer_kwargs["overfit_batches"] = 10
        trainer_kwargs.pop("val_check_interval", None)
        # Disable callbacks for overfit test (no checkpointing or visualization needed)
        trainer_kwargs["callbacks"] = []
        trainer_kwargs["accumulate_grad_batches"] = 1

    trainer = pl.Trainer(**trainer_kwargs)

    # Start fine-tuning
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Starting Fine-tuning Script")

    arguments = parse_args()

    if arguments.debug or arguments.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main(arguments)

    logger.info("Fine-tuning finished successfully.")
