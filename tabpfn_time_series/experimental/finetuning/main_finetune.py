import os
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


from tabpfn_time_series.experimental.finetuning.utils.logging_utils import (
    configure_logging,
)
from tabpfn_time_series.experimental.utils.general import OUTPUT_ROOT
from tabpfn_time_series.experimental.finetuning.data.data_module import (
    TimeSeriesDataModule,
    OverfitTestDataModule,
    SyntheticDataModule,
)
from tabpfn_time_series.experimental.finetuning.lightning_model import (
    FinetuneTabPFNModule,
)
from tabpfn_time_series.experimental.finetuning.callbacks import (
    MetricWorseningVisualizationCallback,
    ValidationVisualizationCallback,
)


logger = logging.getLogger(__name__)

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TabPFN for Time Series.")

    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=[],
        help="Tags to add to the run.",
    )

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
        type=int,
        default=None,
        metavar="N_SERIES",
        help="Run an overfitting test on a small subset of N_SERIES from the dataset.",
    )
    parser.add_argument(
        "--use_synthetic_data",
        action="store_true",
        help="Use synthetic data for debugging and testing.",
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

    parser.add_argument(
        "--validate_before_training",
        action="store_true",
        help="Run a full validation loop before starting training.",
    )

    parser.add_argument(
        "--validate_dataset",
        action="store_true",
        help="Load all data to check for issues without training.",
    )

    # Data args
    parser.add_argument("--dataset", type=str, help="Name of the dataset to use.")

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
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for dataloading. If None, will use all available cores.",
    )
    parser.add_argument("--past_length", type=int, default=1024, help="Context length.")
    parser.add_argument(
        "--future_length",
        type=int,
        default=48,
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
    parser.add_argument(
        "--visualize_every_val",
        action="store_true",
        help="Enable visualization of random samples after every validation run.",
    )

    # Wandb args
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tabpfn-ts-finetuning",
        help="Wandb project name.",
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity.")

    # Synthetic data args
    synth_group = parser.add_argument_group("Synthetic Data Generation")
    synth_group.add_argument(
        "--synthetic_num_train_series",
        type=int,
        default=100,
        help="Number of training series to generate.",
    )
    synth_group.add_argument(
        "--synthetic_num_val_series",
        type=int,
        default=20,
        help="Number of validation series to generate.",
    )
    synth_group.add_argument(
        "--synthetic_series_length",
        type=int,
        default=512,
        help="Length of each generated series.",
    )
    synth_group.add_argument(
        "--synthetic_slope", type=float, default=0.1, help="Slope of linear trend."
    )
    synth_group.add_argument(
        "--synthetic_intercept",
        type=float,
        default=0.0,
        help="Intercept of linear trend.",
    )
    synth_group.add_argument(
        "--synthetic_noise_std",
        type=float,
        default=0.1,
        help="Standard deviation of noise.",
    )

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


def _iterate_dataloader(dataloader, name: str):
    """Helper to iterate through a dataloader and log progress."""
    logger.info(f"Checking {name} data...")
    count = 0
    for i, _ in enumerate(tqdm(dataloader, desc=f"Checking {name} data")):
        count += 1

    if count == 0:
        logger.warning(f"{name} dataloader is empty.")
    else:
        logger.info(f"Processed a total of {count} {name} batches.")
        logger.info(f"{name} data loaded successfully.")


def validate_dataset(data_module: TimeSeriesDataModule):
    """
    Iterates through the entire dataset to check for integrity issues.
    """
    logger.info("--- Validating dataset integrity ---")
    logger.info("This will iterate through all data without training.")
    data_module.setup("fit")
    _iterate_dataloader(data_module.train_dataloader(), "training")

    data_module.setup("validate")
    _iterate_dataloader(data_module.val_dataloader(), "validation")


def main(args):
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Configure logging based on args
    log_level = logging.INFO
    if args.debug or args.verbose:
        log_level = logging.DEBUG

    if args.overfit_test:
        args.tags.append("overfit_test")

    # Define configs
    tabpfn_model_config = {
        "device": args.accelerator,
        "model_path": args.tabpfn_model_path,
        "inference_precision": args.precision,
    }
    if args.debug:
        tabpfn_model_config["n_estimators"] = 2
        args.past_length = 30
        args.future_length = 20

    training_config = {
        "lr": args.lr,
        "l2_sp_lambda": args.l2_sp_lambda,
        "use_schedulefree": args.use_schedulefree,
    }

    if args.use_synthetic_data:
        assert args.dataset is None, (
            "Dataset cannot be provided when using synthetic data"
        )
        args.tags.append("synthetic")
        args.dataset = "synthetic"

    # Set up directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{timestamp}_{args.dataset.replace('/', '_')}"
        f"_past{args.past_length}_future{args.future_length}"
        f"_lr{args.lr}_lambda{args.l2_sp_lambda}"
        f"_sampling{not args.disable_sampling}_seed{args.seed}"
    )
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Centralized logging setup
    configure_logging(output_dir, run_name, log_level=log_level)

    logger.info("--- Starting Fine-tuning ---")
    logger.info("Arguments:")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"  {arg_name}: {arg_value}")

    # The storage path for gift_eval datasets is typically at the repo root + /gift_eval/data
    storage_path = None
    if not args.use_synthetic_data:
        storage_path = Path(os.getenv("DATASET_STORAGE_PATH"))
        if not storage_path.exists():
            logger.error(f"Dataset storage path not found: {storage_path}")
            logger.error(
                "Please run `cd gift_eval && ./setup.sh` to download the data."
            )
            exit(1)

    model = FinetuneTabPFNModule(
        training_config=training_config,
        tabpfn_model_config=tabpfn_model_config,
        seed=args.seed,
    )

    # Setup data module
    if args.use_synthetic_data:
        logger.info("--- Using Synthetic Data ---")
        data_module = SyntheticDataModule(
            model=model.get_cpu_regressor_for_preprocessing(),
            num_train_series=args.synthetic_num_train_series,
            num_val_series=args.synthetic_num_val_series,
            series_length=args.synthetic_series_length,
            slope=args.synthetic_slope,
            intercept=args.synthetic_intercept,
            noise_std=args.synthetic_noise_std,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            past_length=args.past_length,
            future_length=args.future_length,
            enable_sampling=not args.disable_sampling,
            use_train_as_val=bool(args.overfit_test),
        )
        args.dataset = "synthetic"  # for logging purposes
    else:
        data_module_kwargs = {
            "dataset_name": args.dataset,
            "dataset_storage_path": storage_path,
            "model": model.get_cpu_regressor_for_preprocessing(),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "past_length": args.past_length,
            "future_length": args.future_length,
            "enable_sampling": not args.disable_sampling,
        }

        if args.overfit_test:
            data_module = OverfitTestDataModule(
                n_series=args.overfit_test,
                **data_module_kwargs,
            )
        else:
            data_module = TimeSeriesDataModule(**data_module_kwargs)

    if args.validate_dataset:
        try:
            validate_dataset(data_module)
            logger.info("--- Dataset validation successful ---")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}", exc_info=True)
            sys.exit(1)

    # Set up trainer
    if not args.no_wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            save_dir=str(output_dir),
            tags=[args.dataset] + args.tags,
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

    if args.visualize_every_val:
        val_visualization_callback = ValidationVisualizationCallback(
            output_dir=output_dir / "visualization_every_val",
            num_samples=5,
            random_seed=args.seed,
        )
        callbacks.append(val_visualization_callback)
        logger.info(
            f"Periodic validation visualization enabled (output dir: {val_visualization_callback.output_dir})"
        )

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
        logger.info(
            f"--- Running in overfit test mode with {args.overfit_test} series ---"
        )

        # For overfit test, we want the fastest feedback loop.
        if args.accumulate_grad_batches != 1:
            logger.warning(
                f"Overfit test mode: Overriding `accumulate_grad_batches` "
                f"(was {args.accumulate_grad_batches}) to 1 for fastest feedback."
            )
        trainer_kwargs["accumulate_grad_batches"] = 1

        # Validate once per epoch. Since batch_size=1 and we forced
        # accumulate_grad_batches=1, an epoch has N_SERIES batches/training steps.
        # We set the validation interval to match the number of series.
        val_check_interval = args.overfit_test
        if args.val_check_interval != val_check_interval:
            logger.info(
                f"Overfit test mode: Setting `val_check_interval` to {val_check_interval} "
                f"to validate exactly once per epoch (was {args.val_check_interval})."
            )
        trainer_kwargs["val_check_interval"] = val_check_interval

        # For overfit test, we usually disable callbacks like checkpointing.
        # We'll keep visualization callbacks if they were explicitly enabled.
        if trainer_kwargs["callbacks"]:
            logger.info(
                "Overfit test mode: Disabling ModelCheckpoint and EarlyStopping callbacks, but keeping visualization callbacks."
            )
            trainer_kwargs["callbacks"] = [
                cb
                for cb in trainer_kwargs["callbacks"]
                if isinstance(
                    cb,
                    (
                        MetricWorseningVisualizationCallback,
                        ValidationVisualizationCallback,
                    ),
                )
            ]
        if args.overfit_test and args.use_synthetic_data:
            args.max_steps = max(
                args.max_steps, args.overfit_test * 5
            )  # Ensure enough steps
            logger.info(
                f"Overfit test with synthetic data: Ensuring max_steps is at least {args.max_steps}"
            )
        elif args.max_steps > 512:
            logger.warning(
                f"Overfit test mode is on, but max_steps ({args.max_steps}) is high. "
                "Consider reducing it for a quick check (e.g., --max_steps 256)."
            )

    trainer = pl.Trainer(**trainer_kwargs)

    if args.validate_before_training:
        logger.info("--- Running validation before training ---")
        trainer.validate(model, datamodule=data_module)

    # Start fine-tuning
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    # The root logger is now configured by the `configure_logging` function
    # inside the `main` function. We can still log here, but the format
    # will be basic until `configure_logging` is called.
    logger.info("Starting Fine-tuning Script")

    arguments = parse_args()

    main(arguments)

    logger.info("Fine-tuning finished successfully.")
