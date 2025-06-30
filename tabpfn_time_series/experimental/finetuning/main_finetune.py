import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint


from tabpfn_time_series.experimental.utils.general import OUTPUT_ROOT
from tabpfn_time_series.experimental.finetuning.data.data_module import (
    TimeSeriesDataModule,
)
from tabpfn_time_series.experimental.finetuning.lightning_model import (
    FinetuneTabPFNModule,
)


logger = logging.getLogger(__name__)

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TabPFN for Time Series.")

    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")

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
        default=1000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=200,
        help="Run validation every N training steps.",
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
    parser.add_argument("--past_length", type=int, default=1024, help="Context length.")
    parser.add_argument(
        "--future_length",
        type=int,
        default=1024,
        help="Prediction length for training samples.",
    )
    parser.add_argument(
        "--accelerator", type=str, default="auto", help="PyTorch Lightning accelerator."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        help="Floating point precision.",
    )

    # Wandb args
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="tabpfn-ts-finetuning",
        help="Wandb project name.",
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity.")
    parser.add_argument(
        "--save_model", action="store_true", help="Save the fine-tuned model weights."
    )

    args = parser.parse_args()
    logger.info("Arguments:")
    for arg_name, arg_value in vars(args).items():
        logger.info(f"  {arg_name}: {arg_value}")

    return args


def main(args):
    # Set up logging and output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define configs
    tabpfn_model_config = {
        "device": args.accelerator,
        "model_path": args.tabpfn_model_path,
        "inference_precision": "float32" if args.precision == "32-true" else "float16",
    }
    if args.debug:
        tabpfn_model_config["n_estimators"] = 2

    training_config = {
        "lr": args.lr,
        "l2_sp_lambda": args.l2_sp_lambda,
    }

    # The storage path for gift_eval datasets is typically at the repo root + /gift_eval/data
    storage_path = Path(os.getenv("DATASET_STORAGE_PATH"))
    if not storage_path.exists():
        logger.error(f"Dataset storage path not found: {storage_path}")
        logger.error("Please run `cd gift_eval && ./setup.sh` to download the data.")
        exit(1)

    print("Initiating FinetuneTabPFNModule")
    model = FinetuneTabPFNModule(
        training_config=training_config,
        tabpfn_model_config=tabpfn_model_config,
    )

    print("Initiating TimeSeriesDataModule")
    data_module = TimeSeriesDataModule(
        dataset_name=args.dataset,
        dataset_storage_path=storage_path,
        model=model.get_cpu_regressor_for_preprocessing(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        past_length=args.past_length if not args.debug else 5,
        future_length=args.future_length if not args.debug else 3,
    )

    # Set up trainer
    run_name = f"{args.dataset}_lr{args.lr}_lambda{args.l2_sp_lambda}"
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        save_dir=str(output_dir),
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints" / run_name),
        filename="{epoch}-{step}-{val/mae:.4f}",
        save_top_k=1,
        monitor="val/mae",
        mode="min",
    )

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        accelerator=args.accelerator,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )

    # Start fine-tuning
    trainer.fit(model, datamodule=data_module)

    # Save the fine-tuned model weights
    logger.info("=" * 20)
    if args.save_model:
        if checkpoint_callback.best_model_path:
            logger.info(
                f"Loading best model from: {checkpoint_callback.best_model_path}"
            )
            best_model = FinetuneTabPFNModule.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )

            model_save_dir = output_dir / "models"
            model_save_dir.mkdir(parents=True, exist_ok=True)
            model_save_path = model_save_dir / f"{run_name}.pt"

            torch.save(best_model.regressor.model_.state_dict(), model_save_path)
            logger.info(
                f"Successfully saved fine-tuned model weights to: {model_save_path}"
            )
        else:
            logger.warning("No best model checkpoint found to save.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Starting Fine-tuning Script")

    arguments = parse_args()

    if arguments.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    main(arguments)

    logger.info("Fine-tuning finished successfully.")
