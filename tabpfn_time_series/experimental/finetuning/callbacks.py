import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import random

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

logger = logging.getLogger(__name__)


class MonitorMixin:
    """Mixin class that provides monitor/mode functionality similar to ModelCheckpoint and EarlyStopping.

    This encapsulates the common pattern of monitoring a metric and determining when it worsens/improves.
    """

    def __init__(self, monitor: str, mode: str):
        """Initialize monitoring functionality.

        Args:
            monitor: Metric to monitor (e.g., "val/mase", "val/sql")
            mode: "min" triggers when metric increases (worse), "max" when it decreases (worse)
        """
        self.monitor = monitor
        self.mode = mode

        # Initialize best metric value and comparison operation
        if mode == "min":
            self.best_metric = float("inf")
            self.monitor_op = lambda current, best: current > best
        elif mode == "max":
            self.best_metric = float("-inf")
            self.monitor_op = lambda current, best: current < best
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

    def check_metric_worsened(self, trainer: pl.Trainer) -> tuple[bool, float]:
        """Check if the monitored metric has worsened.

        Args:
            trainer: PyTorch Lightning trainer instance

        Returns:
            Tuple of (metric_worsened: bool, current_metric_value: float)
        """
        current_metric = trainer.logged_metrics.get(self.monitor)
        if current_metric is None:
            logger.warning(f"Metric '{self.monitor}' not found in logged metrics")
            return False, float("nan")

        current_metric = float(current_metric)
        metric_worsened = self.monitor_op(current_metric, self.best_metric)

        if not metric_worsened:
            # Update best metric if it improved
            self.best_metric = current_metric

        return metric_worsened, current_metric


class MetricWorseningVisualizationCallback(Callback, MonitorMixin):
    """PyTorch Lightning callback to visualize predictions when a validation metric worsens.

    This callback monitors a single validation metric and automatically generates
    visualizations of random validation samples collected across different validation runs
    when the metric gets worse compared to the previous validation run.

    Uses MonitorMixin to follow the same monitor/mode pattern as ModelCheckpoint and EarlyStopping.
    """

    def __init__(
        self,
        monitor: str = "val/mase",
        mode: str = "min",
        output_dir: str = "output/visualizations",
        max_samples: int = 10,
        enable_visualization: bool = True,
        random_seed: Optional[int] = None,
    ):
        """Initialize the callback.

        Args:
            monitor: Metric to monitor (e.g., "val/mase", "val/sql")
            mode: "min" triggers when metric increases (worse), "max" when it decreases (worse)
            output_dir: Directory to save visualization plots
            max_samples: Maximum number of validation samples to store across all validation runs
            enable_visualization: Whether to enable visualization
            random_seed: Random seed for reproducible sample selection (optional)
        """
        Callback.__init__(self)
        MonitorMixin.__init__(self, monitor=monitor, mode=mode)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_samples = max_samples
        self.enable_visualization = enable_visualization
        self.random_seed = random_seed

        # Initialize random number generator
        self.rng = random.Random(random_seed)

        # Store validation samples across different validation runs
        self.stored_samples: List[Dict[str, Any]] = []
        # Track current validation run samples before random selection
        self.current_val_run_samples: List[Dict[str, Any]] = []

        logger.info("Initialized MetricWorseningVisualizationCallback:")
        logger.info(f"  Monitor: {self.monitor} (mode: {self.mode})")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Max samples: {self.max_samples} (across all validation runs)")
        logger.info(f"  Visualization enabled: {self.enable_visualization}")
        logger.info(f"  Random seed: {self.random_seed}")

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Store validation samples from current run for potential random selection."""
        if not self.enable_visualization:
            return

        # Extract prediction data from validation outputs
        if "prediction_data" in outputs:
            prediction_data = outputs["prediction_data"].copy()
            # Add the metrics for this sample
            prediction_data["sample_metrics"] = {
                metric: outputs[metric]
                for metric in ["mse", "mae", "r2", "mase", "sql"]
                if metric in outputs
            }
            # Add metadata about which validation run this came from
            prediction_data["validation_step"] = trainer.global_step
            prediction_data["validation_epoch"] = trainer.current_epoch

            self.current_val_run_samples.append(prediction_data)
        else:
            logger.warning(
                f"No prediction_data found in validation outputs for batch {batch_idx}. "
                "Make sure the Lightning module returns prediction_data."
            )

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Check for metric worsening and update stored samples with random selection from current run."""
        if not self.enable_visualization:
            return

        # Randomly select samples from current validation run to add to stored samples
        self._update_stored_samples_with_random_selection()

        # Get the current metric value from logged metrics
        current_metric = trainer.logged_metrics.get(self.monitor)
        current_metric = float(current_metric)

        # Check if metric worsened using the monitor operation
        metric_worsened = self.monitor_op(current_metric, self.best_metric)

        if metric_worsened:
            logger.info(
                f"At step {trainer.global_step}, metric {self.monitor} worsened: {self.best_metric:.4f} → {current_metric:.4f} "
                f"Generating visualizations using {len(self.stored_samples)} stored samples from different validation runs..."
            )

            self._visualize_validation_samples(
                global_step=trainer.global_step,
                current_metric=current_metric,
                previous_metric=self.best_metric,
            )
        else:
            # Update best metric if it improved
            self.best_metric = current_metric
            logger.debug(
                f"Metric {self.monitor} improved/maintained: {current_metric:.4f} "
                f"(global step {trainer.global_step}). Stored samples: {len(self.stored_samples)}"
            )

        # Clear current validation run data (but keep stored samples)
        self._clear_current_validation_run_data()

    def _update_stored_samples_with_random_selection(self):
        """Update stored samples by randomly selecting from current validation run."""
        if not self.current_val_run_samples:
            return

        # If we haven't reached max capacity, add random samples from current run
        if len(self.stored_samples) < self.max_samples:
            # Calculate how many samples we can add
            available_slots = self.max_samples - len(self.stored_samples)
            num_to_add = min(available_slots, len(self.current_val_run_samples))

            # Randomly select samples to add
            samples_to_add = self.rng.sample(self.current_val_run_samples, num_to_add)
            self.stored_samples.extend(samples_to_add)

            logger.debug(
                f"Added {num_to_add} random samples from current validation run. "
                f"Total stored samples: {len(self.stored_samples)}"
            )
        else:
            # We're at capacity, randomly replace some existing samples
            num_to_replace = min(len(self.current_val_run_samples), self.max_samples)

            # Randomly select which stored samples to replace
            replace_indices = self.rng.sample(
                range(len(self.stored_samples)), num_to_replace
            )
            # Randomly select which current samples to use as replacements
            replacement_samples = self.rng.sample(
                self.current_val_run_samples, num_to_replace
            )

            # Replace the selected samples
            for idx, new_sample in zip(replace_indices, replacement_samples):
                self.stored_samples[idx] = new_sample

            logger.debug(
                f"Replaced {num_to_replace} stored samples with random samples from current validation run. "
                f"Total stored samples: {len(self.stored_samples)}"
            )

    def _clear_current_validation_run_data(self):
        """Clear data stored for the current validation run only."""
        self.current_val_run_samples = []

    def _visualize_validation_samples(
        self,
        global_step: int,
        current_metric: float,
        previous_metric: float,
    ) -> None:
        """Generate and save visualizations for stored validation samples."""
        if not self.stored_samples:
            logger.warning("No validation samples stored for visualization")
            return

        # Create validation-run-specific output directory
        run_dir = self.output_dir / f"step_{global_step}_metric_worsening"
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Visualizing {len(self.stored_samples)} validation samples "
            f"collected from different validation runs (triggering at step {global_step})"
        )

        # Generate visualizations
        for i, sample in enumerate(self.stored_samples):
            try:
                self._create_and_save_plot(
                    sample=sample,
                    sample_idx=i,
                    global_step=global_step,
                    current_metric=current_metric,
                    previous_metric=previous_metric,
                    output_dir=run_dir,
                )

            except Exception as e:
                logger.error(f"Failed to visualize sample {i + 1}: {e}")
                plt.close()  # Ensure plot is closed even on error

        logger.info(f"Visualization complete. Plots saved to {run_dir}")

    def _create_and_save_plot(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        global_step: int,
        current_metric: float,
        previous_metric: float,
        output_dir: Path,
    ) -> None:
        """Create and save a single visualization plot using cached prediction."""
        # Extract data from the cached validation results
        y_train = sample["y_train_raw"]
        y_test = sample["y_test_raw"]
        full_pred = sample["full_pred_on_test"]
        batch_idx = sample["batch_idx"]
        sample_validation_step = sample.get("validation_step", "unknown")
        sample_validation_epoch = sample.get("validation_epoch", "unknown")

        # Use the median prediction for visualization (like the original visualize_samples.py)
        prediction = full_pred["median"]

        # Create plot
        plt.figure(figsize=(15, 7))

        # Plot training context
        train_indices = range(len(y_train))
        plt.plot(
            train_indices,
            y_train,
            label="Training Context (Input)",
            linewidth=2,
            color="blue",
        )

        # Plot ground truth and prediction
        future_indices = range(len(y_train), len(y_train) + len(y_test))
        plt.plot(
            future_indices,
            y_test,
            label="Ground Truth (Future)",
            color="green",
            linewidth=2,
        )
        plt.plot(
            future_indices,
            prediction,
            label="Model Prediction (from validation)",
            color="red",
            linestyle="--",
            linewidth=2,
        )

        # Add confidence intervals if available
        if "quantiles" in full_pred:
            quantiles = full_pred["quantiles"]
            # Typically quantiles are [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # Use 0.1 and 0.9 quantiles for 80% confidence interval
            if len(quantiles) >= 9:  # Ensure we have enough quantiles
                lower_bound = quantiles[0]  # 0.1 quantile
                upper_bound = quantiles[-1]  # 0.9 quantile
                plt.fill_between(
                    future_indices,
                    lower_bound,
                    upper_bound,
                    alpha=0.2,
                    color="red",
                    label="80% Prediction Interval",
                )

        # Create title with metric information
        metric_name = self.monitor.replace("val/", "").upper()
        metric_change = f"{metric_name}: {previous_metric:.4f} → {current_metric:.4f}"

        # Add sample-specific metrics if available
        sample_metrics_info = ""
        if "sample_metrics" in sample:
            sample_metrics = sample["sample_metrics"]
            sample_metrics_info = f" | Sample MASE: {sample_metrics.get('mase', 'N/A'):.3f}, SQL: {sample_metrics.get('sql', 'N/A'):.3f}"

        plt.title(
            f"Sample {sample_idx + 1} (from step {sample_validation_step}, epoch {sample_validation_epoch}) - "
            f"Triggered at Step {global_step} - Metric Worsened: {metric_change}{sample_metrics_info}"
        )
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_path = (
            output_dir
            / f"sample_{sample_idx + 1}_from_step_{sample_validation_step}_batch_{batch_idx}.png"
        )
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close()

        logger.debug(f"Saved visualization to {plot_path}")
