import logging
from dataclasses import dataclass

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn.metrics import mean_absolute_error, r2_score
from gluonts.evaluation.metrics import mse

from tabpfn import TabPFNRegressor

from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn_time_series.experimental.utils.metrics import compute_mase, compute_sql


logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation metrics."""

    mse: float
    mae: float
    r2: float
    mase: float
    sql: float


TABPFN_ENABLE_FINETUNING_KWARGS = {
    "ignore_pretraining_limits": True,
    "differentiable_input": False,
    "fit_mode": "batched",
}

TABPFN_FINETUNING_FIXED_BATCH_SIZE = 1

PRECISION_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


class FinetuneTabPFNModule(pl.LightningModule):
    """PyTorch Lightning module for TabPFN Time Series fine-tuning.

    This pipeline is currently only designed for batch size of 1.
    """

    def __init__(
        self,
        training_config: dict,
        tabpfn_model_config: dict,
        seed: int = 42,
    ):
        super().__init__()
        self.seed = seed
        self.tabpfn_model_config = self._parse_model_config(tabpfn_model_config)
        self.regressor = TabPFNRegressor(
            **self.tabpfn_model_config,
            **TABPFN_ENABLE_FINETUNING_KWARGS,
            random_state=self.seed,
        )
        self.regressor.initialize_model()
        self.training_config = training_config
        self.save_hyperparameters(ignore=["regressor"])

        self.original_params_ = None

    def get_cpu_regressor_for_preprocessing(self) -> TabPFNRegressor:
        """Create a CPU-based copy of the regressor for data preprocessing."""
        cpu_config = self.tabpfn_model_config.copy()
        cpu_config["device"] = "cpu"

        cpu_regressor = TabPFNRegressor(
            **cpu_config,
            **TABPFN_ENABLE_FINETUNING_KWARGS,
            random_state=self.seed,
        )
        cpu_regressor.initialize_model()
        return cpu_regressor

    def forward(self, X_tests_preprocessed):
        """Forward pass through the regressor."""
        return self.regressor.forward(X_tests_preprocessed)

    def training_step(self, batch, batch_idx):
        """Execute a single training step."""
        # Unpack batch
        logger.debug(f"Training step batch: {batch_idx}")

        # The batch is already processed by the collate_fn and has a batch size of 1
        X_train_preprocessed = batch["X_train_preprocessed"]
        y_train_standardized = batch["y_train_standardized"]
        cat_ixs = batch["cat_ixs"]
        conf = batch["conf"]
        X_test_preprocessed = batch["X_test_preprocessed"]
        y_test_standardized = batch["y_test_standardized"]
        normalized_bardist = batch["normalized_bardist"][0]  # Un-list

        # Forward pass
        self.regressor.fit_from_preprocessed(
            X_train_preprocessed,
            y_train_standardized,
            cat_ixs,
            conf,
        )

        # Lazily initialize original_params_ after model is loaded
        if self.original_params_ is None:
            self.original_params_ = {
                name: p.clone().detach()
                for name, p in self.regressor.model_.named_parameters()
            }

        averaged_pred_logits, _, _ = self.forward(X_test_preprocessed)

        # Calculate loss
        loss_fn = normalized_bardist
        pred_loss_per_sample = loss_fn(
            averaged_pred_logits,
            y_test_standardized.to(self.device),
        )
        pred_loss = pred_loss_per_sample.mean()

        if torch.isinf(pred_loss):
            logger.warning(f"Batch: {batch_idx}, Train loss is inf")

        tying_loss = self._weight_tying_loss(
            current_model=self.regressor.model_,
            original_params=self.original_params_,
            l2_sp_lambda=self.training_config["l2_sp_lambda"],
            device=self.device,
        )

        loss = pred_loss + tying_loss

        self.log(
            "train/pred_loss", pred_loss, on_step=True, on_epoch=False, batch_size=1
        )
        self.log(
            "train/tying_loss", tying_loss, on_step=True, on_epoch=False, batch_size=1
        )
        self.log("train/total_loss", loss, on_step=True, on_epoch=False, batch_size=1)

        return loss

    def validation_step(self, batch, batch_idx):
        """Execute a single validation step."""
        eval_model = clone_model_for_evaluation(
            original_model=self.regressor,
            eval_init_args=self.tabpfn_model_config,
            model_class=TabPFNRegressor,
        )

        # Remove batch dimension before converting to numpy
        x_train_raw_numpy = batch["X_train_raw"][0].cpu().numpy()
        y_train_raw_numpy = batch["y_train_raw"][0].cpu().numpy()
        x_test_raw_numpy = batch["X_test_raw"][0].cpu().numpy()
        y_test_raw_numpy = batch["y_test_raw"][0].cpu().numpy()

        eval_model.fit(x_train_raw_numpy, y_train_raw_numpy)
        full_pred_on_test = eval_model.predict(x_test_raw_numpy, output_type="full")

        metrics_on_test = self._compute_metrics(
            full_pred=full_pred_on_test,
            y_train=y_train_raw_numpy,
            y_test=y_test_raw_numpy,
        )

        # Log metrics and prepare return dictionary
        result_dict = {}
        for field_name in EvalResult.__dataclass_fields__:
            field_value = getattr(metrics_on_test, field_name)
            self.log(
                f"val/{field_name}",
                field_value,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
            result_dict[field_name] = field_value

        return result_dict

    @staticmethod
    def _compute_metrics(
        full_pred: dict[str, np.ndarray],
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> EvalResult:
        """Calculate standard regression metrics for model evaluation."""

        return EvalResult(
            mse=mse(y_test, full_pred["mean"]),
            r2=r2_score(y_test, full_pred["mean"]),
            mae=mean_absolute_error(y_test, full_pred["median"]),
            mase=compute_mase(
                y_test=y_test,
                y_pred=full_pred["median"],
                y_train=y_train,
            ),
            sql=compute_sql(
                y_test=y_test,
                pred_quantiles=full_pred["quantiles"],
                y_train=y_train,
            ),
        )

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        return torch.optim.Adam(
            self.regressor.model_.parameters(), lr=self.training_config["lr"]
        )

    @staticmethod
    def _parse_model_config(raw_model_config: dict) -> dict:
        new_model_config = raw_model_config.copy()
        new_model_config["inference_precision"] = PRECISION_MAP[
            raw_model_config["inference_precision"]
        ]
        return new_model_config

    @staticmethod
    def _weight_tying_loss(
        current_model: torch.nn.Module,
        original_params: dict[str, torch.Tensor],
        l2_sp_lambda: float,
        device: torch.device,
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
