import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import hydra
import lightning.pytorch as pl
import omegaconf
import torch
import torch.nn.functional as F
import torchmetrics
from omegaconf import DictConfig
from torch.optim import Optimizer

pylogger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class GlucoseModule(pl.LightningModule):
    """Lightning module for glucose prediction.

    Wraps a CNN-LSTM model with L1 loss and RMSE tracking.
    Predictions and ground-truth are denormalized back to mg/dL
    using the training-set glucose statistics before computing metrics.
    """

    # CGM sampling interval
    INTERVAL_MINUTES: int = 15

    def __init__(
        self,
        model: DictConfig,
        pred_horizon_minutes: int = 60,
        glucose_mean: float = 0.0,
        glucose_std: float = 1.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if pred_horizon_minutes % self.INTERVAL_MINUTES != 0:
            raise ValueError(f"pred_horizon_minutes ({pred_horizon_minutes}) must be a multiple of {self.INTERVAL_MINUTES}")

        self.save_hyperparameters(logger=False, ignore=("metadata",))

        metric = torchmetrics.MeanSquaredError(squared=False)
        self.train_rmse = metric.clone()
        self.val_rmse = metric.clone()
        self.test_rmse = metric.clone()

        self.pred_length: int = pred_horizon_minutes // self.INTERVAL_MINUTES
        # Optimal input length is 3× the prediction horizon
        self.input_length: int = 3 * self.pred_length
        self.model: torch.nn.Module = hydra.utils.instantiate(model, input_length=self.input_length)
        self.glucose_mean = glucose_mean
        self.glucose_std = glucose_std

        pylogger.info(
            f"PH: {pred_horizon_minutes} min ({self.pred_length} steps), "
            f"input: 3×PH = {self.input_length} steps, "
            f"total window: {self.input_length + self.pred_length} steps"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run auto-regressive prediction over the full sequence."""
        return self.model(x, self.input_length)

    def _denormalize_glucose(self, x: torch.Tensor) -> torch.Tensor:
        """Convert z-normalized glucose values back to mg/dL."""
        return x * self.glucose_std + self.glucose_mean

    def _step(self, batch: torch.Tensor, split: str) -> Dict[str, torch.Tensor]:
        """Shared train/val/test step.

        Args:
            batch: (N, L, 3) z-normalized tensor [glucose, bolus, carbs].
            split: one of "train", "val", "test" — used for metric/log keys.
        """
        # Ground-truth: last pred_length glucose values, denormalized
        gt_y = self._denormalize_glucose(batch[:, -self.pred_length :, 0])
        # Prediction: run model then extract the same horizon
        hat_y = self._denormalize_glucose(self(batch)[:, -self.pred_length :, 0])

        loss = F.l1_loss(hat_y, gt_y)

        rmse_metric: torchmetrics.Metric = getattr(self, f"{split}_rmse")
        rmse_metric.update(hat_y.clone().detach(), gt_y.clone().detach())

        self.log_dict(
            {f"rmse/{split}": rmse_metric, f"loss/{split}": loss},
            prog_bar=True,
            on_epoch=True,
        )
        return {"loss": loss}

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._step(batch, split="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._step(batch, split="val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self._step(batch, split="test")

    def configure_optimizers(
        self,
    ) -> Union[List[Optimizer], Tuple[List[Optimizer], List[Any]]]:
        opt: Optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3.2")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module."""
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    _: pl.LightningModule = hydra.utils.instantiate(cfg.nn.module, _recursive_=False)


if __name__ == "__main__":
    main()
