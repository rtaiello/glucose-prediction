import logging
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import hydra
import lightning.pytorch as pl
import omegaconf
import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

pylogger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


class GlucoseModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        # example
        metric = torchmetrics.MeanSquaredError(squared=False)
        self.train_rmse = metric.clone()
        self.val_rmse = metric.clone()
        self.test_rmse = metric.clone()

        self.model = hydra.utils.instantiate(model)
        self.input_length = 12
        self.pred_length = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.model(x, self.input_length)

    def _step(self, batch: Dict[str, torch.Tensor], split: str) -> Mapping[str, Any]:
        x = batch
        gt_y = batch[:, -self.pred_length :, 0] * 63.60143682 + 160.87544032

        # example
        hat_y = self(x)[:, -self.pred_length :, 0] * 63.60143682 + 160.87544032
        loss = F.l1_loss(hat_y, gt_y)

        metrics = getattr(self, f"{split}_rmse")
        metrics.update(hat_y.clone().detach(), gt_y.clone().detach())

        self.log_dict(
            {
                f"rmse/{split}": metrics,
                f"loss/{split}": loss,
            },
            prog_bar=True,
            on_epoch=True,
        )

        return {"loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="train")

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="val")

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._step(batch=batch, split="test")

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3.2")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    _: pl.LightningModule = hydra.utils.instantiate(cfg.nn.module, _recursive_=False)


if __name__ == "__main__":
    main()
