import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import torch
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, ListConfig

# Force the execution of __init__.py if this file is executed directly.
import glucose_prediction  # noqa

pylogger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))

torch.set_float32_matmul_precision("high")


def seed_index_everything(train_cfg: DictConfig, sampling_seed: int = 42) -> Optional[int]:
    """Derive a deterministic seed from ``train_cfg.seed_index`` and seed all RNGs.

    Uses a fixed ``sampling_seed`` to generate an array of candidate seeds,
    then picks the one at position ``seed_index``. This makes different
    seed_index values produce different but reproducible experiments.
    """
    if "seed_index" in train_cfg and train_cfg.seed_index is not None:
        seed_index: int = train_cfg.seed_index
        np.random.seed(sampling_seed)
        seeds = np.random.randint(np.iinfo(np.int32).max, size=max(42, seed_index + 1))
        seed = int(seeds[seed_index])
        pl.seed_everything(seed)
        pylogger.info(f"Setting seed {seed} from seeds[{seed_index}]")
        return seed
    else:
        pylogger.warning("The seed has not been set! The reproducibility is not guaranteed.")
        return None


def build_callbacks(cfg: ListConfig) -> List[Callback]:
    """Instantiate Lightning callbacks from a list of Hydra configs."""
    callbacks: List[Callback] = []
    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))
    return callbacks


def _build_logger(logging_cfg: DictConfig, fast_dev_run: bool) -> Optional[Logger]:
    """Instantiate the experiment logger from config, or return None.

    The ``logging.logger`` field selects a named sub-config (e.g. ``"wandb"``,
    ``"csv"``), or ``null`` to use Lightning's default TensorBoard logger.
    """
    logger_choice: Optional[str] = logging_cfg.get("logger", None)
    if logger_choice is None or not isinstance(logger_choice, str):
        return None

    logger_cfg = logging_cfg[logger_choice]
    is_wandb = logger_cfg["_target_"].endswith("WandbLogger")
    if fast_dev_run and is_wandb:
        pylogger.info("Setting the logger in 'offline' mode")
        logger_cfg.mode = "offline"

    pylogger.info(f"Instantiating <{logger_cfg['_target_'].split('.')[-1]}>")
    return hydra.utils.instantiate(logger_cfg)


def run(cfg: DictConfig) -> str:
    """Full train/test pipeline.

    Args:
        cfg: Hydra run configuration (conf/default.yaml).

    Returns:
        Path to the storage directory used by this run.
    """
    seed_index_everything(cfg.train)

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
        cfg.train.trainer.accelerator = "cpu"
        cfg.nn.data.num_workers.train = 0
        cfg.nn.data.num_workers.val = 0
        cfg.nn.data.num_workers.test = 0

    if cfg.core.get("tags", None) is None:
        cfg.core.tags = ["develop"]
    pylogger.info(f"Tags: {cfg.core.tags}")

    # Derive dataset example_len from the prediction horizon so windows match.
    # Input length = 3 × PH (optimal for all horizons per literature).
    interval = 15  # CGM interval in minutes
    pred_horizon: int = cfg.nn.module.get("pred_horizon_minutes", 60)
    pred_length = pred_horizon // interval
    input_length = 3 * pred_length
    example_len = input_length + pred_length  # 4 × PH in steps
    cfg.nn.data.dataset.example_len = example_len
    pylogger.info(f"Window: {input_length} input (3×PH) + {pred_length} pred = {example_len} total steps")

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    datamodule.setup(stage=None)

    # Instantiate model with train statistics for denormalization
    pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.module,
        glucose_mean=datamodule.train_mean[0],
        glucose_std=datamodule.train_std[0],
        _recursive_=False,
    )

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks)
    storage_dir: str = cfg.core.storage_dir
    logger: Optional[Logger] = _build_logger(cfg.train.logging, fast_dev_run)

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        logger=logger if logger is not None else True,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    if fast_dev_run:
        pylogger.info("Skipping testing in 'fast_dev_run' mode!")
    else:
        if datamodule.test_dataset is not None and trainer.checkpoint_callback.best_model_path is not None:
            pylogger.info("Starting testing!")
            trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

    if logger is not None and hasattr(logger, "experiment") and hasattr(logger.experiment, "finish"):
        logger.experiment.finish()

    return storage_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3.2")
def main(cfg: omegaconf.DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
