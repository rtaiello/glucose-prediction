import logging
import os
from typing import Optional

import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT

pylogger = logging.getLogger(__name__)


class GlucoseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        accelerator: str,
    ):
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = accelerator is not None and str(accelerator) == "gpu"

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        data_dir = f"{PROJECT_ROOT}/data/raw/patients"

        patients = os.listdir(data_dir)

        patients = [int(p.replace(".csv", "")) for p in patients]
        # 70% of patients for training random idx
        patients_training = np.random.choice(patients, int(len(patients) * 0.7), replace=False)
        # 10% of patients for validation of the remaining patients
        patients_validation = np.random.choice(
            list(set(patients) - set(patients_training)), int(len(patients) * 0.1), replace=False
        )
        # 20% of patients for testing of the remaining patients
        patients_testing = list(set(patients) - set(patients_training) - set(patients_validation))

        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_dataset is None):
            self.train_dataset = torch.utils.data.ConcatDataset(
                [
                    hydra.utils.instantiate(
                        self.dataset, raw_df=pd.read_csv(os.path.join(data_dir, "{}.csv".format(p))), _recursive_=False
                    )
                    for p in patients_training
                ]
            )
            self.val_dataset = torch.utils.data.ConcatDataset(
                [
                    hydra.utils.instantiate(
                        self.dataset, raw_df=pd.read_csv(os.path.join(data_dir, "{}.csv".format(p))), _recursive_=False
                    )
                    for p in patients_validation
                ]
            )
        if stage is None or stage == "test":
            self.test_dataset = torch.utils.data.ConcatDataset(
                [
                    hydra.utils.instantiate(
                        self.dataset, raw_df=pd.read_csv(os.path.join(data_dir, "{}.csv".format(p))), _recursive_=False
                    )
                    for p in patients_testing
                ]
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            pin_memory=self.pin_memory,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.dataset=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3.2")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    m: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    m.setup()

    for _ in tqdm(m.train_dataloader()):
        pass


if __name__ == "__main__":
    main()
