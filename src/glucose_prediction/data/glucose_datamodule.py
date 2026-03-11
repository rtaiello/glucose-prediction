import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import hydra
import lightning.pytorch as pl
import numpy as np
import omegaconf
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

pylogger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _compute_stats(data_dir: str, patient_ids: List[int]) -> Tuple[List[float], List[float]]:
    """Compute per-feature mean and std from raw CSVs of the given patients.

    Args:
        data_dir: directory containing ``{patient_id}.csv`` files.
        patient_ids: patient IDs to include in the computation.

    Returns:
        (mean, std) — each a list of 3 floats for [glucose, bolus, carbs].
    """
    frames: List[np.ndarray] = []
    for p in patient_ids:
        df = pd.read_csv(os.path.join(data_dir, f"{p}.csv"))
        df.replace(to_replace=-1, value=np.nan, inplace=True)
        frames.append(df[["GlucoseValue", "Normal", "CarbInput"]].to_numpy(dtype=np.float32))
    all_data = np.concatenate(frames, axis=0)
    mean = [float(np.nanmean(all_data[:, i])) for i in range(all_data.shape[1])]
    std = [float(np.nanstd(all_data[:, i])) for i in range(all_data.shape[1])]
    return mean, std


class GlucoseDataModule(pl.LightningDataModule):
    """Lightning DataModule that splits patients into train/val/test (70/10/20).

    Normalization statistics are computed from training patients only and
    applied consistently to all splits.
    """

    def __init__(
        self,
        dataset: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        accelerator: str,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory: bool = accelerator is not None and str(accelerator) == "gpu"

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.train_mean: Optional[List[float]] = None
        self.train_std: Optional[List[float]] = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        data_dir = f"{PROJECT_ROOT}/data/raw/patients"

        patients = os.listdir(data_dir)
        patients = [int(p.replace(".csv", "")) for p in patients if ".csv" in p]

        patients_training = np.random.choice(patients, int(len(patients) * 0.7), replace=False)
        remaining = list(set(patients) - set(patients_training))
        patients_validation = np.random.choice(remaining, int(len(patients) * 0.1), replace=False)
        patients_testing = list(set(remaining) - set(patients_validation))

        # Compute normalization stats from training patients only
        self.train_mean, self.train_std = _compute_stats(data_dir, patients_training.tolist())
        pylogger.info(f"Train stats — mean: {self.train_mean}, std: {self.train_std}")

        def _make_dataset(patient_ids: Union[np.ndarray, List[int]]) -> ConcatDataset:
            return ConcatDataset(
                [
                    hydra.utils.instantiate(
                        self.dataset,
                        raw_df=pd.read_csv(os.path.join(data_dir, f"{p}.csv")),
                        external_mean=self.train_mean,
                        external_std=self.train_std,
                        _recursive_=False,
                    )
                    for p in patient_ids
                ]
            )

        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_dataset is None):
            self.train_dataset = _make_dataset(patients_training)
            self.val_dataset = _make_dataset(patients_validation)
        if stage is None or stage == "test":
            self.test_dataset = _make_dataset(patients_testing)

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
        return f"{self.__class__.__name__}({self.dataset=}, {self.num_workers=}, {self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.3.2")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule."""
    m: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    m.setup()

    for _ in tqdm(m.train_dataloader()):
        pass


if __name__ == "__main__":
    main()
