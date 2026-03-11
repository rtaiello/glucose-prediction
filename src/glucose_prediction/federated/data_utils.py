"""Data utilities for federated learning: patient splitting and dataset construction."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, DataLoader

from glucose_prediction.data.replace_bg_dataset import ReplaceBGDataset

MIN_WINDOWS = 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_all_patient_ids(data_dir: str) -> List[int]:
    """Scan data_dir for CSV files and return sorted patient IDs."""
    return sorted(
        int(f.replace(".csv", ""))
        for f in os.listdir(data_dir)
        if f.endswith(".csv")
    )


def split_patients(all_ids: List[int], n_train: int = 180, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Deterministic shuffle + split into train and test sets."""
    rng = np.random.default_rng(seed)
    shuffled = list(all_ids)
    rng.shuffle(shuffled)
    train_ids = shuffled[:n_train]
    test_ids = shuffled[n_train:]
    return train_ids, test_ids


def build_patient_dataset(
    patient_id: int,
    data_dir: str,
    global_mean: List[float],
    global_std: List[float],
    example_len: int = 16,
) -> Optional[ReplaceBGDataset]:
    """Build a ReplaceBGDataset for a single patient, returning None if insufficient data.

    Guards against zero-std features by substituting 1.0.
    """
    safe_std = [s if s > 0.0 else 1.0 for s in global_std]
    try:
        df = pd.read_csv(os.path.join(data_dir, f"{patient_id}.csv"))
        dataset = ReplaceBGDataset(df, example_len, global_mean, safe_std)
    except (ValueError, Exception):
        return None

    if len(dataset) < MIN_WINDOWS:
        return None
    return dataset


def build_test_dataloader(
    test_ids: List[int],
    data_dir: str,
    global_mean: List[float],
    global_std: List[float],
    example_len: int = 16,
    batch_size: int = 32,
) -> DataLoader:
    """Build a DataLoader over the ConcatDataset of all valid test patients."""
    datasets = []
    for pid in test_ids:
        ds = build_patient_dataset(pid, data_dir, global_mean, global_std, example_len)
        if ds is not None:
            datasets.append(ds)
    if not datasets:
        raise RuntimeError("No valid test patients found.")
    concat = ConcatDataset(datasets)
    return DataLoader(concat, batch_size=batch_size, shuffle=False, num_workers=0)
