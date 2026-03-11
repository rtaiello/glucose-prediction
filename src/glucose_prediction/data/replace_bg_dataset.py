"""Replace-BG dataset loader.

Code inspired by https://github.com/r-cui/GluPred/tree/master
"""

import datetime
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

pylogger = logging.getLogger(__name__)

# CGM readings are sampled every 15 minutes
INTERVAL_MINUTES = 15


class ReplaceBGDataset(Dataset):
    """Sliding-window dataset over a single patient's time-series.

    Each sample is a ``(example_len, 3)`` tensor with z-normalized
    [glucose, bolus, carbs] values.
    """

    def __init__(
        self,
        raw_df: pd.DataFrame,
        example_len: int,
        external_mean: List[float],
        external_std: List[float],
        unimodal: bool = False,
    ) -> None:
        raw_df.replace(to_replace=-1, value=np.nan, inplace=True)
        self.example_len = example_len
        self.unimodal = unimodal

        self.data, self.times = self._parse_features(raw_df)
        self.example_indices = self._extract_windows(self.times)
        self._standardise(external_mean, external_std)

        pylogger.info(f"Dataset loaded, total examples: {len(self)}")

        # Sanity check: no NaN should survive after standardisation
        for i in range(len(self)):
            if torch.isnan(self[i]).any():
                raise ValueError("NaN detected in dataset!")

    @staticmethod
    def _str2dt(s: str) -> datetime.datetime:
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

    def _parse_features(self, raw_df: pd.DataFrame) -> Tuple[np.ndarray, List[datetime.datetime]]:
        """Extract feature columns and timestamps from the raw dataframe.

        Returns:
            data: (N_rows, 3) float32 array [glucose, bolus, carbs].
            times: list of datetime objects aligned with data rows.
        """
        times = [self._str2dt(s) for s in raw_df["time"]]
        glucose = raw_df["GlucoseValue"].to_numpy(dtype=np.float32)
        bolus = raw_df["Normal"].to_numpy(dtype=np.float32)
        carbs = raw_df["CarbInput"].to_numpy(dtype=np.float32)

        bolus[np.isnan(bolus)] = 0.0
        carbs[np.isnan(carbs)] = 0.0

        data = np.stack([glucose, bolus, carbs], axis=1)  # (N_rows, 3)
        return data, times

    def _extract_windows(self, times: List[datetime.datetime]) -> List[Tuple[int, int]]:
        """Find all valid sliding windows of length ``example_len``.

        A window is valid when:
        - No feature value is NaN within the window.
        - The time span equals exactly ``example_len * 15`` minutes (no gaps).

        Returns:
            List of (start_row, end_row) inclusive index pairs.
        """
        indices: List[Tuple[int, int]] = []
        total_len = self.data.shape[0]
        max_gap = datetime.timedelta(minutes=self.example_len * INTERVAL_MINUTES)

        def _scan_from(start: int) -> Tuple[List[Tuple[int, int]], int]:
            end = start
            found: List[Tuple[int, int]] = []
            while end < total_len:
                if np.any(np.isnan(self.data[end, :])):
                    break
                if end - start + 1 >= self.example_len:
                    window_start = end - self.example_len + 1
                    if (times[end] - times[window_start]) <= max_gap:
                        found.append((window_start, end))
                end += 1
            return found, end

        i = 0
        while i < total_len:
            if not np.any(np.isnan(self.data[i, :])):
                windows, next_i = _scan_from(i)
                indices.extend(windows)
                i = next_i + 1
            else:
                i += 1
        return indices

    def _standardise(self, mean: List[float], std: List[float]) -> None:
        """Apply z-normalization in-place using the provided statistics."""
        self.mean = mean
        self.std = std
        for i in range(self.data.shape[1]):
            self.data[:, i] = (self.data[:, i] - mean[i]) / std[i]

    def __len__(self) -> int:
        return len(self.example_indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a single window as a ``(example_len, 3)`` float tensor."""
        start_row, end_row = self.example_indices[idx]
        return torch.from_numpy(self.data[start_row : end_row + 1, :])
