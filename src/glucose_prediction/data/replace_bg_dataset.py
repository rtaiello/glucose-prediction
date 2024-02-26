"""
Code inspired by https://github.com/r-cui/GluPred/tree/master
"""

import datetime

import numpy as np
import torch
from torch.utils.data import Dataset


class ReplaceBGDataset(Dataset):
    """The Replace-BG dataset for Torch training."""

    def __init__(self, raw_df, example_len, external_mean=None, external_std=None, unimodal=False):
        """
        Args
            raw_df: dataframe
            example_len: int
            external_mean: [float]
                If none, self fit.
            external_std: [float]
                If none, self fit.
            unimodal: bool
                If True, data contains glucose only
        """
        raw_df.replace(to_replace=-1, value=np.nan, inplace=True)
        self.example_len = example_len
        self.unimodal = unimodal
        self.data, self.times = self._initial(raw_df)  # (len, n_features)
        self.example_indices = self._example_indices(self.times)
        self._standardise(external_mean, external_std)
        print("Dataset loaded, total examples: {}.".format(len(self)))

        # post check
        for i in range(len(self)):
            if torch.isnan(self[i]).any():
                raise ValueError("NaN detected in dataset!")

    @staticmethod
    def str2dt(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

    def _initial(self, raw_df):
        times = [self.str2dt(s) for s in raw_df["time"]]
        glucose = raw_df["GlucoseValue"].to_numpy(dtype=np.float32)
        bolus = raw_df["Normal"].to_numpy(dtype=np.float32)
        carbs = raw_df["CarbInput"].to_numpy(dtype=np.float32)

        bolus[np.isnan(bolus)] = 0.0
        carbs[np.isnan(carbs)] = 0.0

        return (
            np.array(
                [
                    glucose,
                    bolus,
                    carbs,
                ],
                dtype=np.float32,
            ).T,
            times,
        )

    def _example_indices(self, times):
        """Extract every possible example from the dataset, st. all data entry in this example is not missing.

        Returns:
            [(start_row, end_row)]
                Starting and ending indices for each possible example from this dataframe.
        """
        res = []
        total_len = self.data.shape[0]

        def look_ahead(start):
            end = start
            res = []
            while end < total_len:
                if np.any(np.isnan(self.data[end, :])):
                    break
                if end - start + 1 >= self.example_len:
                    # check that between start and end, there is the difference of self.example_len * 15 minutes
                    gap_min = self.example_len * 15
                    if (times[end] - times[end - self.example_len + 1]) <= datetime.timedelta(minutes=gap_min):
                        res.append((end - self.example_len + 1, end))
                end += 1
            return res, end

        i = 0
        while i < total_len:
            if not np.any(np.isnan(self.data[i, :])):
                temp_res, temp_end = look_ahead(i)
                res += temp_res
                i = temp_end + 1
            else:
                i += 1
        return res

    def _standardise(self, external_mean=None, external_std=None):
        if external_mean is None and external_std is None:
            mean = []
            std = []
            for i in range(self.data.shape[1]):
                mean.append(np.nanmean(self.data[:, i]))
                std.append(np.nanstd(self.data[:, i]))
        else:
            mean = external_mean
            std = external_std
        self.mean = mean
        self.std = std
        print("Standardising with mean: {} and std: {}.".format(mean, std))
        for i in range(self.data.shape[1]):
            self.data[:, i] = (self.data[:, i] - mean[i]) / std[i]

    def __len__(self):
        return len(self.example_indices)

    def __getitem__(self, idx):
        """
        Args:
            idx: int
        Returns:
            (example_len, channels)
        """
        start_row, end_row = self.example_indices[idx]
        res = torch.from_numpy(self.data[start_row : end_row + 1, :])
        # print(f"start_row: {self.times[start_row]}, end_row: {self.times[end_row +1]}")
        return res
