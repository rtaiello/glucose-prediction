"""Federated learning client: wraps a single patient's data and local model."""

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from glucose_prediction.data.replace_bg_dataset import ReplaceBGDataset
from glucose_prediction.federated.train_utils import get_model_weights, make_model, set_model_weights, train_n_steps


class FederatedClient:
    """Holds a patient's local dataset and performs local training rounds."""

    def __init__(
        self,
        patient_id: int,
        dataset: ReplaceBGDataset,
        global_mean: List[float],
        global_std: List[float],
        batch_size: int = 32,
        lr: float = 1e-3,
        device: torch.device = torch.device("cpu"),
        input_length: int = 12,
        pred_length: int = 4,
    ) -> None:
        self.patient_id = patient_id
        self.global_mean = global_mean
        self.global_std = global_std
        self.lr = lr
        self.device = device
        self.input_length = input_length
        self.pred_length = pred_length

        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.model = make_model(input_length=input_length)

    def set_weights(self, weights: Dict) -> None:
        set_model_weights(self.model, weights)

    def get_weights(self) -> Dict:
        return get_model_weights(self.model)

    def get_n_samples(self) -> int:
        return len(self.dataloader.dataset)

    def local_train(self, n_steps: int) -> float:
        """Run n_steps gradient updates with a fresh Adam optimizer. Returns mean loss."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return train_n_steps(
            self.model,
            self.dataloader,
            optimizer,
            self.device,
            self.global_mean,
            self.global_std,
            n_steps,
            self.input_length,
            self.pred_length,
        )
