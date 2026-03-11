"""Pure PyTorch training utilities for federated learning."""

import copy
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from glucose_prediction.modules.cnn_lstm import CNN_LSTM


def make_model(input_length: int = 12) -> CNN_LSTM:
    """Create a fresh randomly-initialized CNN_LSTM on CPU."""
    return CNN_LSTM(single_pred=True, d_in=3, input_length=input_length)


def train_n_steps(
    model: CNN_LSTM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_mean: List[float],
    global_std: List[float],
    n_steps: int,
    input_length: int = 12,
    pred_length: int = 4,
) -> float:
    """Run exactly n_steps gradient updates, cycling the dataloader as needed.

    Returns mean L1 loss over those steps (in mg/dL).
    """
    model.train()
    model.to(device)
    glucose_mean = global_mean[0]
    glucose_std = global_std[0]

    total_loss = 0.0
    step = 0
    data_iter = iter(dataloader)

    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch, input_length)
        hat_y = output[:, -pred_length:, 0] * glucose_std + glucose_mean
        gt_y = batch[:, -pred_length:, 0] * glucose_std + glucose_mean
        loss = F.l1_loss(hat_y, gt_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        step += 1

    return total_loss / n_steps if n_steps > 0 else 0.0


def get_model_weights(model: CNN_LSTM) -> Dict:
    """Return a deep-copied CPU state dict."""
    return copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})


def set_model_weights(model: CNN_LSTM, weights: Dict) -> None:
    """Load a state dict into the model (strict)."""
    model.load_state_dict(weights, strict=True)
