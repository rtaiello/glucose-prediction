"""Evaluation utilities: per-patient metrics and baseline/federated evaluation loops."""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from glucose_prediction.federated.data_utils import build_patient_dataset, build_test_dataloader
from glucose_prediction.federated.train_utils import make_model, train_n_steps
from glucose_prediction.modules.cnn_lstm import CNN_LSTM

pylogger = logging.getLogger(__name__)


@dataclass
class PatientResult:
    patient_id: int
    mae: float
    rmse: float
    n_windows: int


def evaluate_model(
    model: CNN_LSTM,
    dataloader: DataLoader,
    device: torch.device,
    global_mean: List[float],
    global_std: List[float],
    input_length: int = 12,
    pred_length: int = 4,
) -> Tuple[float, float]:
    """Evaluate model on a dataloader, returning (MAE, RMSE) in mg/dL."""
    model.eval()
    model.to(device)
    glucose_mean = global_mean[0]
    glucose_std = global_std[0]

    all_hat = []
    all_gt = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch, input_length)
            hat_y = output[:, -pred_length:, 0] * glucose_std + glucose_mean
            gt_y = batch[:, -pred_length:, 0] * glucose_std + glucose_mean
            all_hat.append(hat_y.cpu())
            all_gt.append(gt_y.cpu())

    hat = torch.cat(all_hat, dim=0)
    gt = torch.cat(all_gt, dim=0)
    mae = F.l1_loss(hat, gt).item()
    rmse = torch.sqrt(F.mse_loss(hat, gt)).item()
    return mae, rmse


def evaluate_local_baseline(
    train_ids: List[int],
    test_ids: List[int],
    data_dir: str,
    global_mean: List[float],
    global_std: List[float],
    total_steps: int,
    example_len: int = 16,
    input_length: int = 12,
    pred_length: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float, List[PatientResult]]:
    """Train one model per training patient for total_steps gradient steps each.

    Evaluates every model on the shared held-out test set.
    Returns (macro_avg_mae, macro_avg_rmse, per_patient_results).
    """
    test_loader = build_test_dataloader(test_ids, data_dir, global_mean, global_std, example_len, batch_size)

    results: List[PatientResult] = []
    for pid in tqdm(train_ids, desc="Local baseline"):
        ds = build_patient_dataset(pid, data_dir, global_mean, global_std, example_len)
        if ds is None:
            pylogger.warning(f"Patient {pid}: insufficient data, skipping.")
            continue

        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        model = make_model(input_length=input_length)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_n_steps(model, train_loader, optimizer, device, global_mean, global_std, total_steps, input_length, pred_length)

        mae, rmse = evaluate_model(model, test_loader, device, global_mean, global_std, input_length, pred_length)
        results.append(PatientResult(patient_id=pid, mae=mae, rmse=rmse, n_windows=len(ds)))

    if not results:
        return 0.0, 0.0, []

    avg_mae = sum(r.mae for r in results) / len(results)
    avg_rmse = sum(r.rmse for r in results) / len(results)
    return avg_mae, avg_rmse, results


def evaluate_centralized(
    train_ids: List[int],
    test_ids: List[int],
    data_dir: str,
    global_mean: List[float],
    global_std: List[float],
    total_steps: int,
    example_len: int = 16,
    input_length: int = 12,
    pred_length: int = 4,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Train a single model on pooled training data for total_steps gradient steps.

    Returns (mae, rmse) in mg/dL.
    """
    datasets = []
    for pid in tqdm(train_ids, desc="Centralized — loading data"):
        ds = build_patient_dataset(pid, data_dir, global_mean, global_std, example_len)
        if ds is not None:
            datasets.append(ds)
    if not datasets:
        raise RuntimeError("No valid training patients for centralized baseline.")

    train_loader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = build_test_dataloader(test_ids, data_dir, global_mean, global_std, example_len, batch_size)

    model = make_model(input_length=input_length)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Wrap in a simple progress bar by chunking into reporting intervals
    chunk = max(1, total_steps // 20)
    done = 0
    with tqdm(total=total_steps, desc="Centralized — training") as pbar:
        while done < total_steps:
            this_chunk = min(chunk, total_steps - done)
            train_n_steps(model, train_loader, optimizer, device, global_mean, global_std, this_chunk, input_length, pred_length)
            done += this_chunk
            pbar.update(this_chunk)

    return evaluate_model(model, test_loader, device, global_mean, global_std, input_length, pred_length)


def evaluate_federated(
    global_model: CNN_LSTM,
    test_ids: List[int],
    data_dir: str,
    global_mean: List[float],
    global_std: List[float],
    example_len: int = 16,
    input_length: int = 12,
    pred_length: int = 4,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Evaluate the global federated model on the held-out test set."""
    test_loader = build_test_dataloader(test_ids, data_dir, global_mean, global_std, example_len, batch_size)
    return evaluate_model(global_model, test_loader, device, global_mean, global_std, input_length, pred_length)
