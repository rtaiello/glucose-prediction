"""Evaluation utilities: per-patient metrics and baseline/federated evaluation loops."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from glucose_prediction.federated.data_utils import build_patient_dataset, build_test_dataloader
from glucose_prediction.federated.train_utils import make_model, train_n_steps
from glucose_prediction.modules.cnn_lstm import CNN_LSTM

pylogger = logging.getLogger(__name__)

HYPO_THRESH = 70.0
HYPER_THRESH = 180.0


@dataclass
class PatientResult:
    patient_id: int
    mae: float
    rmse: float
    r2: float
    n_windows: int


@dataclass
class ZoneMetrics:
    """Per-zone classification recall and overall zone accuracy."""
    hypo_recall: float     # sensitivity for hypoglycemia  (<70 mg/dL)
    normal_recall: float   # sensitivity for normal range  (70-180 mg/dL)
    hyper_recall: float    # sensitivity for hyperglycemia (>180 mg/dL)
    hypo_n: int            # # true hypo timesteps in test set
    normal_n: int
    hyper_n: int
    overall_acc: float     # fraction of timesteps with correct zone label


def _zone_labels(values: torch.Tensor) -> torch.Tensor:
    """Map glucose values (mg/dL) to zone indices: 0=hypo, 1=normal, 2=hyper."""
    labels = torch.ones(values.shape, dtype=torch.long)
    labels[values < HYPO_THRESH] = 0
    labels[values > HYPER_THRESH] = 2
    return labels


def _compute_zone_metrics(hat: torch.Tensor, gt: torch.Tensor) -> ZoneMetrics:
    """Compute per-zone recall and overall accuracy from denormalized predictions."""
    hat_flat = hat.reshape(-1)
    gt_flat = gt.reshape(-1)
    hat_z = _zone_labels(hat_flat)
    gt_z = _zone_labels(gt_flat)

    total = gt_flat.shape[0]
    overall_acc = (hat_z == gt_z).sum().item() / total if total > 0 else 0.0

    def _recall(zone_id: int) -> Tuple[float, int]:
        mask = gt_z == zone_id
        n = mask.sum().item()
        if n == 0:
            return 0.0, 0
        tp = ((hat_z == zone_id) & mask).sum().item()
        return tp / n, n

    hypo_recall, hypo_n = _recall(0)
    normal_recall, normal_n = _recall(1)
    hyper_recall, hyper_n = _recall(2)
    return ZoneMetrics(
        hypo_recall=hypo_recall, normal_recall=normal_recall, hyper_recall=hyper_recall,
        hypo_n=hypo_n, normal_n=normal_n, hyper_n=hyper_n, overall_acc=overall_acc,
    )


def evaluate_model(
    model: CNN_LSTM,
    dataloader: DataLoader,
    device: torch.device,
    global_mean: List[float],
    global_std: List[float],
    input_length: int = 12,
    pred_length: int = 4,
) -> Tuple[float, float, float, ZoneMetrics]:
    """Evaluate model on a dataloader, returning (MAE, RMSE, R², ZoneMetrics) in mg/dL."""
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

    # R² = 1 - SS_res / SS_tot  (how much variance the model explains vs. always predicting the mean)
    gt_flat = gt.reshape(-1)
    hat_flat = hat.reshape(-1)
    ss_res = ((gt_flat - hat_flat) ** 2).sum().item()
    ss_tot = ((gt_flat - gt_flat.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    zone_metrics = _compute_zone_metrics(hat, gt)
    return mae, rmse, r2, zone_metrics


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
    model_save_dir: Optional[str] = None,
) -> Tuple[float, float, float, ZoneMetrics, List[PatientResult]]:
    """Train one model per training patient for total_steps gradient steps each.

    Evaluates every model on the shared held-out test set.
    Returns (macro_avg_mae, macro_avg_rmse, macro_avg_r2, macro_avg_zone_metrics, per_patient_results).
    If model_save_dir is set, saves each patient's state dict to {model_save_dir}/patient_{pid}.pt.
    """
    if model_save_dir is not None:
        Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    test_loader = build_test_dataloader(test_ids, data_dir, global_mean, global_std, example_len, batch_size)

    results: List[PatientResult] = []
    all_zone_metrics: List[ZoneMetrics] = []
    for pid in tqdm(train_ids, desc="Local baseline"):
        ds = build_patient_dataset(pid, data_dir, global_mean, global_std, example_len)
        if ds is None:
            pylogger.warning(f"Patient {pid}: insufficient data, skipping.")
            continue

        train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        model = make_model(input_length=input_length)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        train_n_steps(model, train_loader, optimizer, device, global_mean, global_std, total_steps, input_length, pred_length)

        if model_save_dir is not None:
            torch.save(model.state_dict(), Path(model_save_dir) / f"patient_{pid}.pt")

        mae, rmse, r2, zone_metrics = evaluate_model(model, test_loader, device, global_mean, global_std, input_length, pred_length)
        results.append(PatientResult(patient_id=pid, mae=mae, rmse=rmse, r2=r2, n_windows=len(ds)))
        all_zone_metrics.append(zone_metrics)

    if not results:
        return 0.0, 0.0, 0.0, ZoneMetrics(0.0, 0.0, 0.0, 0, 0, 0, 0.0), []

    avg_mae = sum(r.mae for r in results) / len(results)
    avg_rmse = sum(r.rmse for r in results) / len(results)
    avg_r2 = sum(r.r2 for r in results) / len(results)

    # Macro-average zone recalls; counts from first model (same test set for all models)
    n = len(all_zone_metrics)
    avg_zones = ZoneMetrics(
        hypo_recall=sum(zm.hypo_recall for zm in all_zone_metrics) / n,
        normal_recall=sum(zm.normal_recall for zm in all_zone_metrics) / n,
        hyper_recall=sum(zm.hyper_recall for zm in all_zone_metrics) / n,
        hypo_n=all_zone_metrics[0].hypo_n,
        normal_n=all_zone_metrics[0].normal_n,
        hyper_n=all_zone_metrics[0].hyper_n,
        overall_acc=sum(zm.overall_acc for zm in all_zone_metrics) / n,
    )
    return avg_mae, avg_rmse, avg_r2, avg_zones, results


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
    model_save_path: Optional[str] = None,
) -> Tuple[float, float, float, ZoneMetrics]:
    """Train a single model on pooled training data for total_steps gradient steps.

    Returns (mae, rmse, r2, zone_metrics) in mg/dL.
    If model_save_path is set, saves the model state dict there.
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

    chunk = max(1, total_steps // 20)
    done = 0
    with tqdm(total=total_steps, desc="Centralized — training") as pbar:
        while done < total_steps:
            this_chunk = min(chunk, total_steps - done)
            train_n_steps(model, train_loader, optimizer, device, global_mean, global_std, this_chunk, input_length, pred_length)
            done += this_chunk
            pbar.update(this_chunk)

    if model_save_path is not None:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)

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
) -> Tuple[float, float, float, ZoneMetrics]:
    """Evaluate the global federated model on the held-out test set."""
    test_loader = build_test_dataloader(test_ids, data_dir, global_mean, global_std, example_len, batch_size)
    return evaluate_model(global_model, test_loader, device, global_mean, global_std, input_length, pred_length)
