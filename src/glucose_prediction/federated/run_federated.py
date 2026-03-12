"""CLI entry point for the federated learning simulation.

Supports parallel execution via --phase:
  local       — train one model per patient, save results
  centralized — train single pooled model, save results
  fedavg      — run FedAvg federation, save results
  report      — load saved results and print comparison table + plot
  all         — run all phases sequentially (default)

Since the patient split and normalization stats are deterministic from --seed,
each phase can run independently in parallel and produce identical splits.

Usage:
    # Parallel (launch all three, then report):
    uv run python -m glucose_prediction.federated.run_federated --phase local       --device cuda:0 &
    uv run python -m glucose_prediction.federated.run_federated --phase centralized --device cuda:0 &
    uv run python -m glucose_prediction.federated.run_federated --phase fedavg      --device cuda:0 &
    wait && uv run python -m glucose_prediction.federated.run_federated --phase report

    # Sequential (default):
    uv run python -m glucose_prediction.federated.run_federated --device cuda:0
"""

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = str(PROJECT_ROOT / "data" / "raw" / "patients")
RESULTS_DIR = str(PROJECT_ROOT / "results" / "federated")
MODELS_DIR = str(PROJECT_ROOT / "results" / "federated" / "models")

INPUT_LENGTH = 12
PRED_LENGTH = 4
EXAMPLE_LEN = 16

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
pylogger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated Learning Simulation for Glucose Prediction")
    parser.add_argument(
        "--phase", choices=["all", "local", "centralized", "fedavg", "report"], default="all",
        help="Which phase to run (default: all). Use local/centralized/fedavg in parallel, then report.",
    )
    parser.add_argument("--n-train", type=int, default=180)
    parser.add_argument("--n-rounds", type=int, default=50)
    parser.add_argument("--local-steps", type=int, default=100)
    parser.add_argument("--client-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-test", type=int, default=None)
    return parser.parse_args()


def _setup(args):
    """Shared setup: split patients and compute normalization stats."""
    from glucose_prediction.data.glucose_datamodule import _compute_stats
    from glucose_prediction.federated.data_utils import load_all_patient_ids, split_patients

    all_ids = load_all_patient_ids(DATA_DIR)
    train_ids, test_ids = split_patients(all_ids, n_train=args.n_train, seed=args.seed)
    if args.max_test is not None:
        test_ids = test_ids[: args.max_test]
    global_mean, global_std = _compute_stats(DATA_DIR, train_ids)

    print(f"Total patients: {len(all_ids)} | Train: {len(train_ids)} | Test: {len(test_ids)} (held-out)")
    print(f"Normalization  Mean: {[f'{m:.3f}' for m in global_mean]}  Std: {[f'{s:.3f}' for s in global_std]}\n")
    return train_ids, test_ids, global_mean, global_std


def _print_header(args):
    n_selected = max(1, int(args.n_train * args.client_fraction))
    local_budget = args.n_rounds * args.local_steps
    cent_budget = args.n_rounds * n_selected * args.local_steps
    print(f"\n{'='*60}")
    print(f"Federated Learning Simulation  [phase: {args.phase}]")
    print(f"{'='*60}")
    print(f"  Training patients  : {args.n_train}")
    print(f"  FL rounds          : {args.n_rounds}")
    print(f"  Local steps/round  : {args.local_steps}")
    print(f"  Client fraction    : {args.client_fraction} (~{n_selected} clients/round)")
    print(f"  Local budget       : {local_budget} steps/patient  (n_rounds × local_steps)")
    print(f"  Centralized budget : {cent_budget} steps  (n_rounds × n_selected × local_steps)")
    print(f"  Seed / Device      : {args.seed} / {args.device}")
    print(f"{'='*60}\n")


def _save_norm_stats(global_mean, global_std) -> None:
    """Save normalization stats alongside models so they can be loaded later."""
    path = Path(MODELS_DIR) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mean": global_mean, "std": global_std}, indent=2))


def run_local(args, train_ids, test_ids, global_mean, global_std) -> dict:
    from glucose_prediction.federated.evaluate import evaluate_local_baseline

    total_steps = args.n_rounds * args.local_steps
    model_save_dir = str(Path(MODELS_DIR) / "local")
    print(f"Phase: Local Baseline  ({total_steps} steps per patient)")
    local_mae, local_rmse, local_r2, zone_metrics, per_patient = evaluate_local_baseline(
        train_ids=train_ids, test_ids=test_ids, data_dir=DATA_DIR,
        global_mean=global_mean, global_std=global_std,
        total_steps=total_steps, example_len=EXAMPLE_LEN,
        input_length=INPUT_LENGTH, pred_length=PRED_LENGTH,
        batch_size=args.batch_size, lr=args.lr, device=torch.device(args.device),
        model_save_dir=model_save_dir,
    )
    print(f"\nLocal baseline  →  MAE: {local_mae:.3f} mg/dL | RMSE: {local_rmse:.3f} mg/dL | R²: {local_r2:.4f}")
    print(f"  Zone recall    →  Hypo: {zone_metrics.hypo_recall:.1%} | Normal: {zone_metrics.normal_recall:.1%} | Hyper: {zone_metrics.hyper_recall:.1%}\n")
    print(f"  Models saved   →  {model_save_dir}/patient_{{pid}}.pt")
    return {
        "mae": local_mae, "rmse": local_rmse, "r2": local_r2, "n_patients": len(per_patient),
        "zone_metrics": dataclasses.asdict(zone_metrics),
        "per_patient": [
            {"patient_id": r.patient_id, "mae": r.mae, "rmse": r.rmse, "r2": r.r2, "n_windows": r.n_windows}
            for r in per_patient
        ],
    }


def run_centralized(args, train_ids, test_ids, global_mean, global_std) -> dict:
    from glucose_prediction.federated.evaluate import evaluate_centralized

    n_selected = max(1, int(args.n_train * args.client_fraction))
    total_steps = args.n_rounds * n_selected * args.local_steps
    model_save_path = str(Path(MODELS_DIR) / "centralized.pt")
    print(f"Phase: Centralized  ({total_steps} steps = {args.n_rounds} rounds × {n_selected} clients × {args.local_steps} steps)")
    cent_mae, cent_rmse, cent_r2, zone_metrics = evaluate_centralized(
        train_ids=train_ids, test_ids=test_ids, data_dir=DATA_DIR,
        global_mean=global_mean, global_std=global_std,
        total_steps=total_steps, example_len=EXAMPLE_LEN,
        input_length=INPUT_LENGTH, pred_length=PRED_LENGTH,
        batch_size=args.batch_size, lr=args.lr, device=torch.device(args.device),
        model_save_path=model_save_path,
    )
    print(f"\nCentralized     →  MAE: {cent_mae:.3f} mg/dL | RMSE: {cent_rmse:.3f} mg/dL | R²: {cent_r2:.4f}")
    print(f"  Zone recall    →  Hypo: {zone_metrics.hypo_recall:.1%} | Normal: {zone_metrics.normal_recall:.1%} | Hyper: {zone_metrics.hyper_recall:.1%}")
    print(f"  Model saved    →  {model_save_path}\n")
    return {"mae": cent_mae, "rmse": cent_rmse, "r2": cent_r2, "total_steps": total_steps, "zone_metrics": dataclasses.asdict(zone_metrics)}


def run_fedavg(args, train_ids, test_ids, global_mean, global_std) -> dict:
    from glucose_prediction.federated.client import FederatedClient
    from glucose_prediction.federated.evaluate import evaluate_federated
    from glucose_prediction.federated.data_utils import build_patient_dataset
    from glucose_prediction.federated.server import FederatedServer

    device = torch.device(args.device)
    model_save_path = Path(MODELS_DIR) / "fedavg.pt"
    print(f"Phase: FedAvg  ({args.n_rounds} rounds × {args.local_steps} steps, fraction={args.client_fraction})")

    clients = []
    skipped = 0
    for pid in train_ids:
        ds = build_patient_dataset(pid, DATA_DIR, global_mean, global_std, EXAMPLE_LEN)
        if ds is None:
            skipped += 1
            continue
        clients.append(FederatedClient(
            patient_id=pid, dataset=ds, global_mean=global_mean, global_std=global_std,
            batch_size=args.batch_size, lr=args.lr, device=device,
            input_length=INPUT_LENGTH, pred_length=PRED_LENGTH,
        ))
    print(f"Clients ready: {len(clients)} (skipped {skipped})\n")

    server = FederatedServer(clients=clients, device=device, input_length=INPUT_LENGTH, seed=args.seed)
    server.run_federation(
        n_rounds=args.n_rounds, n_local_steps=args.local_steps,
        client_fraction=args.client_fraction, verbose=True,
    )

    # Save global model
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(server.global_model.state_dict(), model_save_path)

    fed_mae, fed_rmse, fed_r2, zone_metrics = evaluate_federated(
        global_model=server.global_model, test_ids=test_ids, data_dir=DATA_DIR,
        global_mean=global_mean, global_std=global_std, example_len=EXAMPLE_LEN,
        input_length=INPUT_LENGTH, pred_length=PRED_LENGTH,
        batch_size=args.batch_size, device=device,
    )
    print(f"\nFedAvg          →  MAE: {fed_mae:.3f} mg/dL | RMSE: {fed_rmse:.3f} mg/dL | R²: {fed_r2:.4f}")
    print(f"  Zone recall    →  Hypo: {zone_metrics.hypo_recall:.1%} | Normal: {zone_metrics.normal_recall:.1%} | Hyper: {zone_metrics.hyper_recall:.1%}")
    print(f"  Model saved    →  {model_save_path}\n")
    return {"mae": fed_mae, "rmse": fed_rmse, "r2": fed_r2, "round_history": server.round_history, "zone_metrics": dataclasses.asdict(zone_metrics)}


def _save_phase(phase_name: str, data: dict, args) -> None:
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = Path(RESULTS_DIR) / f"{phase_name}.json"
    payload = {
        "config": {
            "n_train": args.n_train, "n_rounds": args.n_rounds,
            "local_steps": args.local_steps, "total_steps": args.n_rounds * args.local_steps,
            "client_fraction": args.client_fraction, "seed": args.seed,
            "device": args.device, "batch_size": args.batch_size, "lr": args.lr,
        },
        "results": data,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {phase_name} results → {path}")


def _load_phase(phase_name: str) -> dict:
    path = Path(RESULTS_DIR) / f"{phase_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}. Run --phase {phase_name} first.")
    return json.loads(path.read_text())["results"]


def run_report(args) -> None:
    from glucose_prediction.federated.evaluate import ZoneMetrics
    from glucose_prediction.federated.results import print_comparison_table, save_comparison_plot, save_results_json

    local = _load_phase("local")
    cent = _load_phase("centralized")
    fed = _load_phase("fedavg")

    def _zones(phase_data: dict) -> Optional[ZoneMetrics]:
        zm = phase_data.get("zone_metrics")
        return ZoneMetrics(**zm) if zm is not None else None

    local_zones = _zones(local)
    cent_zones = _zones(cent)
    fed_zones = _zones(fed)

    print_comparison_table(
        local["mae"], local["rmse"], local.get("r2"),
        cent["mae"], cent["rmse"], cent.get("r2"),
        fed["mae"], fed["rmse"], fed.get("r2"),
        local_zones=local_zones, cent_zones=cent_zones, fed_zones=fed_zones,
    )

    if local_zones is None:
        print("\n[Note] Zone metrics not available — re-run all phases with the updated code to include them.")

    combined = {
        "config": {
            "n_train": args.n_train, "n_rounds": args.n_rounds,
            "local_steps": args.local_steps, "total_steps": args.n_rounds * args.local_steps,
            "client_fraction": args.client_fraction, "seed": args.seed,
        },
        "local_baseline": local,
        "centralized": cent,
        "fedavg": fed,
    }
    save_results_json(combined, RESULTS_DIR)
    save_comparison_plot(
        local["mae"], local["rmse"], local.get("r2"),
        cent["mae"], cent["rmse"], cent.get("r2"),
        fed["mae"], fed["rmse"], fed.get("r2"),
        RESULTS_DIR,
        local_zones=local_zones, cent_zones=cent_zones, fed_zones=fed_zones,
        per_patient_results=local.get("per_patient"),
    )


def main() -> None:
    args = parse_args()
    _print_header(args)

    if args.phase == "report":
        run_report(args)
        return

    train_ids, test_ids, global_mean, global_std = _setup(args)

    if args.phase == "local":
        data = run_local(args, train_ids, test_ids, global_mean, global_std)
        _save_phase("local", data, args)
        _save_norm_stats(global_mean, global_std)

    elif args.phase == "centralized":
        data = run_centralized(args, train_ids, test_ids, global_mean, global_std)
        _save_phase("centralized", data, args)
        _save_norm_stats(global_mean, global_std)

    elif args.phase == "fedavg":
        data = run_fedavg(args, train_ids, test_ids, global_mean, global_std)
        _save_phase("fedavg", data, args)
        _save_norm_stats(global_mean, global_std)

    elif args.phase == "all":
        local_data = run_local(args, train_ids, test_ids, global_mean, global_std)
        _save_phase("local", local_data, args)
        cent_data = run_centralized(args, train_ids, test_ids, global_mean, global_std)
        _save_phase("centralized", cent_data, args)
        fed_data = run_fedavg(args, train_ids, test_ids, global_mean, global_std)
        _save_phase("fedavg", fed_data, args)
        _save_norm_stats(global_mean, global_std)
        run_report(args)


if __name__ == "__main__":
    main()
