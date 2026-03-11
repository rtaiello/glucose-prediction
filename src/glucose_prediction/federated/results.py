"""Result reporting: rich table, JSON saving, matplotlib comparison plot."""

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table


def print_comparison_table(
    local_mae: float,
    local_rmse: float,
    cent_mae: float,
    cent_rmse: float,
    fed_mae: float,
    fed_rmse: float,
) -> None:
    """Print a 3-method comparison table: Local (avg) | Centralized | FedAvg."""
    console = Console()
    table = Table(title="Glucose Prediction — Method Comparison")
    table.add_column("Method", style="bold cyan")
    table.add_column("MAE (mg/dL)", justify="right")
    table.add_column("RMSE (mg/dL)", justify="right")

    table.add_row("Local Baseline (avg)", f"{local_mae:.3f}", f"{local_rmse:.3f}")
    table.add_row("Centralized", f"{cent_mae:.3f}", f"{cent_rmse:.3f}")
    table.add_row("FedAvg", f"{fed_mae:.3f}", f"{fed_rmse:.3f}")

    sign = lambda v: "+" if v >= 0 else ""
    for label, mae, rmse in [
        ("Δ Centralized − Local", cent_mae - local_mae, cent_rmse - local_rmse),
        ("Δ FedAvg − Local", fed_mae - local_mae, fed_rmse - local_rmse),
    ]:
        table.add_row(
            f"[dim]{label}[/dim]",
            f"[dim]{sign(mae)}{mae:.3f}[/dim]",
            f"[dim]{sign(rmse)}{rmse:.3f}[/dim]",
        )

    console.print(table)


def save_results_json(results_dict: Dict[str, Any], output_dir: str) -> None:
    """Save structured results to results.json."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {path}")


def save_comparison_plot(
    local_mae: float,
    local_rmse: float,
    cent_mae: float,
    cent_rmse: float,
    fed_mae: float,
    fed_rmse: float,
    fl_round_history: List[float],
    output_dir: str,
) -> None:
    """Save a 2-panel figure: grouped bar chart (3 methods) + FL loss curve."""
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: grouped bar chart
    methods = [
        ("Local (avg)", [local_mae, local_rmse], "#4C72B0"),
        ("Centralized", [cent_mae, cent_rmse], "#55A868"),
        ("FedAvg", [fed_mae, fed_rmse], "#DD8452"),
    ]
    x = [0, 1]
    total_width = 0.7
    width = total_width / len(methods)

    for i, (label, values, color) in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        ax1.bar([xi + offset for xi in x], values, width=width * 0.9, label=label, color=color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(["MAE (mg/dL)", "RMSE (mg/dL)"])
    ax1.set_ylabel("Error (mg/dL)")
    ax1.set_title("Method Comparison")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: FL round loss curve
    if fl_round_history:
        ax2.plot(range(1, len(fl_round_history) + 1), fl_round_history, marker="o", markersize=4, color="#DD8452")
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Avg Client L1 Loss (mg/dL)")
        ax2.set_title("FedAvg Training Loss per Round")
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No round history", ha="center", va="center", transform=ax2.transAxes)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {path}")
