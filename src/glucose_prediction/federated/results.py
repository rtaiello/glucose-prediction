"""Result reporting: rich table, JSON saving, matplotlib comparison plot."""

import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.table import Table

from glucose_prediction.federated.evaluate import ZoneMetrics


def print_comparison_table(
    local_mae: float,
    local_rmse: float,
    local_r2: Optional[float],
    cent_mae: float,
    cent_rmse: float,
    cent_r2: Optional[float],
    fed_mae: float,
    fed_rmse: float,
    fed_r2: Optional[float],
    local_zones: Optional[ZoneMetrics] = None,
    cent_zones: Optional[ZoneMetrics] = None,
    fed_zones: Optional[ZoneMetrics] = None,
) -> None:
    """Print a 3-method comparison table with MAE/RMSE/R² and optional zone recall."""
    console = Console()
    table = Table(title="Glucose Prediction — Method Comparison")
    table.add_column("Method", style="bold cyan")
    table.add_column("MAE (mg/dL)", justify="right")
    table.add_column("RMSE (mg/dL)", justify="right")

    has_r2 = local_r2 is not None and cent_r2 is not None and fed_r2 is not None
    if has_r2:
        table.add_column("R²", justify="right")

    has_zones = local_zones is not None and cent_zones is not None and fed_zones is not None
    if has_zones:
        table.add_column("Hypo Recall\n(<70)", justify="right")
        table.add_column("Normal Recall\n(70–180)", justify="right")
        table.add_column("Hyper Recall\n(>180)", justify="right")
        table.add_column("Zone Acc", justify="right")

    rows = [
        ("Local Baseline (avg)", local_mae, local_rmse, local_r2, local_zones),
        ("Centralized", cent_mae, cent_rmse, cent_r2, cent_zones),
        ("FedAvg", fed_mae, fed_rmse, fed_r2, fed_zones),
    ]
    for label, mae, rmse, r2, zones in rows:
        row = [label, f"{mae:.3f}", f"{rmse:.3f}"]
        if has_r2:
            row.append(f"{r2:.4f}")
        if has_zones:
            row += [
                f"{zones.hypo_recall:.1%}",
                f"{zones.normal_recall:.1%}",
                f"{zones.hyper_recall:.1%}",
                f"{zones.overall_acc:.1%}",
            ]
        table.add_row(*row)

    sign = lambda v: "+" if v >= 0 else ""
    for label, mae, rmse, r2_delta in [
        ("Δ Centralized − Local", cent_mae - local_mae, cent_rmse - local_rmse,
         (cent_r2 - local_r2) if has_r2 else None),
        ("Δ FedAvg − Local", fed_mae - local_mae, fed_rmse - local_rmse,
         (fed_r2 - local_r2) if has_r2 else None),
    ]:
        row = [
            f"[dim]{label}[/dim]",
            f"[dim]{sign(mae)}{mae:.3f}[/dim]",
            f"[dim]{sign(rmse)}{rmse:.3f}[/dim]",
        ]
        if has_r2:
            row.append(f"[dim]{sign(r2_delta)}{r2_delta:.4f}[/dim]")
        if has_zones:
            row += ["", "", "", ""]
        table.add_row(*row)

    console.print(table)

    if has_zones:
        total = local_zones.hypo_n + local_zones.normal_n + local_zones.hyper_n
        console.print(f"\n[dim]Test set zone distribution ({total:,} prediction timesteps):[/dim]")
        console.print(f"[dim]  Hypo   (<70 mg/dL):    {local_zones.hypo_n:,}[/dim]")
        console.print(f"[dim]  Normal (70–180 mg/dL): {local_zones.normal_n:,}[/dim]")
        console.print(f"[dim]  Hyper  (>180 mg/dL):   {local_zones.hyper_n:,}[/dim]")


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
    local_r2: Optional[float],
    cent_mae: float,
    cent_rmse: float,
    cent_r2: Optional[float],
    fed_mae: float,
    fed_rmse: float,
    fed_r2: Optional[float],
    output_dir: str,
    local_zones: Optional[ZoneMetrics] = None,
    cent_zones: Optional[ZoneMetrics] = None,
    fed_zones: Optional[ZoneMetrics] = None,
    per_patient_results: Optional[List[Dict]] = None,
) -> None:
    """Save a multi-panel comparison figure.

    Panels:
      1. MAE, RMSE & R² grouped bar chart (3 methods)
      2. Zone recall grouped bar chart (if zone metrics available)
      3. Per-patient MAE distribution for local baseline (if per_patient_results available)
    """
    os.makedirs(output_dir, exist_ok=True)

    has_r2 = local_r2 is not None and cent_r2 is not None and fed_r2 is not None
    has_zones = local_zones is not None and cent_zones is not None and fed_zones is not None
    has_per_patient = per_patient_results is not None and len(per_patient_results) > 0

    n_panels = 1 + int(has_zones) + int(has_per_patient)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels + 1, 5))
    if n_panels == 1:
        axes = [axes]

    methods = ["Local (avg)", "Centralized", "FedAvg"]
    colors = ["#4C72B0", "#55A868", "#DD8452"]
    total_width = 0.7
    width = total_width / 3

    # --- Panel 1: MAE, RMSE (and R² on twin axis) ---
    ax1 = axes[0]
    x = [0, 1]
    for i, (label, values, color) in enumerate(zip(
        methods,
        [[local_mae, local_rmse], [cent_mae, cent_rmse], [fed_mae, fed_rmse]],
        colors,
    )):
        offset = (i - 1) * width
        ax1.bar([xi + offset for xi in x], values, width=width * 0.9, label=label, color=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["MAE (mg/dL)", "RMSE (mg/dL)"])
    ax1.set_ylabel("Error (mg/dL)")
    ax1.set_title("Regression Metrics")
    ax1.grid(axis="y", alpha=0.3)

    if has_r2:
        # Add R² as a secondary plot: text annotations above each method's bars
        for i, (label, r2_val, color) in enumerate(zip(methods, [local_r2, cent_r2, fed_r2], colors)):
            ax1.annotate(
                f"R²={r2_val:.3f}",
                xy=(i * 0.01, 1.02), xycoords="axes fraction",
                ha="center", fontsize=7, color=color,
            )
        # Also add a small R² grouped bar on its own x position
        x_r2 = [2]
        for i, (label, r2_val, color) in enumerate(zip(methods, [local_r2, cent_r2, fed_r2], colors)):
            offset = (i - 1) * width
            ax1.bar([2 + offset], [r2_val], width=width * 0.9, color=color, alpha=0.9)
        ax1.set_xticks([0, 1, 2])
        ax1.set_xticklabels(["MAE (mg/dL)", "RMSE (mg/dL)", "R²"])

    ax1.legend()

    panel_idx = 1

    # --- Panel 2: Zone recall (optional) ---
    if has_zones:
        ax2 = axes[panel_idx]
        panel_idx += 1
        zones_list = [local_zones, cent_zones, fed_zones]
        zone_xlabels = [
            f"Hypo\n(<70\nn={local_zones.hypo_n:,})",
            f"Normal\n(70–180\nn={local_zones.normal_n:,})",
            f"Hyper\n(>180\nn={local_zones.hyper_n:,})",
        ]
        x2 = [0, 1, 2]
        for i, (label, zones, color) in enumerate(zip(methods, zones_list, colors)):
            recalls = [zones.hypo_recall, zones.normal_recall, zones.hyper_recall]
            offset = (i - 1) * width
            bars = ax2.bar([xi + offset for xi in x2], recalls, width=width * 0.9, label=label, color=color)
            for bar, val in zip(bars, recalls):
                if val > 0.02:
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.0%}",
                        ha="center", va="bottom", fontsize=7,
                    )
        ax2.set_xticks(x2)
        ax2.set_xticklabels(zone_xlabels)
        ax2.set_ylabel("Recall (sensitivity)")
        ax2.set_title("Zone Classification Recall")
        ax2.set_ylim(0, 1.15)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

    # --- Panel 3: Per-patient MAE distribution (optional) ---
    if has_per_patient:
        ax3 = axes[panel_idx]
        maes = sorted([r["mae"] for r in per_patient_results])
        n = len(maes)
        mean_mae = np.mean(maes)
        median_mae = np.median(maes)
        p10 = np.percentile(maes, 10)
        p90 = np.percentile(maes, 90)

        ax3.hist(maes, bins=30, color="#4C72B0", alpha=0.7, edgecolor="white")
        ax3.axvline(mean_mae, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_mae:.1f}")
        ax3.axvline(median_mae, color="orange", linestyle="-", linewidth=1.5, label=f"Median: {median_mae:.1f}")
        ax3.axvspan(p10, p90, alpha=0.1, color="blue", label=f"P10–P90: [{p10:.1f}, {p90:.1f}]")
        ax3.set_xlabel("MAE (mg/dL)")
        ax3.set_ylabel("# patients")
        ax3.set_title(f"Local Baseline — Per-Patient MAE\n(n={n} patients)")
        ax3.legend(fontsize=8)
        ax3.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {path}")
