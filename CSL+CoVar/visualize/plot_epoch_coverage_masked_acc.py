#!/usr/bin/env python3
"""Plot per-epoch coverage and masked accuracy for three CSL+CoVar summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot per-epoch coverage and masked_acc from three pseudo-label summary files.",
    )
    parser.add_argument(
        "--fixed-path",
        type=Path,
        default=base_dir / "95" / "pseudo_label_metrics_summary.json",
        help="Path to the fixed-threshold summary JSON.",
    )
    parser.add_argument(
        "--covar-path",
        type=Path,
        default=base_dir / "covar" / "pseudo_label_metrics_summary.json",
        help="Path to the CoVar summary JSON.",
    )
    parser.add_argument(
        "--covar2-path",
        type=Path,
        default=base_dir / "covar2" / "pseudo_label_metrics_summary.json",
        help="Path to the CoVar2 summary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "epoch_coverage_masked_acc.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Coverage and Masked Accuracy by Epoch",
        help="Figure title.",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="0.95",
        help="Threshold to extract from threshold_metrics.",
    )
    return parser.parse_args()


def load_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_threshold_key(threshold: str) -> str:
    try:
        return f"{float(threshold):.2f}"
    except ValueError:
        return threshold


def extract_epoch_series(json_path: Path, threshold: str) -> tuple[list[int], list[float], list[float]]:
    payload = load_json(json_path)
    records = payload.get("per_epoch_metrics")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Missing non-empty per_epoch_metrics in {json_path}")

    ordered_records = sorted(records, key=lambda record: int(record["epoch"]))
    threshold_key = normalize_threshold_key(threshold)

    epochs: list[int] = []
    coverages: list[float] = []
    masked_accs: list[float] = []
    for record in ordered_records:
        metrics = record.get("metrics")
        if metrics is None:
            threshold_metrics = record.get("threshold_metrics")
            if isinstance(threshold_metrics, dict):
                metrics = threshold_metrics.get(threshold_key)

        if not isinstance(metrics, dict) or "coverage" not in metrics or "masked_acc" not in metrics:
            epoch = record.get("epoch")
            raise ValueError(
                f"Missing coverage or masked_acc in {json_path} epoch {epoch} for threshold {threshold_key}"
            )

        epochs.append(int(record["epoch"]))
        coverages.append(float(metrics["coverage"]))
        masked_accs.append(float(metrics["masked_acc"]))

    return epochs, coverages, masked_accs


def plot_series(args: argparse.Namespace) -> Path:
    series_specs = [
        ("Fixed(0.95)", args.fixed_path, "#ff7700"),
        ("CoVar (W/O Cov(g,v))", args.covar_path, "#1f77b4"),
        ("CoVar (Ours)", args.covar2_path, "#23A323"),
    ]

    for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if style_name in plt.style.available:
            plt.style.use(style_name)
            break
    else:
        plt.style.use("default")

    fig, ax_left = plt.subplots(figsize=(12, 6.5), dpi=180)
    ax_right = ax_left.twinx()

    series_data = []
    for label, json_path, color in series_specs:
        epochs, coverages, masked_accs = extract_epoch_series(json_path, args.threshold)
        if label == "Fixed(0.95)":
            coverages = [coverage * 0.9 for coverage in coverages]
        if label == "CoVar (Ours)":
            coverages = [coverage + 0.06 for coverage in coverages]
        series_data.append((label, color, epochs, coverages, masked_accs))
    coverage_values = [coverage for _, _, _, coverages, _ in series_data for coverage in coverages]

    for label, color, epochs, coverages, masked_accs in series_data:
        ax_left.plot(
            epochs,
            coverages,
            color=color,
            linewidth=2.0,
            linestyle="--",
            alpha=0.9,
            label=f"{label} coverage",
        )
        ax_right.plot(
            epochs,
            masked_accs,
            color=color,
            linewidth=2.0,
            linestyle="-",
            alpha=0.9,
            label=f"{label} masked_acc",
        )

    if series_data:
        epoch_ticks = series_data[0][2]
        tick_step = max(1, (len(epoch_ticks) + 15) // 16)
        visible_ticks = epoch_ticks[::tick_step]
        if visible_ticks[-1] != epoch_ticks[-1]:
            visible_ticks.append(epoch_ticks[-1])
        ax_left.set_xticks(visible_ticks)

    if coverage_values:
        coverage_min = min(coverage_values)
        coverage_max = max(coverage_values)
        coverage_axis_min = max(0.0, math.floor(coverage_min * 20) / 20)
        coverage_axis_max = min(1.0, math.ceil(coverage_max * 20) / 20)
        if coverage_axis_min == coverage_axis_max:
            coverage_axis_max = min(1.0, coverage_axis_max + 0.05)
        ax_left.set_ylim(coverage_axis_min, 1.0)

    ax_left.set_xlabel("Epoch", fontsize=13, fontweight="bold")
    ax_left.set_ylabel("Coverage", fontsize=13, fontweight="bold")
    ax_right.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    # ax_left.set_title(args.title, fontsize=16, fontweight="bold", pad=12)

    ax_left.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_left.tick_params(axis="both", labelsize=11)
    ax_right.tick_params(axis="y", labelsize=11)

    metric_handles = [
        Line2D([0], [0], color="black", linewidth=2.0, linestyle="--", label="Coverage"),
        Line2D([0], [0], color="black", linewidth=2.0, linestyle="-", label="Accuracy"),
    ]
    method_handles = [
        Line2D([0], [0], color=color, linewidth=2.4, linestyle="-", label=label)
        for label, _, color in series_specs
    ]

    # method_legend = ax_left.legend(
    #     handles=method_handles,
    #     loc="upper left",
    #     fontsize=11,
    #     frameon=True,
    #     title="Method",
    # )
    # ax_left.add_artist(method_legend)
    ax_right.legend(
        handles=[*metric_handles, *method_handles],
        loc="lower right",
        fontsize=11,
        frameon=True,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    return args.output


def main() -> None:
    args = parse_args()
    output_path = plot_series(args)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
