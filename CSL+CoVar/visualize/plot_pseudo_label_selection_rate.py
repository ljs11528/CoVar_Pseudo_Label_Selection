#!/usr/bin/env python3
"""Plot the two-panel pseudo-label selection-rate figure used across CoVar experiments.

This script combines metrics from two subprojects:

- segmentation: CSL+CoVar pseudo_label_metrics_summary.json
- classification: per-epoch selection-rate series from SimPLE+CoVar or a manual export

The figure is configured through a JSON file so one command can reproduce the exact
layout from the screenshot while still supporting different experiment paths.

Example config:

{
  "output_path": "selection_rate_comparison.png",
  "figure_size": [12, 5],
  "segmentation": {
    "title": "Pseudo-label Selection Rate (Segmentation)",
    "curves": [
      {
        "label": "W/ CRCV(1/4)",
        "source": {
          "type": "csl_summary",
          "path": "CSL+CoVar/exp/pascal/CSL/r101/1_4/pseudo_label_metrics_summary.json"
        }
      }
    ]
  },
  "classification": {
    "title": "Pseudo-label Selection Rate (Classification)",
    "curves": [
      {
        "label": "W/ CRCV(4000)",
        "values": [0.04, 0.12, 0.22, 0.33, 0.41, 0.46, 0.49],
        "x_values": [0, 50, 100, 150, 200, 250, 300]
      }
    ]
  }
}

Supported sources:

- curve-level inline series:
  {"label": "...", "values": [...], "x_values": [...]}
- CSL summary JSON:
  {"type": "csl_summary", "path": "...", "field": "selection_rate_cumulative"}
- generic JSON list:
  {"type": "json_list", "path": "...", "key": "rates"}
- generic JSON records:
    {"type": "json_records", "path": "...", "records_key": "history", "x_field": "epoch", "y_field": "metrics.coverage"}
- SimPLE+CoVar classification summary JSON:
    {"type": "classification_summary", "path": "...", "metric": "coverage"}
- CSV column:
  {"type": "csv_column", "path": "...", "y_column": "selection_rate"}
- CSV ratio:
  {"type": "csv_ratio", "path": "...", "numerator_column": "selected", "denominator_column": "total"}

Notes:
- CSL+CoVar already persists segmentation trend data in per_epoch_cumulative.
- SimPLE+CoVar now persists classification trend data in pseudo_label_metrics_summary.json
    under per_epoch_metrics[*].metrics, and this script can read that format directly via
    the classification_summary source type.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


DEFAULT_CONFIG_TEMPLATE = {
    "output_path": "selection_rate_comparison.png",
    "figure_size": [12, 5],
    "dpi": 200,
    "share_y": True,
    "segmentation": {
        "title": "Pseudo-label Selection Rate (Segmentation)",
        "x_label": "Epoch",
        "y_label": "Selection Rate",
        "curves": [
            {
                "label": "W/ CRCV(1/4)",
                "source": {
                    "type": "csl_summary",
                    "path": "CSL+CoVar/exp/pascal/CSL/r101/1_4/pseudo_label_metrics_summary.json",
                },
            }
        ],
    },
    "classification": {
        "title": "Pseudo-label Selection Rate (Classification)",
        "x_label": "Epoch",
        "y_label": "Selection Rate",
        "curves": [
            {
                "label": "W/ CRCV(4000)",
                "source": {
                    "type": "classification_summary",
                    "path": "SimPLE+CoVar/path/to/pseudo_label_metrics_summary.json",
                    "metric": "coverage"
                },
            }
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the two-panel pseudo-label selection-rate comparison figure.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the JSON config describing the segmentation and classification curves.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path override. Defaults to output_path inside the config.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="seaborn-v0_8-whitegrid",
        help="Matplotlib style to apply before plotting.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="Optional DPI override. Defaults to dpi inside the config or 200.",
    )
    parser.add_argument(
        "--write-example-config",
        type=Path,
        default=None,
        help="Write an example JSON config to the given path and exit.",
    )
    return parser.parse_args()


def write_example_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG_TEMPLATE, handle, indent=2)
    print(f"Wrote example config to: {config_path}")


def load_json(json_path: Path) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_nested_value(data: Any, key_path: str) -> Any:
    current = data
    for part in key_path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


def resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_x_values(length: int, spec: Dict[str, Any]) -> List[float]:
    if "x_values" in spec:
        x_values = [float(value) for value in spec["x_values"]]
    else:
        start = float(spec.get("x_start", 0.0))
        step = float(spec.get("x_step", 1.0))
        x_values = [start + index * step for index in range(length)]

    if len(x_values) != length:
        raise ValueError(
            f"Expected {length} x values but got {len(x_values)} for curve '{spec.get('label', '<unknown>')}'."
        )
    return x_values


def load_csl_summary(source: Dict[str, Any], base_dir: Path) -> Tuple[List[float], List[float]]:
    json_path = resolve_path(source["path"], base_dir)
    payload = load_json(json_path)
    records = payload.get("per_epoch_cumulative")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Missing non-empty per_epoch_cumulative in {json_path}")

    field = source.get("field", "selection_rate_cumulative")
    x_field = source.get("x_field", "epoch")
    x_values = [float(record[x_field]) for record in records]
    y_values = [float(record[field]) for record in records]
    return x_values, y_values


def load_json_list(source: Dict[str, Any], base_dir: Path) -> Tuple[List[float], List[float]]:
    json_path = resolve_path(source["path"], base_dir)
    payload = load_json(json_path)
    list_key = source["key"]
    values = get_nested_value(payload, list_key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Expected a non-empty list at '{list_key}' in {json_path}")

    y_values = [float(value) for value in values]
    x_values = build_x_values(len(y_values), source)
    return x_values, y_values


def load_json_records(source: Dict[str, Any], base_dir: Path) -> Tuple[List[float], List[float]]:
    json_path = resolve_path(source["path"], base_dir)
    payload = load_json(json_path)
    records_key = source["records_key"]
    records = get_nested_value(payload, records_key)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty record list at '{records_key}' in {json_path}")

    x_field = source["x_field"]
    y_field = source["y_field"]
    x_values = [float(get_nested_value(record, x_field)) for record in records]
    y_values = [float(get_nested_value(record, y_field)) for record in records]
    return x_values, y_values


def load_classification_summary(source: Dict[str, Any], base_dir: Path) -> Tuple[List[float], List[float]]:
    json_path = resolve_path(source["path"], base_dir)
    payload = load_json(json_path)
    records = payload.get("per_epoch_metrics")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Missing non-empty per_epoch_metrics in {json_path}")

    x_field = source.get("x_field", "epoch")
    metric = source.get("metric")
    if metric is not None:
        y_field = f"metrics.{metric}"
    else:
        y_field = source.get("y_field", "metrics.coverage")

    x_values = [float(get_nested_value(record, x_field)) for record in records]
    y_values = [float(get_nested_value(record, y_field)) for record in records]
    return x_values, y_values


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file is empty: {csv_path}")
    return rows


def load_csv_column(source: Dict[str, Any], base_dir: Path) -> Tuple[List[float], List[float]]:
    csv_path = resolve_path(source["path"], base_dir)
    rows = load_csv_rows(csv_path)
    y_column = source["y_column"]
    y_values = [float(row[y_column]) for row in rows]

    x_column = source.get("x_column")
    if x_column is not None:
        x_values = [float(row[x_column]) for row in rows]
    else:
        x_values = build_x_values(len(y_values), source)
    return x_values, y_values


def load_csv_ratio(source: Dict[str, Any], base_dir: Path) -> Tuple[List[float], List[float]]:
    csv_path = resolve_path(source["path"], base_dir)
    rows = load_csv_rows(csv_path)
    numerator_column = source["numerator_column"]
    denominator_column = source["denominator_column"]

    y_values = []
    for row in rows:
        numerator = float(row[numerator_column])
        denominator = float(row[denominator_column])
        y_values.append(numerator / denominator if denominator else 0.0)

    x_column = source.get("x_column")
    if x_column is not None:
        x_values = [float(row[x_column]) for row in rows]
    else:
        x_values = build_x_values(len(y_values), source)
    return x_values, y_values


def load_curve_series(curve: Dict[str, Any], base_dir: Path) -> Tuple[List[float], List[float]]:
    if "values" in curve:
        y_values = [float(value) for value in curve["values"]]
        if not y_values:
            raise ValueError(f"Curve '{curve.get('label', '<unknown>')}' has no values.")
        x_values = build_x_values(len(y_values), curve)
        return x_values, y_values

    source = curve.get("source")
    if not isinstance(source, dict):
        raise ValueError(
            f"Curve '{curve.get('label', '<unknown>')}' must provide either 'values' or a 'source' object."
        )

    source_type = source.get("type")
    if source_type == "csl_summary":
        return load_csl_summary(source, base_dir)
    if source_type == "json_list":
        return load_json_list(source, base_dir)
    if source_type == "json_records":
        return load_json_records(source, base_dir)
    if source_type == "classification_summary":
        return load_classification_summary(source, base_dir)
    if source_type == "csv_column":
        return load_csv_column(source, base_dir)
    if source_type == "csv_ratio":
        return load_csv_ratio(source, base_dir)

    raise ValueError(
        f"Unsupported source type '{source_type}' for curve '{curve.get('label', '<unknown>')}'."
    )


def apply_panel_options(axis: plt.Axes, panel: Dict[str, Any], default_title: str) -> None:
    axis.set_title(panel.get("title", default_title))
    axis.set_xlabel(panel.get("x_label", "Epoch"))
    axis.set_ylabel(panel.get("y_label", "Selection Rate"))
    axis.grid(True, linestyle="--", alpha=0.5)

    if "x_lim" in panel:
        axis.set_xlim(panel["x_lim"])
    if "y_lim" in panel:
        axis.set_ylim(panel["y_lim"])
    if "x_ticks" in panel:
        axis.set_xticks(panel["x_ticks"])
    if "y_ticks" in panel:
        axis.set_yticks(panel["y_ticks"])


def plot_panel(axis: plt.Axes, panel: Dict[str, Any], default_title: str, base_dir: Path) -> None:
    curves = panel.get("curves", [])
    if not curves:
        raise ValueError(f"Panel '{default_title}' does not define any curves.")

    for curve in curves:
        label = curve.get("label")
        if not label:
            raise ValueError(f"Panel '{default_title}' contains a curve without a label.")
        x_values, y_values = load_curve_series(curve, base_dir)
        axis.plot(x_values, y_values, label=label, linewidth=2)

    apply_panel_options(axis, panel, default_title)


def validate_panel_names(config: Dict[str, Any]) -> None:
    for required_key in ("segmentation", "classification"):
        if required_key not in config:
            raise ValueError(f"Config must contain a '{required_key}' section.")


def main() -> None:
    args = parse_args()

    if args.write_example_config is not None:
        write_example_config(args.write_example_config)
        return

    if args.config is None:
        raise SystemExit("Either --config or --write-example-config must be provided.")

    config_path = args.config.resolve()
    config = load_json(config_path)
    validate_panel_names(config)
    base_dir = config_path.parent

    try:
        plt.style.use(args.style)
    except OSError as exc:
        raise SystemExit(f"Unknown matplotlib style '{args.style}': {exc}") from exc

    figure_size = tuple(config.get("figure_size", [12, 5]))
    dpi = int(args.dpi or config.get("dpi", 200))
    share_y = bool(config.get("share_y", True))

    figure, axes = plt.subplots(1, 2, figsize=figure_size, dpi=dpi, sharey=share_y)
    plot_panel(axes[0], config["segmentation"], "Pseudo-label Selection Rate (Segmentation)", base_dir)
    plot_panel(axes[1], config["classification"], "Pseudo-label Selection Rate (Classification)", base_dir)

    legend_panel = config.get("legend_panel", "classification")
    legend_axis = axes[0] if legend_panel == "segmentation" else axes[1]
    legend_loc = config.get("legend_loc", "upper right")
    legend_axis.legend(loc=legend_loc)

    figure.tight_layout()

    output_path = args.output
    if output_path is None:
        output_path = resolve_path(config.get("output_path", "selection_rate_comparison.png"), base_dir)
    else:
        output_path = output_path.resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()