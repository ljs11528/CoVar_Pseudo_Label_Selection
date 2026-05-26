#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


METRIC_META = {
    "coverage": {
        "title": "Pseudo-label Selection Rate (Classification)",
        "y_label": "Selection Rate",
        "legend_label": "Coverage",
    },
    "masked_acc": {
        "title": "Pseudo-label Accuracy (Selected, Classification)",
        "y_label": "Pseudo-label Accuracy",
        "legend_label": "Masked Accuracy",
    },
    "full_acc": {
        "title": "Pseudo-label Accuracy (All, Classification)",
        "y_label": "Pseudo-label Accuracy",
        "legend_label": "Full Accuracy",
    },
}

DUAL_AXIS_META = {
    "title": "Pseudo-label Selection Rate vs Full Accuracy (Classification)",
    "left_metric": "coverage",
    "right_metric": "full_acc",
}

STYLE_FALLBACKS = {
    "seaborn-v0_8-whitegrid": ["seaborn-v0_8-whitegrid", "seaborn-whitegrid"],
    "seaborn-whitegrid": ["seaborn-whitegrid", "seaborn-v0_8-whitegrid"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot classification pseudo-label metrics from pseudo_label_metrics_summary.json files.",
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more pseudo_label_metrics_summary.json files.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="*",
        default=None,
        help="Optional legend labels. Defaults to each JSON parent directory name.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_META.keys()),
        default="coverage",
        help="Metric to extract from per_epoch_metrics[*].metrics.",
    )
    parser.add_argument(
        "--dual-axis",
        action="store_true",
        help="Plot coverage on the left axis and full_acc on the right axis for each input.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output figure path. Defaults to the first JSON directory.",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save extracted plot-ready JSON. Defaults next to the figure.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title override.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="seaborn-v0_8-whitegrid",
        help="Matplotlib style name.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI.",
    )
    return parser.parse_args()


def load_summary(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def use_plot_style(style_name: str) -> None:
    candidates = STYLE_FALLBACKS.get(style_name, [style_name])
    last_error = None
    for candidate in candidates:
        try:
            plt.style.use(candidate)
            return
        except OSError as exc:
            last_error = exc

    raise SystemExit(f"Unknown matplotlib style '{style_name}': {last_error}") from last_error


def default_labels(inputs: List[Path]) -> List[str]:
    labels = []
    for json_path in inputs:
        parent_name = json_path.parent.name
        labels.append(parent_name or json_path.stem)
    return labels


def extract_metric_series(summary: Dict, metric: str) -> Tuple[List[int], List[float]]:
    records = summary.get("per_epoch_metrics", [])
    if not records:
        raise ValueError("JSON does not contain per_epoch_metrics.")

    pairs = []
    for record in records:
        metrics = record.get("metrics", {})
        if "epoch" not in record or metric not in metrics:
            continue
        pairs.append((int(record["epoch"]), float(metrics[metric])))

    if not pairs:
        raise ValueError(f"JSON does not contain usable '{metric}' values in per_epoch_metrics.")

    epochs, values = zip(*pairs)
    return list(epochs), list(values)


def extract_dual_metric_series(
    summary: Dict,
    left_metric: str,
    right_metric: str,
) -> Tuple[List[int], List[float], List[float]]:
    records = summary.get("per_epoch_metrics", [])
    if not records:
        raise ValueError("JSON does not contain per_epoch_metrics.")

    triples = []
    for record in records:
        metrics = record.get("metrics", {})
        if "epoch" not in record:
            continue
        if left_metric not in metrics or right_metric not in metrics:
            continue
        triples.append(
            (
                int(record["epoch"]),
                float(metrics[left_metric]),
                float(metrics[right_metric]),
            )
        )

    if not triples:
        raise ValueError(
            "JSON does not contain usable dual-axis values in per_epoch_metrics."
        )

    epochs = [epoch for epoch, _, _ in triples]
    left_values = [left_value for _, left_value, _ in triples]
    right_values = [right_value for _, _, right_value in triples]
    return epochs, left_values, right_values


def build_plot_payload(inputs: List[Path], labels: List[str], metric: str) -> Dict:
    series = []
    for json_path, label in zip(inputs, labels):
        summary = load_summary(json_path)
        epochs, values = extract_metric_series(summary, metric)
        series.append(
            {
                "label": label,
                "json_path": str(json_path.resolve()),
                "epochs": epochs,
                "values": values,
            }
        )

    return {
        "metric": metric,
        "series": series,
    }


def build_dual_axis_payload(
    inputs: List[Path],
    labels: List[str],
    left_metric: str,
    right_metric: str,
) -> Dict:
    series = []
    for json_path, label in zip(inputs, labels):
        summary = load_summary(json_path)
        epochs, left_values, right_values = extract_dual_metric_series(
            summary,
            left_metric,
            right_metric,
        )
        series.append(
            {
                "label": label,
                "json_path": str(json_path.resolve()),
                "epochs": epochs,
                "left_values": left_values,
                "right_values": right_values,
            }
        )

    return {
        "mode": "dual_axis",
        "left_metric": left_metric,
        "right_metric": right_metric,
        "series": series,
    }


def plot_series(plot_payload: Dict, metric: str, output_path: Path, title: str, dpi: int) -> None:
    meta = METRIC_META[metric]
    plt.figure(figsize=(8.5, 5), dpi=dpi)
    for series in plot_payload["series"]:
        plt.plot(series["epochs"], series["values"], linewidth=2, label=series["label"])

    plt.xlabel("Epoch")
    plt.ylabel(meta["y_label"])
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_dual_axis(plot_payload: Dict, output_path: Path, title: str, dpi: int) -> None:
    left_metric = plot_payload["left_metric"]
    right_metric = plot_payload["right_metric"]
    left_meta = METRIC_META[left_metric]
    right_meta = METRIC_META[right_metric]

    fig, ax_left = plt.subplots(figsize=(8.5, 5), dpi=dpi)
    ax_right = ax_left.twinx()
    colors = plt.rcParams.get("axes.prop_cycle", None)
    color_values = colors.by_key().get("color", []) if colors is not None else []

    for index, series in enumerate(plot_payload["series"]):
        color = color_values[index % len(color_values)] if color_values else None
        ax_left.plot(
            series["epochs"],
            series["left_values"],
            linewidth=2,
            color=color,
            label=f'{series["label"]} ({left_meta["legend_label"]})',
        )
        ax_right.plot(
            series["epochs"],
            series["right_values"],
            linewidth=2,
            linestyle="--",
            color=color,
            label=f'{series["label"]} ({right_meta["legend_label"]})',
        )

    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel(left_meta["y_label"])
    ax_right.set_ylabel(right_meta["y_label"])
    ax_left.set_title(title)
    ax_left.grid(True, linestyle="--", alpha=0.5)

    handles = []
    labels = []
    for axis in (ax_left, ax_right):
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    ax_left.legend(handles, labels, loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_plot_payload(plot_payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(plot_payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    inputs = [path.resolve() for path in args.inputs]
    labels = args.labels or default_labels(inputs)
    if len(labels) != len(inputs):
        raise SystemExit(f"Expected {len(inputs)} labels but got {len(labels)}.")

    use_plot_style(args.style)

    if args.dual_axis:
        left_metric = DUAL_AXIS_META["left_metric"]
        right_metric = DUAL_AXIS_META["right_metric"]
        plot_payload = build_dual_axis_payload(inputs, labels, left_metric, right_metric)
        default_output = inputs[0].parent / "classification_coverage_vs_full_acc.png"
        title = args.title or DUAL_AXIS_META["title"]
    else:
        plot_payload = build_plot_payload(inputs, labels, args.metric)
        default_output = inputs[0].parent / f"classification_{args.metric}.png"
        title = args.title or METRIC_META[args.metric]["title"]

    output_path = (args.output_path or default_output).resolve()
    save_json_path = (args.save_json or output_path.with_suffix(".json")).resolve()

    save_plot_payload(plot_payload, save_json_path)
    if args.dual_axis:
        plot_dual_axis(plot_payload, output_path, title, args.dpi)
    else:
        plot_series(plot_payload, args.metric, output_path, title, args.dpi)

    print(f"Saved plot data to: {save_json_path}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()