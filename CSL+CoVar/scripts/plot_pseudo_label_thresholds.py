import argparse
import json
import os

import matplotlib.pyplot as plt


METRICS = ["masked_acc", "full_acc", "coverage", "miou"]


def _load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_thresholds(raw_thresholds):
    if not raw_thresholds:
        return [round(0.90 + i * 0.01, 2) for i in range(10)]
    return sorted([round(float(v), 2) for v in raw_thresholds])


def _get_threshold_entry(summary_by_threshold, threshold):
    key_candidates = [
        f"{threshold:.2f}",
        str(threshold),
        f"{threshold:.1f}",
    ]
    for key in key_candidates:
        if key in summary_by_threshold:
            return summary_by_threshold[key]
    raise KeyError(f"Cannot find threshold={threshold:.2f} in summary_metrics_by_threshold")


def extract_series(summary):
    if "summary_metrics_by_threshold" not in summary:
        raise KeyError("JSON must contain summary_metrics_by_threshold")

    summary_by_threshold = summary["summary_metrics_by_threshold"]
    thresholds = _normalize_thresholds(summary.get("monitor_thresholds", None))

    y_values = {metric: [] for metric in METRICS}
    for threshold in thresholds:
        entry = _get_threshold_entry(summary_by_threshold, threshold)
        for metric in METRICS:
            if metric not in entry:
                raise KeyError(f"Missing metric '{metric}' under threshold={threshold:.2f}")
            y_values[metric].append(float(entry[metric]))

    return thresholds, y_values


def plot_threshold_curves(thresholds, y_values, save_path, title):
    plt.figure(figsize=(10, 6))

    style_map = {
        "masked_acc": {"marker": "o", "linestyle": "-"},
        "full_acc": {"marker": "s", "linestyle": "--"},
        "coverage": {"marker": "^", "linestyle": "-."},
        "miou": {"marker": "d", "linestyle": ":"},
    }

    for metric in METRICS:
        style = style_map[metric]
        plt.plot(
            thresholds,
            y_values[metric],
            label=metric,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=6,
        )

    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.xticks(thresholds, [f"{thr:.2f}" for thr in thresholds])
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot masked_acc/full_acc/coverage/miou vs threshold from pseudo_label_metrics_summary.json"
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to pseudo_label_metrics_summary.json",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save output figure. Defaults to JSON directory/pseudo_label_threshold_curves.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Pseudo Label Metrics vs Threshold",
        help="Figure title",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary = _load_json(args.json_path)
    thresholds, y_values = extract_series(summary)

    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(args.json_path),
            "pseudo_label_threshold_curves.png",
        )

    plot_threshold_curves(thresholds, y_values, save_path, args.title)
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    main()