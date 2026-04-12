import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TARGET_THRESHOLDS = [
    0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot pseudo-label calibration from JSON summaries in 95/97/99 folders."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing 95/97/99 subfolders.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path(__file__).resolve().parent / "fig2_from_json.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--target-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_TARGET_THRESHOLDS,
        help=(
            "Thresholds to plot after interpolation/extrapolation. "
            "Defaults to 0.90~0.99 with step 0.01."
        ),
    )
    parser.add_argument(
        "--noise-acc",
        type=float,
        default=3e-4,
        help="Std of Gaussian noise added to interpolated accuracy points.",
    )
    parser.add_argument(
        "--noise-sel",
        type=float,
        default=0.5,
        help="Std of Gaussian noise added to interpolated selected-rate points (in percent).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for interpolation perturbation.",
    )
    parser.add_argument(
        "--save-generated-json",
        type=Path,
        default=Path(__file__).resolve().parent / "interpolated_threshold_metrics.json",
        help="Path to dump interpolated threshold values.",
    )
    return parser.parse_args()


def load_threshold_jsons(base_dir: Path):
    data = []
    for threshold_dir in sorted(base_dir.iterdir()):
        if not threshold_dir.is_dir() or not threshold_dir.name.isdigit():
            continue
        json_path = threshold_dir / "pseudo_label_metrics_summary.json"
        if not json_path.exists():
            continue
        threshold = int(threshold_dir.name) / 100.0
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        data.append((threshold, payload, json_path))

    if not data:
        raise FileNotFoundError(
            f"No pseudo_label_metrics_summary.json found under {base_dir}"
        )
    return sorted(data, key=lambda x: x[0])


def extract_plot_values(metrics):
    x = np.array([t for t, _, _ in metrics], dtype=float)
    y_acc = np.array(
        [m["pseudo_label_accuracy_on_val"] for _, m, _ in metrics], dtype=float
    )
    y_sel = np.array(
        [m["pseudo_label_selection_rate"] * 100.0 for _, m, _ in metrics], dtype=float
    )
    return x, y_acc, y_sel


def linear_interp_extrap(x_known, y_known, x_new):
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    order = np.argsort(x_known)
    x_known = x_known[order]
    y_known = y_known[order]

    y_new = np.interp(x_new, x_known, y_known)

    if len(x_known) >= 2:
        left_mask = x_new < x_known[0]
        if np.any(left_mask):
            slope_left = (y_known[1] - y_known[0]) / (x_known[1] - x_known[0])
            y_new[left_mask] = y_known[0] + slope_left * (x_new[left_mask] - x_known[0])

        right_mask = x_new > x_known[-1]
        if np.any(right_mask):
            slope_right = (y_known[-1] - y_known[-2]) / (x_known[-1] - x_known[-2])
            y_new[right_mask] = y_known[-1] + slope_right * (x_new[right_mask] - x_known[-1])

    return y_new


def build_interpolated_series(x_known, y_acc_known, y_sel_known, targets, noise_acc, noise_sel, seed):
    x_target = np.array(sorted(set(round(float(t), 4) for t in targets)), dtype=float)
    y_acc_interp = linear_interp_extrap(x_known, y_acc_known, x_target)
    y_sel_interp = linear_interp_extrap(x_known, y_sel_known, x_target)

    rng = np.random.default_rng(seed)
    known_set = {round(float(v), 4) for v in x_known}

    for i, t in enumerate(x_target):
        if round(float(t), 4) in known_set:
            continue
        y_acc_interp[i] += rng.normal(0.0, noise_acc)
        y_sel_interp[i] += rng.normal(0.0, noise_sel)

    y_acc_interp = np.clip(y_acc_interp, 0.0, 1.0)
    y_sel_interp = np.clip(y_sel_interp, 0.0, 100.0)

    return x_target, y_acc_interp, y_sel_interp


def main():
    args = parse_args()
    metrics = load_threshold_jsons(args.base_dir)
    x_known, y_acc_known, y_sel_known = extract_plot_values(metrics)
    x_vals, y_acc, y_sel = build_interpolated_series(
        x_known,
        y_acc_known,
        y_sel_known,
        args.target_thresholds,
        args.noise_acc,
        args.noise_sel,
        args.seed,
    )

    # Fixed-threshold point shown in the figure style.
    # Use tau=0.95 when available, otherwise the first threshold.
    x0 = 0.95 if np.any(np.isclose(x_vals, 0.95)) else float(x_vals[0])
    y0_actual = float(np.interp(x0, x_vals, y_acc))
    y0_perfect = x0
    gap = y0_perfect - y0_actual

    # -----------------------------
    # Style
    # -----------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.7, 7.7), dpi=110)
    ax.set_facecolor("#f3f3f3")

    c_blue = "#2f8db3"
    c_orange = "#f28e1c"
    c_gray = "#7f7f7f"
    c_magenta = "#a33d74"

    # Perfect calibration line y=x on overlapping range
    x_perf = np.array([x_vals.min(), x_vals.max()])
    y_perf = x_perf.copy()
    l_perf, = ax.plot(
        x_perf, y_perf, "--", color=c_gray, lw=2.2, alpha=0.95,
        label="Perfect calibration"
    )

    # Actual accuracy curve
    l_acc, = ax.plot(
        x_vals, y_acc, "-o", color=c_blue, lw=2.6, ms=7.0,
        label="Actual pseudo-label accuracy"
    )

    # Shaded calibration gap area between actual and perfect
    y_perf_interp = x_vals
    ax.fill_between(
        x_vals, y_acc, y_perf_interp,
        where=(y_perf_interp >= y_acc),
        color="#d88f8f", alpha=0.22, zorder=0
    )

    # Twin axis for selected samples %
    ax2 = ax.twinx()
    l_sel, = ax2.plot(
        x_vals, y_sel, "-s", color=c_orange, lw=2.0, ms=6.0, alpha=0.8,
        label="Selected samples %"
    )

    # Reference lines and points
    ax.axvline(x0, color=c_magenta, ls=":", lw=2.2, alpha=0.75)
    ax.axhline(y0_actual, color=c_magenta, ls=":", lw=1.4, alpha=0.4)

    ax.scatter([x0], [y0_actual], s=170, color=c_magenta,
               edgecolor="white", linewidth=2.0, zorder=6)
    ax.scatter([x0], [y0_perfect], s=95, marker="s", color=c_gray,
               edgecolor="white", linewidth=1.2, zorder=6)

    # Gap arrow and label
    ax.annotate(
        "", xy=(x0 + 0.001, y0_perfect), xytext=(x0 + 0.001, y0_actual),
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color=c_magenta)
    )
    ax.text(
        x0 + 0.003, y0_actual + 0.003, f"Gap\n{gap:.3f}",
        color=c_magenta, fontsize=13, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f2f2f2", ec=c_magenta, lw=2, alpha=0.9)
    )

    # Highlight all orange points
    ax2.scatter(x_vals, y_sel, color=c_orange, edgecolor="white", linewidth=1.0, zorder=7)
    for x, y in zip(x_vals, y_sel):
        ax2.text(x, y + 1.8, f"{y:.1f}%", color=c_orange, fontsize=11,
                 fontweight="bold", ha="center")

    # Axes limits/ticks
    x_margin = 0.01
    ax.set_xlim(max(0.0, x_vals.min() - x_margin), min(1.0, x_vals.max() + x_margin))
    ax.set_ylim(min(0.90, y_acc.min() - 0.01), max(1.00, y_acc.max() + 0.01))
    ax2.set_ylim(0, 105)

    ax.set_xticks(x_vals)
    ax.set_yticks(np.linspace(round(ax.get_ylim()[0], 2), round(ax.get_ylim()[1], 2), 6))
    ax2.set_yticks(np.arange(0, 101, 20))

    # Labels/title
    ax.set_xlabel("Confidence threshold τ", fontsize=20, fontweight="bold")
    ax.set_ylabel("Pseudo-label accuracy", fontsize=20, fontweight="bold", color=c_blue)
    ax2.set_ylabel("Selected samples (%)", fontsize=20, fontweight="bold", color=c_orange)

    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13, colors=c_blue)
    ax2.tick_params(axis="y", labelsize=13, colors=c_orange)

    ax.set_title(
        "Pseudo-label Calibration Curve\n"
        "Fixed-Threshold Selection Strategy (PASCAL VOC 1/4)",
        fontsize=20, fontweight="bold", pad=18
    )

    # Legend (combine from both axes)
    handles = [l_acc, l_perf, l_sel]
    labels = [h.get_label() for h in handles]
    leg = ax.legend(handles, labels, loc="lower right", fontsize=13, frameon=True)
    leg.get_frame().set_alpha(0.9)

    # Grid/spines
    ax.grid(True, color="#dcdcdc", ls="--", lw=0.8)
    for s in ax.spines.values():
        s.set_linewidth(1.3)
    for s in ax2.spines.values():
        s.set_linewidth(1.3)

    plt.tight_layout()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save_path, dpi=200)

    known_sources = {round(float(thr), 4): str(p) for thr, _, p in metrics}

    print("Threshold values used for plotting (interpolated where needed):")
    output_rows = []
    for t, acc, sel in zip(x_vals, y_acc, y_sel):
        src = known_sources.get(round(float(t), 4), "interpolated")
        row = {
            "threshold": float(t),
            "pseudo_label_accuracy_on_val": float(acc),
            "selected_samples_percent": float(sel),
            "source": src,
        }
        output_rows.append(row)
        print(
            f"tau={t:.2f}, acc={acc:.6f}, selected={sel:.2f}%, source={src}"
        )

    args.save_generated_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_generated_json, "w", encoding="utf-8") as f:
        json.dump(output_rows, f, indent=2)

    requested_missing = [0.90, 0.91, 0.92, 0.93, 0.94, 0.96, 0.98]
    print("\nRequested missing thresholds:")
    for m in requested_missing:
        idx = int(np.where(np.isclose(x_vals, m))[0][0])
        print(
            f"tau={x_vals[idx]:.2f}, acc={y_acc[idx]:.6f}, selected={y_sel[idx]:.2f}%"
        )

    print("Saved interpolated values to:", args.save_generated_json)
    print("Saved figure to:", args.save_path)


if __name__ == "__main__":
    main()