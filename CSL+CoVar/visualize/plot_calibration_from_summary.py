import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot pseudo-label calibration curve from summary_metrics_by_threshold."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path(
            "/data/zxf_test/user1/kd/covar/CoVar_Pseudo_Label_Selection/"
            "CSL+CoVar/visualize/95/pseudo_label_metrics_summary.json"
        ),
        help="Path to the pseudo_label_metrics_summary.json file.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path(__file__).resolve().parent / "fig2_calibration.png",
        help="Output image path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    by_thr = data["summary_metrics_by_threshold"]

    # Sort thresholds numerically
    thresholds = sorted(by_thr.keys(), key=float)
    x_vals = np.array([float(t) for t in thresholds])
    y_acc = np.array([by_thr[t]["masked_acc"] for t in thresholds])
    y_cov = np.array([by_thr[t]["coverage"] for t in thresholds])
    y_sel = y_cov * 100.0
    y_combo = (y_acc + y_cov) / 2.0 + 0.08

    # Reference point at tau=0.95
    x0 = 0.95
    idx0 = int(np.argmin(np.abs(x_vals - x0)))
    y0_actual = y_acc[idx0]
    y0_perfect = x_vals[idx0]
    gap = y0_perfect - y0_actual

    # ---- Style ----
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.7, 7.7), dpi=110)
    ax.set_facecolor("#f3f3f3")

    c_blue = "#2f8db3"
    c_orange = "#f28e1c"
    c_gray = "#7f7f7f"
    c_magenta = "#a33d74"

    # Perfect calibration y=x
    x_perf = np.array([x_vals.min(), x_vals.max()])
    l_perf, = ax.plot(
        x_perf, x_perf, "--", color=c_gray, lw=2.2, alpha=0.95,
        label="Perfect calibration"
    )

    # Blue: masked_acc
    l_acc, = ax.plot(
        x_vals, y_acc, "-o", color=c_blue, lw=2.6, ms=7.0,
        label="Actual pseudo-label accuracy"
    )

    # Green dashed: (masked_acc + coverage) / 2 + 0.08
    c_green = "#2ca02c"
    # l_combo, = ax.plot(
    #     x_vals, y_combo, "--D", color=c_green, lw=2.0, ms=5.5, alpha=0.85,
    #     label="(Acc + Coverage) / 2 + 0.08"
    # )

    # Shaded gap
    ax.fill_between(
        x_vals, y_acc, x_vals,
        where=(x_vals >= y_acc),
        color="#d88f8f", alpha=0.22, zorder=0
    )

    # Orange: coverage (twin axis)
    ax2 = ax.twinx()
    l_sel, = ax2.plot(
        x_vals, y_sel, "-s", color=c_orange, lw=2.0, ms=6.0, alpha=0.8,
        label="Coverage %"
    )

    # Reference lines / gap annotation at tau=0.95
    ax.axvline(x0, color=c_magenta, ls=":", lw=2.2, alpha=0.75)
    ax.axhline(y0_actual, color=c_magenta, ls=":", lw=1.4, alpha=0.4)

    ax.scatter([x0], [y0_actual], s=170, color=c_magenta,
               edgecolor="white", linewidth=2.0, zorder=6)
    ax.scatter([x0], [y0_perfect], s=95, marker="s", color=c_gray,
               edgecolor="white", linewidth=1.2, zorder=6)

    ax.annotate(
        "", xy=(x0 + 0.001, y0_perfect), xytext=(x0 + 0.001, y0_actual),
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color=c_magenta)
    )
    ax.text(
        x0 + 0.003, y0_actual + 0.003, f"Gap\n{gap:.3f}",
        color=c_magenta, fontsize=13, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f2f2f2", ec=c_magenta, lw=2, alpha=0.9)
    )

    # Orange percentage labels
    highlights = [x_vals[0], 0.95, x_vals[-1]]
    ax2.scatter(x_vals, y_sel, color=c_orange, edgecolor="white", linewidth=1.0, zorder=7)
    for x, y in zip(x_vals, y_sel):
        if any(abs(x - h) < 1e-6 for h in highlights):
            ax2.text(x, y + 1.8, f"{y:.1f}%", color=c_orange, fontsize=11,
                     fontweight="bold", ha="center")

    # Axes
    x_margin = 0.01
    ax.set_xlim(x_vals.min() - x_margin, x_vals.max() + x_margin)
    y_all = np.concatenate([y_acc, y_combo])
    ax.set_ylim(min(0.90, y_all.min() - 0.01), max(1.00, y_all.max() + 0.01))
    ax2.set_ylim(0, 105)

    ax.set_xticks(x_vals)
    ax.set_yticks(np.linspace(round(ax.get_ylim()[0], 2), round(ax.get_ylim()[1], 2), 6))
    ax2.set_yticks(np.arange(0, 101, 20))

    ax.set_xlabel("Maximum Confidence threshold τ", fontsize=20, fontweight="bold")
    ax.set_ylabel("Pseudo-label accuracy", fontsize=20, fontweight="bold", color=c_blue)
    ax2.set_ylabel("Coverage (%)", fontsize=20, fontweight="bold", color=c_orange)

    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13, colors=c_blue)
    ax2.tick_params(axis="y", labelsize=13, colors=c_orange)

    ax.set_title(
        "Pseudo-label Calibration Curve\n"
        "Fixed-Threshold Selection Strategy (PASCAL VOC 1/4)",
        fontsize=20, fontweight="bold", pad=18
    )

    handles = [l_acc, l_perf, l_sel]
    labels = [h.get_label() for h in handles]
    leg = ax.legend(handles, labels, loc="lower right", fontsize=13, frameon=True)
    leg.get_frame().set_alpha(0.9)

    ax.grid(True, color="#dcdcdc", ls="--", lw=0.8)
    for s in ax.spines.values():
        s.set_linewidth(1.3)
    for s in ax2.spines.values():
        s.set_linewidth(1.3)

    plt.tight_layout()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save_path, dpi=200)

    print("Plotted thresholds:")
    for t, acc, sel in zip(x_vals, y_acc, y_sel):
        print(f"  tau={t:.2f}  masked_acc={acc:.6f}  coverage={sel:.2f}%")
    print("Combo line values:")
    for t, v in zip(x_vals, y_combo):
        print(f"  tau={t:.2f}  combo={v:.6f}")
    print(f"Gap at tau={x0}: {gap:.4f}")
    print("Saved figure to:", args.save_path)


if __name__ == "__main__":
    main()
