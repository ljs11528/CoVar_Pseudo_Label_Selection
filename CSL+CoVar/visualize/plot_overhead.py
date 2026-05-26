"""
Parse [DynamicThreshold Overhead] lines from output.log and plot:
  - Left Y-axis (lines): Epoch Train Time  vs.  Epoch Train Time + Dynamic Compute Time
  - Right Y-axis (bars): Peak Extra GPU Memory (per batch)
"""

import re
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_LOG = os.path.join(
    os.path.dirname(__file__),
    "../exp/pascal/CSL/r101/1_4/output.log",
)
OUT_IMAGE = os.path.join(os.path.dirname(__file__), "overhead_plot.png")

PATTERN = re.compile(
    r"\[DynamicThreshold Overhead\] Epoch (\d+) \| "
    r"Epoch Train Time: ([\d.]+)s \| "
    r"Dynamic Threshold Compute Time: ([\d.]+)s "
    r"\([\d.]+% of epoch\) \| "
    r"Peak Extra GPU Memory \(per batch\): ([\d.]+) MB"
)

# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
def parse_log(log_path):
    epochs, train_times, dyn_times, mem_peaks = [], [], [], []
    with open(log_path, "r") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                epochs.append(int(m.group(1)))
                train_times.append(float(m.group(2)))
                dyn_times.append(float(m.group(3)))
                mem_peaks.append(float(m.group(4)))
    return (
        np.array(epochs),
        np.array(train_times),
        np.array(dyn_times),
        np.array(mem_peaks),
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(epochs, train_times, dyn_times, mem_peaks, out_path):
    train_plus_dyn = train_times + dyn_times

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=150)
    ax2 = ax1.twinx()

    c_train    = "#1f77b4"   # blue  – Epoch Train Time
    c_combined = "#ff7f0e"   # orange – Train + Dynamic
    c_mem      = "#d62728"   # red   – GPU memory line

    bar_width = 0.38
    offset = bar_width / 2

    # --- Grouped bars (left axis) -------------------------------------------
    bar1 = ax1.bar(
        epochs - offset, train_times,
        width=bar_width,
        color=c_train, alpha=0.80,
        zorder=2,
        label="Epoch Train Time (s) (Fixed Baseline)",
    )
    bar2 = ax1.bar(
        epochs + offset, train_plus_dyn,
        width=bar_width,
        color=c_combined, alpha=0.80,
        zorder=2,
        label="Epoch Train Time (s) (CoVar)",
    )

    ax1.set_xlabel("Training Epoch", fontsize=13)
    ax1.set_ylabel("Time (s)", fontsize=13, color="black")
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.set_ylim(400, 800)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Line (right axis) --------------------------------------------------
    line, = ax2.plot(
        epochs, mem_peaks,
        marker="o", linewidth=2, markersize=6,
        color=c_mem, zorder=4,
        label="Peak Extra GPU Memory (per batch)",
    )
    ax2.set_ylabel("Peak Extra GPU Memory (MB)", fontsize=13, color=c_mem)
    ax2.tick_params(axis="y", labelcolor=c_mem)
    ax2.set_ylim(60, 70)

    # --- Legend -------------------------------------------------------------
    handles = [bar1, bar2, line]
    labels  = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper right", fontsize=10, framealpha=0.9)

    plt.title(
        "Dynamic Threshold Overhead per Epoch\n"
        "(CSL+CoVar · Pascal VOC · 1/4 split · ResNet-101)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    log_path = os.path.abspath(log_path)
    if not os.path.isfile(log_path):
        sys.exit(f"ERROR: log file not found: {log_path}")

    epochs, train_times, dyn_times, mem_peaks = parse_log(log_path)
    if len(epochs) == 0:
        sys.exit("ERROR: no [DynamicThreshold Overhead] lines found in the log.")

    print(f"Parsed {len(epochs)} epochs (0 … {epochs[-1]})")
    plot(epochs, train_times, dyn_times, mem_peaks, OUT_IMAGE)
