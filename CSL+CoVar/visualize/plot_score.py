import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, LogFormatter

# ── load data ──────────────────────────────────────────────────────────────────
json_path = "../exp/pascal/CSL/r101/1_4/score_stats.json"
with open(json_path) as f:
    data = json.load(f)

epochs     = [d["epoch"]     for d in data]
score_max  = [d["score_max"] for d in data]
score_mean = [d["score_mean"] for d in data]
score_min  = [d["score_min"] for d in data]

# replace any 0 with a very small positive number for log scale
score_min_log = [v if v > 0 else 1e-30 for v in score_min]

# ── figure layout ──────────────────────────────────────────────────────────────
# Three rows sharing the x-axis; different y-ranges per row.
fig = plt.figure(figsize=(10, 8))
gs  = gridspec.GridSpec(3, 1, hspace=0.15)

ax_max  = fig.add_subplot(gs[0])
ax_mean = fig.add_subplot(gs[1])
ax_min  = fig.add_subplot(gs[2])

COLOR_MAX  = "#d62728"   # red
COLOR_MEAN = "#1f77b4"   # blue
COLOR_MIN  = "#2ca02c"   # green

# ── score_max  [4.2, 4.3] ─────────────────────────────────────────────────────
ax_max.plot(epochs, score_max, color=COLOR_MAX, linewidth=1.5, marker='o',
            markersize=2.5, label="score_max")
ax_max.set_ylim(4.20, 4.30)
ax_max.set_ylabel("score_max", color=COLOR_MAX)
ax_max.tick_params(axis='y', labelcolor=COLOR_MAX)
ax_max.set_yticks(np.arange(4.20, 4.31, 0.02))
ax_max.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax_max.set_xlim(epochs[0], epochs[-1])
ax_max.set_title("Score Statistics over Epochs", fontsize=13, fontweight='bold')
ax_max.grid(True, linestyle='--', alpha=0.5)

# ── score_mean  [0.2, 0.6] ────────────────────────────────────────────────────
ax_mean.plot(epochs, score_mean, color=COLOR_MEAN, linewidth=1.5, marker='s',
             markersize=2.5, label="score_mean")
ax_mean.set_ylim(0.20, 0.60)
ax_mean.set_ylabel("score_mean", color=COLOR_MEAN)
ax_mean.tick_params(axis='y', labelcolor=COLOR_MEAN)
ax_mean.set_yticks(np.arange(0.20, 0.61, 0.08))
ax_mean.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
ax_mean.set_xlim(epochs[0], epochs[-1])
ax_mean.grid(True, linestyle='--', alpha=0.5)

# ── score_min  log10 scale, range [2e-25, 1e0] ────────────────────────────────
ax_min.semilogy(epochs, score_min_log, color=COLOR_MIN, linewidth=1.5,
                marker='^', markersize=2.5, label="score_min")
ax_min.set_ylim(2e-25, 1.0)
ax_min.set_ylabel("score_min  (log₁₀)", color=COLOR_MIN)
ax_min.tick_params(axis='y', labelcolor=COLOR_MIN)
ax_min.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
ax_min.yaxis.set_major_formatter(
    matplotlib.ticker.LogFormatterSciNotation(base=10, labelOnlyBase=False)
)
ax_min.set_xlim(epochs[0], epochs[-1])
ax_min.set_xlabel("Epoch", fontsize=11)
ax_min.grid(True, linestyle='--', alpha=0.5, which='both')

# ── shared x-label & save ─────────────────────────────────────────────────────
for ax in (ax_max, ax_mean):
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelbottom=True)

out_path = "../exp/pascal/CSL/r101/1_4/score_stats.png"
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
