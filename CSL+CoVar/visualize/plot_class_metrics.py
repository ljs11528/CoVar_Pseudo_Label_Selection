import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_majority_mask(txt_path, n_classes=21):
    """
    Read per-class pixel counts from txt, rank by 'generated' descending,
    top 50% (ceil) are majority (alpha=1.0), rest minority (alpha=0.5).
    Returns a bool array of length n_classes indexed by class_idx.
    """
    rows = []
    with open(txt_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            rows.append((int(row['class_idx']), int(row['generated'])))
    rows.sort(key=lambda r: r[1], reverse=True)
    n_majority = (n_classes + 1) // 2  # ceil(21/2) = 11
    majority_indices = {r[0] for r in rows[:n_majority]}
    is_majority = np.array([i in majority_indices for i in range(n_classes)])
    return is_majority

def main():
    covar_path = "/data/zxf_test/user1/kd/covar/CoVar_Pseudo_Label_Selection/CSL+CoVar/visualize/covar/pseudo_label_metrics_summary.json"
    covar2_path = "/data/zxf_test/user1/kd/covar/CoVar_Pseudo_Label_Selection/CSL+CoVar/visualize/covar2/pseudo_label_metrics_summary.json"
    fixed_path = "/data/zxf_test/user1/kd/covar/CoVar_Pseudo_Label_Selection/CSL+CoVar/visualize/95/pseudo_label_metrics_summary.json"
    txt_path = "/data/zxf_test/user1/kd/covar/CoVar_Pseudo_Label_Selection/CSL+CoVar/visualize/pseudo_per_class_epoch_76.txt"

    # Load data
    covar_data = load_json(covar_path)["summary_metrics"]
    covar2_data = load_json(covar2_path)["summary_metrics"]
    
    # 95 path might not have "summary_metrics" at top level, check if it has "summary_metrics_by_threshold"
    fixed_json = load_json(fixed_path)
    if "summary_metrics" in fixed_json:
        fixed_data = fixed_json["summary_metrics"]
    else:
        fixed_data = fixed_json["summary_metrics_by_threshold"]["0.95"]

    categories = [
        "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
        "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", 
        "potted plant", "sheep", "sofa", "train", "tv/monitor"
    ]

    x = np.arange(len(categories))
    width = 0.25

    # Majority/minority split: top 50% by generated pixel count (from txt)
    is_majority = load_majority_mask(txt_path, n_classes=len(categories))
    alphas = np.where(is_majority, 1.0, 0.5)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(15, 8), dpi=150)

    # Coverage (Left Axis - Bars)
    covar_cov = covar_data["cat_coverage"]
    covar2_cov = covar2_data["cat_coverage"]
    fixed_cov = fixed_data["cat_coverage"]

    # Colors from the image roughly
    c_covar_bar = "#1f77b4" # Blue
    c_covar2_bar = "#2ca02c" # Green 
    c_fixed_bar = "#ff7f0e" # Orange

    for i in range(len(categories)):
        a = alphas[i]
        ax1.bar(x[i] - width, covar_cov[i], width, color=c_covar_bar, alpha=a)
        ax1.bar(x[i],          covar2_cov[i], width, color=c_covar2_bar, alpha=a)
        ax1.bar(x[i] + width,  fixed_cov[i],  width, color=c_fixed_bar,  alpha=a)

    ax1.set_xlabel('Class Name', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Coverage', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    # Accuracy (Right Axis - Lines)
    ax2 = ax1.twinx()
    covar_acc = covar_data["cat_acc"]
    covar2_acc = covar2_data["cat_acc"]
    fixed_acc = fixed_data["cat_acc"]

    # Colors for lines
    c_covar_line = "#17becf" # Cyan-ish
    c_covar2_line = "#9467bd" # Purple
    c_fixed_line = "#d62728" # Red

    ax2.plot(x, covar_acc, marker='o', color=c_covar_line, lw=2.5, ms=6)
    ax2.plot(x, covar2_acc, marker='^', ls='-', color=c_covar2_line, lw=2.5, ms=6)
    ax2.plot(x, fixed_acc, marker='s', ls='--', color=c_fixed_line, lw=2.5, ms=6)

    ax2.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=12)

    # Explicit legend handles with a standard opaque legend box.
    legend_handles = [
        Patch(facecolor='gray', alpha=1.0, edgecolor='none', label='Majority class (opaque)'),
        Patch(facecolor=c_covar2_bar, edgecolor='none', label='Fixed(0.95) - Coverage (Bar)'),
        Patch(facecolor=c_covar_bar, edgecolor='none', label='W/O Cov(g, v) - Coverage (Bar)'),
        Patch(facecolor=c_fixed_bar, edgecolor='none', label='W/ Cov(g, v) - Coverage (Bar)'),
        Patch(facecolor='gray', alpha=0.5, edgecolor='none', label='Minority class (α=0.5)'),
        Line2D([0], [0], color=c_covar_line, marker='o', lw=2.5, ms=6, label='Fixed(0.95) - Accuracy (Line)'),
        Line2D([0], [0], color=c_covar2_line, marker='^', lw=2.5, ms=6, ls='-', label='W/O Cov(g, v) - Accuracy (Line)'),
        Line2D([0], [0], color=c_fixed_line, marker='s', lw=2.5, ms=6, ls='--', label='W/ Cov(g, v) - Accuracy (Line)'),
    ]
    leg = ax2.legend(
        handles=legend_handles,
        loc='lower left',
        ncol=2,
        fontsize=12,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        framealpha=1.0,
    )
    leg.set_zorder(1000)
    
    plt.title('Comparison of Pseudo-Label Selection Rate and Accuracy Across Categories', fontsize=16, fontweight='bold', pad=40)
    plt.tight_layout()
    plt.savefig('/data/zxf_test/user1/kd/plot_category_comparison.png', dpi=300)
    print("Saved figure to /data/zxf_test/user1/kd/plot_category_comparison.png")

if __name__ == "__main__":
    main()
