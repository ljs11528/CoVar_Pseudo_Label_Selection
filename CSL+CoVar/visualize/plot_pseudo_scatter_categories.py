import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D


DEFAULT_THRESHOLD = 0.95
DEFAULT_X_LIMITS = (-0.1, 1.0)
DEFAULT_Y_LIMITS = (0.9, 1.0)
VALID_CATEGORY_KEYS = ("A", "B", "C", "D")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot rcv-vs-mc scatter points grouped by mc threshold and selection flag."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to the pseudo scatter txt file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <input_stem>_grouped_scatter.pdf.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="mc threshold used to split the four categories.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Point transparency in the scatter plot.",
    )
    parser.add_argument(
        "--size",
        type=float,
        default=18.0,
        help="Marker size for each sample point.",
    )
    parser.add_argument(
        "--x-scale",
        choices=("log", "linear"),
        default="log",
        help="Scale used for the rcv axis. Defaults to log because rcv often spans multiple orders of magnitude; the plot falls back to linear if the visible x-range includes 0.",
    )
    parser.add_argument(
        "--legend-count",
        action="append",
        default=[],
        metavar="CATEGORY=COUNT",
        help="Override the displayed legend/sample count for a category, e.g. --legend-count A=75185. Can be repeated for A/B/C/D.",
    )
    return parser.parse_args()


def parse_legend_count_overrides(entries):
    overrides = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --legend-count value '{entry}'. Expected CATEGORY=COUNT."
            )

        category_key, count_text = entry.split("=", 1)
        category_key = category_key.strip().upper()
        count_text = count_text.strip()

        if category_key not in VALID_CATEGORY_KEYS:
            raise ValueError(
                f"Invalid category '{category_key}' in --legend-count. Expected one of {VALID_CATEGORY_KEYS}."
            )

        try:
            count_value = int(count_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid count '{count_text}' for category '{category_key}'. Expected an integer."
            ) from exc

        if count_value < 0:
            raise ValueError(
                f"Invalid count '{count_value}' for category '{category_key}'. Expected a non-negative integer."
            )

        overrides[category_key] = count_value

    return overrides


def load_points(input_path):
    rows = []
    with input_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split()
        expected_header = ["rcv", "mc", "selected"]
        if header[:3] != expected_header:
            raise ValueError(
                f"Expected header {expected_header}, but got {header}."
            )

        for line_number, line in enumerate(handle, start=2):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Line {line_number} must contain rcv, mc, selected: {line.rstrip()}"
                )

            rows.append(
                {
                    "rcv": float(parts[0]),
                    "mc": float(parts[1]),
                    "selected": int(float(parts[2])),
                }
            )

    if not rows:
        raise ValueError(f"No sample points found in {input_path}.")

    return rows


def build_categories(points, threshold):
    categories = {
        "A": {
            "label": f"A: mc >= {threshold:.2f}, selected = 1",
            "color": "#4C46A8",
            "marker": "o",
            "points": [],
        },
        "B": {
            "label": f"B: mc < {threshold:.2f}, selected = 1",
            "color": "#DF6D84",
            "marker": "^",
            "size_scale": 1.2,
            "legend_markersize": 8,
            "points": [],
        },
        "C": {
            "label": f"C: mc >= {threshold:.2f}, selected = 0",
            "color": "#57B7A8",
            "marker": "s",
            "points": [],
        },
        "D": {
            "label": f"D: mc < {threshold:.2f}, selected = 0",
            "color": "#8CCEF2",
            "marker": "D",
            "points": [],
        },
    }

    for point in points:
        mc = point["mc"]
        selected = point["selected"]

        if selected == 1 and mc >= threshold:
            category = "A"
        elif selected == 1 and mc < threshold:
            category = "B"
        elif selected == 0 and mc >= threshold:
            category = "C"
        else:
            category = "D"

        categories[category]["points"].append((point["rcv"], mc))

    return categories


def get_display_count(category_key, category, legend_count_overrides):
    return legend_count_overrides.get(category_key, len(category["points"]))


def plot_categories(
    categories,
    threshold,
    output_path,
    x_scale,
    alpha,
    marker_size,
    title,
    legend_count_overrides,
):
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter_handles = {}

    def build_marker_handle(category_key):
        category = categories[category_key]
        return Line2D(
            [],
            [],
            linestyle="none",
            marker=category["marker"],
            markersize=category.get("legend_markersize", 7),
            markerfacecolor=category["color"],
            markeredgecolor=category["color"],
        )

    for key in ("D", "C", "A", "B"):
        category = categories[key]
        if not category["points"]:
            continue

        x_values, y_values = zip(*category["points"])
        display_count = get_display_count(key, category, legend_count_overrides)
        scatter_handles[key] = ax.scatter(
            x_values,
            y_values,
            s=marker_size * category.get("size_scale", 1.0),
            c=category["color"],
            marker=category["marker"],
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=f"{category['label']} (n={display_count})",
        )

    threshold_handle = ax.axhline(
        threshold,
        color="#444444",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
        label=f"MC = {threshold:.2f}",
    )
    ax.set_xlabel("RCV")
    ax.set_ylabel("MC")
    # ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)

    if x_scale == "log" and DEFAULT_X_LIMITS[0] > 0:
        ax.set_xscale("log")

    ax.set_xlim(*DEFAULT_X_LIMITS)
    ax.set_ylim(*DEFAULT_Y_LIMITS)
    legend_handles = [
        scatter_handles[key] for key in ("A", "B", "C", "D") if key in scatter_handles
    ]
    legend_labels = [
        f"{categories[key]['label']} (n={get_display_count(key, categories[key], legend_count_overrides)})"
        for key in VALID_CATEGORY_KEYS
        if key in scatter_handles
    ]
    legend_handles.append(threshold_handle)
    legend_labels.append(f"mc = {threshold:.2f}")
    legend_handles.extend(
        [
            (build_marker_handle("A"), build_marker_handle("C")),
            (build_marker_handle("A"), build_marker_handle("B")),
        ]
    )
    legend_labels.extend(
        [
            f"Fixed={threshold:.2f}: A+C",
            f"Ours: A+B",
        ]
    )
    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        frameon=True,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_path = args.output
    if output_path is None:
        output_path = args.input_path.with_name(
            f"{args.input_path.stem}_grouped_scatter.pdf"
        )

    points = load_points(args.input_path)
    categories = build_categories(points, args.threshold)
    legend_count_overrides = parse_legend_count_overrides(args.legend_count)
    title = f"Pseudo Label Scatter by Category ({args.input_path.stem})"
    plot_categories(
        categories=categories,
        threshold=args.threshold,
        output_path=output_path,
        x_scale=args.x_scale,
        alpha=args.alpha,
        marker_size=args.size,
        title=title,
        legend_count_overrides=legend_count_overrides,
    )

    print(f"Saved scatter plot to {output_path}")
    for key in ("A", "B", "C", "D"):
        print(f"{key}: {len(categories[key]['points'])}")


if __name__ == "__main__":
    main()