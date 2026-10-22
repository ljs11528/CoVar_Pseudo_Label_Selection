"""Reproduce the post-hoc P1 summary from checksummed, unmodified small outputs.

No model fitting, threshold selection, or new confidence intervals occur here.
All stored quantities are fractions; figures convert AP to percent/percentage points.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


STAGES = ("early", "middle", "late")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(summary, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    colors = {"raw_MC": "#222222", "h_prototype_margin": "#3984b8", "h_product": "#d48b2c",
              "h_uniform": "#309978", "h_e": "#bd4266"}
    labels = {"raw_MC": "MC", "h_prototype_margin": "h(MC, prototype margin)",
              "h_product": "h(MC, product)", "h_uniform": "h(MC, uniform)", "h_e": "h(MC, e)"}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1, 1.15]})
    for method, color in colors.items():
        values = [summary["stages"][s]["ranking"]["1024"]["conflict"][method]["error_ap"]*100 for s in STAGES]
        ax[0].plot(STAGES, values, "o-", label=labels[method], color=color, linewidth=1.8)
    ax[0].set(ylabel="Error AP (%)", title="A. Conflict pixels: fixed teacher, seed 42")
    ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
    ax[0].grid(axis="y", alpha=.2)
    rows = summary["stages"]["late"]["primary_contrasts"]
    names = ["F+e minus F_base", "h(MC,e) minus h(MC,proto)",
             "h(MC,e) minus h(MC,product)", "h(MC,e) minus mean shuffled AP"]
    for i, row in enumerate(rows):
        x = float(row["delta_error_ap"])*100
        lo, hi = float(row["ci_low"])*100, float(row["ci_high"])*100
        ax[1].errorbar(x, i, xerr=[[x-lo], [hi-x]], fmt="o", color="#bd4266", capsize=4)
        ax[1].text(hi+.06, i+.10, f"{x:+.3f}", fontsize=9)
    ax[1].axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax[1].set(yticks=range(4), yticklabels=names, xlabel="Delta error AP (percentage points)",
              title="B. Late primary contrasts: 98.75% image CIs", xlim=(-4.1, .45))
    ax[1].invert_yaxis()
    ax[1].grid(axis="x", alpha=.2)
    fig.text(.5, .015, "2000 paired image bootstraps; fixed teacher/heads. Three stages are not independent seeds.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1), w_pad=2)
    for suffix in ("png", "svg"):
        fig.savefig(target / f"phase1_primary.{suffix}", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
    groups = [("raw_e", "raw_uniform", "raw_permutation_mean"),
              ("h_e", "h_uniform", "h_permutation_mean")]
    for panel, methods in zip(ax, groups):
        for method, label, color in zip(methods, ("Aligned e", "Uniform evidence", "Mean of 10 shuffled APs"),
                                         ("#bd4266", "#309978", "#777777")):
            values = [summary["stages"][s]["permutation_comparison"][method]*100 for s in STAGES]
            panel.plot(STAGES, values, "o-", label=label, color=color)
        panel.set(ylabel="Conflict error AP (%)")
        panel.grid(axis="y", alpha=.2)
        panel.legend(fontsize=8)
    ax[0].set_title("Raw scores: class alignment helps vs shuffling")
    ax[1].set_title("Refitted h(MC, score): alignment loses its advantage")
    fig.text(.5, .015, "Point estimates; shuffled models are fitted separately, not ensembled. No new CI is inferred.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    for suffix in ("png", "svg"):
        fig.savefig(target / f"phase1_raw_vs_head.{suffix}", dpi=180)
    plt.close(fig)
    for path in target.glob("phase1_*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous summaries; choose a new directory")
    receipt = json.loads((args.snapshot / "snapshot_receipt.json").read_text())
    for filename, item in receipt["files"].items():
        path = args.snapshot / filename
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Snapshot integrity failure: {filename}")
    summary = {"scope": "Post-hoc summary of the original seed42 fixed-checkpoint diagnostics; not corrected BN results",
               "input_snapshot_receipt_sha256": digest(args.snapshot / "snapshot_receipt.json"),
               "verified_files": len(receipt["files"]), "source_files": receipt["files"],
               "units": "fractions for AP, accuracy, coverage and deltas; mIoU is not estimated by this script",
               "stages": {}}
    combined = {name: [] for name in ("ranking", "primary_contrasts", "complementarity", "calibration",
                                     "precision_targets", "conditional_selected", "head_cv")}
    for stage in STAGES:
        run = args.snapshot / "experiments" / f"frozen_seed42_{stage}_7c4c5c09"
        record = {"ranking": {}}
        for cap in (1024, 4096):
            rows = read_csv(run / f"analysis_{cap}/ranking.csv")
            combined["ranking"].extend(dict(stage=stage, cap=cap, **row) for row in rows)
            by_scope = {}
            for row in rows:
                by_scope.setdefault(row["scope"], {})[row["method"]] = {
                    key: float(row[key]) for key in ("error_ap", "auroc", "aurc_continuous", "error_prevalence",
                                                    "precision_at_0.5", "precision_at_0.9", "precision_at_0.95")}
            record["ranking"][str(cap)] = by_scope
        main_dir = run / "analysis_1024"
        record["primary_contrasts"] = read_csv(main_dir / "primary_contrasts.csv")
        combined["primary_contrasts"].extend(record["primary_contrasts"])
        a_rows = read_csv(main_dir / "A_complementarity.csv")
        combined["complementarity"].extend(dict(stage=stage, **row) for row in a_rows)
        record["complementarity"] = {}
        for scope in ("all", "high_mc"):
            by_group = {row["group"]: row for row in a_rows if row["scope"] == scope}
            groups = {key: float(row["coverage"]) for key, row in by_group.items() if row["coverage"]}
            assert abs(sum(groups.values())-1) < 1e-12
            wrong = groups["classifier_wrong_prototype_correct"] + groups["both_wrong_agree"] + groups["both_wrong_disagree"]
            conflict = groups["classifier_wrong_prototype_correct"] + groups["classifier_correct_prototype_wrong"] + groups["both_wrong_disagree"]
            error_cover = float(by_group["classifier_error"]["probability_e_positive"])
            record["complementarity"][scope] = dict(groups=groups, classifier_accuracy=1-wrong,
                prototype_accuracy=groups["both_correct"]+groups["classifier_wrong_prototype_correct"],
                conflict_coverage=conflict, conflict_error_fraction=(conflict-groups["classifier_correct_prototype_wrong"])/conflict,
                error_coverage_by_e=error_cover, error_blind_fraction=1-error_cover,
                prototype_false_alarm_to_rescue_ratio=groups["classifier_correct_prototype_wrong"]/groups["classifier_wrong_prototype_correct"])
            assert abs(wrong-record["ranking"]["1024"][scope]["raw_MC"]["error_prevalence"]) < 1e-10
        rank = record["ranking"]["1024"]["conflict"]
        comparison = {name: rank[name]["error_ap"] for name in ("raw_e", "raw_uniform", "raw_top2", "h_e", "h_uniform", "h_top2")}
        for prefix in ("raw", "h"):
            comparison[prefix+"_permutation_mean"] = statistics.mean(rank[f"{prefix}_permutation_{i}"]["error_ap"] for i in range(10))
        record["permutation_comparison"] = comparison
        for source, key in (("calibration_and_mass", "calibration"), ("precision_targets", "precision_targets"), ("head_cv", "head_cv")):
            rows = read_csv(main_dir / (source+".csv"))
            combined[key].extend(dict(stage=stage, **row) for row in rows)
            record[key] = rows
        best_cv = {}
        for row in record["head_cv"]:
            name = row["method"]
            if name not in best_cv or float(row["cv_nll"]) < best_cv[name]["cv_nll"]:
                best_cv[name] = {"cv_nll": float(row["cv_nll"]), "l2": float(row["l2"])}
        record["best_cv"] = best_cv
        metadata = json.loads((main_dir / "analysis_metadata.json").read_text())
        record["temperature"] = metadata["temperature"]
        record["bootstrap_metadata"] = metadata
        conditionals = read_csv(main_dir / "conditional.csv")
        selected = [row for row in conditionals if row["stratum"].startswith(("MC[", "predicted_class="))
                    or row["stratum"] in ("boundary", "interior")]
        combined["conditional_selected"].extend(dict(stage=stage, **row) for row in selected)
        record["secondary_intervals"] = [row for row in read_csv(main_dir / "paired_image_bootstrap.csv")
            if row["metric"] == "error_ap" and ((row["score"] == "F_plus" and row["delta_vs"] == "F_base")
                                              or (row["score"] == "h_e" and row["delta_vs"] in ("h_MC", "h_uniform")))]
        summary["stages"][stage] = record
    summary["gate"] = json.loads((args.snapshot / "experiments/P1_gate_seed42.json").read_text())
    args.output.mkdir(parents=True)
    for name, rows in combined.items():
        write_csv(args.output / (name+".csv"), rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    if args.figures:
        plot(summary, args.output)
    generated = {p.name: dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(args.output.iterdir())}
    (args.output / "receipt.json").write_text(json.dumps(dict(source_tool_sha256=digest(Path(__file__)), files=generated), indent=2)+"\n")
    print(json.dumps(dict(verified_input_files=summary["verified_files"], output=str(args.output), files=len(generated))))


if __name__ == "__main__":
    main()
