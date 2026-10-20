"""Run A -> B -> C -> D on a completed, image-isolated frozen export."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

import numpy as np
import torch

from .analysis_utils import (calibration, choose_temperature, fit_selected_head, head_features,
                            permutation_features, predict_head, read_pool, temperature_features)
from .export_frozen import atomic_json
from .diagnostic_tables import calibration_intervals, conditional_rows, support
from .gpu_bootstrap import KEYS, paired_image_bootstrap_gpu
from .prepare_data import digest, source_revision
from .ranking_metrics import Ranking, selection_summary


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def complementary_table(pool):
    error = pool["error"].astype(bool)
    prototype_correct = pool["prototype_label"] == pool["GT"]
    agree = pool["prototype_label"] == pool["pseudo_label"]
    output = []
    for scope, mask in {"all":np.ones(len(error), bool), "high_mc":pool["mc"] >= .95}.items():
        groups = {"both_correct":~error & prototype_correct,
                  "classifier_wrong_prototype_correct":error & prototype_correct,
                  "classifier_correct_prototype_wrong":~error & ~prototype_correct,
                  "both_wrong_agree":error & ~prototype_correct & agree,
                  "both_wrong_disagree":error & ~prototype_correct & ~agree}
        for name, group in groups.items():
            selected = mask & group
            denominator = pool["sampling_weight"][mask].sum()
            mass = pool["sampling_weight"][selected].sum()
            output.append(dict(scope=scope, group=name, sampled_pixels=int(selected.sum()),
                images=len(np.unique(pool["image"][selected])), estimated_pixels=float(mass),
                coverage=float(mass/denominator) if denominator else None,
                mean_mc=float(np.average(pool["mc"][selected], weights=pool["sampling_weight"][selected])) if mass else None,
                mean_e=float(np.average(pool["competitor_evidence"][selected], weights=pool["sampling_weight"][selected])) if mass else None))
        for name, group in {"classifier_error":error, "classifier_correct":~error}.items():
            selected = mask & group
            mass = pool["sampling_weight"][selected].sum()
            output.append(dict(scope=scope, group=name, sampled_pixels=int(selected.sum()),
                images=len(np.unique(pool["image"][selected])), estimated_pixels=float(mass),
                probability_e_positive=float(np.average(pool["competitor_evidence"][selected] > 0,
                    weights=pool["sampling_weight"][selected])) if mass else None,
                probability_strict_conflict=float(np.average(pool["strict_conflict"][selected],
                    weights=pool["sampling_weight"][selected])) if mass else None))
    return output


def coverage_at_precision(error, risk, weight, target):
    """Descriptive test-GT curve, including random fractional selection in ties."""
    mass, errors, _ = Ranking(error, risk).groups(weight)
    cum_mass, cum_error = mass.cumsum(), errors.cumsum()
    previous_mass, previous_error = cum_mass-mass, cum_error-errors
    rate = errors/mass
    target_risk = 1-target
    eligible = cum_error <= target_risk*cum_mass+1e-12
    maximum = float(cum_mass[eligible].max()) if eligible.any() else 0.
    crossing = ((previous_error <= target_risk*previous_mass+1e-12)
                & (cum_error > target_risk*cum_mass) & (rate > target_risk))
    if crossing.any():
        selected = previous_mass[crossing]+(target_risk*previous_mass[crossing]-previous_error[crossing])/(rate[crossing]-target_risk)
        maximum = max(maximum, float(selected.max()))
    return maximum/cum_mass[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--cap", type=int, choices=[1024, 4096], default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    export = args.export.resolve()
    package = Path(__file__).resolve().parent
    protocol = json.loads((package / "protocol.json").read_text())
    if not (export / "COMPLETE.json").exists():
        raise ValueError("Frozen export incomplete")
    manifest = json.loads((export / "manifest.json").read_text())
    if bool(manifest.get("synthetic", False)) != args.synthetic_smoke:
        raise ValueError("Synthetic caches require an explicit smoke flag; never mix them with formal results")
    if not args.synthetic_smoke:
        expected = dict(D_sup=protocol["partition"]["supervised_images"], D_rel=protocol["partition"]["verifier_images"], validation=protocol["dataset"]["diagnostic_images"])
        if manifest["images"] != expected:
            raise ValueError("Formal export image counts differ from the locked protocol")
    if manifest["signature"]["protocol_sha256"] != digest(package / "protocol.json"):
        raise ValueError("Export and analysis protocols differ")
    replicates = (protocol["statistics"]["bootstrap_replicates"] if args.cap == 1024 else 0)
    if args.bootstrap is not None and args.bootstrap != replicates:
        raise ValueError("Cannot override the locked formal bootstrap budget")
    output = export / f"analysis_{args.cap}"
    output.mkdir(exist_ok=True)
    signature = dict(source_commit=source_revision(package), protocol_sha256=digest(package / "protocol.json"),
                     export_manifest_sha256=digest(export / "manifest.json"), cap=args.cap,
                     analysis_tool_sha256=digest(__file__), bootstrap_replicates=replicates)
    signature["synthetic_smoke"] = args.synthetic_smoke
    status_file = output / "analysis_status.json"
    def update(phase, **extra):
        atomic_json(status_file, dict(state="running", phase=phase, signature=signature,
                    updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **extra))
    if (output / "COMPLETE.json").exists():
        if json.loads((output / "COMPLETE.json").read_text())["signature"] != signature:
            raise ValueError("Existing analysis signature differs")
        print("Existing completed analysis preserved", flush=True)
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    update("read_fixed_role_caches")
    train = read_pool(export, "D_rel", args.cap)
    test = read_pool(export, "validation", args.cap)
    if set(train["image_ids"]) & set(test["image_ids"]):
        raise ValueError("Verifier/evaluation images overlap")
    update("A_classifier_prototype_complementarity")
    write_csv(output / "A_complementarity.csv", complementary_table(test))
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(3).split(np.zeros(len(train["error"])), groups=train["image"]))
    for fitting, heldout in folds:
        if set(train["image"][fitting]) & set(train["image"][heldout]):
            raise ValueError("Pixel-level cross-validation leakage")
    chosen_temperature, temperature_cv, train_fusion = choose_temperature(train, folds,
                                        protocol["scores"]["q_temperature_grid"], args.device)
    write_csv(output / "temperature_cv.csv", temperature_cv)
    test_fusion = temperature_features(test, chosen_temperature, args.device)
    risk = {"raw_MC":-test["mc"], "raw_entropy":test["entropy"], "raw_margin":-test["margin"],
            "raw_residual_entropy_high":test["residual_entropy"], "raw_residual_entropy_low":-test["residual_entropy"],
            "raw_prototype_support":-test["prototype_support"], "raw_prototype_margin":-test["prototype_margin"],
            "raw_prototype_disagreement":test["prototype_disagreement"].astype(float),
            "raw_strict_conflict":test["strict_conflict"].astype(float),
            "raw_JS":test_fusion["probability_prototype_JS"], "raw_product":-test_fusion["product_support_original_label"],
            "raw_uniform":test["uniform_competitor"], "raw_top2":test["top2_competitor"], "raw_e":test["competitor_evidence"]}
    cv, calibration_rows = [], []
    models = output / "heads"
    models.mkdir(exist_ok=True)
    probabilities, dev_predictions = {"raw_MC":test["mc"]}, {"raw_MC":train["mc"]}
    phases = [("B_score_controls", ["h_MC", "h_entropy", "h_margin", "h_prototype_support", "h_prototype_margin", "h_JS", "h_product", "h_uniform", "h_top2", "h_e"]),
              ("C_competitor_identity_permutation", [f"h_permutation_{seed}" for seed in protocol["scores"]["permutation_seeds"]]),
              ("D_conditional_increment", ["F_base", "F_plus", "F_simple_base", "F_simple_plus"])]
    for phase, methods in phases:
        for method in methods:
            update(phase, current_head=method)
            if method.startswith("h_permutation_"):
                seed = int(method.rsplit("_", 1)[1])
                train_perm = permutation_features(train, seed, args.device)
                test_perm = permutation_features(test, seed, args.device)
                risk[f"raw_permutation_{seed}"] = test_perm
            else:
                train_perm = test_perm = None
            train_features = head_features(train, method, train_fusion, train_perm)
            test_features = head_features(test, method, test_fusion, test_perm)
            head, l2, values, dev_oof = fit_selected_head(train, train_features, folds, protocol["verifier"], args.device)
            prediction = predict_head(head, test_features, args.device)
            probabilities[method] = prediction
            dev_predictions[method] = dev_oof
            risk[method] = 1-prediction
            cv.extend(dict(method=method, **row) for row in values)
            torch.save(dict(state_dict=head.cpu().state_dict(), signature=signature, method=method,
                            chosen_l2=l2, prototype_temperature=chosen_temperature,
                            fitting_role="D_rel only", features_dimension=train_features.shape[1]), models / (method+".pt"))
            del head
            write_csv(output / "head_cv.csv", cv)
    np.savez_compressed(output / "predictions.npz", image_index=test["image"], GT=test["GT"], error=test["error"],
                        sampling_weight=test["sampling_weight"],
                        **{"probability__"+name:value for name, value in probabilities.items()})
    scopes = {"all":np.ones(len(test["error"]), bool), "conflict":test["strict_conflict"].astype(bool),
              "high_mc":test["mc"] >= .95,
              "high_mc_conflict":test["strict_conflict"].astype(bool) & (test["mc"] >= .95)}
    coverages = tuple(protocol["statistics"]["coverage_grid"])
    ranking_rows, precision_rows, curves, bootstrap_rows, support_rows, primary_rows, calibration_ci_rows = [], [], [], [], [], [], []
    bootstrap_methods = ["raw_MC", "raw_entropy", "raw_margin", "raw_prototype_margin", "raw_product", "raw_e",
                         "h_MC", "h_prototype_margin", "h_product", "h_uniform", "h_e", "F_base", "F_plus", "F_simple_base", "F_simple_plus"]
    bootstrap_methods += [f"h_permutation_{seed}" for seed in protocol["scores"]["permutation_seeds"]]
    for scope, selected in scopes.items():
        info = support(test, selected, protocol["statistics"]["minimum_subgroup_support"])
        support_rows.append(dict(scope=scope, **info))
        if not selected.any():
            continue
        e, w = test["error"][selected], test["sampling_weight"][selected]
        local_risk = {name:value[selected] for name, value in risk.items()}
        for name, value in local_risk.items():
            ranking = Ranking(e, value)
            ranking_rows.append(dict(scope=scope, method=name, **info, **ranking.evaluate(w, coverages)))
            coverage, error_rate, _ = ranking.curve(w)
            indices = np.unique(np.linspace(0, len(coverage)-1, min(1000, len(coverage))).astype(int))
            curves.extend(dict(scope=scope, method=name, coverage=float(coverage[j]), risk=float(error_rate[j])) for j in indices)
            for target in protocol["statistics"]["precision_targets"]:
                row = dict(scope=scope, method=name, target_precision=target,
                           test_curve_coverage=coverage_at_precision(e, value, w, target),
                           test_curve_scope="descriptive test-GT curve; not a deployable threshold")
                if name in dev_predictions:
                    dev_risk = -dev_predictions[name] if name == "raw_MC" else 1-dev_predictions[name]
                    cutoff = Ranking(train["error"], dev_risk).threshold_at_precision(train["sampling_weight"], target)
                    if cutoff:
                        summary = selection_summary(e, value <= cutoff["threshold"], w)
                        row.update(threshold=cutoff["threshold"], threshold_source="D_rel image-OOF predictions (MC uses raw confidence)",
                                   achieved_test_coverage=summary["coverage"], achieved_test_precision=summary["precision"])
                precision_rows.append(row)
        for name, prediction in probabilities.items():
            p = prediction[selected]
            stats = calibration(1-e, p, w)
            mass = np.sum(w*p)
            stats.update(effective_coverage=float(mass/w.sum()), weighted_error_rate=float(np.sum(w*p*e)/mass) if mass else None,
                         ESS_population_pixel_estimate=float(mass**2/np.sum(w*p*p)) if mass else None)
            calibration_rows.append(dict(scope=scope, method=name, **stats))
        if replicates and info["adequate_support"]:
            update("paired_image_bootstrap", scope=scope)
            calibration_ci_rows.extend(dict(scope=scope, **row) for row in calibration_intervals(
                test, probabilities, selected, replicates, protocol["statistics"]["bootstrap_seed"]))
            intervals, samples = paired_image_bootstrap_gpu(e, {name:local_risk[name] for name in bootstrap_methods}, w,
                test["image"][selected], replicates=replicates, seed=protocol["statistics"]["bootstrap_seed"],
                baseline="h_e", coverages=coverages, comparators=("F_base", "F_simple_base", "h_prototype_margin", "h_product", "h_uniform"),
                device=args.device, return_samples=True, all_images=len(test["image_ids"]))
            bootstrap_rows.extend(dict(scope=scope, **row) for row in intervals)
            np.savez_compressed(output / f"bootstrap_{scope}.npz", **samples)
            if scope == "conflict":
                j = list(KEYS).index("error_ap")
                permutations = np.stack([samples[f"h_permutation_{seed}"][:, j] for seed in protocol["scores"]["permutation_seeds"]])
                points = {row["method"]:row["error_ap"] for row in ranking_rows if row["scope"] == scope}
                contrasts = [
                    ("conditional_plus_e_minus_base", points["F_plus"]-points["F_base"], samples["F_plus"][:, j]-samples["F_base"][:, j]),
                    ("e_head_minus_prototype_margin", points["h_e"]-points["h_prototype_margin"], samples["h_e"][:, j]-samples["h_prototype_margin"][:, j]),
                    ("e_head_minus_product_fusion", points["h_e"]-points["h_product"], samples["h_e"][:, j]-samples["h_product"][:, j]),
                    ("e_head_minus_mean_permuted_AP", points["h_e"]-np.mean([points[f"h_permutation_{seed}"] for seed in protocol["scores"]["permutation_seeds"]]), samples["h_e"][:, j]-permutations.mean(0)),
                ]
                alpha = (1-protocol["statistics"]["primary_familywise_CI"])/2
                for name, point, differences in contrasts:
                    finite = differences[np.isfinite(differences)]
                    lo, hi = np.quantile(finite, [alpha, 1-alpha])
                    primary_rows.append(dict(contrast=name, scope=scope, stage=manifest["signature"]["stage"],
                        delta_error_ap=float(point), ci_low=float(lo), ci_high=float(hi), confidence=1-2*alpha,
                        valid_replicates=len(finite), family_size=4,
                        interpretation="late stage is primary; earlier stages are stage-consistency diagnostics"))
    write_csv(output / "ranking.csv", ranking_rows)
    write_csv(output / "risk_coverage.csv", curves)
    write_csv(output / "precision_targets.csv", precision_rows)
    write_csv(output / "calibration_and_mass.csv", calibration_rows)
    write_csv(output / "calibration_intervals.csv", calibration_ci_rows)
    write_csv(output / "conditional.csv", conditional_rows(test, risk, protocol["statistics"]["minimum_subgroup_support"]))
    write_csv(output / "support.csv", support_rows)
    write_csv(output / "paired_image_bootstrap.csv", bootstrap_rows)
    write_csv(output / "primary_contrasts.csv", primary_rows)
    atomic_json(output / "analysis_metadata.json", dict(signature=signature, temperature=chosen_temperature,
        data_roles=dict(fit="D_rel", evaluate="validation", prototype="D_sup"), images=dict(D_rel=len(train["image_ids"]), validation=len(test["image_ids"])),
        stages_not_independent=True, bootstrap_unit="full validation image pool, including images with zero subgroup observations",
        primary_familywise_CI=protocol["statistics"]["primary_familywise_CI"], secondary_CI=.95,
        limitation="CIs conditional on fixed teacher, prototypes and fitted verifiers; no training-seed population guarantee"))
    complete = dict(state="complete", signature=signature, completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    methods=len(risk), heads=len(probabilities)-1, primary_contrasts=primary_rows,
                    files={str(p.relative_to(output)):dict(bytes=p.stat().st_size, sha256=digest(p))
                           for p in sorted(output.iterdir()) if p.is_file() and p.name not in ("analysis_status.json", "COMPLETE.json")})
    atomic_json(output / "COMPLETE.json", complete)
    atomic_json(status_file, complete)
    print(json.dumps(dict(state="complete", cap=args.cap, heads=len(probabilities)-1)), flush=True)


if __name__ == "__main__":
    main()
