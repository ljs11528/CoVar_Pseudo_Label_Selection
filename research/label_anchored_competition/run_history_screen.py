"""Screen existing cross-stage caches; never fit a model or update a teacher."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import torch

from .anchor_trust import apply_model
from .analyze_frozen import coverage_at_precision
from .bounded_probe import image_folds
from .history_competition import (aggregate_sufficient, build_pair, current_features,
    read_aligned, recovery_intervals, scope_masks, screen_gate, sufficient_rows, support)
from .prepare_data import digest, source_revision
from .ranking_metrics import Ranking
from .run_score_audit import write_csv, write_json


def checked_file(folder, receipt, name):
    path = folder/name
    expected = receipt["files"][name]
    if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
        raise ValueError("Saved baseline artifact changed: "+name)
    return path


def baseline_replay(root, pool, current_stage, spec):
    source = root/spec["ranking"]["saved_audit"]
    all_receipt = json.loads((source/"COMPLETE.json").read_text())
    folder = source/("seed42_"+current_stage)
    if digest(folder/"COMPLETE.json") != all_receipt["stage_receipts"][current_stage]:
        raise ValueError("Prior baseline stage receipt changed")
    receipt = json.loads((folder/"COMPLETE.json").read_text())
    if receipt["state"] != "complete" or receipt["signature"]["source_commit"] != spec["ranking"]["source_commit"]:
        raise ValueError("Unexpected saved baseline source")
    role = pool["role"]
    previous_pool_file = checked_file(folder, receipt, role+"_pool.npz")
    previous_predictions_file = checked_file(folder, receipt, role+"_predictions.npz")
    with np.load(previous_pool_file, allow_pickle=False) as archive:
        old_ids = archive["image_ids"].tolist()
        old_image, old_pixel, old_gt = (archive[k].copy() for k in ("image", "pixel_indices", "GT"))
    if set(old_ids) != set(pool["image_ids"]) or len(old_ids) != len(set(old_ids)):
        raise ValueError("Saved baseline image set differs")
    old_lookup = {name:i for i,name in enumerate(old_ids)}
    order = []
    for i, name in enumerate(pool["image_ids"]):
        old = np.flatnonzero(old_image == old_lookup[name])
        new = np.flatnonzero(pool["image"] == i)
        if not np.array_equal(old_pixel[old], pool["pixel_indices"][new]) or not np.array_equal(old_gt[old], pool["GT"][new]):
            raise ValueError("Saved baseline pixel population differs")
        order.extend(old.tolist())
    order = np.asarray(order, dtype=np.int64)
    predictions = {name:np.full(len(pool["GT"]), np.nan) for name in ("MC", "MM")}
    def check_kind(model):
        if model["kind"] not in ("MC", "MM"):
            raise ValueError("Anchor model cannot stand in for MC/MM")
        if "parent" in model:
            check_kind(model["parent"])
    provenance = dict(role=role, current_stage=current_stage, previous_source=spec["ranking"]["source_commit"],
        source_receipt_sha256=digest(folder/"COMPLETE.json"), files={})
    if role == "D_rel":
        model_file = checked_file(folder, receipt, "outer_models.json")
        folds_file = checked_file(folder, receipt, "folds.json")
        models = json.loads(model_file.read_text()); old_folds = json.loads(folds_file.read_text())
        folds = image_folds(pool["image_ids"], pool["image"], spec["folds"]["outer"], spec["folds"]["namespace"])
        for fold, (train, held) in enumerate(folds):
            selected = {pool["image_ids"][i] for i in np.unique(pool["image"][held])}
            training = {pool["image_ids"][i] for i in np.unique(pool["image"][train])}
            if selected != set(old_folds[fold]["heldout_images"]) or training != set(old_folds[fold]["train_images"]):
                raise ValueError("Saved baseline fold membership differs")
            for name in predictions:
                model = models[f"fold{fold}__{name}"]; check_kind(model)
                predictions[name][held] = apply_model(model, pool, held)
        provenance["files"]["folds.json"] = digest(folds_file)
    else:
        model_file = checked_file(folder, receipt, "models.json")
        models = json.loads(model_file.read_text())
        for name in predictions:
            check_kind(models[name]); predictions[name] = apply_model(models[name], pool)
    for path in (model_file, previous_pool_file, previous_predictions_file):
        provenance["files"][path.name] = digest(path)
    with np.load(previous_predictions_file, allow_pickle=False) as archive:
        errors = {name:float(np.max(np.abs(values-archive[name][order]))) for name,values in predictions.items()}
    if not all(np.isfinite(v).all() for v in predictions.values()) or max(errors.values()) > 1e-10:
        raise ValueError("Frozen MC/MM replay differs beyond numeric tolerance")
    provenance["max_abs_difference"] = errors
    provenance["new_fits"] = 0
    return predictions, provenance


def rank_frozen(pool, risks, spec):
    rows, curves = [], []
    folds = image_folds(pool["image_ids"], pool["image"], spec["folds"]["outer"], spec["folds"]["namespace"]) if pool["role"] == "D_rel" else []
    minimum = spec["screen"]["minimum_support"]
    for scope, mask in scope_masks(pool).items():
        groups = [("pooled", np.flatnonzero(mask))]+[(str(i), held[mask[held]]) for i,(_,held) in enumerate(folds)]
        for fold, ix in groups:
            selected = np.zeros(len(mask), bool); selected[ix] = True
            info = support(pool, selected, minimum)
            for name, risk in risks.items():
                row = dict(pair=pool["pair"], role=pool["role"], scope=scope, fold=fold, method=name, **info)
                if len(ix):
                    ranking = Ranking(pool["error"][ix], risk[ix])
                    weight = pool["sampling_weight"][ix]
                    row.update(ranking.evaluate(weight, spec["ranking"]["coverages"]))
                    for target in spec["ranking"]["precision_targets"]:
                        row[f"coverage_at_precision_{target:g}"] = coverage_at_precision(pool["error"][ix], risk[ix], weight, target)
                    if scope in ("all", "high_mc"):
                        grid = np.linspace(.01, 1., 100)
                        points = ranking.evaluate(weight, grid)
                        curves.extend(dict(pair=pool["pair"], role=pool["role"], scope=scope, fold=fold,
                            method=name, coverage=float(c), risk=1-points[f"precision_at_{c:g}"]) for c in grid)
                rows.append(row)
        if folds:
            for name in risks:
                parts = [r for r in rows if r["scope"] == scope and r["method"] == name and r["fold"] in ("0", "1", "2")]
                keys = ["auroc", "error_ap", "error_prevalence", "aurc_continuous"]
                keys += [f"precision_at_{c:g}" for c in spec["ranking"]["coverages"]]
                keys += [f"coverage_at_precision_{c:g}" for c in spec["ranking"]["precision_targets"]]
                rows.append(dict(pair=pool["pair"], role=pool["role"], scope=scope, fold="mean", method=name,
                    folds=len(parts), adequate_support=all(r["adequate_support"] for r in parts),
                    **{k:float(np.mean([r.get(k, np.nan) for r in parts])) for k in keys}))
                if scope in ("all", "high_mc"):
                    for c in np.linspace(.01, 1., 100):
                        values = [r["risk"] for r in curves if r["scope"] == scope and r["method"] == name and r["fold"] in ("0", "1", "2") and r["coverage"] == float(c)]
                        curves.append(dict(pair=pool["pair"], role=pool["role"], scope=scope, fold="mean", method=name,
                            coverage=float(c), risk=float(np.mean(values)) if len(values) == 3 else np.nan))
    return rows, curves


def conditional_recovery(pool, alternatives, spec):
    groups = []; cfg = spec["strata"]
    mb = np.searchsorted(cfg["mc_edges"], pool["mc"], side="right")-1
    db = np.searchsorted(cfg["margin_edges"], pool["margin"], side="right")-1
    for a in range(len(cfg["mc_edges"])-1):
        for b in range(len(cfg["margin_edges"])-1):
            groups.append(("MC_margin", a*4+b, (mb == a) & (db == b), dict(mc_bin=a, margin_bin=b)))
    for c in range(cfg["classes"]):
        groups.append(("predicted_class", c, pool["pseudo_label"] == c, dict(predicted_class=c)))
        for boundary in cfg["boundary"]:
            groups.append(("class_boundary", c*2+boundary, (pool["pseudo_label"] == c) & (pool["boundary"] == boundary),
                           dict(predicted_class=c, boundary=boundary)))
    for boundary in cfg["boundary"]:
        groups.append(("boundary", boundary, pool["boundary"] == boundary, dict(boundary=boundary)))
    rows = []
    for stratum, group, mask, extra in groups:
        ix = np.flatnonzero(mask); w = pool["sampling_weight"][ix]; e = pool["error"][ix].astype(bool)
        denominator = float(w@e); mass = float(w.sum())
        info = support(pool, mask, spec["screen"]["minimum_support"])
        for method, candidate in alternatives.items():
            recovered = float(w@(e & (candidate[ix] == pool["GT"][ix])))
            rows.append(dict(pair=pool["pair"], role=pool["role"], stratum=stratum, group=group, method=method,
                **extra, **info, mass=mass, error_mass=denominator, recovered_mass=recovered,
                error_recovery=recovered/denominator if denominator else np.nan,
                set_truth_coverage=(mass-denominator+recovered)/mass if mass else np.nan))
    return rows


def wide_sufficient(rows):
    """Store shared denominators once while retaining every fixed method."""
    output = {}
    variable = ("recovered_mass", "gained_vs_now_mass", "lost_vs_now_mass", "alternative_changed_mass")
    for row in rows:
        key = (row["pair"], row["role"], row["scope"], row["image_id"])
        if key not in output:
            output[key] = {k:v for k,v in row.items() if k not in (*variable, "method")}
        output[key].update({row["method"]+"__"+k:row[k] for k in variable})
    return list(output.values())


def save_role(root, output, raw, spec, update):
    results = {}
    for pair in spec["pairs"]:
        folder = output/pair["name"]; folder.mkdir(exist_ok=True)
        role = raw["role"]
        update("derive_history", role=role, pair=pair["name"])
        pool, alternatives, risks, invariant = build_pair(raw, pair, spec)
        update("replay_baselines", role=role, pair=pair["name"])
        predictions, replay = baseline_replay(root, pool, pair["current"], spec)
        risks.update({name:-score for name,score in predictions.items()})
        write_json(folder/(role+"_baseline_replay.json"), replay)
        write_json(folder/(role+"_permutation.json"), dict(seeds=spec["permutation"]["seeds"], maximum_invariant_difference=invariant,
            argmax_agreement_preserved=True, identity_keyed=True, no_new_fitting=True))
        update("candidate_statistics", role=role, pair=pair["name"])
        rows = sufficient_rows(pool, alternatives, spec)
        write_csv(folder/(role+"_image_sufficient.csv"), wide_sufficient(rows))
        candidate_table = aggregate_sufficient(rows, spec)
        write_csv(folder/(role+"_candidate_recovery.csv"), candidate_table)
        if role == "D_rel":
            fold_table = []
            folds = image_folds(pool["image_ids"], pool["image"], spec["folds"]["outer"], spec["folds"]["namespace"])
            for fold, (_, held) in enumerate(folds):
                held_images = set(pool["image"][held].tolist())
                parts = [r for r in rows if r["image_index"] in held_images]
                fold_table.extend(dict(row, fold=fold) for row in aggregate_sufficient(parts, spec))
            write_csv(folder/(role+"_candidate_recovery_folds.csv"), fold_table)
            uncertainty = recovery_intervals(rows, spec)
            write_csv(folder/(role+"_recovery_intervals.csv"), uncertainty)
        else:
            uncertainty = []
        del rows
        write_csv(folder/(role+"_conditional_recovery.csv"), conditional_recovery(pool, alternatives, spec))
        update("frozen_rankings", role=role, pair=pair["name"])
        ranking, curves = rank_frozen(pool, risks, spec)
        write_csv(folder/(role+"_ranking.csv"), ranking)
        write_csv(folder/(role+"_risk_coverage.csv"), curves)
        results[pair["name"]] = dict(candidate_recovery=candidate_table, intervals=uncertainty)
    return results


def run(root, output, spec, synthetic=False):
    if output.exists():
        raise FileExistsError("Preserve previous audit runs")
    output.mkdir(parents=True)
    package = Path(__file__).resolve().parent; started = time.monotonic()
    signature = dict(source_commit=source_revision(package), protocol_sha256=digest(package/"history_protocol.json"),
                     synthetic=bool(synthetic), no_new_model_fitting=True, no_teacher_updates=True)
    write_json(output/"effective_protocol.json", spec)
    def update(phase, **extra):
        elapsed = time.monotonic()-started
        if elapsed > spec["execution"]["max_screen_hours"]*3600:
            raise TimeoutError("Registered screen time limit exceeded")
        record = dict(state="running", phase=phase, **extra, elapsed_seconds=elapsed, signature=signature,
                      updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(output/"status.json", record)
        print(json.dumps({k:v for k,v in record.items() if k != "signature"}), flush=True)
    try:
        torch.set_num_threads(4)
        update("read_D_rel")
        raw, checks = read_aligned(root, "D_rel", spec, update, synthetic=synthetic)
        write_json(output/"D_rel_alignment.json", checks)
        results = save_role(root, output, raw, spec, update)
        gate = screen_gate([r for x in results.values() for r in x["candidate_recovery"]],
                           [r for x in results.values() for r in x["intervals"]], spec)
        gate.update(frozen_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), signature=signature)
        write_json(output/"screen_gate.json", gate)
        del results
        if gate["passed"]:
            # Hand back to the sequential executor before any validation read.
            # The predeclared probe is a separate stage, never an automatic teacher run.
            np.savez_compressed(output/"eligible_D_rel_pool.npz", **{k:np.asarray(v) for k,v in raw.items()})
            state = dict(state="screen_complete_probes_eligible", signature=signature,
                elapsed_seconds=time.monotonic()-started, validation_read=False, new_fits=0,
                decision=gate["decision"], completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat())
            write_json(output/"SCREEN_COMPLETE.json", state); write_json(output/"status.json", state)
            return state
        drel_ids = set(raw["image_ids"]); del raw
        write_json(output/"PROBES_SKIPPED.json", dict(reason=gate["decision"], new_fits=0,
            screen_gate_sha256=digest(output/"screen_gate.json"), teacher_updates=0))
        update("read_validation_after_failed_screen")
        raw, checks = read_aligned(root, "validation", spec, update, synthetic=synthetic)
        if set(raw["image_ids"]) & drel_ids:
            raise ValueError("D_rel and validation overlap")
        write_json(output/"validation_alignment.json", checks)
        save_role(root, output, raw, spec, update)
        del raw
        complete = dict(state="complete", signature=signature, elapsed_seconds=time.monotonic()-started,
            completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), gate=gate["decision"], new_fits=0, teacher_updates=0,
            numpy=np.__version__, torch=torch.__version__,
            files={str(f.relative_to(output)):dict(bytes=f.stat().st_size, sha256=digest(f))
                for f in sorted(output.rglob("*")) if f.is_file() and f.name != "status.json"})
        write_json(output/"COMPLETE.json", complete); write_json(output/"status.json", complete)
        print(json.dumps({k:v for k,v in complete.items() if k != "files"}), flush=True)
        return complete
    except Exception as exc:
        write_json(output/"FAILED.json", dict(state="failed", signature=signature, error=repr(exc),
                   elapsed_seconds=time.monotonic()-started, scientific_conclusion="not evaluated from engineering failure"))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name("history_protocol.json").read_text())
    run(args.root, args.output, spec)


if __name__ == "__main__":
    main()
