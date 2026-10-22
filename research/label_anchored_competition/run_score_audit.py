"""Execute the fixed nested-image-CV development audit without teacher updates."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import torch

from .bounded_probe import (apply_probe, choose_candidate, fit_basis, fit_correction_path,
                            fit_mc, image_folds, logit_metrics, mc_logits, sigmoid, transform_basis)
from .gpu_bootstrap import KEYS, paired_image_bootstrap_gpu
from .prepare_data import digest, source_revision
from .ranking_metrics import Ranking, selection_summary
from .analyze_frozen import coverage_at_precision
from .score_audit_diagnostics import (distributions, joint_tables, read_audit_pool,
                                      scopes, signals, support_indices)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(value, indent=2)+"\n")
    tmp.replace(path)


def key(seed, budget, method):
    return f"s{seed}_u{budget}_{method}"


def nested_fit(pool, x, spec, output, device, update):
    if pool["role"] != "D_rel":
        raise ValueError("Only D_rel may enter nested fitting")
    n = len(pool["error"])
    y, w = 1-pool["error"], pool["sampling_weight"]
    cfg, namespace = spec["correction"], spec["folds"]["namespace"]
    outer = image_folds(pool["image_ids"], pool["image"], spec["folds"]["outer"], namespace)
    mc_cache, fold_rows, selections, candidates, curve_rows, models = {}, [], [], [], [], {}
    outer_models = {}
    baseline_oof = np.full(n, np.nan)
    oof = {key(s, b, m): np.full(n, np.nan) for s in cfg["fit_seeds"] for b in cfg["budgets"] for m in x}
    def calibrate(indices):
        identity = tuple(np.unique(pool["image"][indices]))
        if identity not in mc_cache:
            model = fit_mc(pool["logit_mc"][indices], y[indices], w[indices], spec["mc_calibrator"])
            mc_cache[identity] = (model, mc_logits(model, pool["logit_mc"]))
        return mc_cache[identity]
    def path(train, held, method, form, seed, context):
        mc_model, offset = calibrate(train)
        basis = fit_basis(x[method][train], w[train], form, cfg["knot_quantiles"])
        train_x, held_x = transform_basis(basis, x[method][train]), transform_basis(basis, x[method][held])
        def curve(row):
            curve_rows.append(dict(seed=seed, method=method, form=form, **context,
                                   heldout_weight=float(w[held].sum()), **row))
        values = fit_correction_path(train_x, offset[train], y[train], w[train], held_x, offset[held],
                                    y[held], w[held], cfg, seed, device, curve)
        fitted = {b: dict(mc=mc_model, form=form, basis=basis, coefficients=v["coefficients"].tolist()) for b, v in values.items()}
        return values, fitted
    for fold, (train, held) in enumerate(outer):
        images = np.unique(pool["image"][train])
        inner_local = image_folds([pool["image_ids"][i] for i in images],
            np.searchsorted(images, pool["image"][train]), spec["folds"]["inner"], namespace+"|inner")
        inner = [(train[a], train[b]) for a, b in inner_local]
        def ids(index):
            return [pool["image_ids"][i] for i in np.unique(pool["image"][index])]
        fold_rows.append(dict(outer=fold, train_images=ids(train), heldout_images=ids(held),
                              inner=[dict(train_images=ids(a), heldout_images=ids(b)) for a, b in inner]))
        for a, b in inner:
            if set(ids(a)) & set(ids(b)) or not set(ids(a)+ids(b)) <= set(ids(train)):
                raise ValueError("Nested image-role leakage")
        _, offset = calibrate(train)
        baseline_oof[held] = offset[held]
    # This manifest is saved before any correction coefficients are fitted.
    write_json(output / "folds.json", fold_rows)
    for seed in cfg["fit_seeds"]:
        full_candidates = {(m, f, b): np.full(n, np.nan) for m in x for f in ("linear", "additive") for b in cfg["budgets"]}
        for fold, (train, held) in enumerate(outer):
            images = np.unique(pool["image"][train])
            inner = image_folds([pool["image_ids"][i] for i in images], np.searchsorted(images, pool["image"][train]),
                                spec["folds"]["inner"], namespace+"|inner")
            inner_pred = {(m, f, b): np.full(len(train), np.nan) for m in x for f in ("linear", "additive") for b in cfg["budgets"]}
            inner_zero = np.full(len(train), np.nan)
            for j, (local_a, local_b) in enumerate(inner):
                a, b = train[local_a], train[local_b]
                _, offset = calibrate(a)
                inner_zero[local_b] = offset[b]
                update("nested_inner", seed=seed, outer=fold, inner=j)
                for method in x:
                    for form in ("linear", "additive"):
                        values, _ = path(a, b, method, form, seed, dict(level="inner", outer=fold, inner=j))
                        for budget, value in values.items():
                            inner_pred[method, form, budget][local_b] = value["heldout_logits"]
            zero_nll = logit_metrics(y[train], inner_zero, w[train])["nll"]
            chosen = {}
            for method in x:
                for budget in cfg["budgets"]:
                    loss = dict(zero=zero_nll)
                    for form in ("linear", "additive"):
                        pred = inner_pred[method, form, budget]
                        if not np.isfinite(pred).all():
                            raise ValueError("Incomplete inner OOF predictions")
                        loss[form] = logit_metrics(y[train], pred, w[train])["nll"]
                    chosen[method, budget] = choose_candidate(loss, cfg["selection_tolerance"])
                    selections.append(dict(seed=seed, budget=budget, method=method, outer=fold,
                        level="nested_inner_selection", chosen=chosen[method, budget], **{f"nll_{f}":v for f,v in loss.items()}))
            mc_model, outer_offset = calibrate(train)
            update("outer_candidates", seed=seed, outer=fold)
            for method in x:
                values_by_form = {}
                fitted_by_form = {}
                for form in ("linear", "additive"):
                    values, fitted = path(train, held, method, form, seed, dict(level="outer_candidate", outer=fold, inner=-1))
                    values_by_form[form] = values
                    fitted_by_form[form] = fitted
                    for budget, value in values.items():
                        full_candidates[method, form, budget][held] = value["heldout_logits"]
                for budget in cfg["budgets"]:
                    form = chosen[method, budget]
                    pred = outer_offset[held] if form == "zero" else values_by_form[form][budget]["heldout_logits"]
                    oof[key(seed, budget, method)][held] = pred
                    outer_models[f"fold{fold}_"+key(seed, budget, method)] = (dict(mc=mc_model, form="zero")
                        if form == "zero" else fitted_by_form[form][budget])
            write_csv(output / "selection.csv", selections)
            write_csv(output / "training_curves.csv", curve_rows)
        update("final_D_rel_selection_and_refit", seed=seed)
        full = np.arange(n)
        mc_model, full_offset = calibrate(full)
        for method in x:
            chosen = {}
            for budget in cfg["budgets"]:
                loss = dict(zero=logit_metrics(y, baseline_oof, w)["nll"])
                for form in ("linear", "additive"):
                    pred = full_candidates[method, form, budget]
                    if not np.isfinite(pred).all():
                        raise ValueError("Incomplete candidate OOF predictions")
                    metrics = logit_metrics(y, pred, w)
                    loss[form] = metrics["nll"]
                    candidates.append(dict(seed=seed, budget=budget, method=method, form=form, **metrics,
                        error_ap=Ranking(pool["error"], -pred).evaluate(w, ())["error_ap"]))
                chosen[budget] = choose_candidate(loss, cfg["selection_tolerance"])
                selections.append(dict(seed=seed, budget=budget, method=method, outer=-1,
                    level="final_refit_selection", chosen=chosen[budget], **{f"nll_{f}":v for f,v in loss.items()}))
            by_form = {}
            for form in sorted(set(chosen.values())-{"zero"}):
                _, fitted = path(full, full, method, form, seed, dict(level="full_refit_in_sample", outer=-1, inner=-1))
                by_form[form] = fitted
            for budget, form in chosen.items():
                name = key(seed, budget, method)
                models[name] = dict(mc=mc_model, form="zero") if form == "zero" else by_form[form][budget]
                models[name].update(feature_set=method, budget=budget, seed=seed, fitting_role="D_rel")
    for values in oof.values():
        if not np.isfinite(values).all():
            raise ValueError("Incomplete outer OOF predictions")
    oof["MC"] = baseline_oof
    models["MC"] = dict(mc=calibrate(np.arange(n))[0], form="zero", fitting_role="D_rel")
    write_json(output / "models.json", models)
    write_json(output / "outer_models.json", outer_models)
    write_csv(output / "selection.csv", selections)
    write_csv(output / "candidate_oof.csv", candidates)
    write_csv(output / "training_curves.csv", curve_rows)
    # These are candidate OOF curves, not another search over update counts.
    grouped = {}
    for row in curve_rows:
        if row["level"] == "outer_candidate":
            identity = (row["seed"], row["method"], row["form"], row["update"])
            grouped.setdefault(identity, []).append(row)
    write_csv(output / "oof_curves.csv", [dict(seed=k[0], method=k[1], form=k[2], update=k[3],
        oof_nll=float(np.average([r["heldout_nll"] for r in rows], weights=[r["heldout_weight"] for r in rows])),
        folds=len(rows)) for k,rows in grouped.items()])
    return models, oof


def evaluate(pool, predictions, spec):
    ranking, workpoints = [], []
    risks = {name:-values for name, values in predictions.items()}
    risks.update(raw_MC=-pool["mc"], raw_logit_MC=-pool["logit_mc"], raw_entropy=pool["entropy"], raw_margin=-pool["margin"])
    for scope, selected in scopes(pool).items():
        index = np.flatnonzero(selected)
        info = support_indices(pool, index, spec["strata"]["minimum_support"])
        for method, risk in risks.items():
            e, r, w = pool["error"][index], risk[index], pool["sampling_weight"][index]
            result = Ranking(e, r).evaluate(w, spec["statistics"]["coverage_grid"])
            if method in predictions:
                result.update(logit_metrics(1-e, predictions[method][index], w))
            ranking.append(dict(role=pool["role"], scope=scope, method=method, **info, **result))
            for target in spec["statistics"]["precision_targets"]:
                workpoints.append(dict(role=pool["role"], scope=scope, method=method, target_precision=target,
                    descriptive_curve_coverage=coverage_at_precision(e, r, w, target),
                    scope_note="GT-descriptive curve; not a deployment threshold"))
    return ranking, workpoints, risks


def development_intervals(pool, predictions, spec, device, update):
    statistics = spec["statistics"]
    prefix = f"s{statistics['primary_seed']}_u{statistics['primary_budget']}_"
    names = {m: prefix+m for m in spec["feature_sets"]}
    risks = {m:-predictions[n] for m,n in names.items()}
    risks["MC"] = -predictions["MC"]
    rows = []
    for scope in ("all", "conflict"):
        selected = scopes(pool)[scope]
        info = support_indices(pool, np.flatnonzero(selected), spec["strata"]["minimum_support"])
        if not info["adequate_support"]:
            rows.append(dict(scope=scope, adequate_support=False))
            continue
        update("nested_D_rel_image_bootstrap", scope=scope)
        _, samples = paired_image_bootstrap_gpu(pool["error"][selected], {m:r[selected] for m,r in risks.items()},
            pool["sampling_weight"][selected], pool["image"][selected], replicates=statistics["bootstrap_replicates"],
            seed=statistics["bootstrap_seed"], baseline="MC", coverages=(), device=device, return_samples=True,
            all_images=len(pool["image_ids"]))
        ap = KEYS.index("error_ap")
        pairs = statistics["primary_contrasts"]+[["e", "MC"]]
        for a,b in pairs:
            delta = samples[a][:, ap]-samples[b][:, ap]
            finite = delta[np.isfinite(delta)]
            confidence = statistics["primary_confidence"] if scope == "conflict" and [a,b] in statistics["primary_contrasts"] else .95
            lo,hi = np.quantile(finite, [(1-confidence)/2, 1-(1-confidence)/2])
            point_a = Ranking(pool["error"][selected], risks[a][selected]).evaluate(pool["sampling_weight"][selected], ())["error_ap"]
            point_b = Ranking(pool["error"][selected], risks[b][selected]).evaluate(pool["sampling_weight"][selected], ())["error_ap"]
            rows.append(dict(scope=scope, method=a, comparator=b, delta_error_ap=point_a-point_b,
                confidence=confidence, ci_low=float(lo), ci_high=float(hi), valid_replicates=len(finite),
                adequate_support=True, unit="paired full D_rel images; conditional on fitted nested OOF predictions"))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve completed and failed audit directories")
    args.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    spec = json.loads((package / "score_audit_protocol.json").read_text())
    manifest = json.loads((args.export / "manifest.json").read_text())
    if manifest.get("synthetic"):
        raise ValueError("Formal audit rejects synthetic exports")
    stage = manifest["signature"]["stage"]
    if stage not in spec["stages"]:
        raise ValueError("Unregistered stage")
    started = time.monotonic()
    signature = dict(stage=stage, source_commit=source_revision(package), protocol_sha256=digest(package / "score_audit_protocol.json"),
                     export_manifest_sha256=digest(args.export / "manifest.json"), old_bn_checkpoint=True)
    def update(phase, **extra):
        if time.monotonic()-started > spec["execution"]["maximum_stage_hours"]*3600:
            raise TimeoutError("Registered stage time budget exceeded")
        write_json(args.output / "status.json", dict(state="running", phase=phase, **extra, signature=signature,
                   elapsed_seconds=time.monotonic()-started, updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    torch.set_num_threads(4)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        update("read_D_rel_cache")
        train, train_check = read_audit_pool(args.export, "D_rel", spec, lambda **kw:update("read_D_rel_cache", **kw))
        train_x = signals(train, spec["feature_sets"])
        models, oof = nested_fit(train, train_x, spec, args.output, args.device, update)
        # All feature/form/budget selection is now complete, before reading validation.
        write_json(args.output / "FITTING_COMPLETE.json", dict(signature=signature, role="D_rel only", models_sha256=digest(args.output/"models.json")))
        update("nested_D_rel_metrics")
        ranking, precision, train_risks = evaluate(train, oof, spec)
        intervals = development_intervals(train, oof, spec, args.device, update)
        write_csv(args.output / "D_rel_intervals.csv", intervals)
        update("read_validation_after_fitting")
        test, test_check = read_audit_pool(args.export, "validation", spec, lambda **kw:update("read_validation_cache", **kw))
        if set(train["image_ids"]) & set(test["image_ids"]):
            raise ValueError("Cross-role image overlap")
        test_x = signals(test, spec["feature_sets"])
        predictions = {name: apply_probe(model, test_x[model["feature_set"]] if name != "MC" else np.zeros((len(test["error"]),0)),
                                               test["logit_mc"]) for name,model in models.items()}
        update("descriptive_validation_metrics")
        rows, points, test_risks = evaluate(test, predictions, spec)
        ranking.extend(rows); precision.extend(points)
        deployed = []
        for name in predictions:
            for target in spec["statistics"]["precision_targets"]:
                cutoff = Ranking(train["error"], -oof[name]).threshold_at_precision(train["sampling_weight"], target)
                if cutoff:
                    deployed.append(dict(method=name, target_precision=target, cutoff=cutoff["threshold"],
                        **selection_summary(test["error"], -predictions[name] <= cutoff["threshold"], test["sampling_weight"]),
                        threshold_source="nested D_rel OOF; final refit may shift score scale"))
        write_csv(args.output / "ranking.csv", ranking)
        write_csv(args.output / "precision_targets.csv", precision)
        write_csv(args.output / "D_rel_threshold_validation.csv", deployed)
        distribution_rows = distributions(train, spec)+distributions(test, spec)
        write_csv(args.output / "distributions.csv", distribution_rows)
        update("joint_strata_and_feature_shift")
        prefix = f"s{spec['statistics']['primary_seed']}_u{spec['statistics']['primary_budget']}_"
        joint, support = [], []
        for pool, risks in ((train,train_risks),(test,test_risks)):
            selected = {name:r for name,r in risks.items() if name == "MC" or name.startswith(prefix)}
            jr,sr = joint_tables(pool, selected, spec)
            joint.extend(jr);support.extend(sr)
        write_csv(args.output / "joint_conditional.csv", joint)
        write_csv(args.output / "joint_support.csv", support)
        write_json(args.output / "real_cache_checks.json", dict(D_rel=train_check, validation=test_check))
        for role, pool, values in (("D_rel",train,oof),("validation",test,predictions)):
            np.savez_compressed(args.output / (role+"_predictions.npz"), image_index=pool["image"], image_ids=np.asarray(pool["image_ids"]),
                pixel_indices=pool["pixel_indices"], error=pool["error"], sampling_weight=pool["sampling_weight"],
                **{"correctness_logit__"+name:value for name,value in values.items()})
        complete = dict(state="complete", signature=signature, stage=stage, elapsed_seconds=time.monotonic()-started,
            completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), torch=torch.__version__, numpy=np.__version__,
            scope="post-hoc existing-cache development; no teacher training; validation never selects",
            files={f.name:dict(bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(args.output.iterdir()) if f.name not in ("status.json","COMPLETE.json")})
        write_json(args.output / "COMPLETE.json", complete)
        write_json(args.output / "status.json", complete)
        print(json.dumps(dict(state="complete",stage=stage,elapsed_seconds=complete["elapsed_seconds"])),flush=True)
    except Exception as exc:
        write_json(args.output / "FAILED.json", dict(state="failed", signature=signature, error=repr(exc), elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == "__main__":
    main()
