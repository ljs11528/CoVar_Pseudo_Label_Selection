"""Fixed low-dimensional probes and identity controls for the anchor-trust audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .bounded_probe import fit_mc, mc_logits, weighted_quantile
from .core import pixel_identity_keys, residual_permutation
from .prepare_data import digest
from .score_audit_diagnostics import read_audit_pool


BASE_CANDIDATES = ("MC", "MM_linear", "MM_additive")
FEATURE_COUNTS = {"G": 5, "S": 8, "T": 12}


def variants(spec):
    return ["real"] + [f"sham{s}" for s in spec["permutation"]["seeds"]]


def candidate_sets(spec):
    result = {"MM": list(BASE_CANDIDATES)}
    for variant in variants(spec):
        for family in FEATURE_COUNTS:
            names = list(BASE_CANDIDATES)
            names += [f"{variant}_{f}" for f in ("G", "S", "T")[:list(FEATURE_COUNTS).index(family)+1]]
            result[f"{variant}_{family}"] = names
    return result


def select(losses, names, tolerance):
    chosen = names[0]
    for name in names[1:]:
        if losses[name] < losses[chosen] - tolerance:
            chosen = name
    return chosen


def geometry(logits, similarities, counts):
    """All inputs are observables; counts move with their similarity in shams."""
    z, s = np.asarray(logits, np.float64), np.asarray(similarities, np.float64)
    n, classes = z.shape
    if counts.ndim == 1:
        counts = np.broadcast_to(counts, z.shape)
    index = np.arange(n)
    a = z.argmax(1)
    rival = z.copy()
    rival[index, a] = -np.inf
    b = rival.argmax(1)
    c = s.argmax(1)
    z_other_max = rival.max(1)
    r = np.exp(rival-z_other_max[:, None])
    r /= r.sum(1)[:, None]
    logit_mc = z[index, a] - (z_other_max + np.log(np.exp(rival-z_other_max[:, None]).sum(1)))
    p = np.exp(z-z.max(1)[:, None]); p /= p.sum(1)[:, None]
    m, p2 = p[index, a], p[index, b]
    delta = s-s[index, a, None]
    d = np.maximum(0, delta)
    uniform = d.sum(1)/(classes-1)
    e = (r*d).sum(1)
    db = d[index, b]
    product = np.log((r*np.exp(delta)).sum(1))
    other_s = s.copy(); other_s[index, a] = -np.inf
    proto_risk = other_s.max(1)-s[index, a]
    ha, hb, hc = (np.log1p(counts[index, k])/np.log1p(1191) for k in (a, b, c))
    features = np.column_stack([proto_risk, product, uniform, e, db, ha, hb, hc,
                               e*ha, e*hc, db*ha, db*hb])
    return dict(features=features, mc=m, p2=p2, margin=m-p2, logit_mc=logit_mc,
                entropy=-(p*np.log(np.maximum(p, 1e-300))).sum(1), a=a, b=b, c=c,
                strict_conflict=proto_risk > 0, count_a=counts[index,a],
                count_b=counts[index,b], count_c=counts[index,c])


def shuffle_tuples(logits, similarities, counts, image_id, pixels, seed):
    n, classes = logits.shape
    a = logits.argmax(1)
    residual_slots = np.arange(classes)[None,:] != a[:,None]
    source = np.broadcast_to(np.arange(classes), (n,classes))[residual_slots].reshape(n,classes-1)
    keys = pixel_identity_keys(image_id, pixels)
    perm = residual_permutation(keys, classes-1, seed).numpy()
    source = np.take_along_axis(source, perm, 1)
    full_counts = np.broadcast_to(counts, (n,classes)).copy()
    shuffled_s = np.asarray(similarities).copy()
    shuffled_s[residual_slots] = np.take_along_axis(similarities, source, 1).ravel()
    full_counts[residual_slots] = counts[source].ravel()
    return shuffled_s, full_counts


def read_pool(export: Path, role, spec, progress=None):
    pool, checks = read_audit_pool(export, role, spec, progress)
    manifest = json.loads((export/"manifest.json").read_text())
    with np.load(export/"prototypes.npz", allow_pickle=False) as bank:
        counts = bank["image_counts"].astype(np.int64)
    all_rows = []
    checks["permutation_invariant_max_abs_diff"] = 0.
    checks["fp32_score_max_abs_diff"] = 0.
    checks["probability_identity_max_abs_diff"] = 0.
    for entry in manifest["files"]:
        if entry["role"] != role:
            continue
        path = export/entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Cache changed between audited reads")
        with np.load(path, allow_pickle=False) as cache:
            n = min(spec["cap"], len(cache["GT"]))
            z, s = cache["logits"][:n], cache["prototype_similarities"][:n]
            real = geometry(z, s, counts)
            row = {k: real[k] for k in ("mc", "p2", "margin", "logit_mc", "entropy", "b", "count_a", "count_b", "count_c")}
            row["anchor__real"] = real["features"]
            for k in ("mc", "margin", "entropy", "logit_mc"):
                diff = float(np.max(np.abs(real[k]-cache["score__"+k][:n])))
                checks["fp32_score_max_abs_diff"] = max(checks["fp32_score_max_abs_diff"], diff)
                if diff > 2e-5:
                    raise ValueError("FP64 observables differ from original cache")
            np.testing.assert_array_equal(real["strict_conflict"], cache["score__strict_conflict"][:n])
            np.testing.assert_array_equal(real["c"], cache["score__prototype_label"][:n])
            checks["probability_identity_max_abs_diff"] = max(checks["probability_identity_max_abs_diff"],
                float(np.max(np.abs(real["p2"]-(real["mc"]-real["margin"])))))
            for seed in spec["permutation"]["seeds"]:
                ss, nn = shuffle_tuples(z, s, counts, entry["image_id"], cache["pixel_indices"][:n], seed)
                sham = geometry(z, ss, nn)
                np.testing.assert_array_equal(np.sort(s,1), np.sort(ss,1))
                np.testing.assert_array_equal(real["strict_conflict"], sham["strict_conflict"])
                for k in ("mc", "margin", "entropy", "logit_mc"):
                    np.testing.assert_array_equal(real[k], sham[k])
                # Prototype ties use the argmax index. h_c is defined below by the
                # real winning tuple, preserving it even under exact residual ties.
                sham["features"][:,7] = real["features"][:,7]
                sham["features"][:,9] = sham["features"][:,3]*sham["features"][:,7]
                invariant = float(np.max(np.abs(real["features"][:,[0,2,5,7]]-sham["features"][:,[0,2,5,7]])))
                checks["permutation_invariant_max_abs_diff"] = max(checks["permutation_invariant_max_abs_diff"], invariant)
                if invariant > 1e-12:
                    raise ValueError("Permutation violated its invariant controls")
                row[f"anchor__sham{seed}"] = sham["features"]
            all_rows.append(row)
    for name in all_rows[0]:
        if name in pool:
            pool["cached_"+name] = pool[name]
        pool[name] = np.concatenate([r[name] for r in all_rows])
    pool["prototype_counts"] = counts
    return pool, checks


def fit_basis64(x, weight, form="linear", knots=(), interaction=False):
    x, weight = np.asarray(x, np.float64), np.asarray(weight, np.float64)
    mean = np.average(x, weights=weight, axis=0)
    scale = np.sqrt(np.average((x-mean)**2, weights=weight, axis=0))
    scale = np.where(scale > 1e-10, scale, 1.)
    z = (x-mean)/scale
    columns, existing = [], []
    candidates = []
    for j in range(x.shape[1]):
        candidates.append((dict(kind="linear", column=j), z[:,j]))
        if form == "additive":
            for knot in np.unique(weighted_quantile(z[:,j], weight, knots)):
                candidates.append((dict(kind="hinge", column=j, knot=float(knot)), np.maximum(0,z[:,j]-knot)))
    if interaction:
        candidates.append((dict(kind="interaction", columns=[0,1]), z[:,0]*z[:,1]))
    for item, raw in candidates:
        center = float(np.average(raw, weights=weight))
        std = float(np.sqrt(np.average((raw-center)**2, weights=weight)))
        if std <= 1e-10:
            continue
        value = (raw-center)/std
        if any(np.allclose(value, sign*old, atol=1e-10, rtol=1e-10) for old in existing for sign in (1,-1)):
            continue
        item.update(center=center, scale=std)
        columns.append(item); existing.append(value)
    return dict(mean=mean.tolist(), scale=scale.tolist(), columns=columns, intercept=True)


def design(basis, x):
    z = (np.asarray(x,np.float64)-basis["mean"])/basis["scale"]
    columns = [np.ones(len(x))]
    for item in basis["columns"]:
        if item["kind"] == "interaction":
            value = z[:,0]*z[:,1]
        else:
            value = z[:,item["column"]]
            if item["kind"] == "hinge":
                value = np.maximum(0,value-item["knot"])
        columns.append((value-item["center"])/item["scale"])
    return np.column_stack(columns)


def newton(x, offset, correct, weight, config, device, callback=None):
    arrays = [np.asarray(v,np.float64) for v in (x, offset, correct, weight)]
    if not all(np.isfinite(a).all() for a in arrays) or not (arrays[-1] > 0).all():
        raise ValueError("Nonfinite/invalid fitting inputs")
    xx, oo, yy, ww = [torch.as_tensor(a, dtype=torch.float64, device=device) for a in arrays]
    ww = ww/ww.sum()
    beta = torch.zeros(xx.shape[1],dtype=torch.float64,device=device)
    ridge = config["l2"]
    identity = torch.eye(len(beta),dtype=beta.dtype,device=device)
    def objective(b):
        logits = oo+xx@b
        return (ww*(torch.nn.functional.softplus(logits)-yy*logits)).sum()+ridge/2*(b@b)
    for step in range(config["max_iterations"]+1):
        pred = oo+xx@beta
        p = pred.sigmoid()
        gradient = xx.T@(ww*(p-yy))+ridge*beta
        maxgrad = float(gradient.abs().max())
        if callback:
            callback(step, beta.cpu().numpy().copy(), float(objective(beta)), maxgrad)
        if maxgrad <= config["gradient_tolerance"]:
            return beta.cpu().numpy(), dict(iterations=step, max_gradient=maxgrad, converged=True,
                                           objective=float(objective(beta)))
        if step == config["max_iterations"]:
            break
        hessian = xx.T@(xx*(ww*p*(1-p))[:,None])+ridge*identity
        direction = torch.linalg.solve(hessian, gradient)
        previous = objective(beta)
        for power in range(config["line_search_steps"]):
            trial = beta-(.5**power)*direction
            if bool(objective(trial) <= previous+1e-4*(gradient@(trial-beta))):
                beta = trial
                break
        else:
            raise ValueError("FP64 Newton line search failed")
    raise ValueError("Fixed FP64 optimizer did not converge; do not interpret as a negative experiment")


def apply_model(model, pool, indices=None, variant=None):
    ix = np.arange(len(pool["error"])) if indices is None else indices
    if model["kind"] == "MC":
        return mc_logits(model["mc"],pool["logit_mc"][ix])
    baseline = apply_model(model["parent"],pool,ix,variant)
    if model["kind"] == "MM":
        x = np.column_stack([pool["logit_mc"][ix],pool["margin"][ix]])
        return baseline+design(model["basis"],x)@np.asarray(model["coefficients"])
    if model["kind"] != "anchor":
        raise ValueError("Unknown model kind")
    mask = pool["strict_conflict"][ix].astype(bool)
    x = pool["anchor__"+(variant or model["variant"])][ix[mask],:model["inputs"]]
    baseline[mask] += design(model["basis"],x)@np.asarray(model["coefficients"])
    return baseline


def fit_candidates(pool, train, held, spec, device, context, curve):
    if pool["role"] != "D_rel":
        raise ValueError("Only D_rel training images may fit models")
    if len(held) and not np.array_equal(train,held) and set(pool["image"][train]) & set(pool["image"][held]):
        raise ValueError("Training and held-out images overlap")
    y, w = 1-pool["error"],pool["sampling_weight"]
    mc = dict(kind="MC",mc=fit_mc(pool["logit_mc"][train],y[train],w[train],spec["mc_calibrator"]))
    models = {"MC":mc}
    def fit(name, raw, indices, parent, form, interaction, kind, **extra):
        if not len(indices):
            return parent
        basis = fit_basis64(raw[indices],w[indices],form,spec["baseline"]["knots"],interaction)
        xx = design(basis,raw[indices])
        offset = apply_model(parent,pool,indices)
        hh = held if kind == "MM" else held[pool["strict_conflict"][held].astype(bool)]
        held_design = design(basis,raw[hh])
        held_offset = apply_model(parent,pool,hh)
        def trace(step,beta,objective,gradient):
            logits = held_offset+held_design@beta
            nll = float(np.average(np.logaddexp(0,logits)-y[hh]*logits,weights=w[hh])) if len(hh) else None
            curve(dict(**context,candidate=name,iteration=step,objective=objective,max_gradient=gradient,
                       heldout_nll=nll,heldout_mass=float(w[hh].sum()),fit_scope="all" if kind=="MM" else "conflict"))
        beta, diagnostics = newton(xx,offset,y[indices],w[indices],spec["optimizer"],device,trace)
        return dict(kind=kind,parent=parent,basis=basis,coefficients=beta.tolist(),fit=diagnostics,**extra)
    base_x = np.column_stack([pool["logit_mc"],pool["margin"]])
    for form in ("linear","additive"):
        name = "MM_"+form
        models[name] = fit(name,base_x,train,mc,form,form=="additive","MM")
    selected_train = train[pool["strict_conflict"][train].astype(bool)]
    for variant in variants(spec):
        for family,count in FEATURE_COUNTS.items():
            name = f"{variant}_{family}"
            models[name] = fit(name,pool["anchor__"+variant][:,:count],selected_train,models["MM_additive"],
                               "linear",False,"anchor",variant=variant,inputs=count)
    return models
