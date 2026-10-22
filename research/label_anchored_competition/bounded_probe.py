"""Low-dimensional, MC-offset probes for the locked post-hoc audit."""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch.nn import functional as F


def image_folds(image_ids, image_index, count, namespace):
    if len(image_ids) != len(set(image_ids)) or len(image_ids) < count:
        raise ValueError("Unique independent images required for folds")
    order = sorted(range(len(image_ids)), key=lambda i: hashlib.sha256(
        f"{namespace}|{image_ids[i]}".encode()).hexdigest())
    assignment = np.empty(len(image_ids), dtype=np.int32)
    assignment[order] = np.arange(len(order)) % count
    return [(np.flatnonzero(assignment[image_index] != j),
             np.flatnonzero(assignment[image_index] == j)) for j in range(count)]


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values, kind="stable")
    x, w = np.asarray(values)[order], np.asarray(weights, dtype=np.float64)[order]
    cumulative = w.cumsum()
    positions = np.searchsorted(cumulative, np.asarray(quantiles)*cumulative[-1], side="left")
    return x[np.minimum(positions, len(x)-1)]


def sigmoid(logits):
    return .5*(1+np.tanh(np.asarray(logits, dtype=np.float64)/2))


def logit_metrics(correct, logits, weights):
    correct, logits = np.asarray(correct), np.asarray(logits, dtype=np.float64)
    return dict(nll=float(np.average(np.logaddexp(0, logits)-correct*logits, weights=weights)),
                brier=float(np.average((sigmoid(logits)-correct)**2, weights=weights)))


def fit_mc(logit_mc, correct, weights, config, role="D_rel"):
    if role != "D_rel":
        raise ValueError("MC calibration may only fit D_rel training images")
    x, y, w = (np.asarray(v, dtype=np.float64) for v in (logit_mc, correct, weights))
    if not (np.isfinite(x).all() and np.isfinite(w).all() and (w > 0).all()):
        raise ValueError("Invalid MC calibration inputs")
    w = w/w.sum()
    mean = float(w@x)
    scale = float(np.sqrt(w@((x-mean)**2)))
    scale = scale if scale > 1e-10 else 1.
    z = (x-mean)/scale
    theta = np.array([scale, mean], dtype=np.float64)
    penalty, tolerance = config["l2"], config["projected_gradient_tolerance"]
    def objective(t):
        logits = t[0]*z+t[1]
        return float(w@(np.logaddexp(0, logits)-y*logits)+penalty/2*(t@t))
    converged = False
    for iteration in range(config["max_iterations"]):
        logits = theta[0]*z+theta[1]
        probability = sigmoid(logits)
        residual = w*(probability-y)
        gradient = np.array([residual@z, residual.sum()])+penalty*theta
        projected = gradient.copy()
        if theta[0] <= 1e-6+1e-12 and projected[0] > 0:
            projected[0] = 0
        if np.max(np.abs(projected)) < tolerance:
            converged = True
            break
        curvature = w*probability*(1-probability)
        hessian = np.array([[curvature@(z*z)+penalty, curvature@z],
                            [curvature@z, curvature.sum()+penalty]])
        direction = np.linalg.solve(hessian, gradient)
        if theta[0] <= 1e-6+1e-12 and gradient[0] > 0:
            direction = np.array([0., gradient[1]/hessian[1, 1]])
        previous = objective(theta)
        for power in range(40):
            trial = theta-(.5**power)*direction
            trial[0] = max(1e-6, trial[0])
            if objective(trial) <= previous+1e-4*(gradient@(trial-theta)):
                theta = trial
                break
        else:
            raise ValueError("MC calibrator line search failed")
    if not converged:
        raise ValueError("Fixed MC calibrator did not converge within registered budget")
    return dict(mean=mean, scale=scale, a=float(theta[0]), b=float(theta[1]),
                iterations=iteration, objective=objective(theta), converged=True)


def mc_logits(model, values):
    return model["a"]*(np.asarray(values, dtype=np.float64)-model["mean"])/model["scale"]+model["b"]


def fit_basis(features, weights, form, quantiles, role="D_rel"):
    if role != "D_rel" or form not in ("linear", "additive"):
        raise ValueError("Only registered bases fitted on D_rel are permitted")
    x, w = np.asarray(features, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    mean = np.average(x, weights=w, axis=0)
    scale = np.sqrt(np.average((x-mean)**2, weights=w, axis=0))
    scale = np.where(scale > 1e-10, scale, 1.)
    z = (x-mean)/scale
    columns, normalized_columns = [], []
    for j in range(x.shape[1]):
        specifications = [(j, None)]
        if form == "additive":
            specifications += [(j, float(q)) for q in np.unique(weighted_quantile(z[:, j], w, quantiles))]
        for column, knot in specifications:
            raw = z[:, column] if knot is None else np.maximum(0, z[:, column]-knot)
            center = float(np.average(raw, weights=w))
            std = float(np.sqrt(np.average((raw-center)**2, weights=w)))
            if std > 1e-10:
                normalized = (raw-center)/std
                # A hinge at the minimum duplicates the linear term. Cross-input
                # duplicates also alter effective ridge strength if kept twice.
                if any(np.allclose(normalized, previous, atol=1e-10, rtol=1e-10)
                       for previous in normalized_columns):
                    continue
                columns.append(dict(column=column, knot=knot, mean=center, scale=std))
                normalized_columns.append(normalized)
    return dict(mean=mean.tolist(), scale=scale.tolist(), columns=columns, form=form)


def transform_basis(basis, features):
    z = (np.asarray(features, dtype=np.float64)-np.asarray(basis["mean"]))/np.asarray(basis["scale"])
    output = []
    for item in basis["columns"]:
        value = z[:, item["column"]]
        if item["knot"] is not None:
            value = np.maximum(0, value-item["knot"])
        output.append((value-item["mean"])/item["scale"])
    return np.stack(output, 1).astype(np.float32) if output else np.zeros((len(z), 0), dtype=np.float32)


def choose_candidate(nll, tolerance):
    chosen = "zero"
    for name in ("linear", "additive"):
        if nll[name] < nll[chosen]-tolerance:
            chosen = name
    return chosen


def fit_correction_path(train_x, train_offset, correct, weight, heldout_x, heldout_offset,
                        heldout_correct, heldout_weight, config, seed, device, log=None, role="D_rel"):
    if role != "D_rel":
        raise ValueError("Only D_rel may fit correction coefficients")
    if not all(np.isfinite(v).all() for v in (train_x, train_offset, weight, heldout_x, heldout_offset)):
        raise ValueError("Nonfinite probe inputs")
    x = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    offset = torch.as_tensor(train_offset, dtype=torch.float32, device=device)
    target = torch.as_tensor(correct, dtype=torch.float32, device=device)
    w = torch.as_tensor(weight, dtype=torch.float32, device=device)
    coefficient = torch.nn.Parameter(torch.zeros(x.shape[1], device=device))
    optimizer = torch.optim.Adam([coefficient], lr=config["learning_rate"], weight_decay=config["l2"])
    generator = torch.Generator(device=device).manual_seed(seed+1)
    average_weight = w.mean()
    result = {}
    for step in range(max(config["budgets"])+1):
        if step in config["curve_updates"] or step in config["budgets"]:
            beta = coefficient.detach().cpu().numpy().copy()
            # Use FP64 offsets for exact zero restoration and stable risk ranking.
            pred = np.asarray(heldout_offset, dtype=np.float64)+heldout_x@beta
            if step in config["budgets"]:
                result[step] = dict(coefficients=beta, heldout_logits=pred)
            if log is not None:
                train_pred = np.asarray(train_offset, dtype=np.float64)+train_x@beta
                log(dict(update=step, train_nll=logit_metrics(correct, train_pred, weight)["nll"],
                    heldout_nll=logit_metrics(heldout_correct, pred, heldout_weight)["nll"],
                    coefficient_norm=float(np.linalg.norm(beta))))
        if step == max(config["budgets"]):
            break
        if not x.shape[1]:
            continue
        index = torch.randint(len(x), (config["batch_pixels"],), generator=generator, device=device)
        logits = offset[index]+x[index]@coefficient
        loss = (F.binary_cross_entropy_with_logits(logits, target[index], reduction="none")*w[index]).mean()/average_weight
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite correction loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return result


def apply_probe(model, features, logit_mc):
    baseline = mc_logits(model["mc"], logit_mc)
    if model["form"] == "zero":
        return baseline
    return baseline+transform_basis(model["basis"], features)@np.asarray(model["coefficients"], dtype=np.float32)
