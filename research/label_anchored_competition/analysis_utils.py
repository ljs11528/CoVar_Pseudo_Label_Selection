"""Image-role checked cache loading and fixed-budget correctness-head fitting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .core import CorrectnessHead, permuted_evidence, pixel_identity_keys, score_prediction
from .prepare_data import digest
from .ranking_metrics import Ranking


def read_pool(export, role, cap, device="cuda:0"):
    if role not in ("D_rel", "validation"):
        raise ValueError("Invalid verifier/evaluation data role")
    manifest = json.loads((export / "manifest.json").read_text())
    pieces = []
    ids = []
    for entry in manifest["files"]:
        if entry["role"] != role:
            continue
        path = export / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Export cache hash differs")
        with np.load(path, allow_pickle=False) as cache:
            if cache["role"].item() != role or cache["image_id"].item() != entry["image_id"]:
                raise ValueError("Data-role cache identity mismatch")
            n = min(cap, len(cache["pixel_indices"]))
            row = dict(image=np.full(n, len(ids), dtype=np.int32), GT=cache["GT"][:n],
                       error=(~cache["correct"][:n]).astype(np.int8), boundary=cache["boundary"][:n],
                       logits=cache["logits"][:n], similarities=cache["prototype_similarities"][:n],
                       pixel_indices=cache["pixel_indices"][:n],
                       sampling_weight=np.full(n, cache["valid_pixel_count"].item()/n, dtype=np.float64))
            row.update({key.removeprefix("score__"):cache[key][:n] for key in cache.files if key.startswith("score__")})
            row["pixel_keys"] = pixel_identity_keys(entry["image_id"], row["pixel_indices"]).numpy()
        pieces.append(row)
        ids.append(entry["image_id"])
    if len(ids) != manifest["images"][role] or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate images in diagnostic pool")
    pool = {key:np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    pool["image_ids"] = ids
    pool["role"] = role
    if not np.array_equal(pool["pseudo_label"], pool["logits"].argmax(1)):
        raise ValueError("Cached classifier label differs from original logits")
    if not np.array_equal(pool["error"], (pool["pseudo_label"] != pool["GT"]).astype(np.int8)):
        raise ValueError("Cached correctness differs from evaluation GT")
    return pool


@torch.no_grad()
def temperature_features(pool, temperature, device="cuda:0"):
    output = {key:[] for key in ("product_support_original_label", "logit_product_support", "probability_prototype_JS")}
    for start in range(0, len(pool["error"]), 65536):
        scores = score_prediction(torch.from_numpy(pool["logits"][start:start+65536]).to(device),
                                  torch.from_numpy(pool["similarities"][start:start+65536]).to(device), temperature)
        for key in output:
            output[key].append(scores[key].cpu().numpy())
    return {key:np.concatenate(value) for key, value in output.items()}


@torch.no_grad()
def permutation_features(pool, seed, device="cuda:0"):
    output = []
    for start in range(0, len(pool["error"]), 65536):
        z = torch.from_numpy(pool["logits"][start:start+65536]).to(device)
        s = torch.from_numpy(pool["similarities"][start:start+65536]).to(device)
        scores = score_prediction(z, s)
        values = permuted_evidence(scores["residual_probability"], scores["residual_counterevidence"],
                                  torch.from_numpy(pool["pixel_keys"][start:start+65536]).to(device), seed)
        output.append(values.cpu().numpy())
    return np.concatenate(output)


def choose_temperature(pool, folds, grid, device):
    """Select one fusion temperature on D_rel only; never access test labels."""
    if pool["role"] != "D_rel":
        raise ValueError("Temperature tuning is restricted to verifier images")
    candidates, calculated = [], {}
    for temperature in grid:
        values = temperature_features(pool, temperature, device)
        calculated[temperature] = values
        metrics = [Ranking(pool["error"][test], -values["product_support_original_label"][test]).evaluate(pool["sampling_weight"][test])["error_ap"]
                   for _, test in folds]
        candidates.append(dict(temperature=temperature, cv_error_ap=float(np.nanmean(metrics))))
    if not any(np.isfinite(x["cv_error_ap"]) for x in candidates):
        raise ValueError("Verifier data cannot support fusion temperature selection")
    chosen = max(candidates, key=lambda row:(row["cv_error_ap"], -row["temperature"]))["temperature"]
    return chosen, candidates, calculated[chosen]


def head_features(pool, method, fusion, permutation=None):
    m = pool["logit_mc"]
    second = {
        "h_MC":np.zeros_like(m), "h_entropy":pool["entropy"], "h_margin":pool["margin"],
        "h_prototype_support":pool["prototype_support"], "h_prototype_margin":pool["prototype_margin"],
        "h_JS":fusion["probability_prototype_JS"], "h_product":fusion["product_support_original_label"],
        "h_uniform":pool["uniform_competitor"], "h_top2":pool["top2_competitor"], "h_e":pool["competitor_evidence"],
    }
    if method in second:
        return np.stack((m, second[method]), 1).astype(np.float32)
    if method.startswith("h_permutation_"):
        if permutation is None:
            raise ValueError("Missing predeclared permutation features")
        return np.stack((m, permutation), 1).astype(np.float32)
    if method in ("F_base", "F_plus"):
        extra = pool["competitor_evidence"] if method == "F_plus" else np.zeros_like(m)
        return np.stack((m, pool["margin"], pool["entropy"], pool["residual_entropy"],
                         pool["prototype_support"], pool["prototype_margin"], fusion["logit_product_support"],
                         fusion["probability_prototype_JS"], extra), 1).astype(np.float32)
    if method in ("F_simple_base", "F_simple_plus"):
        extra = pool["competitor_evidence"] if method == "F_simple_plus" else np.zeros_like(m)
        return np.stack((m, pool["prototype_margin"], fusion["logit_product_support"], extra), 1).astype(np.float32)
    raise ValueError(f"Unknown preregistered head {method}")


def fit_head(features, correct, weights, config, l2, device):
    torch.manual_seed(config["fit_seed"])
    head = CorrectnessHead(features.shape[1]).to(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    target = torch.as_tensor(correct, dtype=torch.float32, device=device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=device)
    head.fit_standardization(x, w)
    average_weight = w.mean()
    generator = torch.Generator(device=device).manual_seed(config["fit_seed"]+1)
    optimizer = torch.optim.Adam(head.parameters(), lr=config["learning_rate"], weight_decay=l2)
    head.train()
    for _ in range(config["updates"]):
        index = torch.randint(len(x), (config["batch_pixels"],), generator=generator, device=device)
        loss = (F.binary_cross_entropy_with_logits(head(x[index]), target[index], reduction="none")*w[index]).mean()/average_weight
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite correctness-head training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return head.eval()


@torch.no_grad()
def predict_head(head, features, device):
    output = []
    for start in range(0, len(features), 65536):
        output.append(head.weights(torch.from_numpy(features[start:start+65536]).to(device)).cpu().numpy())
    return np.concatenate(output)


def calibration(correct, probability, weights):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1-1e-7)
    return dict(brier=float(np.average((probability-correct)**2, weights=weights)),
                nll=float(np.average(-(correct*np.log(probability)+(1-correct)*np.log1p(-probability)), weights=weights)))


def fit_selected_head(pool, train_features, folds, config, device):
    if pool["role"] != "D_rel":
        raise ValueError("Only D_rel may fit correctness heads")
    correct = 1-pool["error"]
    results = []
    out_of_fold = {}
    for l2 in config["l2_grid"]:
        metrics = []
        heldout = np.full(len(correct), np.nan, dtype=np.float32)
        for train, validation in folds:
            head = fit_head(train_features[train], correct[train], pool["sampling_weight"][train], config, l2, device)
            prediction = predict_head(head, train_features[validation], device)
            heldout[validation] = prediction
            metrics.append(calibration(correct[validation], prediction, pool["sampling_weight"][validation])["nll"])
            del head
        results.append(dict(l2=l2, cv_nll=float(np.mean(metrics))))
        if not np.isfinite(heldout).all():
            raise ValueError("D_rel image folds failed to cover every sampled pixel")
        out_of_fold[l2] = heldout
    chosen = min(results, key=lambda row:(row["cv_nll"], row["l2"]))["l2"]
    head = fit_head(train_features, correct, pool["sampling_weight"], config, chosen, device)
    return head, chosen, results, out_of_fold[chosen]
