"""Read-only real-cache checks, distributions and fixed joint strata."""
import json
import numpy as np

from .bounded_probe import weighted_quantile
from .prepare_data import digest
from .ranking_metrics import Ranking


def read_audit_pool(export, role, spec, progress=None):
    if role not in ("D_rel", "validation"):
        raise ValueError("Unsupported audit role")
    manifest = json.loads((export / "manifest.json").read_text())
    if digest(export / "manifest.json") != json.loads((export / "COMPLETE.json").read_text())["manifest_sha256"]:
        raise ValueError("Completed manifest changed")
    if manifest["signature"]["protocol_sha256"] != spec["parent_protocol_sha256"]:
        raise ValueError("Parent protocol mismatch")
    if digest(export / "prototypes.npz") != manifest["prototype_sha256"]:
        raise ValueError("Prototype file checksum mismatch")
    with np.load(export / "prototypes.npz", allow_pickle=False) as bank:
        prototypes = bank["prototypes"].copy()
        prototype_ids = set(bank["image_ids"].tolist())
        support = bank["image_counts"].copy()
    pieces, identities = [], []
    masses = np.zeros(22, dtype=np.float64)
    sums, squares = np.zeros((22, prototypes.shape[1])), np.zeros((22, prototypes.shape[1]))
    check = dict(cache_shards_verified=0, prototype_similarity_max_abs_diff=0.,
                 independent_e_max_abs_diff=0., normalized_feature_max_error=0.)
    names = ("mc", "logit_mc", "margin", "entropy", "prototype_margin", "prototype_support",
             "prototype_label", "strict_conflict", "competitor_evidence", "uniform_competitor", "top2_competitor")
    for entry in manifest["files"]:
        if entry["role"] != role:
            continue
        path = export / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Cache shard changed")
        with np.load(path, allow_pickle=False) as cache:
            if cache["role"].item() != role or cache["image_id"].item() != entry["image_id"] or entry["image_id"] in prototype_ids:
                raise ValueError("Role/prototype image isolation failure")
            n = min(spec["cap"], len(cache["GT"]))
            row = {name: cache["score__"+name][:n] for name in names}
            z, similarity, features = cache["logits"][:n], cache["prototype_similarities"][:n], cache["features"][:n]
            if not all(np.isfinite(v).all() for v in (z, similarity, features)):
                raise ValueError("Nonfinite real cache")
            label = z.argmax(1)
            gt = cache["GT"][:n]
            error = (gt != label).astype(np.int8)
            if not np.array_equal(error, ~cache["correct"][:n]) or not np.array_equal(label, cache["score__pseudo_label"][:n]):
                raise ValueError("Real-cache correctness mismatch")
            sim_error = float(np.max(np.abs(features@prototypes.T-similarity)))
            norm_error = float(np.max(np.abs(np.linalg.norm(features, axis=1)-1)))
            rival_z = z.astype(np.float64).copy()
            rival_z[np.arange(n), label] = -np.inf
            residual = np.exp(rival_z-rival_z.max(1)[:, None])
            residual /= residual.sum(1)[:, None]
            difference = similarity.astype(np.float64)-similarity[np.arange(n), label, None]
            independent_e = (residual*np.maximum(0, difference)).sum(1)
            evidence_error = float(np.max(np.abs(independent_e-row["competitor_evidence"])))
            if sim_error > 2e-5 or norm_error > 2e-5 or evidence_error > 2e-5:
                raise ValueError("Real feature/prototype/evidence consistency audit failed")
            for key, value in [("prototype_similarity_max_abs_diff", sim_error),
                               ("normalized_feature_max_error", norm_error), ("independent_e_max_abs_diff", evidence_error)]:
                check[key] = max(check[key], value)
            weight = float(cache["valid_pixel_count"].item()/n)
            row.update(image=np.full(n, len(identities), dtype=np.int32), error=error, GT=gt, pseudo_label=label,
                       boundary=cache["boundary"][:n], pixel_indices=cache["pixel_indices"][:n],
                       sampling_weight=np.full(n, weight, dtype=np.float64), residual_max=residual.max(1),
                       product_delta=np.log((residual*np.exp(difference/spec["temperature"])).sum(1)))
            f = features.astype(np.float64)
            for k in [-1, *np.unique(label).tolist()]:
                selected = np.ones(n, bool) if k == -1 else label == k
                slot = k+1
                masses[slot] += selected.sum()*weight
                sums[slot] += f[selected].sum(0)*weight
                squares[slot] += (f[selected]**2).sum(0)*weight
            pieces.append(row)
            identities.append(entry["image_id"])
            check["cache_shards_verified"] += 1
            if progress and len(identities) % 100 == 0:
                progress(role=role, images=len(identities))
    if len(identities) != spec["role_counts"][role] or len(set(identities)) != len(identities):
        raise ValueError("Incomplete image pool")
    pool = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    pool.update(image_ids=identities, role=role)
    moments = []
    for k in range(22):
        mass = masses[k]
        mean = sums[k]/mass if mass else np.zeros_like(sums[k])
        variance = np.maximum(0, squares[k]/mass-mean**2) if mass else np.zeros_like(mean)
        moments.append(dict(predicted_class=k-1, estimated_pixels=float(mass),
                            feature_mean=mean.tolist(), feature_variance=variance.tolist()))
    check.update(role=role, feature_moments=moments, prototype_support_images=support.tolist())
    return pool, check


def signals(pool, feature_sets):
    columns = dict(prototype_risk=-pool["prototype_margin"], product_correction=pool["product_delta"],
                   uniform=pool["uniform_competitor"], e=pool["competitor_evidence"])
    return {name: np.stack([columns[key] for key in keys], 1) for name, keys in feature_sets.items()}


def weighted_correlation(x, y, w):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    x = x-np.average(x, weights=w)
    y = y-np.average(y, weights=w)
    denominator = np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
    return float(np.sum(w*x*y)/denominator) if denominator else None


def weighted_midrank(x, w):
    order = np.argsort(x, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(np.asarray(x)[order]) != 0)+1]
    mass = np.add.reduceat(np.asarray(w)[order], starts)
    ranks = (mass.cumsum()-.5*mass)/mass.sum()
    out = np.empty(len(x), dtype=np.float64)
    out[order] = np.repeat(ranks, np.diff(np.r_[starts, len(x)]))
    return out


def distributions(pool, spec):
    output = []
    for name, selected in scopes(pool).items():
        w = pool["sampling_weight"][selected]
        if not len(w):
            continue
        e, top2 = pool["competitor_evidence"][selected], pool["top2_competitor"][selected]
        row = dict(role=pool["role"], scope=name, pixels=int(selected.sum()),
            error_rate=float(np.average(pool["error"][selected], weights=w)),
            e_top2_Pearson=weighted_correlation(e, top2, w),
            e_top2_weighted_Spearman=weighted_correlation(weighted_midrank(e, w), weighted_midrank(top2, w), w),
            e_zero_fraction=float(np.average(e == 0, weights=w)))
        for key in ("mc", "logit_mc", "prototype_margin", "competitor_evidence", "uniform_competitor", "residual_max"):
            value = pool[key][selected]
            row[key+"_mean"] = float(np.average(value, weights=w))
            for quantile, qvalue in zip(spec["distributions"]["quantiles"], weighted_quantile(value, w, spec["distributions"]["quantiles"])):
                row[f"{key}_q{quantile:g}"] = float(qvalue)
        for threshold in spec["distributions"]["residual_max_thresholds"]:
            row[f"residual_max_ge_{threshold:g}"] = float(np.average(pool["residual_max"][selected] >= threshold, weights=w))
        output.append(row)
    return output


def scopes(pool):
    return dict(all=np.ones(len(pool["error"]), bool), conflict=pool["strict_conflict"].astype(bool),
                high_mc=pool["mc"] >= .95, high_mc_conflict=pool["strict_conflict"].astype(bool) & (pool["mc"] >= .95))


def support_indices(pool, indices, limits):
    e = pool["error"][indices].astype(bool)
    image = pool["image"][indices]
    info = dict(sampled_pixels=len(indices), errors=int(e.sum()), correct=int((~e).sum()),
                images=len(np.unique(image)), error_images=len(np.unique(image[e])), correct_images=len(np.unique(image[~e])))
    info["adequate_support"] = bool(info["errors"] >= limits["sampled_error_pixels"]
        and info["correct"] >= limits["sampled_correct_pixels"] and info["error_images"] >= limits["images_with_errors"]
        and info["correct_images"] >= limits["images_with_correct_pixels"])
    return info


def joint_tables(pool, risks, spec):
    mc_bin = np.searchsorted(spec["strata"]["mc_edges"], pool["mc"], side="right")-1
    pm_bin = np.searchsorted(spec["strata"]["prototype_margin_edges"], pool["prototype_margin"], side="right")-1
    nm, np_ = len(spec["strata"]["mc_edges"])-1, len(spec["strata"]["prototype_margin_edges"])-1
    if (mc_bin < 0).any() or (mc_bin >= nm).any() or (pm_bin < 0).any() or (pm_bin >= np_).any():
        raise ValueError("Sample outside locked stratum edges")
    joint = mc_bin*np_+pm_bin
    cb = pool["pseudo_label"]*2+pool["boundary"].astype(int)
    output, supports = [], []
    for kind, code, count in (("MC_prototype", joint, nm*np_), ("class_boundary", cb, 42), ("four_way", joint*42+cb, nm*np_*42)):
        order = np.argsort(code, kind="stable")
        cuts = np.searchsorted(code[order], np.arange(count+1))
        for group in range(count):
            index = order[cuts[group]:cuts[group+1]]
            metadata = dict(role=pool["role"], stratum=kind, group=group,
                            **support_indices(pool, index, spec["strata"]["minimum_support"]))
            if kind in ("MC_prototype", "four_way"):
                j = group if kind == "MC_prototype" else group//42
                metadata.update(mc_bin=j//np_, prototype_margin_bin=j%np_)
            if kind in ("class_boundary", "four_way"):
                k = group if kind == "class_boundary" else group%42
                metadata.update(predicted_class=k//2, boundary=bool(k%2))
            supports.append(metadata)
            if kind == "four_way" or not len(index):
                continue
            for name, risk in risks.items():
                output.append(dict(**metadata, method=name,
                    **Ranking(pool["error"][index], risk[index]).evaluate(pool["sampling_weight"][index], (.9,))))
    return output, supports
