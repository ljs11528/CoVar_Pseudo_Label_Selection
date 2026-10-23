"""Fixed historical alternatives and image-level recovery statistics, no fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .core import pixel_identity_keys, residual_permutation
from .prepare_data import digest


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    shifted = z-z.max(1, keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(1, keepdims=True))


def other_argmax(z, a):
    other = np.asarray(z, dtype=np.float64).copy()
    other[np.arange(len(a)), a] = -np.inf
    return other.argmax(1)


def label_logit(z, a):
    z = np.asarray(z, dtype=np.float64)
    other = z.copy()
    other[np.arange(len(a)), a] = -np.inf
    maximum = other.max(1)
    return z[np.arange(len(a)), a]-(maximum+np.log(np.exp(other-maximum[:, None]).sum(1)))


def current_features(z):
    z = np.asarray(z, dtype=np.float64)
    lp = log_softmax(z); p = np.exp(lp)
    a = z.argmax(1); b = other_argmax(z, a); ix = np.arange(len(a))
    return dict(mc=p[ix, a], margin=p[ix, a]-p[ix, b], p2=p[ix, b],
                logit_mc=label_logit(z, a), entropy=-(p*lp).sum(1), pseudo_label=a, b=b)


def pair_values(current, past):
    z, u = np.asarray(current, np.float64), np.asarray(past, np.float64)
    if z.shape != u.shape or z.ndim != 2 or z.shape[1] < 2:
        raise ValueError("Aligned multiclass logits required")
    if not np.isfinite(z).all() or not np.isfinite(u).all():
        raise ValueError("Nonfinite historical logits")
    cur = current_features(z); a, b = cur["pseudo_label"], cur["b"]
    lp, lq = log_softmax(z), log_softmax(u)
    arithmetic = np.logaddexp(lp, lq)-np.log(2.)
    geometric = (lp+lq)/2
    h = other_argmax(u, a)
    alternatives = dict(now=b, history=h, arithmetic=other_argmax(arithmetic, a),
                        geometric=other_argmax(geometric, a))
    ix = np.arange(len(a)); q = np.exp(lq)
    history_a = label_logit(u, a)
    arithmetic_a = label_logit(arithmetic, a)
    geometric_a = label_logit(geometric, a)
    past_a = u.argmax(1)
    risks = dict(raw_MC=-cur["mc"], raw_margin=-cur["margin"], raw_entropy=cur["entropy"],
        arithmetic_current_a=-arithmetic_a, geometric_current_a=-geometric_a,
        past_current_a=-history_a, past_b_vs_a=u[ix, b]-u[ix, a])
    past_b = other_argmax(u, past_a)
    ordinary = np.column_stack([q.max(1), -(q*lq).sum(1), q[ix, a], past_a != a,
                                arithmetic_a, geometric_a])
    identity = np.column_stack([u[ix, b]-u[ix, a], h == b, (past_a == b) | (past_b == b)])
    return dict(current=cur, alternatives=alternatives, risks=risks, past_label=past_a,
                H0=ordinary, H1=np.column_stack([ordinary, identity]))


def shuffled_past(current, past, image_id, pixels, seed, namespace):
    z, u = np.asarray(current), np.asarray(past)
    a = z.argmax(1); n, classes = z.shape
    slots = np.arange(classes)[None, :] != a[:, None]
    source = np.broadcast_to(np.arange(classes), z.shape)[slots].reshape(n, classes-1)
    keys = pixel_identity_keys(namespace+"|"+image_id, pixels)
    order = residual_permutation(keys, classes-1, seed).numpy()
    source = np.take_along_axis(source, order, 1)
    result = u.copy()
    result[slots] = np.take_along_axis(u, source, 1).ravel()
    np.testing.assert_array_equal(np.sort(result, axis=1), np.sort(u, axis=1))
    np.testing.assert_array_equal(result[np.arange(n), a], u[np.arange(n), a])
    # Tied maxima can change np.argmax identity under permutation. Compare the
    # fixed-a logit against every other logit, and separately audit unique maxima.
    maximum_ties = (u == u.max(1, keepdims=True)).sum(1) > 1
    unique = ~maximum_ties
    np.testing.assert_array_equal((result.argmax(1) == a)[unique], (u.argmax(1) == a)[unique])
    old, new = current_features(u), current_features(result)
    error = max(float(np.max(np.abs(old[k]-new[k]))) for k in ("mc", "entropy"))
    error = max(error, float(np.max(np.abs(np.exp(log_softmax(u))[np.arange(n), a]-
                                           np.exp(log_softmax(result))[np.arange(n), a]))))
    if error > 1e-12:
        raise ValueError("History permutation changed an invariant")
    # The locked semantic invariant requires actual argmax agreement. Ties that
    # violate it must stop the formal run, rather than silently relax the rule.
    if np.any((result.argmax(1) == a) != (u.argmax(1) == a)):
        raise ValueError("History argmax tie violates fixed-a agreement invariant")
    return result, error


def validate_aligned_records(records, spec):
    stages = list(spec["input_exports"])
    reference = records[stages[0]]
    for stage, row in records.items():
        for key in spec["alignment"]["fields"]:
            if not np.array_equal(row[key], reference[key]):
                raise ValueError(f"Cross-stage mismatch: {stage}/{key}")
        z, gt, pixels, yx = (row[k] for k in ("logits", "GT", "pixel_indices", "yx"))
        size = np.asarray(row["original_size"])
        if size.shape != (2,) or np.any(size <= 0):
            raise ValueError("Invalid original dimensions")
        h, w = map(int, size)
        if z.shape != (len(gt), spec["classes"]) or yx.shape != (len(gt), 2):
            raise ValueError("Invalid aligned pixel shape")
        if pixels.shape != (len(gt),) or len(np.unique(pixels)) != len(pixels):
            raise ValueError("Duplicate or mismatched pixel indices")
        if not len(gt) or not np.isin(gt, np.arange(spec["classes"])).all():
            raise ValueError("Invalid/void diagnostic labels")
        if np.any(pixels < 0) or np.any(pixels >= h*w):
            raise ValueError("Out-of-range pixel")
        if not np.array_equal(yx, np.column_stack([pixels//w, pixels % w])):
            raise ValueError("Pixel coordinate mapping differs")
        valid = int(row["valid_pixel_count"])
        if not len(gt) <= valid <= h*w or not np.isin(row["boundary"], [False, True]).all():
            raise ValueError("Invalid valid-pixel count or boundary mask")
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite logits")
        if not np.array_equal(z.argmax(1) == gt, row["correct"]):
            raise ValueError("Current-stage correctness cache differs")
        if not np.array_equal(z.argmax(1), row["score__pseudo_label"]):
            raise ValueError("Cached pseudo-label differs")
    return len(reference["GT"])


def array_digest(array):
    a = np.asarray(array)
    h = hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def read_aligned(root: Path, role, spec, progress, synthetic=False):
    if role not in spec["role_counts"]:
        raise ValueError("Unregistered role")
    manifests, entries, source, bank_ids = {}, {}, {}, None
    base_protocol = digest(Path(__file__).with_name("protocol.json"))
    for stage, name in spec["input_exports"].items():
        folder = root/"experiments"/name
        manifest = json.loads((folder/"manifest.json").read_text())
        complete = json.loads((folder/"COMPLETE.json").read_text())
        if complete["state"] != "complete" or digest(folder/"manifest.json") != complete["manifest_sha256"]:
            raise ValueError("Export is incomplete or its manifest changed")
        sig = manifest["signature"]
        if bool(manifest.get("synthetic", False)) != bool(synthetic):
            raise ValueError("Synthetic/formal export mixing")
        if sig["stage"] != stage or sig["seed"] != spec["teacher_seed"] or sig != complete["signature"]:
            raise ValueError("Teacher stage/seed identity mismatch")
        if sig["protocol_sha256"] != base_protocol:
            raise ValueError("Frozen parent protocol mismatch")
        if digest(folder/"prototypes.npz") != manifest["prototype_sha256"]:
            raise ValueError("Original prototype identity changed")
        with np.load(folder/"prototypes.npz", allow_pickle=False) as bank:
            ids = set(bank["image_ids"].tolist())
        if bank_ids is None:
            bank_ids = ids
        elif ids != bank_ids:
            raise ValueError("D_sup identities changed across stages")
        files = [r for r in manifest["files"] if r["role"] == role]
        mapping = {r["image_id"]: r for r in files}
        if len(mapping) != len(files) or len(files) != spec["role_counts"][role]:
            raise ValueError("Incomplete/duplicate image role")
        if set(mapping) & ids:
            raise ValueError("D_sup overlaps diagnostic images")
        manifests[stage], entries[stage] = manifest, mapping
        source[stage] = dict(manifest_sha256=digest(folder/"manifest.json"), signature=sig,
                             prototype_sha256=manifest["prototype_sha256"])
    first = next(iter(entries)); ids = list(entries[first])
    if any(set(m) != set(ids) for m in entries.values()):
        raise ValueError("Cross-stage image sets differ")
    if len({s["signature"]["partition_receipt_sha256"] for s in source.values()}) != 1:
        raise ValueError("Partition identity changed across stages")
    if len({s["signature"]["checkpoint_sha256"] for s in source.values()}) != len(entries):
        raise ValueError("Different stages unexpectedly reference one checkpoint")
    pieces, logits = [], {s: [] for s in entries}
    aligned, shard_receipts = [], []
    for image_index, identity in enumerate(ids):
        records = {}
        for stage in entries:
            entry = entries[stage][identity]
            folder = root/"experiments"/spec["input_exports"][stage]
            path = folder/entry["file"]
            if path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                raise ValueError("Unsafe cache path")
            if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                raise ValueError("Original cache shard changed")
            with np.load(path, allow_pickle=False) as cache:
                names = list(spec["alignment"]["fields"])+["logits", "correct", "score__pseudo_label", "score__strict_conflict"]
                row = {k: cache[k].copy() for k in names}
                for key, expected in (("checkpoint_sha256", manifests[stage]["signature"]["checkpoint_sha256"]),
                                      ("protocol_sha256", base_protocol),
                                      ("prototype_sha256", manifests[stage]["prototype_sha256"])):
                    if cache[key].item() != expected:
                        raise ValueError("Shard provenance mismatch: "+key)
            if row["image_id"].item() != identity or row["role"].item() != role:
                raise ValueError("Shard role/identity mismatch")
            records[stage] = row
            shard_receipts.append(dict(stage=stage, role=role, image_id=identity, **{k:entry[k] for k in ("file", "bytes", "sha256")}))
        count = validate_aligned_records(records, spec)
        n = min(spec["cap"], count); row = records[first]
        piece = {k:row[k][:n] for k in ("GT", "boundary", "pixel_indices", "yx")}
        piece.update(image=np.full(n, image_index, np.int32),
                     sampling_weight=np.full(n, float(row["valid_pixel_count"])/n))
        for stage in entries:
            logits[stage].append(records[stage]["logits"][:n])
            piece["conflict_"+stage] = records[stage]["score__strict_conflict"][:n]
        pieces.append(piece)
        aligned.append(dict(image_id=identity, role=role, full_cached_pixels=count, analyzed_pixels=n,
            valid_pixels=int(row["valid_pixel_count"]), original_size=row["original_size"].tolist(),
            fields_equal=True, shared_array_sha256={k:array_digest(row[k]) for k in ("pixel_indices", "yx", "GT", "boundary")}))
        if (image_index+1) % 50 == 0:
            progress("read_aligned", role=role, images=image_index+1)
    pool = {key:np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    pool.update(role=role, image_ids=ids)
    for stage in logits:
        pool["logits_"+stage] = np.concatenate(logits[stage])
    checks = dict(role=role, images=len(ids), shards_verified=len(shard_receipts),
                  all_full_pixel_arrays_equal=True, full_pixel_positions=sum(x["full_cached_pixels"] for x in aligned),
                  analyzed_pixel_positions=len(pool["GT"]), sources=source, per_image=aligned, shards=shard_receipts)
    return pool, checks


def image_slices(pool):
    counts = np.bincount(pool["image"], minlength=len(pool["image_ids"]))
    starts = np.r_[0, counts.cumsum()]
    if not np.array_equal(pool["image"], np.repeat(np.arange(len(counts)), counts)):
        raise ValueError("Expected image-contiguous audited pixels")
    return [slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:])]


def build_pair(pool, pair, spec, include_probe_features=False):
    values = pair_values(pool["logits_"+pair["current"]], pool["logits_"+pair["past"]])
    result = dict(pool)
    result.update(values["current"])
    result.update(error=(result["pseudo_label"] != pool["GT"]).astype(np.int8),
                  strict_conflict=pool["conflict_"+pair["current"]], pair=pair["name"],
                  past_label=values["past_label"])
    if include_probe_features:
        result.update(H0=values["H0"], H1=values["H1"])
    alternatives = dict(values["alternatives"]); max_error = 0.
    slices = image_slices(pool)
    for seed in spec["permutation"]["seeds"]:
        rows = []
        for i, identity in enumerate(pool["image_ids"]):
            ix = slices[i]
            shuffled, error = shuffled_past(pool["logits_"+pair["current"]][ix],
                pool["logits_"+pair["past"]][ix], identity, pool["pixel_indices"][ix], seed, spec["permutation"]["namespace"])
            item = pair_values(pool["logits_"+pair["current"]][ix], shuffled)
            rows.append(item if include_probe_features else dict(alternatives=item["alternatives"]))
            max_error = max(max_error, error)
        alternatives[f"sham{seed}"] = np.concatenate([r["alternatives"]["history"] for r in rows])
        if include_probe_features:
            result[f"H0_sham{seed}"] = np.concatenate([r["H0"] for r in rows])
            result[f"H1_sham{seed}"] = np.concatenate([r["H1"] for r in rows])
    result["history_alternative"] = alternatives["history"]
    return result, alternatives, values["risks"], max_error


def scope_masks(pool):
    same = pool["past_label"] == pool["pseudo_label"]
    persistent = pool["history_alternative"] == pool["b"]
    conflict = pool["strict_conflict"].astype(bool)
    return dict(all=np.ones(len(same), bool), high_mc=pool["mc"] >= .95, very_high_mc=pool["mc"] >= .995,
        past_agrees=same, past_disagrees=~same, alternative_persists=persistent,
        alternative_changes=~persistent, conflict=conflict, nonconflict=~conflict)


def support(pool, mask, minimum):
    selected = np.flatnonzero(mask); error = pool["error"][selected].astype(bool)
    images = pool["image"][selected]
    row = dict(sampled_pixels=len(selected), errors=int(error.sum()), correct=int((~error).sum()),
               images=len(np.unique(images)), error_images=len(np.unique(images[error])),
               correct_images=len(np.unique(images[~error])))
    row["adequate_support"] = bool(row["errors"] >= minimum["error_pixels"] and row["correct"] >= minimum["correct_pixels"]
        and row["error_images"] >= minimum["error_images"] and row["correct_images"] >= minimum["correct_images"])
    return row


def sufficient_rows(pool, alternatives, spec):
    output = []; e = pool["error"].astype(bool); w = pool["sampling_weight"]
    now_hit = alternatives["now"] == pool["GT"]
    old_error = pool["past_label"] != pool["GT"]
    slices = image_slices(pool)
    for scope, mask in scope_masks(pool).items():
        for i, identity in enumerate(pool["image_ids"]):
            sl = slices[i]
            ix = np.flatnonzero(mask[sl])+sl.start
            ee, ww = e[ix], w[ix]
            common = dict(pair=pool["pair"], role=pool["role"], scope=scope, image_id=identity, image_index=i,
                pixels=len(ix), error_pixels=int(ee.sum()), correct_pixels=int((~ee).sum()),
                mass=float(ww.sum()), error_mass=float(ww@ee), correct_mass=float(ww@(~ee)),
                past_correct_current_error_mass=float(ww@(ee & ~old_error[ix])),
                past_error_current_correct_mass=float(ww@(~ee & old_error[ix])),
                both_correct_mass=float(ww@(~ee & ~old_error[ix])),
                both_wrong_same_mass=float(ww@(ee & old_error[ix] & (pool["past_label"][ix] == pool["pseudo_label"][ix]))),
                both_wrong_different_mass=float(ww@(ee & old_error[ix] & (pool["past_label"][ix] != pool["pseudo_label"][ix]))))
            for method, candidate in alternatives.items():
                hit = candidate[ix] == pool["GT"][ix]
                output.append(dict(common, method=method,
                    recovered_mass=float(ww@(ee & hit)),
                    gained_vs_now_mass=float(ww@(ee & hit & ~now_hit[ix])),
                    lost_vs_now_mass=float(ww@(ee & ~hit & now_hit[ix])),
                    alternative_changed_mass=float(ww@(candidate[ix] != alternatives["now"][ix]))))
    return output


def aggregate_sufficient(rows, spec):
    output = []
    keys = list(dict.fromkeys((r["pair"], r["role"], r["scope"], r["method"]) for r in rows))
    minimum = spec["screen"]["minimum_support"]
    for pair, role, scope, method in keys:
        parts = [r for r in rows if (r["pair"], r["role"], r["scope"], r["method"]) == (pair, role, scope, method)]
        sum_keys = [k for k in parts[0] if k.endswith("_mass") or k in ("mass", "pixels", "error_pixels", "correct_pixels")]
        total = {k:sum(r[k] for r in parts) for k in sum_keys}
        error_images = sum(r["error_pixels"] > 0 for r in parts)
        correct_images = sum(r["correct_pixels"] > 0 for r in parts)
        denominator = total["error_mass"]; mass = total["mass"]
        output.append(dict(pair=pair, role=role, scope=scope, method=method, **total,
            image_pool=len(parts), images=sum(r["pixels"] > 0 for r in parts),
            error_images=error_images, correct_images=correct_images,
            adequate_support=error_images >= minimum["error_images"] and correct_images >= minimum["correct_images"]
                and total["error_pixels"] >= minimum["error_pixels"] and total["correct_pixels"] >= minimum["correct_pixels"],
            error_recovery=total["recovered_mass"]/denominator if denominator else np.nan,
            set_truth_coverage=(total["correct_mass"]+total["recovered_mass"])/mass if mass else np.nan,
            error_rate=denominator/mass if mass else np.nan,
            gain_vs_now=total["gained_vs_now_mass"]/denominator if denominator else np.nan,
            loss_vs_now=total["lost_vs_now_mass"]/denominator if denominator else np.nan,
            candidate_change_rate=total["alternative_changed_mass"]/mass if mass else np.nan))
    return output


def recovery_intervals(rows, spec):
    result = []; cfg = spec["screen"]
    keys = list(dict.fromkeys((r["pair"], r["role"], r["scope"]) for r in rows))
    for pair, role, scope in keys:
        selected = [r for r in rows if (r["pair"], r["role"], r["scope"]) == (pair, role, scope)]
        methods = list(dict.fromkeys(r["method"] for r in selected))
        vectors = {m:sorted([r for r in selected if r["method"] == m], key=lambda r:r["image_id"]) for m in methods}
        ids = [r["image_id"] for r in vectors[methods[0]]]
        if any([r["image_id"] for r in v] != ids for v in vectors.values()):
            raise ValueError("Methods do not share full bootstrap image pool")
        n = len(ids)
        rng = np.random.default_rng(cfg["bootstrap_seed"])
        counts = np.stack([np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(cfg["bootstrap_replicates"])])
        denominator = np.array([r["error_mass"] for r in vectors[methods[0]]])
        den = counts@denominator
        sample, point = {}, {}
        for method, vector in vectors.items():
            np.testing.assert_array_equal(denominator, [r["error_mass"] for r in vector])
            numerator = np.array([r["recovered_mass"] for r in vector])
            sample[method] = np.divide(counts@numerator, den, out=np.full(len(den), np.nan), where=den > 0)
            point[method] = numerator.sum()/denominator.sum() if denominator.sum() else np.nan
        sham = [f"sham{s}" for s in spec["permutation"]["seeds"]]
        sample["sham_mean"] = np.mean([sample[s] for s in sham], axis=0)
        point["sham_mean"] = float(np.mean([point[s] for s in sham]))
        for comparator in cfg["comparators"]+sham:
            confidence = cfg["primary_confidence"] if role == "D_rel" and pair == cfg["primary_pair"] and scope == cfg["primary_scope"] and comparator in cfg["comparators"] else cfg["secondary_confidence"]
            differences = sample["history"]-sample[comparator]
            finite = differences[np.isfinite(differences)]
            lo, hi = np.quantile(finite, [(1-confidence)/2, 1-(1-confidence)/2]) if len(finite) else (np.nan, np.nan)
            result.append(dict(pair=pair, role=role, scope=scope, method="history", comparator=comparator,
                metric="error_recovery", estimator="pooled_image_mass_ratio", delta=point["history"]-point[comparator],
                confidence=confidence, ci_low=float(lo), ci_high=float(hi), valid_replicates=len(finite), image_pool=n,
                sufficient_bootstrap=len(finite) >= .99*cfg["bootstrap_replicates"]))
    return result


def screen_gate(ranking, intervals, spec):
    if any(r["role"] != "D_rel" for r in ranking+intervals):
        raise ValueError("Validation cannot enter the screen gate")
    checks = []; cfg = spec["screen"]; late = cfg["primary_pair"]
    middle = spec["pairs"][0]["name"]
    def add(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))
    def find(pair, scope, comparator):
        return next(r for r in intervals if r["pair"] == pair and r["scope"] == scope and r["comparator"] == comparator)
    for comparator in cfg["comparators"]:
        row = find(late, cfg["primary_scope"], comparator)
        add("late_positive_interval_"+comparator, row["ci_low"] > 0 and row["sufficient_bootstrap"], **row)
        row = find(middle, cfg["primary_scope"], comparator)
        add("middle_positive_point_"+comparator, row["delta"] > 0, delta=row["delta"])
    row = find(late, cfg["primary_scope"], "now")
    add("late_minimum_practical_gain", row["delta"] >= cfg["min_late_gain_over_now"], delta=row["delta"])
    for seed in spec["permutation"]["seeds"]:
        row = find(late, cfg["primary_scope"], f"sham{seed}")
        add(f"late_point_beats_sham{seed}", row["delta"] > 0, delta=row["delta"])
    for pair in (middle, late):
        row = find(pair, "all", "now")
        add(pair+"_all_recovery_nonnegative", row["delta"] >= 0, delta=row["delta"])
        support_row = next(r for r in ranking if r["pair"] == pair and r["scope"] == cfg["primary_scope"] and r["method"] == "history")
        add(pair+"_support", support_row["adequate_support"],
            **{k:support_row[k] for k in ("error_images", "correct_images", "error_pixels", "correct_pixels")})
    passed = all(r["passed"] for r in checks)
    return dict(passed=passed, checks=checks, decision="eligible_for_registered_low_dimensional_probes" if passed else cfg["failure_decision"],
                new_probe_fitting_allowed=passed, teacher_training_allowed=False, validation_used=False,
                interpretation=cfg["failure_scope"])
