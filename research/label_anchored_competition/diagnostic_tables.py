"""Descriptive conditional tables and image-bootstrap calibration intervals."""
import numpy as np

from .ranking_metrics import Ranking


def support(pool, selected, limits):
    errors = selected & pool["error"].astype(bool)
    correct = selected & ~pool["error"].astype(bool)
    record = dict(sampled_pixels=int(selected.sum()), sampled_errors=int(errors.sum()),
                  sampled_correct=int(correct.sum()), images=len(np.unique(pool["image"][selected])),
                  error_images=len(np.unique(pool["image"][errors])), correct_images=len(np.unique(pool["image"][correct])))
    record["adequate_support"] = bool(record["error_images"] >= limits["images_with_errors"]
        and record["correct_images"] >= limits["images_with_correct_pixels"]
        and record["sampled_errors"] >= limits["sampled_error_pixels"]
        and record["sampled_correct"] >= limits["sampled_correct_pixels"])
    return record


def conditional_rows(pool, risks, limits):
    selected_methods = [name for name in ("raw_MC", "raw_entropy", "raw_margin", "raw_prototype_margin", "raw_product", "raw_uniform", "raw_e",
                                         "h_MC", "h_prototype_margin", "h_product", "h_uniform", "h_e", "F_base", "F_plus") if name in risks]
    masks = []
    for label, values, edges in [("MC", pool["mc"], [0, .5, .8, .9, .95, .975, .995, 1.000001]),
                                 ("prototype_margin", pool["prototype_margin"], [-2.000001, -.25, -.1, -.05, 0, .05, .1, .25, 2.000001])]:
        for lo, hi in zip(edges[:-1], edges[1:]):
            masks.append((f"{label}[{lo:g},{hi:g})", (values >= lo) & (values < hi)))
    masks.extend((f"predicted_class={c}", pool["pseudo_label"] == c) for c in range(pool["logits"].shape[1]))
    masks.extend((name, mask) for name, mask in [("boundary", pool["boundary"]), ("interior", ~pool["boundary"])])
    for c in range(pool["logits"].shape[1]):
        for lo, hi in zip([0, .9, .95, .995], [.9, .95, .995, 1.000001]):
            for region in (False, True):
                masks.append((f"class={c};MC[{lo:g},{hi:g});boundary={region}",
                              (pool["pseudo_label"] == c) & (pool["mc"] >= lo) & (pool["mc"] < hi) & (pool["boundary"] == region)))
    output = []
    for name, mask in masks:
        info = support(pool, mask, limits)
        if not mask.any():
            output.append(dict(stratum=name, scope="descriptive, no stratum-wise significance claim", **info))
            continue
        for method in selected_methods:
            output.append(dict(stratum=name, method=method, scope="descriptive; broad-scope CIs reported separately", **info,
                               **Ranking(pool["error"][mask], risks[method][mask]).evaluate(pool["sampling_weight"][mask])))
    return output


def calibration_intervals(pool, probabilities, selected, replicates, seed):
    """Linear losses bootstrap cheaply by image sufficient statistics."""
    n_images = len(pool["image_ids"])
    rng = np.random.default_rng(seed)
    counts = np.asarray([np.bincount(rng.integers(n_images, size=n_images), minlength=n_images)
                         for _ in range(replicates)], dtype=np.float64)
    image, weight = pool["image"][selected], pool["sampling_weight"][selected]
    correct = 1-pool["error"][selected]
    denominator = counts@np.bincount(image, weights=weight, minlength=n_images)
    rows, samples = [], {}
    for method, probability in probabilities.items():
        p = np.clip(probability[selected].astype(np.float64), 1e-7, 1-1e-7)
        losses = np.stack(((p-correct)**2, -(correct*np.log(p)+(1-correct)*np.log1p(-p))), 1)
        sufficient = np.stack([np.bincount(image, weights=weight*losses[:, j], minlength=n_images) for j in range(2)], 1)
        samples[method] = (counts@sufficient)/denominator[:, None]
        for j, metric in enumerate(("brier", "nll")):
            values = samples[method][:, j]
            finite = values[np.isfinite(values)]
            lo, hi = np.quantile(finite, [.025, .975])
            rows.append(dict(method=method, metric=metric, ci_low=float(lo), ci_high=float(hi),
                             replicates=len(finite), unit="paired full-pool images"))
    if "F_base" in samples and "F_plus" in samples:
        for j, metric in enumerate(("brier", "nll")):
            values = samples["F_plus"][:, j]-samples["F_base"][:, j]
            finite = values[np.isfinite(values)]
            lo, hi = np.quantile(finite, [.025, .975])
            rows.append(dict(method="F_plus", metric=metric, delta_vs="F_base", delta_ci_low=float(lo),
                             delta_ci_high=float(hi), replicates=len(finite), unit="paired full-pool images"))
    return rows
