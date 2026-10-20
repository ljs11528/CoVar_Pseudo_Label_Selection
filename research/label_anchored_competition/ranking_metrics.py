"""Weighted ranking metrics. Errors are positive; ties are treated as groups."""
from __future__ import annotations

import numpy as np


class Ranking:
    """Sort once; reuse the ordering for image-bootstrap replicates."""

    def __init__(self, error, risk):
        error, risk = np.asarray(error), np.asarray(risk, dtype=float)
        if error.ndim != 1 or risk.shape != error.shape or not len(error):
            raise ValueError("Expected nonempty aligned 1D arrays")
        if not np.isin(error, [0, 1]).all() or np.isnan(risk).any():
            raise ValueError("Binary errors and non-NaN risks required")
        self.order = np.argsort(risk, kind="stable")
        self.error = error[self.order].astype(float)
        self.risk = risk[self.order]
        self.starts = np.r_[0, np.flatnonzero(self.risk[1:] != self.risk[:-1]) + 1]

    def groups(self, weight):
        weight = np.asarray(weight, dtype=float)
        if weight.shape != self.order.shape or not np.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("Invalid sample weights")
        weight = weight[self.order]
        mass = np.add.reduceat(weight, self.starts)
        errors = np.add.reduceat(weight * self.error, self.starts)
        keep = mass > 0
        if not keep.any():
            raise ValueError("No positive sample weight")
        return mass[keep], errors[keep], self.risk[self.starts][keep]

    def evaluate(self, weight, coverages=(0.5, 0.7, 0.9, 0.95)):
        mass, errors, _ = self.groups(weight)
        negatives = mass - errors
        total, positive = mass.sum(), errors.sum()
        negative = negatives.sum()
        negatives_before = np.cumsum(negatives) - negatives
        auroc = (np.sum(errors * (negatives_before + 0.5 * negatives)) / (positive * negative)
                 if positive > 0 and negative > 0 else np.nan)
        high_mass = np.cumsum(mass[::-1])
        high_errors = np.cumsum(errors[::-1])
        ap = (np.sum((errors[::-1] / positive) * (high_errors / high_mass))
              if positive > 0 else np.nan)
        cum_mass, cum_errors = np.cumsum(mass), np.cumsum(errors)
        previous_mass = np.r_[0.0, cum_mass[:-1]]
        previous_errors = np.r_[0.0, cum_errors[:-1]]
        rate = errors / mass
        # Integral of risk over coverage, assuming random fractional selection within ties.
        integral = rate * mass
        nonzero = previous_mass > 0
        integral[nonzero] += ((previous_errors[nonzero] - rate[nonzero] * previous_mass[nonzero])
                              * np.log1p(mass[nonzero] / previous_mass[nonzero]))
        result = {"auroc": float(auroc), "error_ap": float(ap),
                  "error_prevalence": float(positive / total),
                  "aurc_continuous": float(integral.sum() / total)}
        for coverage in coverages:
            if not 0 < coverage <= 1:
                raise ValueError("Coverage must lie in (0,1]")
            selected = coverage * total
            idx = min(int(np.searchsorted(cum_mass, selected, side="left")), len(mass) - 1)
            selected_errors = previous_errors[idx] + rate[idx] * (selected - previous_mass[idx])
            result[f"precision_at_{coverage:g}"] = float(1.0 - selected_errors / selected)
        return result

    def curve(self, weight):
        mass, errors, thresholds = self.groups(weight)
        selected = np.cumsum(mass)
        risk = np.cumsum(errors) / selected
        return selected / selected[-1], risk, thresholds

    def threshold_at_precision(self, weight, precision):
        if not 0 < precision <= 1:
            raise ValueError("Precision must lie in (0,1]")
        coverage, risk, thresholds = self.curve(weight)
        eligible = np.flatnonzero(1.0 - risk >= precision)
        if not len(eligible):
            return None
        i = eligible[-1]
        return {"threshold": float(thresholds[i]), "dev_coverage": float(coverage[i]),
                "dev_precision": float(1.0 - risk[i])}


def selection_summary(error, selected, weight):
    error, selected, weight = map(np.asarray, (error, selected, weight))
    chosen = weight * selected
    total, count, wrong = weight.sum(), chosen.sum(), (chosen * error).sum()
    return {"estimated_pixels": float(count), "coverage": float(count / total) if total else np.nan,
            "precision": float(1.0 - wrong / count) if count else np.nan,
            "estimated_errors": float(wrong)}


def paired_image_bootstrap(error, risks, weight, image_index, replicates=1000, seed=0,
                           baseline="MC", coverages=(0.5, 0.7, 0.9, 0.95), comparators=()):
    """Same resampled images for every score; pixels are never bootstrap units."""
    if baseline not in risks or replicates < 1:
        raise ValueError("Baseline missing or invalid bootstrap count")
    image_index = np.asarray(image_index)
    _, group = np.unique(image_index, return_inverse=True)
    n_images = int(group.max()) + 1
    if n_images < 2:
        raise ValueError("Need at least two images for an image bootstrap")
    weight = np.asarray(weight, dtype=float)
    rankings = {name: Ranking(error, risk) for name, risk in risks.items()}
    samples = {name: [] for name in risks}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        counts = np.bincount(rng.integers(n_images, size=n_images), minlength=n_images)
        resampled_weight = weight * counts[group]
        for name, ranking in rankings.items():
            samples[name].append(ranking.evaluate(resampled_weight, coverages))
    keys = list(next(iter(samples.values()))[0])
    result = []
    for name in risks:
        for metric in keys:
            values = np.asarray([row[metric] for row in samples[name]])
            valid = np.isfinite(values)
            def interval(x):
                return np.quantile(x, [0.025, 0.975]).tolist() if len(x) else [np.nan, np.nan]
            low, high = interval(values[valid])
            for comparator in dict.fromkeys((baseline,) + tuple(comparators)):
                if comparator not in samples:
                    raise ValueError("Bootstrap comparator missing: " + comparator)
                reference = np.asarray([row[metric] for row in samples[comparator]])
                delta = values - reference
                paired_valid = np.isfinite(delta)
                delta_low, delta_high = interval(delta[paired_valid])
                result.append(dict(score=name, metric=metric, ci_low=low, ci_high=high,
                                   delta_vs=comparator, delta_ci_low=delta_low, delta_ci_high=delta_high,
                                   valid_replicates=int(valid.sum()), paired_replicates=int(paired_valid.sum()),
                                   images=n_images))
    return result
