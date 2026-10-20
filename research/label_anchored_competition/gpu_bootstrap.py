"""Exact, tie-aware image bootstrap using FP64 GPU reductions.

Only image multiplicities are resampled. Score ordering and tie groups are fixed;
no histogram approximation, pixel bootstrap, or refitting on the test set.
"""
from __future__ import annotations

import numpy as np
import torch

from .ranking_metrics import Ranking


KEYS = ("auroc", "error_ap", "error_prevalence", "aurc_continuous")


class GPURanking:
    def __init__(self, error, risk, weight, image_index, device="cuda:0"):
        ranking = Ranking(error, risk)
        self.device = device
        self.error = torch.as_tensor(ranking.error, dtype=torch.float64, device=device)
        self.weight = torch.as_tensor(np.asarray(weight)[ranking.order], dtype=torch.float64, device=device)
        self.image = torch.as_tensor(np.asarray(image_index)[ranking.order], dtype=torch.long, device=device)
        groups = np.repeat(np.arange(len(ranking.starts)), np.diff(np.r_[ranking.starts, len(error)]))
        self.group = torch.as_tensor(groups, dtype=torch.long, device=device)
        self.groups = len(ranking.starts)

    @torch.no_grad()
    def evaluate_counts(self, counts, coverages=(.5, .7, .9, .95)):
        count = torch.as_tensor(counts, dtype=torch.float64, device=self.device)
        if count.ndim == 1:
            count = count.unsqueeze(0)
        weight = count[:, self.image] * self.weight
        group = self.group.expand(len(count), -1)
        mass = weight.new_zeros((len(count), self.groups)).scatter_add_(1, group, weight)
        errors = weight.new_zeros(mass.shape).scatter_add_(1, group, weight * self.error)
        cum_mass, cum_error = mass.cumsum(1), errors.cumsum(1)
        total, positive = cum_mass[:, -1], cum_error[:, -1]
        if (total <= 0).any():
            raise ValueError("Bootstrap replicate has no positive mass")
        negative = total - positive
        negatives = mass - errors
        roc = (errors * (cum_mass - cum_error - .5 * negatives)).sum(1) / (positive * negative)
        roc = torch.where((positive > 0) & (negative > 0), roc, torch.full_like(roc, float("nan")))
        high_mass = total[:, None] - cum_mass + mass
        high_error = positive[:, None] - cum_error + errors
        ap = (errors * high_error / high_mass.clamp_min(1e-100)).sum(1) / positive
        ap = torch.where(positive > 0, ap, torch.full_like(ap, float("nan")))
        previous_mass, previous_error = cum_mass - mass, cum_error - errors
        rate = errors / mass.clamp_min(1e-100)
        ratio = mass / torch.where(previous_mass > 0, previous_mass, torch.ones_like(previous_mass))
        correction = (previous_error - rate * previous_mass) * torch.log1p(ratio)
        integral = rate * mass + torch.where(previous_mass > 0, correction, torch.zeros_like(correction))
        columns = [roc, ap, positive / total, integral.sum(1) / total]
        for coverage in coverages:
            selected = (coverage * total).unsqueeze(1)
            index = torch.searchsorted(cum_mass.contiguous(), selected.contiguous(), right=False)
            index = index.clamp_max(self.groups - 1)
            selected_errors = (previous_error.gather(1, index)
                               + rate.gather(1, index) * (selected - previous_mass.gather(1, index)))
            columns.append((1 - selected_errors / selected).squeeze(1))
        return torch.stack(columns, 1).cpu().numpy()


def paired_image_bootstrap_gpu(error, risks, weight, image_index, replicates=1000, seed=0,
                               baseline="MC", coverages=(.5, .7, .9, .95), comparators=(),
                               device="cuda:0", return_samples=False, all_images=None):
    if baseline not in risks or replicates < 1:
        raise ValueError("Missing baseline or invalid bootstrap count")
    if all_images is None:
        _, groups = np.unique(image_index, return_inverse=True)
        n_images = int(groups.max()) + 1
    else:
        groups = np.asarray(image_index, dtype=np.int64)
        n_images = int(all_images)
        if (groups < 0).any() or (groups >= n_images).any():
            raise ValueError("Image indices outside the full evaluation pool")
    if n_images < 2:
        raise ValueError("Image bootstrap needs at least two images")
    rng = np.random.default_rng(seed)
    counts = np.asarray([np.bincount(rng.integers(n_images, size=n_images), minlength=n_images)
                         for _ in range(replicates)])
    batch_size = max(1, min(4, 8000000 // len(error)))
    samples = {}
    for name, risk in risks.items():
        ranking = GPURanking(error, risk, weight, groups, device)
        values = [ranking.evaluate_counts(counts[start:start+batch_size], coverages)
                  for start in range(0, replicates, batch_size)]
        samples[name] = np.concatenate(values)
        del ranking
        print(f"Image bootstrap complete: {name}, {replicates} replicates, {n_images} images", flush=True)
    keys = KEYS + tuple(f"precision_at_{value:g}" for value in coverages)
    result = []
    def interval(values):
        valid = values[np.isfinite(values)]
        return np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
    for name, values in samples.items():
        for j, metric in enumerate(keys):
            low, high = interval(values[:, j])
            for comparator in dict.fromkeys((baseline,) + tuple(comparators)):
                difference = values[:, j] - samples[comparator][:, j]
                dlow, dhigh = interval(difference)
                result.append(dict(score=name, metric=metric, ci_low=float(low), ci_high=float(high),
                                   delta_vs=comparator, delta_ci_low=float(dlow), delta_ci_high=float(dhigh),
                                   valid_replicates=int(np.isfinite(values[:, j]).sum()),
                                   paired_replicates=int(np.isfinite(difference).sum()), images=n_images))
    return (result, samples) if return_samples else result
