"""Pinned native selectors and one-component interventions on frozen predictions."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parent
VARIANTS = (
    "covar_actual", "csl_actual", "csl_alpha2", "covar_alpha8",
    "covar_rcv_only", "covar_unit_only", "covar_embedding_only",
    "covar_centroid_only", "paper_reconstructed", "csl_high_residual_control",
)


def load_native(name):
    folder = ROOT / "vendor" / ("csl_5cac455c" if name == "csl" else "covar_181f147")
    protocol = json.loads((ROOT / "legacy_sources.json").read_text())
    for filename in ("PCOS.py", "semi_supervised_train.py"):
        path = folder / filename
        expected = protocol["vendor_files"][str(path.relative_to(ROOT))]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned source changed: " + str(path))
    spec = importlib.util.spec_from_file_location("validation_native_" + name, folder / "PCOS.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = folder / "semi_supervised_train.py"
    tree = ast.parse(source.read_text())
    method = next(node for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "SemiModule"
                  for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_weight")
    namespace = {"torch": torch, "batch_class_stats": module.batch_class_stats,
                 "get_max_confidence_and_residual_variance": module.get_max_confidence_and_residual_variance}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["get_weight"], module


class Selectors:
    def __init__(self):
        self.covar, self.covar_module = load_native("covar")
        self.csl, self.csl_module = load_native("csl")
        self.covar_self = SimpleNamespace(threshold_strategy="dynamic", g_alpha=1.0,
                                         select_mode="linear", select_lam=.25)

    @torch.no_grad()
    def native(self, probabilities, name="covar", alpha=None):
        ignore = torch.zeros(probabilities.shape[0:1] + probabilities.shape[2:],
                             dtype=torch.long, device=probabilities.device)
        if name == "csl":
            return self.csl(SimpleNamespace(alpha=8.0 if alpha is None else alpha), probabilities, ignore)
        return self.covar(self.covar_self, probabilities, ignore, probabilities.shape[1],
                          alpha=2.0 if alpha is None else alpha)

    @staticmethod
    @torch.no_grad()
    def statistics(probabilities):
        if probabilities.ndim != 4 or probabilities.shape[0] != 1:
            raise ValueError("Interventions require one complete image [1,K,H,W]")
        p = probabilities[0]
        k = p.shape[0]
        m = k - 1
        mc, top = p.max(0)
        nonmax = torch.ones_like(p).scatter_(0, top.unsqueeze(0), 0)
        residual = p * nonmax
        mass = residual.sum(0)
        mu = mass / m
        squared = (residual - mu.unsqueeze(0)) ** 2
        code_v = squared.sum(0) / m
        paper_v = (squared * nonmax).sum(0) / m
        g = m * m / (2 * (1 - mc + 1e-8))
        shape = residual / mass.clamp_min(torch.finfo(p.dtype).tiny).unsqueeze(0)
        concentration = (m * shape.square().sum(0) - 1).clamp_min(0)
        return {"mc": mc, "code_v": code_v, "paper_v": paper_v,
                "code_weighted": g * code_v, "paper_weighted_epsilon": g * paper_v,
                "paper_weighted_stable": .5 * mass * concentration}

    @staticmethod
    @torch.no_grad()
    def spectral(features, reliability, direction=(1, -1), unit="or", alpha=2.0,
                 allow_empty=True, score_coefficients=None):
        shape = features.shape[:-1]
        x = features.reshape(-1, 2)
        reliability = reliability.reshape(-1)
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite features in selector")
        _, singular, vt = torch.linalg.svd(x.T, full_matrices=False)
        order = torch.argsort(-(singular ** 2))
        assignment = vt.T[:, order[:2]].abs().argmax(1)
        means, variances, scores, counts = [], [], [], []
        for group in range(2):
            keep = assignment == group
            points = x[keep]
            count = len(points)
            counts.append(count)
            means.append(points.mean(0) if count else x.new_zeros(2))
            variances.append(points.var(0, unbiased=True) if count > 1 else x.new_zeros(2))
            if count:
                score = ((means[-1] * x.new_tensor(score_coefficients)).sum()
                         if score_coefficients is not None else reliability[keep].mean())
            else:
                score = x.new_tensor(0 if allow_empty else -float("inf"))
            scores.append(score)
        chosen = torch.stack(scores).argmax()
        mean, variance = torch.stack(means)[chosen], torch.stack(variances)[chosen]
        z = (x - mean) / torch.sqrt(variance + 1e-8)
        z = z * x.new_tensor(direction)
        weight = torch.exp(-z[:, 0].square() / alpha) * torch.exp(-z[:, 1].square() / alpha)
        if unit == "or":
            selected = (z[:, 0] > 0) | (z[:, 1] > 0)
        elif unit == "and_member":
            selected = (z[:, 0] > 0) & (z[:, 1] > 0) & (assignment == chosen)
        else:
            raise ValueError(unit)
        weight = torch.where(selected, torch.ones_like(weight), weight)
        info = {"cluster_counts": counts, "chosen_cluster": int(chosen),
                "chosen_empty": counts[int(chosen)] == 0,
                "zero_variance_dimensions": int((variance == 0).sum()),
                "singular_values": singular.detach().cpu().tolist(),
                "unit_coverage": float((weight == 1).float().mean())}
        return weight.reshape(shape), info

    @torch.no_grad()
    def intervention(self, probabilities, variant, scale=1.0):
        s = self.statistics(probabilities)
        a, t = s["mc"], s["code_weighted"]
        x = torch.stack((a, t), -1)
        reliability = a - .25 * t
        direction, unit, alpha, allow_empty = (1, -1), "or", 2.0, True
        if variant == "covar_rcv_only":
            t = s["paper_weighted_epsilon"]
            x = torch.stack((a, t), -1)
            reliability = a - .25 * t
        elif variant == "covar_unit_only":
            unit = "and_member"
        elif variant == "covar_embedding_only":
            x = torch.stack((a.clamp_min(torch.finfo(a.dtype).tiny).log(), -t), -1)
            direction = (1, 1)
        elif variant == "covar_centroid_only":
            reliability = a.clamp_min(torch.finfo(a.dtype).tiny).log() - t
        elif variant == "paper_reconstructed":
            t = s["paper_weighted_stable"]
            x = torch.stack((a.clamp_min(torch.finfo(a.dtype).tiny).log(), -t), -1)
            reliability = x.sum(-1)
            direction, unit, allow_empty = (1, 1), "and_member", False
        elif variant in ("csl_reconstructed", "csl_high_residual_control"):
            x = torch.stack((a, s["code_v"]), -1)
            reliability, alpha = a, 8.0
            if variant == "csl_high_residual_control":
                direction = (1, 1)
        elif variant != "covar_reconstructed":
            raise ValueError(variant)
        # Scale sensitivity changes the SVD/weight coordinates, while reliability
        # scoring remains in its original units. This is a labeled intervention.
        coefficient = None
        if variant in ("covar_reconstructed", "covar_rcv_only", "covar_unit_only"):
            coefficient = (1., -.25 / scale)
        elif variant in ("csl_reconstructed", "csl_high_residual_control"):
            coefficient = (1., 0.)
        x = x * x.new_tensor((1.0, scale))
        return self.spectral(x, reliability, direction, unit, alpha, allow_empty, coefficient)

    @torch.no_grad()
    def all(self, probabilities):
        weights = {"covar_actual": self.native(probabilities)[0],
                   "csl_actual": self.native(probabilities, "csl")[0],
                   "csl_alpha2": self.native(probabilities, "csl", 2.0)[0],
                   "covar_alpha8": self.native(probabilities, "covar", 8.0)[0]}
        details = {}
        for name in VARIANTS[4:]:
            weights[name], details[name] = self.intervention(probabilities, name)
        return weights, details
