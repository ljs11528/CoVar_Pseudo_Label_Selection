"""Numerical primitives for the preregistered label-anchored verification study.

These primitives do not read ground truth from the unlabeled/evaluation pool.
The caller owns image-role isolation and the recorded protocol/feature identity.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Nonfinite {name}")


def _score_inputs(logits, similarities):
    if logits.ndim != 2 or logits.shape[1] < 2 or similarities.shape != logits.shape:
        raise ValueError("Expected equally shaped [pixels, classes>=2] tensors")
    if logits.device != similarities.device:
        raise ValueError("Inputs must share a device")
    # No residual normalization by 1 - FP32 maximum confidence.
    logits, similarities = logits.float(), similarities.float()
    _finite(logits, "logits")
    _finite(similarities, "prototype similarities")
    if bool((similarities.abs() > 1.00001).any()):
        raise ValueError("Cosine similarities are outside [-1,1]")
    return logits, similarities.clamp(-1, 1)


def score_prediction(logits, similarities, temperature=0.1):
    """Return risk signals and observables for the original classifier label.

    `strict_conflict` is a strictly better residual prototype. Argmax disagreement
    is recorded separately because exact prototype ties need not imply e > 0.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Prototype temperature must be finite and positive")
    logits, similarities = _score_inputs(logits, similarities)
    n, classes = logits.shape
    a = logits.argmax(dim=1)
    other = ~F.one_hot(a, classes).bool()
    z_a = logits.gather(1, a[:, None]).squeeze(1)
    z_other = logits[other].reshape(n, classes-1)
    s_a = similarities.gather(1, a[:, None]).squeeze(1)
    s_other = similarities[other].reshape(n, classes-1)
    difference = s_other-s_a[:, None]
    counterevidence = difference.clamp_min(0)
    residual_logp = F.log_softmax(z_other, dim=1)
    residual = residual_logp.exp()
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    logq = F.log_softmax(similarities/temperature, dim=1)
    q = logq.exp()
    logmean = torch.logaddexp(logp, logq)-math.log(2)
    joint = logits+similarities/temperature
    joint_a = joint.gather(1, a[:, None]).squeeze(1)
    joint_other = joint[other].reshape(n, classes-1)
    top_values = p.topk(2, dim=1).values
    result = dict(
        pseudo_label=a,
        mc=logp.gather(1, a[:, None]).squeeze(1).exp(),
        logit_mc=z_a-torch.logsumexp(z_other, dim=1),
        margin=top_values[:, 0]-top_values[:, 1],
        entropy=-(p*logp).sum(dim=1),
        residual_entropy=-(residual*residual_logp).sum(dim=1),
        prototype_support=s_a,
        prototype_margin=s_a-s_other.max(dim=1).values,
        prototype_label=similarities.argmax(dim=1),
        prototype_disagreement=(similarities.argmax(dim=1) != a),
        strict_conflict=(difference.max(dim=1).values > 0),
        probability_prototype_JS=0.5*((p*(logp-logmean)).sum(dim=1)
                                     +(q*(logq-logmean)).sum(dim=1)),
        product_support_original_label=F.log_softmax(joint, dim=1).gather(1, a[:, None]).squeeze(1).exp(),
        logit_product_support=joint_a-torch.logsumexp(joint_other, dim=1),
        uniform_competitor=counterevidence.mean(dim=1),
        top2_competitor=counterevidence.gather(1, z_other.argmax(dim=1)[:, None]).squeeze(1),
        competitor_evidence=(residual*counterevidence).sum(dim=1),
        signed_competitor=(residual*difference).sum(dim=1),
        residual_probability=residual,
        residual_counterevidence=counterevidence,
        residual_probability_variance=p[other].reshape(n, classes-1).var(dim=1, unbiased=False),
    )
    for name, value in result.items():
        _finite(value, name)
    return result


def pixel_identity_keys(image_id, pixel_indices, device=None):
    """Stable signed 64-bit keys, independent of the batch/chunk composition."""
    image_key = int.from_bytes(hashlib.sha256(image_id.encode()).digest()[:8], "little", signed=True)
    indices = torch.as_tensor(pixel_indices, dtype=torch.int64, device=device)
    if bool((indices < 0).any()):
        raise ValueError("Pixel indices must be nonnegative")
    return indices ^ image_key


def _logical_rshift(value, shift):
    return (value >> shift) & ((1 << (64-shift))-1)


def residual_permutation(pixel_keys, classes_minus_one, seed):
    """Deterministic SplitMix64 random-key permutations of residual slots.

    Integer arithmetic wraps modulo 2^64 on CPU/CUDA. Each pixel is keyed by
    identity; no global RNG or batch-dependent random state is consumed.
    """
    if pixel_keys.ndim != 1 or pixel_keys.dtype != torch.int64 or classes_minus_one < 1:
        raise ValueError("Expected int64 pixel keys and a nonempty residual space")
    seed_key = int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "little", signed=True)
    slot = torch.arange(classes_minus_one, device=pixel_keys.device, dtype=torch.int64)
    value = pixel_keys[:, None]+(slot+1)*(-7046029254386353131)+seed_key
    value = (value ^ _logical_rshift(value, 30))*(-4658895280553007687)
    value = (value ^ _logical_rshift(value, 27))*(-7723592293110705685)
    value = value ^ _logical_rshift(value, 31)
    return value.argsort(dim=1, stable=True)


def permuted_evidence(residual_probability, counterevidence, pixel_keys, seed):
    if residual_probability.ndim != 2 or counterevidence.shape != residual_probability.shape:
        raise ValueError("Residual probabilities and evidence must have equal shapes")
    if len(pixel_keys) != len(residual_probability):
        raise ValueError("One identity key is required per pixel")
    permutation = residual_permutation(pixel_keys, residual_probability.shape[1], seed)
    return (residual_probability.gather(1, permutation)*counterevidence).sum(dim=1)


def sample_decoder_features(feature_map, yx, original_size):
    """Sample one image's native pre-classifier feature map at original pixels."""
    if feature_map.ndim != 4 or feature_map.shape[0] != 1 or yx.ndim != 2 or yx.shape[1] != 2:
        raise ValueError("Expected [1,D,Hf,Wf] features and [N,2] pixel coordinates")
    h, w = original_size
    if h < 1 or w < 1 or bool(((yx < 0) | (yx >= yx.new_tensor([h, w]))).any()):
        raise ValueError("Pixel coordinates outside the original image")
    coordinates = yx.to(device=feature_map.device, dtype=torch.float32)
    xy = coordinates[:, [1, 0]]
    grid = 2*xy/xy.new_tensor([max(w-1, 1), max(h-1, 1)])-1
    values = F.grid_sample(feature_map.float(), grid[None, :, None, :],
                           mode="bilinear", padding_mode="border", align_corners=True)
    values = values[0, :, :, 0].T
    if bool((values.norm(dim=1) <= 1e-12).any()):
        raise ValueError("Zero-norm sampled decoder feature")
    return F.normalize(values, dim=1)


class PrototypeAccumulator:
    """Equal-image class prototypes with explicit image-role restrictions."""

    def __init__(self, classes, dimension, allowed_image_ids, device="cpu"):
        self.classes = classes
        self.dimension = dimension
        self.allowed_ids = frozenset(allowed_image_ids)
        self.seen_ids = set()
        self.sums = torch.zeros(classes, dimension, dtype=torch.float64, device=device)
        self.image_counts = torch.zeros(classes, dtype=torch.int64, device=device)
        self.sampled_pixel_counts = torch.zeros_like(self.image_counts)

    @torch.no_grad()
    def add_image(self, image_id, features, labels):
        if image_id not in self.allowed_ids:
            raise ValueError("Prototype image is not in D_sup")
        if image_id in self.seen_ids:
            raise ValueError("Image already contributed to the prototype bank")
        if features.ndim != 2 or features.shape != (len(labels), self.dimension):
            raise ValueError("Feature and label shape mismatch")
        if labels.dtype != torch.int64 or bool(((labels < 0) | (labels >= self.classes)).any()):
            raise ValueError("Only valid supervised class labels may enter prototypes")
        _finite(features, "prototype features")
        if bool((features.norm(dim=1) <= 1e-12).any()):
            raise ValueError("Zero-norm prototype feature")
        features = F.normalize(features.detach().double(), dim=1).to(self.sums.device)
        labels = labels.to(self.sums.device)
        for c in labels.unique().tolist():
            selected = features[labels == c]
            self.sums[c] += selected.mean(dim=0)
            self.image_counts[c] += 1
            self.sampled_pixel_counts[c] += len(selected)
        self.seen_ids.add(image_id)

    @torch.no_grad()
    def finalize(self):
        if bool((self.image_counts == 0).any()):
            raise ValueError("At least one class has no D_sup prototype support")
        means = self.sums/self.image_counts[:, None]
        if bool((means.norm(dim=1) <= 1e-12).any()):
            raise ValueError("Degenerate class prototype mean")
        return F.normalize(means, dim=1).float()


class CorrectnessHead(nn.Module):
    """Low-capacity head; features are always detached from the main model."""

    def __init__(self, input_dimension=2, hidden=16):
        super().__init__()
        self.register_buffer("input_mean", torch.zeros(input_dimension))
        self.register_buffer("input_scale", torch.ones(input_dimension))
        self.network = nn.Sequential(nn.Linear(input_dimension, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    @torch.no_grad()
    def fit_standardization(self, features, weights):
        features, weights = features.detach().float(), weights.detach().float()
        _finite(features, "head inputs")
        _finite(weights, "head sampling weights")
        if features.ndim != 2 or weights.shape != (len(features),) or bool((weights <= 0).any()) or not len(features):
            raise ValueError("Expected nonempty features and positive sampling weights")
        normalized = weights/weights.sum()
        mean = (features*normalized[:, None]).sum(dim=0)
        variance = ((features-mean).square()*normalized[:, None]).sum(dim=0)
        self.input_mean.copy_(mean)
        self.input_scale.copy_(torch.where(variance > 1e-12, variance.sqrt(), torch.ones_like(variance)))

    def forward(self, features):
        standardized = (features.detach().float()-self.input_mean)/self.input_scale
        return self.network(standardized).squeeze(-1)

    @torch.no_grad()
    def weights(self, features):
        return self(features).sigmoid().detach()


def weighted_consistency_loss(logits, pseudo_labels, weights, valid):
    """Normalize by all valid pixels, preserving the preregistered loss mass."""
    if weights.shape != pseudo_labels.shape or valid.shape != weights.shape or valid.dtype != torch.bool:
        raise ValueError("Pseudo-label, weight and valid-mask shapes must match")
    _finite(weights, "unsupervised weights")
    if bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("Weights must be within [0,1]")
    targets = pseudo_labels.detach().clone()
    targets[~valid] = 255
    per_pixel = F.cross_entropy(logits, targets, ignore_index=255, reduction="none")
    return (per_pixel*weights.detach()*valid).sum()/valid.sum().clamp_min(1)
