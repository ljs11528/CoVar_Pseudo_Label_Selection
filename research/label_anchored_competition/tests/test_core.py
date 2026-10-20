import itertools
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.label_anchored_competition.core import (
    CorrectnessHead, PrototypeAccumulator, permuted_evidence, pixel_identity_keys,
    residual_permutation, sample_decoder_features, score_prediction,
    weighted_consistency_loss,
)
from research.label_anchored_competition.prepare_data import disjoint_partitions, partition


class CompetitorInvariants(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)

    def test_numpy_reference_and_original_pseudo_label(self):
        z = np.array([[2., 1., -2., 0.], [-2., 1., .5, -1.]])
        s = np.array([[.1, .5, -.1, .8], [.9, .4, .8, .2]])
        output = score_prediction(torch.tensor(z), torch.tensor(s), temperature=.2)
        for i in range(len(z)):
            a = np.argmax(z[i])
            residual_z = np.delete(z[i], a)
            r = np.exp(residual_z-residual_z.max())
            r /= r.sum()
            c = np.maximum(np.delete(s[i], a)-s[i, a], 0)
            self.assertAlmostEqual(output["competitor_evidence"][i].item(), float(r@c), places=6)
            joint = z[i]+s[i]/.2
            distribution = np.exp(joint-joint.max())
            distribution /= distribution.sum()
            self.assertAlmostEqual(output["product_support_original_label"][i].item(), distribution[a], places=6)
        self.assertTrue(torch.equal(output["pseudo_label"], torch.tensor(z.argmax(axis=1))))

    def test_extreme_confidence_does_not_divide_by_zero(self):
        z = torch.tensor([[1000., 2., 1.], [10000., -10., -20.]])
        s = torch.tensor([[.1, .9, -.1], [.3, .7, .2]])
        output = score_prediction(z, s)
        self.assertTrue(torch.equal(output["mc"], torch.ones(2)))
        self.assertTrue(torch.isfinite(output["competitor_evidence"]).all())
        self.assertTrue(torch.all(output["competitor_evidence"] > 0))
        self.assertTrue(torch.allclose(output["residual_probability"].sum(1), torch.ones(2)))

    def test_top1_boost_preserves_evidence_but_changes_confidence(self):
        z = torch.tensor([[2., 1., 0.], [0., 1., 2.]])
        s = torch.tensor([[.1, .8, .6], [.8, .7, .1]])
        before = score_prediction(z, s)
        boosted = z.clone()
        boosted[torch.arange(len(z)), before["pseudo_label"]] += 4
        after = score_prediction(boosted, s)
        self.assertTrue(torch.equal(before["pseudo_label"], after["pseudo_label"]))
        self.assertTrue(torch.equal(before["residual_probability"], after["residual_probability"]))
        self.assertTrue(torch.equal(before["competitor_evidence"], after["competitor_evidence"]))
        self.assertTrue(torch.all(after["mc"] > before["mc"]))

    def test_competitor_identity_changes_e_when_shape_signals_identical(self):
        z = torch.tensor([[8., 5., 3.], [8., 3., 5.]])
        s = torch.tensor([[.4, .7, .1], [.4, .7, .1]])
        output = score_prediction(z, s)
        for name in ("mc", "entropy", "margin", "residual_entropy", "residual_probability_variance"):
            self.assertTrue(torch.allclose(output[name][0], output[name][1], atol=1e-7), name)
        self.assertGreater(output["competitor_evidence"][0], output["competitor_evidence"][1])

    def test_permutation_expectation_equals_uniform_evidence(self):
        r = torch.tensor([[.7, .2, .1]])
        c = torch.tensor([[.8, 0., .3]])
        values = [(r[:, list(p)]*c).sum(1) for p in itertools.permutations(range(3))]
        self.assertTrue(torch.allclose(torch.stack(values).mean(0), c.mean(1), atol=1e-7))

    def test_keyed_permutations_are_chunk_and_batch_invariant(self):
        keys = pixel_identity_keys("VOC/example", torch.arange(80))
        permutation = residual_permutation(keys, 20, 4)
        chunks = torch.cat([residual_permutation(x, 20, 4) for x in keys.split(13)])
        self.assertTrue(torch.equal(permutation, chunks))
        self.assertTrue(torch.equal(permutation.sort(1).values, torch.arange(20).expand(80, -1)))
        order = torch.randperm(80)
        reordered = residual_permutation(keys[order], 20, 4)
        self.assertTrue(torch.equal(reordered, permutation[order]))
        self.assertFalse(torch.equal(permutation, residual_permutation(keys, 20, 5)))

    def test_permutation_is_read_only(self):
        z = torch.randn(60, 21)
        s = torch.rand(60, 21)*2-1
        before = score_prediction(z, s)
        snapshot = {key:value.clone() for key, value in before.items()}
        keys = pixel_identity_keys("sample", torch.arange(60))
        result = permuted_evidence(before["residual_probability"], before["residual_counterevidence"], keys, 0)
        self.assertTrue(torch.isfinite(result).all())
        for key in before:
            self.assertTrue(torch.equal(snapshot[key], before[key]), key)

    def test_exact_prototype_ties_are_not_strict_conflicts(self):
        output = score_prediction(torch.tensor([[0., 1., 2.]]), torch.tensor([[.4, .4, .4]]))
        self.assertTrue(output["prototype_disagreement"].item())
        self.assertFalse(output["strict_conflict"].item())
        self.assertEqual(output["competitor_evidence"].item(), 0)

    def test_reject_nonfinite_or_noncosine_inputs(self):
        for s in [torch.tensor([[0., float("nan")]]), torch.tensor([[0., 2.]])]:
            with self.assertRaises(ValueError):
                score_prediction(torch.tensor([[1., 0.]]), s)
        with self.assertRaises(ValueError):
            score_prediction(torch.tensor([[1., float("inf")]]), torch.tensor([[.1, .2]]))

    def test_feature_alignment_matches_full_interpolation(self):
        fmap = torch.rand(1, 8, 3, 4)+.1
        positions = torch.tensor([[0, 0], [10, 12], [3, 7], [5, 5]])
        expected = F.interpolate(fmap, (11, 13), mode="bilinear", align_corners=True)
        expected = F.normalize(expected[0, :, positions[:, 0], positions[:, 1]].T, dim=1)
        actual = sample_decoder_features(fmap, positions, (11, 13))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_prototypes_equal_image_weight_and_role_isolation(self):
        bank = PrototypeAccumulator(2, 2, ["sup1", "sup2"])
        bank.add_image("sup1", torch.tensor([[1., 0.]]).repeat(100, 1), torch.zeros(100, dtype=torch.long))
        bank.add_image("sup2", torch.tensor([[0., 1.], [1., 1.]]), torch.tensor([0, 1]))
        prototype = bank.finalize()
        expected = F.normalize(torch.tensor([[1., 1.], [1., 1.]]), dim=1)
        self.assertTrue(torch.allclose(prototype, expected))
        with self.assertRaises(ValueError):
            bank.add_image("rel_image", torch.ones(1, 2), torch.tensor([0]))
        with self.assertRaises(ValueError):
            bank.add_image("sup1", torch.ones(1, 2), torch.tensor([0]))

    def test_missing_and_degenerate_prototypes_fail(self):
        bank = PrototypeAccumulator(2, 2, ["sup"])
        bank.add_image("sup", torch.tensor([[1., 0.]]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            bank.finalize()
        bank = PrototypeAccumulator(1, 2, ["sup1", "sup2"])
        bank.add_image("sup1", torch.tensor([[1., 0.]]), torch.tensor([0]))
        bank.add_image("sup2", torch.tensor([[-1., 0.]]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            bank.finalize()

    def test_head_training_cannot_change_main_features(self):
        features = torch.randn(30, 2, requires_grad=True)
        head = CorrectnessHead()
        head.fit_standardization(features, torch.ones(30))
        F.binary_cross_entropy_with_logits(head(features), torch.arange(30).remainder(2).float()).backward()
        self.assertIsNone(features.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
        weights = head.weights(features)
        self.assertFalse(weights.requires_grad)
        self.assertFalse(any(isinstance(x, (nn.modules.batchnorm._BatchNorm, nn.Dropout)) for x in head.modules()))

    def test_head_weights_batch_regrouping_invariant(self):
        features = torch.randn(100, 2)
        head = CorrectnessHead().eval()
        head.fit_standardization(features, torch.rand(100)+.1)
        whole = head.weights(features)
        chunks = torch.cat([head.weights(x) for x in features.split(7)])
        self.assertTrue(torch.allclose(whole, chunks, atol=1e-7, rtol=1e-6))

    def test_unsupervised_loss_denominator_and_detached_weights(self):
        logits = torch.zeros(3, 2, requires_grad=True)
        weights = torch.tensor([1., .5, .9], requires_grad=True)
        loss = weighted_consistency_loss(logits, torch.tensor([0, 0, 1]), weights,
                                         torch.tensor([True, True, False]))
        self.assertAlmostEqual(loss.item(), .75*np.log(2), places=6)
        loss.backward()
        self.assertIsNone(weights.grad)
        self.assertTrue(torch.equal(logits.grad[2], torch.zeros(2)))
        self.assertGreater(logits.grad[:2].abs().sum(), 0)

    def test_split_is_order_invariant_and_image_leakage_fails(self):
        ids = [f"image_{i}" for i in range(1323)]
        first = partition(ids, .1, 20261020)
        second = partition(list(reversed(ids)), .1, 20261020)
        self.assertEqual(first, second)
        self.assertEqual([len(x) for x in first], [1191, 132])
        with self.assertRaises(ValueError):
            disjoint_partitions(dict(D_sup=["a", "b"], D_rel=["b"]))


if __name__ == "__main__":
    unittest.main()
