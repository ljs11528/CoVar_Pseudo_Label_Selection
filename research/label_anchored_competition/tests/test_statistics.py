import unittest

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from research.label_anchored_competition.analysis_utils import fit_selected_head
from research.label_anchored_competition.analyze_frozen import coverage_at_precision
from research.label_anchored_competition.diagnostic_tables import calibration_intervals
from research.label_anchored_competition.gpu_bootstrap import GPURanking
from research.label_anchored_competition.ranking_metrics import Ranking
from research.label_anchored_competition.train_teacher import training_precision


class StatisticalInvariants(unittest.TestCase):
    def test_weighted_tied_AP_and_AUROC_match_sklearn(self):
        error = np.array([1, 0, 1, 0])
        risk = np.array([.9, .9, .1, .1])
        weight = np.array([1., 3., 2., 4.])
        result = Ranking(error, risk).evaluate(weight)
        self.assertAlmostEqual(result["error_ap"], average_precision_score(error, risk, sample_weight=weight), places=12)
        self.assertAlmostEqual(result["auroc"], roc_auc_score(error, risk, sample_weight=weight), places=12)
        self.assertAlmostEqual(result["error_ap"], .2833333333333333, places=12)

    def test_fp64_bootstrap_reductions_match_CPU_including_empty_images(self):
        error = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        image = np.repeat(np.arange(4), 2)
        risk = np.array([.1, .8, .1, .9, .3, .3, .7, .9])
        weight = np.array([1., 2., 1., 1., 3., 2., 1., 1.])
        counts = np.array([[1, 1, 1, 1, 2], [0, 2, 0, 3, 0], [4, 0, 0, 1, 0]])
        actual = GPURanking(error, risk, weight, image, device="cpu").evaluate_counts(counts, coverages=(.5, .8))
        keys = ["auroc", "error_ap", "error_prevalence", "aurc_continuous", "precision_at_0.5", "precision_at_0.8"]
        for row, multiplicities in zip(actual, counts):
            expected = Ranking(error, risk).evaluate(weight*multiplicities[image], coverages=(.5, .8))
            np.testing.assert_allclose(row, [expected[key] for key in keys], atol=1e-12, rtol=1e-12)

    def test_precision_target_handles_fractional_ties(self):
        self.assertAlmostEqual(coverage_at_precision(np.array([0, 1]), np.array([0., 1.]), np.array([5., 5.]), .8), .625)
        self.assertEqual(coverage_at_precision(np.array([0, 1]), np.array([0., 0.]), np.array([5., 5.]), .8), 0)

    def test_calibration_bootstrap_uses_images_and_preserves_pairing(self):
        error = np.tile([0, 1], 10)
        pool = dict(image_ids=[str(i) for i in range(10)], image=np.repeat(np.arange(10), 2),
                    sampling_weight=np.ones(20), error=error)
        probabilities = dict(F_base=np.tile([.6, .4], 10), F_plus=np.tile([.7, .3], 10))
        rows = calibration_intervals(pool, probabilities, np.ones(20, bool), 64, 2027)
        delta = next(row for row in rows if row.get("delta_vs") == "F_base" and row["metric"] == "brier")
        self.assertAlmostEqual(delta["delta_ci_low"], -.07, places=12)
        self.assertAlmostEqual(delta["delta_ci_high"], -.07, places=12)

    def test_test_role_cannot_fit_a_head(self):
        with self.assertRaises(ValueError):
            fit_selected_head(dict(role="validation"), None, None, None, "cpu")

    def test_training_precision_parses_locked_description(self):
        self.assertEqual(training_precision("16-mixed training; FP32 frozen export"), "16-mixed")
        with self.assertRaises(ValueError):
            training_precision("unsupported")


if __name__ == "__main__":
    unittest.main()
