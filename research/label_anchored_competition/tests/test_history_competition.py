import copy
import json
from pathlib import Path
import unittest

import numpy as np

from ..history_competition import (aggregate_sufficient, pair_values, recovery_intervals,
    screen_gate, shuffled_past, sufficient_rows, validate_aligned_records)


SPEC = json.loads(Path(__file__).resolve().parents[1].joinpath("history_protocol.json").read_text())


class HistoryCompetitionTests(unittest.TestCase):
    def test_candidates_are_fixed_size_and_current_label_is_retained(self):
        p = np.array([[.96, .03, .01], [.1, .2, .7]])
        q = np.array([[.2, .1, .7], [.6, .3, .1]])
        values = pair_values(np.log(p), np.log(q))
        np.testing.assert_array_equal(values["alternatives"]["now"], [1, 1])
        np.testing.assert_array_equal(values["alternatives"]["history"], [2, 0])
        for key, distribution in (("arithmetic", (p+q)/2), ("geometric", np.sqrt(p*q))):
            distribution[np.arange(2), values["current"]["pseudo_label"]] = -1
            np.testing.assert_array_equal(values["alternatives"][key], distribution.argmax(1))
        for a in values["alternatives"].values():
            self.assertTrue(np.all(a != values["current"]["pseudo_label"]))
        np.testing.assert_allclose(values["risks"]["arithmetic_current_a"],
            -np.log(np.array([.58, .4])/(1-np.array([.58, .4]))), atol=1e-14)

    def test_shuffle_invariants_and_chunk_independence(self):
        rng = np.random.default_rng(91)
        current, past = rng.normal(size=(2, 73, 21)); pixels = rng.choice(10000, 73, replace=False)
        full, error = shuffled_past(current, past, "sample", pixels, 401, "test")
        chunks = [shuffled_past(current[a:b], past[a:b], "sample", pixels[a:b], 401, "test")[0]
                  for a,b in ((0, 17), (17, 53), (53, 73))]
        np.testing.assert_array_equal(full, np.concatenate(chunks))
        self.assertLess(error, 1e-12)
        a = current.argmax(1)
        np.testing.assert_array_equal(full[np.arange(len(a)),a], past[np.arange(len(a)),a])
        np.testing.assert_array_equal(full.argmax(1) == a, past.argmax(1) == a)
        self.assertFalse(np.array_equal(full, past))

    def test_full_alignment_detects_corruption_outside_analysis_cap(self):
        row = dict(image_id=np.asarray("i"), role=np.asarray("D_rel"), original_size=np.array([2, 3]),
            valid_pixel_count=np.array(6), pixel_indices=np.arange(6), yx=np.column_stack([np.arange(6)//3, np.arange(6)%3]),
            GT=np.zeros(6, np.int64), boundary=np.zeros(6, bool), correct=np.ones(6, bool),
            logits=np.tile([2., 1., 0.], (6, 1)), score__pseudo_label=np.zeros(6, np.int64))
        spec = copy.deepcopy(SPEC); spec["classes"] = 3; spec["cap"] = 2
        records = {stage:copy.deepcopy(row) for stage in spec["input_exports"]}
        self.assertEqual(validate_aligned_records(records, spec), 6)
        records["late"]["GT"][-1] = 1
        with self.assertRaisesRegex(ValueError, "Cross-stage mismatch"):
            validate_aligned_records(records, spec)

    def test_alignment_checks_correctness_not_only_shapes(self):
        row = dict(image_id=np.asarray("i"), role=np.asarray("D_rel"), original_size=np.array([1, 3]),
            valid_pixel_count=np.array(3), pixel_indices=np.arange(3), yx=np.column_stack([np.zeros(3, int), np.arange(3)]),
            GT=np.zeros(3, np.int64), boundary=np.zeros(3, bool), correct=np.ones(3, bool),
            logits=np.tile([2., 1., 0.], (3, 1)), score__pseudo_label=np.zeros(3, np.int64))
        spec = copy.deepcopy(SPEC); spec["classes"] = 3
        records = {stage:copy.deepcopy(row) for stage in spec["input_exports"]}
        records["middle"]["correct"][1] = False
        with self.assertRaisesRegex(ValueError, "correctness"):
            validate_aligned_records(records, spec)

    def sample(self):
        pool = dict(pair="middle_to_late", role="D_rel", image_ids=["A", "B", "C"], image=np.array([0,0,1,1,2,2]),
            error=np.array([1,0,1,1,0,0]), sampling_weight=np.array([5.,5.,1.,1.,2.,2.]),
            GT=np.array([1,0,2,1,0,0]), pseudo_label=np.zeros(6, int), past_label=np.array([1,0,0,0,0,0]),
            b=np.ones(6,int), history_alternative=np.full(6,2), mc=np.full(6,.97), strict_conflict=np.ones(6,bool))
        alternatives = dict(now=np.ones(6,int), history=np.array([1,2,2,2,2,2]), arithmetic=np.ones(6,int), geometric=np.ones(6,int))
        alternatives.update({f"sham{s}":np.ones(6,int) for s in SPEC["permutation"]["seeds"]})
        return pool, alternatives

    def test_image_sufficient_statistics_preserve_ratio_and_zero_images(self):
        pool, alternatives = self.sample()
        rows = sufficient_rows(pool, alternatives, SPEC)
        result = aggregate_sufficient(rows, SPEC)
        history = next(r for r in result if r["scope"] == "high_mc" and r["method"] == "history")
        self.assertEqual(history["image_pool"], 3)
        self.assertEqual(history["error_images"], 2)
        self.assertAlmostEqual(history["error_recovery"], 6/7)
        self.assertAlmostEqual(history["gain_vs_now"], 1/7)
        self.assertAlmostEqual(history["loss_vs_now"], 1/7)
        self.assertAlmostEqual(history["set_truth_coverage"], 15/16)
        zero = [r for r in rows if r["scope"] == "high_mc" and r["image_id"] == "C"]
        self.assertTrue(zero and all(r["error_mass"] == 0 for r in zero))

    def test_bootstrap_matches_direct_image_resampling(self):
        pool, alternatives = self.sample(); rows = sufficient_rows(pool, alternatives, SPEC)
        spec = copy.deepcopy(SPEC); spec["screen"]["bootstrap_replicates"] = 100
        observed = next(r for r in recovery_intervals(rows, spec) if r["scope"] == "all" and r["comparator"] == "now")
        rng = np.random.default_rng(spec["screen"]["bootstrap_seed"]); values=[]
        for _ in range(100):
            counts = np.bincount(rng.integers(3, size=3), minlength=3)
            w = pool["sampling_weight"]*counts[pool["image"]]; e = pool["error"]
            den = w@e
            if den:
                values.append(float((w*e)@((alternatives["history"] == pool["GT"]).astype(float)-
                                           (alternatives["now"] == pool["GT"]).astype(float))/den))
        expected = np.quantile(values, [.025,.975])
        np.testing.assert_allclose([observed["ci_low"], observed["ci_high"]], expected)

    def test_zero_increment_stops_and_validation_cannot_authorize_probe(self):
        pool, alternatives = self.sample()
        alternatives = {k:alternatives["now"].copy() for k in alternatives}
        rows=[]
        for pair in SPEC["pairs"]:
            pool["pair"] = pair["name"]; rows += sufficient_rows(pool, alternatives, SPEC)
        gate = screen_gate(aggregate_sufficient(rows, SPEC), recovery_intervals(rows, SPEC), SPEC)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["new_probe_fitting_allowed"])
        self.assertFalse(gate["teacher_training_allowed"])
        invalid = aggregate_sufficient(rows, SPEC); invalid[0]["role"] = "validation"
        with self.assertRaisesRegex(ValueError, "Validation"):
            screen_gate(invalid, recovery_intervals(rows, SPEC), SPEC)


if __name__ == "__main__":
    unittest.main()
