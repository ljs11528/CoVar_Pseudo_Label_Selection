import csv
from pathlib import Path
import tempfile
import unittest

from research.label_anchored_competition.run_frozen_queue import decide_gate


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ResearchGate(unittest.TestCase):
    def fixture(self, root):
        outputs = {stage:root / stage for stage in ("early", "middle", "late")}
        for stage, directory in outputs.items():
            primary = [dict(contrast=f"comparison_{j}", delta_error_ap=.01,
                            ci_low=.005, ci_high=.015, valid_replicates=2000) for j in range(4)]
            write_csv(directory / "analysis_1024/primary_contrasts.csv", primary)
            intervals = [dict(scope="all", score="F_plus", delta_vs="F_base", metric="error_ap", delta_ci_low=.001, delta_ci_high=.01),
                         dict(scope="conflict", score="h_e", delta_vs="h_uniform", metric="error_ap", delta_ci_low=.001, delta_ci_high=.02)]
            write_csv(directory / "analysis_1024/paired_image_bootstrap.csv", intervals)
            write_csv(directory / "analysis_4096/ranking.csv", [dict(scope="conflict", method="F_base", error_ap=.6),
                                                                 dict(scope="conflict", method="F_plus", error_ap=.61)])
        return outputs

    def test_all_required_evidence_allows_only_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            result = decide_gate(self.fixture(Path(temp)))
            self.assertTrue(result["pass_to_confirmation"])
            self.assertEqual(result["state"], "candidate_requires_seed43_confirmation")

    def test_one_uncertain_primary_contrast_stops_expansion(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = self.fixture(Path(temp))
            path = outputs["late"] / "analysis_1024/primary_contrasts.csv"
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            rows[2]["ci_low"] = -.001
            write_csv(path, rows)
            self.assertFalse(decide_gate(outputs)["pass_to_confirmation"])

    def test_global_harm_cannot_be_hidden_by_conflict_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = self.fixture(Path(temp))
            path = outputs["middle"] / "analysis_1024/paired_image_bootstrap.csv"
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["delta_ci_low"], rows[0]["delta_ci_high"] = -.01, -.001
            write_csv(path, rows)
            result = decide_gate(outputs)
            self.assertFalse(result["pass_to_confirmation"])
            self.assertEqual(result["clear_all_pixel_AP_degradation_stages"], ["middle"])

    def test_missing_intervals_are_inconclusive(self):
        with tempfile.TemporaryDirectory() as temp:
            outputs = self.fixture(Path(temp))
            (outputs["late"] / "analysis_1024/primary_contrasts.csv").unlink()
            result = decide_gate(outputs)
            self.assertFalse(result["pass_to_confirmation"])
            self.assertEqual(result["state"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
