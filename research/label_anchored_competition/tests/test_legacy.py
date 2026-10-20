import unittest

import torch

from research.label_anchored_competition.legacy_selectors import Selectors


class NativeSelectorFidelity(unittest.TestCase):
    def test_native_weights_match_independent_reconstruction(self):
        torch.manual_seed(1)
        selectors = Selectors()
        for classes in (2, 21):
            for mode in ("random", "uniform", "onehot", "single_pixel"):
                shape = (1, classes, 1, 1) if mode == "single_pixel" else (1, classes, 9, 11)
                logits = torch.randn(shape)*3
                if mode == "uniform":
                    logits.zero_()
                if mode == "onehot":
                    logits.fill_(-1000)
                    logits[:, 0] = 1000
                p = logits.softmax(1)
                for method in ("csl", "covar"):
                    actual = selectors.native(p, method)[0]
                    reconstructed, _ = selectors.intervention(p, method+"_reconstructed")
                    torch.testing.assert_close(actual, reconstructed, rtol=0, atol=2e-5)
                    self.assertTrue(torch.isfinite(actual).all())
                    self.assertTrue(((actual >= 0) & (actual <= 1)).all())

    def test_native_and_paper_residual_variances_are_distinct(self):
        p = torch.tensor([.8, .15, .05], dtype=torch.float64).reshape(1, 3, 1, 1)
        s = Selectors.statistics(p)
        torch.testing.assert_close(s["code_v"]-s["paper_v"], torch.full((1, 1), .2**2/2**3, dtype=p.dtype))
        torch.testing.assert_close(s["paper_weighted_stable"], torch.full((1, 1), .025, dtype=p.dtype))


if __name__ == "__main__":
    unittest.main()
