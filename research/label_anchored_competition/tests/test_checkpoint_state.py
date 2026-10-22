import importlib.util
import io
from pathlib import Path
import unittest

import torch
from torch import nn

from research.label_anchored_competition.checkpoint_state import (
    BN_SNAPSHOT_POLICY, flush_pending_bn_statistics, require_flushed_checkpoint,
)


def original_model():
    source = Path(__file__).resolve().parents[3] / "CSL+CoVar/model/model_helper.py"
    spec = importlib.util.spec_from_file_location("audited_original_model", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.ModelBuilder.__new__(module.ModelBuilder)
    nn.Module.__init__(model)
    model.backbone = nn.BatchNorm2d(2)
    model.decoder = nn.Identity()
    model.saved_bn_stats = []
    return model


class CheckpointBatchNormState(unittest.TestCase):
    def test_old_checkpoint_cannot_be_silently_resumed_as_repaired(self):
        with self.assertRaises(ValueError):
            require_flushed_checkpoint({"protocol_sha256": "unchanged"})
        require_flushed_checkpoint({"bn_snapshot_policy": BN_SNAPSHOT_POLICY})

    def test_dirty_snapshot_differs_from_next_normal_eval(self):
        model = original_model().train()
        model.save_bn_stats(model.backbone)
        model.backbone(torch.arange(64, dtype=torch.float32).reshape(4, 2, 2, 4)/4)
        dirty = {k:v.clone() for k,v in model.state_dict().items()}
        model.restore_bn_stats()
        model.eval()
        resumed = original_model().eval()
        resumed.load_state_dict(dirty, strict=True)
        x = torch.ones(1, 2, 2, 4)
        self.assertFalse(torch.allclose(model.backbone(x), resumed.backbone(x)))
        self.assertFalse(any("saved_bn_stats" in key for key in dirty))

    def test_pre_capture_flush_preserves_parameters_and_eval(self):
        model = original_model().train()
        parameters = {k:v.clone() for k,v in model.named_parameters()}
        rng = torch.get_rng_state().clone()
        model.save_bn_stats(model.backbone)
        model.backbone(torch.arange(64, dtype=torch.float32).reshape(4, 2, 2, 4)/4)
        self.assertEqual(flush_pending_bn_statistics(model), 1)
        self.assertEqual(flush_pending_bn_statistics(model), 0)
        for key,value in model.named_parameters():
            torch.testing.assert_close(value, parameters[key], rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        # Exercise serialization, not just assignment between live tensors.
        saved = torch.load(stream, weights_only=True)
        restored = original_model().eval()
        restored.load_state_dict(saved, strict=True)
        x = torch.ones(1, 2, 2, 4)
        torch.testing.assert_close(restored.backbone(x), x/(1+1e-5)**.5)


if __name__ == "__main__":
    unittest.main()
