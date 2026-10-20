"""Verify real SGD updates, registered scheduler restoration and feature export."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from .core import sample_decoder_features, score_prediction
from .prepare_data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--prepared-partition", required=True, type=Path)
    parser.add_argument("--pretrained", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint_path = args.run / "checkpoints/smoke_only.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained = torch.load(args.pretrained, map_location="cpu", weights_only=True)
    if checkpoint["global_step"] != 2 or len(checkpoint["lr_schedulers"]) != 1:
        raise ValueError("Smoke did not complete exactly two registered-scheduler updates")
    if checkpoint["lr_schedulers"][0]["last_epoch"] != 2:
        raise ValueError("Scheduler count not aligned with optimizer updates")
    optimizer = checkpoint["optimizer_states"][0]
    momenta = [x["momentum_buffer"] for x in optimizer["state"].values() if "momentum_buffer" in x]
    if not momenta or not all(torch.isfinite(x).all() for x in momenta) or not any(x.abs().sum() > 0 for x in momenta):
        raise ValueError("No valid nonzero SGD momentum")
    differences = []
    for name, value in checkpoint["state_dict"].items():
        short = name.removeprefix("model.backbone.")
        if name.startswith("model.backbone.") and short in pretrained and value.is_floating_point():
            differences.append((value-pretrained[short]).abs().max().item())
    if not differences or max(differences) <= 0:
        raise ValueError("Real backbone state did not change")
    spec = json.loads(Path(__file__).with_name("protocol.json").read_text())["teacher"]
    expected_lr = spec["learning_rate"]*(1-2/spec["total_optimizer_steps"])**.9
    if abs(optimizer["param_groups"][0]["lr"]-expected_lr) > 1e-12:
        raise ValueError("Saved learning rate differs from fixed polynomial schedule")
    sys.path.insert(0, str(args.upstream.resolve()))
    import yaml
    from model.model_helper import ModelBuilder
    from dataset.semi import SemiDataset
    cfg = yaml.safe_load((args.upstream / "configs/pascal.yaml").read_text())
    cfg["model"]["backbone"]["kwargs"]["pretrained"] = False
    model = ModelBuilder(cfg["model"])
    model.load_state_dict({name.removeprefix("model."):value for name, value in checkpoint["state_dict"].items() if name.startswith("model.")}, strict=True)
    model = model.cuda().eval()
    captured = []
    handle = model.decoder.classifier.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0].detach()))
    data = SemiDataset("pascal", str(args.data_root), "val", str(args.prepared_partition / "splits/1_8/labeled.txt"))
    images = []
    with torch.no_grad():
        for index in range(2):
            image, mask, identity = data[index]
            captured.clear()
            logits = model(image[None].cuda(), False)[0]
            fmap = captured[-1]
            if fmap.shape[1] != 256 or not torch.isfinite(logits).all():
                raise ValueError("Feature layer shape or output invalid")
            h, w = mask.shape
            yx = torch.tensor([[0, 0], [h//2, w//2], [h-1, w-1]])
            features = sample_decoder_features(fmap, yx, (h, w))
            prototypes = F.normalize(torch.randn(21, 256, device="cuda"), dim=1)
            scores = score_prediction(logits[:, yx[:, 0], yx[:, 1]].T, features@prototypes.T)
            images.append(dict(image_id=identity, logits_shape=list(logits.shape), feature_shape=list(fmap.shape),
                               score_finite=bool(torch.isfinite(scores["competitor_evidence"]).all())))
    handle.remove()
    receipt = dict(state="complete", scope="real D_sup two-step smoke; discarded initialization, no research benefit claim",
                   completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), global_step=2,
                   SGD_momentum_tensors=len(momenta), backbone_max_absolute_change=max(differences),
                   saved_lr=optimizer["param_groups"][0]["lr"], expected_lr=expected_lr, scheduler_state_restorable=True,
                   strict_model_reload=True, feature_checks=images,
                   checkpoint_sha256=digest(checkpoint_path), tool_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
