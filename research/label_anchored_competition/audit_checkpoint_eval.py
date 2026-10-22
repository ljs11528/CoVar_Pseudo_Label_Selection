"""Post-hoc checkpoint/log/cache consistency audit. No fitting or BN update."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--stage", choices=["early", "middle", "late"], required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root, repo = args.root, args.release / "repo"
    package = repo / "research/label_anchored_competition"
    protocol = json.loads((package / "protocol.json").read_text())
    step = protocol["teacher"]["stage_steps"][args.stage]
    teacher = root / "experiments/voc_Dsup_mc_seed42"
    ckpt_file = teacher / "checkpoints" / f"step_{step:06d}.ckpt"
    expected_hash = json.loads((teacher / "COMPLETE.json").read_text())["checkpoint_sha256"][ckpt_file.name]
    actual_hash = digest(ckpt_file)
    if actual_hash != expected_hash:
        raise ValueError("Checkpoint checksum changed")
    upstream = repo / "CSL+CoVar"
    sys.path.insert(0, str(upstream))
    import yaml
    from model.model_helper import ModelBuilder
    from dataset.semi import SemiDataset
    cfg = yaml.safe_load((upstream / "configs/pascal.yaml").read_text())
    if cfg["train"]["eval_mode"] != "original":
        raise ValueError("This exact audit requires original-image evaluation")
    cfg["model"]["backbone"]["kwargs"]["pretrained"] = False
    checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if checkpoint["global_step"] != step:
        raise ValueError("Unexpected update count")
    model = ModelBuilder(cfg["model"])
    model.load_state_dict({k.removeprefix("model."):v for k,v in checkpoint["state_dict"].items() if k.startswith("model.")}, strict=True)
    del checkpoint
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_device(torch.device(args.device))
    model = model.to(args.device).eval().requires_grad_(False)
    export = root / "experiments" / f"frozen_seed42_{args.stage}_7c4c5c09"
    manifest = json.loads((export / "manifest.json").read_text())
    cache = {v["image_id"]:v for v in manifest["full_image_files"][:8]}
    partition = args.release / "prepared_partition"
    dataset = SemiDataset("pascal", str(root / "data/VOCdevkit/VOC2012"), "val", str(partition / "splits/val.txt"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True, shuffle=False)
    log_text = "\n".join(path.read_text(errors="replace") for path in teacher.glob("output.log*"))
    logged = re.findall(rf"validation global_step={step} mIoU=([\d.]+)", log_text)
    result = dict(state="running", scope="post-hoc loaded checkpoint consistency, no parameter/BN updates or head fitting",
                  stage=args.stage, global_step=step, checkpoint_sha256=actual_hash, source_release=manifest["signature"]["source_commit"],
                  torch=torch.__version__, cuda=torch.version.cuda, GPU=torch.cuda.get_device_name(torch.device(args.device)),
                  logged_teacher_mIoU=float(logged[-1]) if logged else None,
                  started_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), modes={}, cache_checks=[])
    started = time.monotonic()
    original_buffers = {k:v.clone() for k,v in model.named_buffers()}
    with torch.inference_mode():
        for mixed in (False, True):
            confusion = torch.zeros(21, 21, dtype=torch.int64, device=args.device)
            for index, (image, mask, lines) in enumerate(loader):
                image, mask = image.to(args.device), mask.to(args.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=mixed):
                    logits = model(image, False)
                label = logits.argmax(1)
                valid = mask != 255
                counts = torch.bincount((mask[valid]*21+label[valid]).reshape(-1), minlength=21*21)
                confusion += counts.reshape(21, 21)
                identity = Path(lines[0].split()[0]).stem
                if not mixed and identity in cache:
                    entry = cache[identity]
                    with np.load(export / entry["file"], allow_pickle=False) as stored:
                        current = logits[0].float().cpu().numpy()
                        raw = stored["logits"]
                        result["cache_checks"].append(dict(image_id=identity,
                            GT_equal=bool(np.array_equal(mask[0].cpu().numpy(), stored["GT"])),
                            max_absolute_logit_difference=float(np.abs(current-raw).max()),
                            changed_argmax=int(np.count_nonzero(current.argmax(0) != raw.argmax(0)))))
                if (index+1) % 100 == 0:
                    state = dict(state="running", mode="FP16" if mixed else "FP32", images=index+1, total=len(dataset),
                                 elapsed_seconds=time.monotonic()-started, stage=args.stage)
                    args.output.with_suffix(".progress.json").write_text(json.dumps(state, indent=2)+"\n")
            matrix = confusion.cpu().numpy()
            intersection = np.diag(matrix)
            union = matrix.sum(0)+matrix.sum(1)-intersection
            iou = intersection/(union+1e-10)*100
            score = float(iou.mean())
            result["modes"]["FP16" if mixed else "FP32"] = dict(mIoU=score, pixel_accuracy=float(intersection.sum()/matrix.sum()),
                delta_mIoU_vs_training_log=score-result["logged_teacher_mIoU"] if logged else None,
                per_class_IoU=iou.tolist(), confusion=matrix.tolist(), images=len(dataset))
    result["buffers_unchanged"] = all(torch.equal(v, original_buffers[k]) for k,v in model.named_buffers())
    if not result["buffers_unchanged"]:
        raise AssertionError("Audit changed model buffers")
    result.update(state="complete", completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
                  tool_sha256=digest(Path(__file__)), limitation="Missing pre-strong-view BN backups cannot be recovered from the old state_dict; log differences do not alone give corrected pseudo-label rankings")
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:result[k] for k in ["state","stage","logged_teacher_mIoU","elapsed_seconds"]}, default=str), flush=True)


if __name__ == "__main__":
    main()
