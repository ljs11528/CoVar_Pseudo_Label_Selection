"""Exercise the complete diagnostic program on explicitly synthetic caches.

This checks workflow compatibility, not method accuracy. The real protocol's
500-update head budget is retained; no scientific result is produced.
"""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core import score_prediction
from .legacy_selectors import Selectors
from .prepare_data import digest, source_revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cap", type=int, choices=[1024, 4096], default=4096)
    parser.add_argument("--test-images", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    package = Path(__file__).resolve().parent
    signature = dict(protocol_sha256=digest(package / "protocol.json"), stage="late", seed=2027,
                     checkpoint_sha256="synthetic-no-checkpoint", source_commit=source_revision(package))
    rng = np.random.default_rng(2027)
    torch.set_num_threads(2)
    torch.manual_seed(2027)
    prototypes = F.normalize(torch.randn(21, 256), dim=1)
    selectors = Selectors()
    files = []
    for role, count in [("D_rel", 6), ("validation", args.test_images)]:
        folder = args.output / "samples" / role
        folder.mkdir(parents=True)
        for index in range(count):
            identity = f"synthetic_{role}_{index}"
            features = F.normalize(torch.randn(64, 256), dim=1)
            logits = torch.randn(64, 21)*3
            similarities = features@prototypes.T
            score = score_prediction(logits, similarities)
            p = logits.T.reshape(1, 21, 8, 8).softmax(1)
            legacy = {"score__legacy_"+name.upper()+"_weight":selectors.native(p, name)[0].reshape(-1).numpy()
                      for name in ("csl", "covar")}
            a = score["pseudo_label"].numpy()
            correct = rng.random(64) > .35
            GT = np.where(correct, a, (a+1) % 21)
            path = folder / f"{identity}.npz"
            np.savez_compressed(path, image_id=np.asarray(identity), role=np.asarray(role), valid_pixel_count=np.asarray(64),
                pixel_indices=np.arange(64), GT=GT, correct=correct, boundary=np.arange(64) % 2 == 0,
                logits=logits.numpy(), prototype_similarities=similarities.numpy(),
                **legacy, **{"score__"+key:value.numpy() for key, value in score.items() if value.ndim == 1})
            files.append(dict(role=role, image_id=identity, file=str(path.relative_to(args.output)), bytes=path.stat().st_size, sha256=digest(path)))
    (args.output / "manifest.json").write_text(json.dumps(dict(synthetic=True, signature=signature, images=dict(D_sup=0, D_rel=6, validation=args.test_images), files=files), indent=2)+"\n")
    (args.output / "COMPLETE.json").write_text(json.dumps(dict(state="complete", synthetic=True, signature=signature))+"\n")
    start = time.monotonic()
    command = [sys.executable, "-u", "-m", "research.label_anchored_competition.analyze_frozen", "--export", str(args.output),
               "--cap", str(args.cap), "--device", "cpu", "--synthetic-smoke"]
    with (args.output / "integration.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    success = result.returncode == 0 and (args.output / f"analysis_{args.cap}/COMPLETE.json").exists()
    receipt = dict(state="complete" if success else "failed", scope="synthetic integration only; not research results", returncode=result.returncode,
                   elapsed_seconds=time.monotonic()-start, completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   protocol_sha256=signature["protocol_sha256"], tool_sha256=digest(__file__),
                   source_files_sha256={p.name:digest(p) for p in package.glob("*.py")}, command=command)
    (args.output / "INTEGRATION_COMPLETE.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in ("state", "returncode", "elapsed_seconds", "scope")}))
    if not success:
        print((args.output / "integration.log").read_text()[-6000:])
        sys.exit(1)


if __name__ == "__main__":
    main()
