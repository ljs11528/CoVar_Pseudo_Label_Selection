"""Run mechanism invariants and write a small, hashed execution receipt."""
import argparse
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    stream = io.StringIO()
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = dict(
        scope="Synthetic numerical, alignment, image-isolation and gradient tests; no empirical benefit claim",
        completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        state="complete" if result.wasSuccessful() else "failed", tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         torch=torch.__version__, numpy=np.__version__, cuda_available=torch.cuda.is_available()),
        source_commit_before_check=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_files_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(root.rglob("*.py")) if "artifacts" not in p.relative_to(root).parts},
        protocol_sha256=hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
        output=stream.getvalue(),
        limitations=["Local CPU environment; formal server CUDA preflight and real-model smoke remain required."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Preserve previous selfcheck receipts; choose a new output")
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({key:receipt[key] for key in ("state", "tests", "failures", "errors", "skipped")}))
    if not result.wasSuccessful():
        print(stream.getvalue())
        sys.exit(1)


if __name__ == "__main__":
    main()
