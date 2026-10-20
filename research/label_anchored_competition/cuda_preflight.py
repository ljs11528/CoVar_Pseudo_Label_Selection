"""Compare mechanism primitives across CPU and the allocated CUDA devices."""
import argparse
import datetime as dt
import json
from pathlib import Path

import torch

from .core import CorrectnessHead, pixel_identity_keys, residual_permutation, score_prediction
from .prepare_data import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(2027)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    z, s = torch.randn(4096, 21)*8, torch.rand(4096, 21)*2-1
    cpu = score_prediction(z, s)
    keys = pixel_identity_keys("cuda-preflight", torch.arange(len(z)))
    order = residual_permutation(keys, 20, 3)
    head = CorrectnessHead()
    features = torch.stack((cpu["logit_mc"], cpu["competitor_evidence"]), 1)
    head.fit_standardization(features, torch.ones(len(z)))
    cpu_weight = head.weights(features)
    records = []
    for index in range(torch.cuda.device_count()):
        device = f"cuda:{index}"
        output = score_prediction(z.to(device), s.to(device))
        errors = {}
        for name in ("logit_mc", "entropy", "residual_entropy", "competitor_evidence", "logit_product_support", "probability_prototype_JS"):
            error = (output[name].cpu()-cpu[name]).abs().max().item()
            if error > 2e-5:
                raise ValueError(f"CPU/CUDA score discrepancy: {index} {name} {error}")
            errors[name] = error
        if not torch.equal(residual_permutation(keys.to(device), 20, 3).cpu(), order):
            raise ValueError("CPU/CUDA identity-keyed permutations differ")
        head.to(device)
        weights = head.weights(features.to(device))
        if (weights.cpu()-cpu_weight).abs().max().item() > 2e-6:
            raise ValueError("CPU/CUDA head weights differ")
        if weights.requires_grad:
            raise ValueError("Detached weights unexpectedly track gradients")
        extreme = score_prediction(torch.tensor([[1000., 3., 1.]], device=device), torch.tensor([[.1, .9, -.1]], device=device))
        if not torch.isfinite(extreme["competitor_evidence"]).all():
            raise ValueError("Nonfinite extreme-confidence evidence")
        records.append(dict(device=index, name=torch.cuda.get_device_name(index), max_absolute_differences=errors,
                            permutations_bitwise_equal=True, detached_weights=True, extreme_confidence_finite=True))
    receipt = dict(state="complete", completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                   scope="synthetic numerical CUDA preflight; no empirical method performance",
                   torch=torch.__version__, cuda=torch.version.cuda, devices=records,
                   core_sha256=digest(Path(__file__).with_name("core.py")), tool_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(dict(state="complete", devices=len(records))), flush=True)


if __name__ == "__main__":
    main()
