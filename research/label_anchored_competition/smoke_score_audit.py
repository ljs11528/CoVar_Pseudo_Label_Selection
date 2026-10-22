"""Short synthetic device check; produces no scientific result or model selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .bounded_probe import fit_correction_path, fit_mc, mc_logits
from .gpu_bootstrap import GPURanking, KEYS
from .ranking_metrics import Ranking


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    torch.set_num_threads(2)
    spec=json.loads(Path(__file__).with_name("score_audit_protocol.json").read_text())
    rng=np.random.default_rng(71)
    x=rng.normal(size=(600,4)).astype(np.float32)
    target=(x[:,0]+rng.normal(size=600)>0).astype(float)
    w=rng.uniform(.5,3,600)
    mc=fit_mc(x[:,1],target,w,spec["mc_calibrator"])
    offset=mc_logits(mc,x[:,1])
    cfg=dict(spec["correction"],budgets=[4,12],curve_updates=[0,4,12],batch_pixels=64)
    result=fit_correction_path(x,offset,target,w,x,offset,target,w,cfg,2027,args.device)
    assert np.isfinite(result[12]["heldout_logits"]).all()
    short=fit_correction_path(x,offset,target,w,x,offset,target,w,dict(cfg,budgets=[4]),2027,args.device)
    np.testing.assert_array_equal(result[4]["coefficients"],short[4]["coefficients"])
    error=1-target
    counts=np.ones((1,30))
    actual=GPURanking(error,-offset,w,np.repeat(np.arange(30),20),device=args.device).evaluate_counts(counts,())
    expected=Ranking(error,-offset).evaluate(w,())
    np.testing.assert_allclose(actual[0],[expected[k] for k in KEYS],atol=1e-12,rtol=1e-12)
    print(json.dumps(dict(state="passed",scope="synthetic device/import/numerical smoke only",device=args.device,
        torch=torch.__version__,numpy=np.__version__,gpu_cpu_metric_max_abs_diff=float(np.max(np.abs(actual[0]-[expected[k] for k in KEYS]))))))


if __name__ == "__main__":
    main()
