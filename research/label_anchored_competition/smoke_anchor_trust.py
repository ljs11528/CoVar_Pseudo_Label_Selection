"""Synthetic complete nested/refit/metrics smoke; no real research outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .anchor_trust import apply_model
from .anchor_trust_analysis import evaluate, intervals, support_tables
from .gpu_bootstrap import GPURanking, KEYS
from .ranking_metrics import Ranking
from .run_anchor_trust import nested_fit, save_pool, load_pool
from .run_score_audit import write_json
from .tests.test_anchor_trust import synthetic_pool, spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    cfg=spec();cfg["statistics"]["bootstrap_replicates"]=30
    pool=synthetic_pool()
    save_pool(args.output/"pool.npz",pool)
    pool=load_pool(args.output/"pool.npz")
    models,oof=nested_fit(pool,cfg,args.output,args.device,lambda *a,**k:None)
    table,curves=evaluate(pool,oof,cfg)
    assert all(np.isfinite(row["risk"]) for row in curves)
    ci=intervals(pool,oof,cfg,args.device,lambda *a,**k:None)
    supports,conditional=support_tables(pool,oof,cfg)
    assert all(np.isfinite(v).all() for v in oof.values())
    for method,model in models.items():
        actual=apply_model(model,pool)
        np.testing.assert_array_equal(actual,apply_model(json.loads(json.dumps(model)),pool))
    error=pool["error"];risk=-oof["MM"];weight=pool["sampling_weight"]
    actual=GPURanking(error,risk,weight,pool["image"],args.device).evaluate_counts(np.ones((1,len(pool["image_ids"]))),(.9,))[0]
    expected=Ranking(error,risk).evaluate(weight,(.9,))
    reference=np.array([expected[k] for k in list(KEYS)+["precision_at_0.9"]])
    np.testing.assert_allclose(actual,reference,atol=1e-12,rtol=1e-12)
    result=dict(state="passed",scope="synthetic engineering check only",device=args.device,
                models=len(models),ranking_rows=len(table),interval_rows=len(ci),support_rows=len(supports),
                conditional_rows=len(conditional),curve_rows=len(curves),metric_max_abs_diff=float(np.max(np.abs(actual-reference))),
                torch=torch.__version__,numpy=np.__version__)
    write_json(args.output/"SMOKE.json",result)
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
