"""Independent NumPy replay and objective-gradient checks; never refits a scorer."""
import argparse
import json
from pathlib import Path

import numpy as np

from .run_anchor_trust import load_pool, load_predictions
from .anchor_trust_analysis import folds_for
from .prepare_data import digest
from .run_score_audit import write_json


def matrix(basis,values):
    z=(values-np.asarray(basis["mean"]))/np.asarray(basis["scale"])
    columns=[np.ones(len(z))]
    for item in basis["columns"]:
        if item["kind"]=="interaction":
            raw=np.prod(z[:,item["columns"]],axis=1)
        elif item["kind"]=="hinge":
            raw=np.clip(z[:,item["column"]]-item["knot"],0,None)
        else:
            raw=z[:,item["column"]]
        columns.append((raw-item["center"])/item["scale"])
    return np.stack(columns,axis=1)


def predict(model,pool,ix):
    if model["kind"]=="MC":
        m=model["mc"]
        return m["b"]+m["a"]*((pool["logit_mc"][ix]-m["mean"])/m["scale"])
    result=predict(model["parent"],pool,ix)
    if model["kind"]=="MM":
        x=np.stack([pool["logit_mc"][ix],pool["margin"][ix]],axis=1)
        return result+np.einsum('ij,j->i',matrix(model["basis"],x),model["coefficients"])
    mask=pool["strict_conflict"][ix].astype(bool)
    x=pool["anchor__"+model["variant"]][ix[mask],:model["inputs"]]
    result[mask]+=np.einsum('ij,j->i',matrix(model["basis"],x),model["coefficients"])
    return result


def gradient(model,pool,train,spec):
    if model["kind"]=="MC":
        m=model["mc"]
        x=np.stack([(pool["logit_mc"][train]-m["mean"])/m["scale"],np.ones(len(train))],axis=1)
        beta=np.array([m["a"],m["b"]]);ridge=spec["mc_calibrator"]["l2"]
    else:
        if model["kind"]=="anchor":
            train=train[pool["strict_conflict"][train].astype(bool)]
            raw=pool["anchor__"+model["variant"]][train,:model["inputs"]]
        else:
            raw=np.stack([pool["logit_mc"][train],pool["margin"][train]],axis=1)
        x=matrix(model["basis"],raw);beta=np.array(model["coefficients"]);ridge=spec["optimizer"]["l2"]
    logits=predict(model,pool,train)
    prob=np.exp(-np.logaddexp(0,-logits))
    w=pool["sampling_weight"][train];w=w/w.sum()
    result=np.einsum('ij,i->j',x,w*(prob-(1-pool["error"][train])))+ridge*beta
    if model["kind"]=="MC" and beta[0]<=1e-6+1e-12 and result[0]>0:
        result[0]=0
    return float(np.max(np.abs(result)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec=json.loads(Path(__file__).with_name("anchor_trust_protocol.json").read_text())
    rows=[]
    for stage in spec["stages"]:
        folder=args.runs/("seed42_"+stage)
        pool=load_pool(folder/"D_rel_pool.npz")
        saved=load_predictions(folder/"D_rel_predictions.npz")
        models=json.loads((folder/"outer_models.json").read_text())
        folds=folds_for(pool,spec)
        for name,model in models.items():
            prefix,method=name.split("__",1);fold=int(prefix.removeprefix("fold"))
            train,held=folds[fold]
            diff=float(np.max(np.abs(predict(model,pool,held)-saved[method][held])))
            grad=gradient(model,pool,train,spec)
            if diff>1e-10 or grad>1.01e-7:
                raise ValueError(f"Independent numerical mismatch: {stage}/{name}, {diff}, {grad}")
            rows.append(dict(stage=stage,model=name,replay_max_abs_diff=diff,projected_gradient_max=grad))
        final=json.loads((folder/"models.json").read_text())
        for name,model in final.items():
            grad=gradient(model,pool,np.arange(len(pool["error"])),spec)
            if grad>1.01e-7:
                raise ValueError("Final selected model gradient mismatch")
            rows.append(dict(stage=stage,model="final__"+name,projected_gradient_max=grad))
    result=dict(state="passed",scope="read-only independent selected-model replay and gradients; no new fitting",
        rows=rows,replays=sum("replay_max_abs_diff" in r for r in rows),gradient_checks=len(rows),
        max_replay_error=max(r.get("replay_max_abs_diff",0) for r in rows),max_projected_gradient=max(r["projected_gradient_max"] for r in rows),
        source_sha256=digest(Path(__file__)))
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"}))


if __name__=="__main__":
    main()
