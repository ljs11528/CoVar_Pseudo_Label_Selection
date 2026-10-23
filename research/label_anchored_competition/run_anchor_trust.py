"""Run the locked anchor-trust cache audit, with all fits before validation reads."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import torch

from .anchor_trust import (apply_model, candidate_sets, fit_candidates, read_pool, select)
from .anchor_trust_analysis import decide_gate, evaluate, folds_for, intervals, support_tables
from .bounded_probe import image_folds, logit_metrics
from .prepare_data import digest, source_revision
from .run_score_audit import write_csv, write_json


def display_name(name):
    return name.removeprefix("real_")


def nested_fit(pool,spec,output,device,update):
    if pool["role"] != "D_rel":
        raise ValueError("Validation cannot enter fitting or model selection")
    n=len(pool["error"]);y=1-pool["error"];w=pool["sampling_weight"]
    families=candidate_sets(spec)
    outer=folds_for(pool,spec)
    fold_records=[]
    for fold,(train,held) in enumerate(outer):
        ids=np.unique(pool["image"][train])
        local=image_folds([pool["image_ids"][i] for i in ids],np.searchsorted(ids,pool["image"][train]),
                          spec["folds"]["inner"],spec["folds"]["namespace"]+"|inner")
        def image_ids(ix):
            return [pool["image_ids"][i] for i in np.unique(pool["image"][ix])]
        fold_records.append(dict(outer=fold,train_images=image_ids(train),heldout_images=image_ids(held),
            inner=[dict(train_images=image_ids(train[a]),heldout_images=image_ids(train[b])) for a,b in local]))
    write_json(output/"folds.json",fold_records)
    candidates={name:np.full(n,np.nan) for name in dict.fromkeys(k for names in families.values() for k in names)}
    oof={display_name(name):np.full(n,np.nan) for name in families}
    oof["MC"]=np.full(n,np.nan)
    for seed in spec["permutation"]["seeds"]:
        oof[f"frozen_sham{seed}"]=np.full(n,np.nan)
    curves=[];selections=[];outer_models={};replay=[]
    def fit(train,held,context):
        update("fit_candidates",**context)
        def curve(row):
            curves.append(row)
        return fit_candidates(pool,train,held,spec,device,context,curve)
    for fold,(train,held) in enumerate(outer):
        ids=np.unique(pool["image"][train])
        local=image_folds([pool["image_ids"][i] for i in ids],np.searchsorted(ids,pool["image"][train]),
                          spec["folds"]["inner"],spec["folds"]["namespace"]+"|inner")
        inner={name:np.full(len(train),np.nan) for name in candidates}
        for j,(a,b) in enumerate(local):
            models=fit(train[a],train[b],dict(level="inner",outer=fold,inner=j))
            for name,model in models.items():
                inner[name][b]=apply_model(model,pool,train[b])
        if not all(np.isfinite(v).all() for v in inner.values()):
            raise ValueError("Incomplete inner OOF")
        losses={name:logit_metrics(y[train],value,w[train])["nll"] for name,value in inner.items()}
        chosen={name:select(losses,options,spec["correction"]["selection_tolerance"]) for name,options in families.items()}
        models=fit(train,held,dict(level="outer",outer=fold,inner=-1))
        for name,model in models.items():
            candidates[name][held]=apply_model(model,pool,held)
        oof["MC"][held]=candidates["MC"][held]
        outer_models[f"fold{fold}__MC"]=models["MC"]
        for family,candidate in chosen.items():
            name=display_name(family)
            model=models[candidate]
            oof[name][held]=candidates[candidate][held]
            outer_models[f"fold{fold}__{name}"]=dict(model,selected_candidate=candidate)
            selections.append(dict(level="outer_selection",outer=fold,method=name,chosen=candidate,
                **{f"nll_{c}":losses[c] for c in families[family]}))
            loaded=json.loads(json.dumps(model))
            error=float(np.max(np.abs(apply_model(loaded,pool,held)-oof[name][held])))
            replay.append(dict(outer=fold,method=name,max_abs_difference=error))
            if error != 0:
                raise ValueError("Saved model replay differs")
        for seed in spec["permutation"]["seeds"]:
            oof[f"frozen_sham{seed}"][held]=apply_model(models[chosen["real_T"]],pool,held,variant=f"sham{seed}")
        write_csv(output/"selection.csv",selections)
        write_csv(output/"optimization_curves.csv",curves)
    if not all(np.isfinite(v).all() for v in candidates.values()):
        raise ValueError("Incomplete outer candidate OOF")
    losses={name:logit_metrics(y,value,w)["nll"] for name,value in candidates.items()}
    final_choices={name:select(losses,options,spec["correction"]["selection_tolerance"]) for name,options in families.items()}
    full=np.arange(n)
    final_candidates=fit(full,full,dict(level="full_refit",outer=-1,inner=-1))
    final_models={"MC":final_candidates["MC"]}
    for family,candidate in final_choices.items():
        name=display_name(family)
        final_models[name]=dict(final_candidates[candidate],selected_candidate=candidate,fitting_role="D_rel")
        selections.append(dict(level="final_selection",outer=-1,method=name,chosen=candidate,
            **{f"nll_{c}":losses[c] for c in families[family]}))
    if not all(np.isfinite(v).all() for v in oof.values()):
        raise ValueError("Incomplete selected OOF")
    write_json(output/"models.json",final_models)
    write_json(output/"outer_models.json",outer_models)
    write_json(output/"model_replay.json",replay)
    write_csv(output/"selection.csv",selections)
    write_csv(output/"optimization_curves.csv",curves)
    write_csv(output/"candidate_oof_nll.csv",[dict(candidate=k,nll=v) for k,v in losses.items()])
    np.savez_compressed(output/"D_rel_predictions.npz",**oof)
    np.savez_compressed(output/"D_rel_candidate_predictions.npz",**candidates)
    return final_models,oof


def save_pool(path,pool):
    np.savez_compressed(path,**{k:np.asarray(v) for k,v in pool.items()})


def load_pool(path):
    with np.load(path,allow_pickle=False) as data:
        pool={k:data[k] for k in data.files}
    pool["role"]=str(pool["role"].item())
    pool["image_ids"]=pool["image_ids"].tolist()
    return pool


def load_predictions(path):
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def run(root,output,spec,device):
    if output.exists():
        raise FileExistsError("Preserve prior complete and failed audit runs")
    output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    started=time.monotonic();stage_elapsed={stage:0. for stage in spec["stages"]}
    signature=dict(source_commit=source_revision(package),protocol_sha256=digest(package/"anchor_trust_protocol.json"),
                   scope="old-BN existing-cache development; no teacher updates")
    stage=None;section_started=None;shared_image_ids=None
    def update(phase,**extra):
        elapsed=time.monotonic()-started
        if elapsed>spec["execution"]["maximum_total_hours"]*3600:
            raise TimeoutError("Fixed total time limit exceeded")
        current=stage_elapsed[stage]+time.monotonic()-section_started if stage is not None else 0
        if current>spec["execution"]["maximum_stage_hours"]*3600:
            raise TimeoutError("Fixed stage time limit exceeded")
        write_json(output/"status.json",dict(state="running",stage=stage,phase=phase,**extra,
                   elapsed_seconds=elapsed,stage_elapsed_seconds=current,signature=signature,
                   updated_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        print(json.dumps(dict(stage=stage,phase=phase,**extra,elapsed_seconds=round(elapsed,2))),flush=True)
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        if device.startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        for stage in spec["stages"]:
            section_started=time.monotonic()
            folder=output/("seed42_"+stage);folder.mkdir()
            export=root/"experiments"/f"frozen_seed42_{stage}_7c4c5c09"
            manifest=json.loads((export/"manifest.json").read_text())
            if manifest.get("synthetic") or manifest["signature"]["stage"]!=stage:
                raise ValueError("Formal run requires the registered real cache stage")
            write_json(folder/"input_signature.json",dict(**signature,stage=stage,
                       export_manifest_sha256=digest(export/"manifest.json"),prototype_sha256=digest(export/"prototypes.npz")))
            update("read_D_rel")
            pool,check=read_pool(export,"D_rel",spec,lambda **kw:update("read_D_rel",**kw))
            if shared_image_ids is None:
                shared_image_ids=list(pool["image_ids"])
            elif shared_image_ids != pool["image_ids"]:
                raise ValueError("Cross-stage image ordering differs; paired stage resampling would be invalid")
            save_pool(folder/"D_rel_pool.npz",pool)
            write_json(folder/"D_rel_cache_checks.json",check)
            models,oof=nested_fit(pool,spec,folder,device,update)
            write_json(folder/"FITTING_COMPLETE.json",dict(signature=signature,models_sha256=digest(folder/"models.json"),
                       outer_models_sha256=digest(folder/"outer_models.json"),completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            stage_elapsed[stage]+=time.monotonic()-section_started
            del pool,models,oof
        stage=None;section_started=None
        write_json(output/"ALL_FITTING_COMPLETE.json",dict(signature=signature,
            stage_models={s:digest(output/("seed42_"+s)/"models.json") for s in spec["stages"]},
            completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        results={}
        for stage in spec["stages"]:
            section_started=time.monotonic()
            folder=output/("seed42_"+stage)
            pool=load_pool(folder/"D_rel_pool.npz");oof=load_predictions(folder/"D_rel_predictions.npz")
            update("D_rel_ranking")
            ranking,curves=evaluate(pool,oof,spec)
            write_csv(folder/"D_rel_ranking.csv",ranking);write_csv(folder/"D_rel_risk_coverage.csv",curves)
            uncertainty=intervals(pool,oof,spec,device,update)
            write_csv(folder/"D_rel_intervals.csv",uncertainty)
            update("D_rel_support")
            support,conditional=support_tables(pool,oof,spec)
            write_csv(folder/"D_rel_support.csv",support);write_csv(folder/"D_rel_conditional_ranking.csv",conditional)
            results[stage]=dict(ranking=ranking,intervals=uncertainty)
            stage_elapsed[stage]+=time.monotonic()-section_started
            del pool,oof
        stage=None;section_started=None
        gate=decide_gate(results,spec)
        write_json(output/"gate.json",gate)
        # Gate is committed to disk before any descriptive validation reads.
        for stage in spec["stages"]:
            section_started=time.monotonic()
            folder=output/("seed42_"+stage)
            export=root/"experiments"/f"frozen_seed42_{stage}_7c4c5c09"
            update("read_validation_after_all_fits")
            pool,check=read_pool(export,"validation",spec,lambda **kw:update("read_validation",**kw))
            train=load_pool(folder/"D_rel_pool.npz")
            if set(train["image_ids"])&set(pool["image_ids"]):
                raise ValueError("Cross-role images overlap")
            del train
            models=json.loads((folder/"models.json").read_text())
            predictions={name:apply_model(model,pool) for name,model in models.items()}
            for seed in spec["permutation"]["seeds"]:
                predictions[f"frozen_sham{seed}"]=apply_model(models["T"],pool,variant=f"sham{seed}")
            update("validation_description")
            ranking,curves=evaluate(pool,predictions,spec)
            support,conditional=support_tables(pool,predictions,spec)
            write_csv(folder/"validation_ranking.csv",ranking);write_csv(folder/"validation_risk_coverage.csv",curves)
            write_csv(folder/"validation_support.csv",support);write_csv(folder/"validation_conditional_ranking.csv",conditional)
            write_json(folder/"validation_cache_checks.json",check)
            save_pool(folder/"validation_pool.npz",pool)
            np.savez_compressed(folder/"validation_predictions.npz",**predictions)
            stage_elapsed[stage]+=time.monotonic()-section_started
            complete=dict(state="complete",signature=signature,stage=stage,elapsed_seconds=stage_elapsed[stage],
                files={f.name:dict(bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(folder.iterdir()) if f.is_file()})
            write_json(folder/"COMPLETE.json",complete)
            del pool,predictions
        stage=None;section_started=None
        complete=dict(state="complete",signature=signature,elapsed_seconds=time.monotonic()-started,
            completed_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),torch=torch.__version__,numpy=np.__version__,
            gate=gate["decision"],stage_receipts={s:digest(output/("seed42_"+s)/"COMPLETE.json") for s in spec["stages"]})
        write_json(output/"COMPLETE.json",complete);write_json(output/"status.json",complete)
        print(json.dumps(complete),flush=True)
    except Exception as exc:
        write_json(output/"FAILED.json",dict(state="failed",signature=signature,stage=stage,error=repr(exc),elapsed_seconds=time.monotonic()-started))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    spec=json.loads(Path(__file__).with_name("anchor_trust_protocol.json").read_text())
    run(args.root,args.output,spec,args.device)


if __name__=="__main__":
    main()
