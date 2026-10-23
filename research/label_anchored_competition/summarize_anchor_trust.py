"""Recompute the fixed gate and compile aggregate anchor-trust results and figures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .anchor_trust_analysis import decide_gate
from .run_score_audit import write_csv, write_json


def read_csv(path):
    def value(v):
        if v in ("True","False"):
            return v=="True"
        if v=="":
            return None
        try:
            return float(v)
        except ValueError:
            return v
    with path.open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    result=[]
    for row in rows:
        parsed={k:value(v) for k,v in row.items()}
        if "fold" in parsed:
            parsed["fold"]=row["fold"]
        result.append(parsed)
    return result


def verify_snapshot(source):
    receipt=json.loads((source/"snapshot_receipt.json").read_text())
    for name,meta in receipt["files"].items():
        data=(source/name).read_bytes()
        if len(data)!=meta["bytes"] or hashlib.sha256(data).hexdigest()!=meta["sha256"]:
            raise ValueError("Local snapshot changed: "+name)
    return len(receipt["files"])


def summarize(source,output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    spec=json.loads(Path(__file__).with_name("anchor_trust_protocol.json").read_text())
    checked=verify_snapshot(source)
    results={}; ranking=[];ci=[];selections=[];supports=[];conditional=[];curves=[];optim=[];replays=[];checks=[]
    for stage in spec["stages"]:
        folder=source/("seed42_"+stage)
        r=read_csv(folder/"D_rel_ranking.csv");intervals=read_csv(folder/"D_rel_intervals.csv")
        results[stage]=dict(ranking=r,intervals=intervals)
        ranking.extend(dict(stage=stage,**row) for row in r+read_csv(folder/"validation_ranking.csv"))
        ci.extend(dict(stage=stage,**row) for row in intervals)
        selections.extend(dict(stage=stage,**row) for row in read_csv(folder/"selection.csv"))
        supports.extend(dict(stage=stage,**row) for role in ("D_rel","validation") for row in read_csv(folder/(role+"_support.csv")))
        conditional.extend(dict(stage=stage,**row) for role in ("D_rel","validation") for row in read_csv(folder/(role+"_conditional_ranking.csv")))
        curves.extend(dict(stage=stage,**row) for role in ("D_rel","validation") for row in read_csv(folder/(role+"_risk_coverage.csv")))
        opt=read_csv(folder/"optimization_curves.csv")
        fits={}
        for row in opt:
            fits[(row["level"],row["outer"],row["inner"],row["candidate"])]=row
        maximum=max(row["max_gradient"] for row in fits.values())
        if maximum>spec["optimizer"]["gradient_tolerance"]:
            raise ValueError("Unconverged formal fit")
        optim.append(dict(stage=stage,fits=len(fits),max_iterations=max(row["iteration"] for row in fits.values()),
                          median_iterations=float(np.median([row["iteration"] for row in fits.values()])),
                          max_final_gradient=maximum,converged=True))
        replay=json.loads((folder/"model_replay.json").read_text())
        if any(row["max_abs_difference"] != 0 for row in replay):
            raise ValueError("Model replay mismatch")
        replays.append(dict(stage=stage,replays=len(replay),maximum_error=max(row["max_abs_difference"] for row in replay)))
        for role in ("D_rel","validation"):
            check=json.loads((folder/(role+"_cache_checks.json")).read_text())
            checks.append(dict(stage=stage,role=role,**{k:v for k,v in check.items() if k not in ("role","feature_moments")}))
        # Independent arithmetic check of the primary estimator from exported folds.
        for scope in ("all","conflict","high_mc","high_mc_conflict"):
            for method in ("MC","MM","G","S","T"):
                subset=[row for row in r if row["scope"]==scope and row["method"]==method]
                parts=[row for row in subset if row["fold"] in ("0","1","2")]
                mean=next(row for row in subset if row["fold"]=="mean")
                for key in ("error_ap","auroc","precision_at_0.9","coverage_at_precision_0.95","nll"):
                    np.testing.assert_allclose(mean[key],sum(row[key] for row in parts)/len(parts),atol=1e-13,rtol=1e-13)
    gate=decide_gate(results,spec)
    remote_gate=json.loads((source/"gate.json").read_text())
    if gate!=remote_gate:
        raise ValueError("Local and server gate disagree")
    for row in supports:
        if row["scope"]=="conflict" and row["sampled_pixels"]:
            total=sum(row[k] for k in ("teacher_wrong_proto_correct","teacher_correct_proto_wrong","both_wrong"))
            np.testing.assert_allclose(total,1.,atol=1e-12,rtol=1e-12)
    methods={"MC","MM","G","S","T","raw_MC","raw_logit_MC","raw_margin","raw_entropy","raw_p2"}
    methods.update(f"sham{s}_T" for s in spec["permutation"]["seeds"])
    primary=[row for row in ranking if row["method"] in methods and row["fold"] in ("mean","pooled")]
    write_csv(output/"primary_ranking.csv",primary)
    write_csv(output/"all_ranking.csv",ranking)
    write_csv(output/"intervals.csv",ci)
    write_csv(output/"selections.csv",selections)
    write_csv(output/"support.csv",supports)
    write_csv(output/"conditional_ranking.csv",conditional)
    write_csv(output/"risk_coverage.csv",curves)
    write_csv(output/"optimizer_convergence.csv",optim)
    write_json(output/"cache_checks.json",checks)
    write_json(output/"gate.json",gate)
    write_json(output/"verification.json",dict(aggregate_files_verified=checked,replays=replays,
        fold_means_independently_recomputed=True,resource_gate_matches_server=True,
        conflict_outcome_partition_verified=True,
        scope="independent aggregate arithmetic and transport integrity; not independent training repetition"))
    draw(primary,supports,ci,spec,output)
    manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output/"manifest.json",manifest)
    print(json.dumps(dict(output=str(output),ranking_rows=len(ranking),gate=gate["decision"],verified_files=checked)))


def draw(ranking,support,intervals,spec,output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.hashsalt":"anchor_trust_audit_20261023"})
    colors={"MC":"#666666","MM":"#0072B2","G":"#009E73","S":"#E69F00","T":"#D55E00"}
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0),sharex=True)
    for ax,scope in zip(axes,("all","conflict")):
        for method,color in colors.items():
            values=[next(row["error_ap"] for row in ranking if row["stage"]==stage and row["role"]=="D_rel" and row["scope"]==scope and row["fold"]=="mean" and row["method"]==method)*100 for stage in spec["stages"]]
            ax.plot(range(3),values,"o-",label=method,color=color,lw=1.6,ms=5)
        sham=np.array([[next(row["error_ap"] for row in ranking if row["stage"]==stage and row["role"]=="D_rel" and row["scope"]==scope and row["fold"]=="mean" and row["method"]==f"sham{s}_T")*100 for s in spec["permutation"]["seeds"]] for stage in spec["stages"]])
        ax.plot(range(3),sham.mean(1),"s--",label="Refitted shuffle mean",color="#AA4499",ms=4)
        ax.fill_between(range(3),sham.min(1),sham.max(1),color="#AA4499",alpha=.12,label="Five-shuffle range")
        ax.set_xticks(range(3),spec["stages"]);ax.set_title("All pixels" if scope=="all" else "Strict prototype conflicts")
        ax.set_ylabel("Error AP (%) · mean within CV folds");ax.grid(axis="y",alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,bbox_to_anchor=(.5,-.02))
    fig.suptitle("Existing-cache development · one VOC teacher with old BN snapshots",fontsize=11)
    fig.tight_layout(rect=(0,.12,1,.96))
    for ext in ("png","svg"):
        fig.savefig(output/("fold_metrics."+ext),dpi=180,bbox_inches="tight",metadata={"Date":None} if ext=="svg" else None)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.1),sharey=True)
    outcomes=[("teacher_wrong_proto_correct","Teacher wrong, anchor correct","#0072B2"),
              ("teacher_correct_proto_wrong","Teacher correct, anchor wrong","#E69F00"),
              ("both_wrong","Both wrong","#CC6677")]
    for ax,role in zip(axes,("D_rel","validation")):
        rows=[next(row for row in support if row["stage"]=="late" and row["role"]==role and row["scope"]=="conflict" and row["stratum"]=="support_c" and row["group"]==band) for band in range(3)]
        bottom=np.zeros(3)
        for key,label,color in outcomes:
            values=np.array([row.get(key,0) or 0 for row in rows])*100
            ax.bar(range(3),values,bottom=bottom,label=label,color=color,width=.65)
            bottom+=values
        ax.set_xticks(range(3),[f"{label}\nE/C images: {int(row['error_images'])}/{int(row['correct_images'])}" for label,row in zip(("<50","50–199","≥200"),rows)])
        ax.set_xlabel("Independent D_sup images supporting the winning prototype")
        ax.set_title("D_rel (development)" if role=="D_rel" else "Validation (previously inspected)")
        ax.set_ylim(0,105);ax.set_ylabel("Conflict pixel mass (%)")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,-.05))
    fig.suptitle("Late stage · fixed support bands · associations, not a causal sample-size effect",fontsize=11)
    fig.tight_layout(rect=(0,.09,1,.96))
    for ext in ("png","svg"):
        fig.savefig(output/("support_conflicts."+ext),dpi=180,bbox_inches="tight",metadata={"Date":None} if ext=="svg" else None)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    summarize(args.source,args.output)


if __name__=="__main__":
    main()
