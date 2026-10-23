"""Fold-aware ranking, paired image intervals and fixed support diagnostics."""
from __future__ import annotations

import numpy as np

from .analyze_frozen import coverage_at_precision
from .bounded_probe import image_folds, logit_metrics, sigmoid
from .gpu_bootstrap import KEYS, paired_image_bootstrap_gpu
from .ranking_metrics import Ranking
from .score_audit_diagnostics import scopes, support_indices


def folds_for(pool, spec):
    return image_folds(pool["image_ids"],pool["image"],spec["folds"]["outer"],spec["folds"]["namespace"])


def basic_metrics(pool, ix, risk, spec, logits=None):
    e,w = pool["error"][ix],pool["sampling_weight"][ix]
    result = Ranking(e,risk[ix]).evaluate(w,spec["statistics"]["coverage_grid"])
    for target in spec["statistics"]["precision_targets"]:
        result[f"coverage_at_precision_{target:g}"] = coverage_at_precision(e,risk[ix],w,target)
    if logits is not None:
        result.update(logit_metrics(1-e,logits[ix],w))
        weight = sigmoid(logits[ix])
        result.update(mean_training_weight=float(np.average(weight,weights=w)),
                      error_fraction_of_weight=float(np.sum(w*weight*e)/np.sum(w*weight)),
                      retained_correct_mass=float(np.sum(w*weight*(1-e))/np.sum(w*(1-e))) if np.sum(w*(1-e)) else np.nan)
    return result


def risks_for(pool, predictions):
    risks = {name:-values for name,values in predictions.items()}
    risks.update(raw_MC=-pool["mc"],raw_logit_MC=-pool["logit_mc"],raw_margin=-pool["margin"],
                 raw_entropy=pool["entropy"],raw_p2=pool["p2"])
    return risks


def evaluate(pool, predictions, spec):
    risks = risks_for(pool,predictions)
    rows, curves = [],[]
    fold_list = folds_for(pool,spec) if pool["role"] == "D_rel" else []
    grid = np.linspace(.01,1,100)
    for scope,mask in scopes(pool).items():
        groups = [("pooled",np.flatnonzero(mask))]
        groups += [(str(fold),held[mask[held]]) for fold,(_,held) in enumerate(fold_list)]
        for fold,ix in groups:
            info = support_indices(pool,ix,spec["strata"]["minimum_support"] if fold=="pooled" else spec["strata"]["fold_minimum_support"])
            if not len(ix):
                continue
            for name,risk in risks.items():
                metrics = basic_metrics(pool,ix,risk,spec,predictions.get(name))
                rows.append(dict(role=pool["role"],scope=scope,fold=fold,method=name,**info,**metrics))
                if scope in ("all","conflict") and name in ("MC","MM","G","S","T","raw_entropy","raw_margin"):
                    points = Ranking(pool["error"][ix],risk[ix]).evaluate(pool["sampling_weight"][ix],grid)
                    curves.extend(dict(role=pool["role"],scope=scope,fold=fold,method=name,coverage=float(c),
                                       risk=1-points[f"precision_at_{c:g}"]) for c in grid)
        if fold_list:
            for name in risks:
                parts = [r for r in rows if r["role"]==pool["role"] and r["scope"]==scope and r["method"]==name and r["fold"] in ("0","1","2")]
                if not parts:
                    continue
                metrics = [k for k in parts[0] if k not in {"role","scope","fold","method",*info}]
                rows.append(dict(role=pool["role"],scope=scope,fold="mean",method=name,folds=len(parts),
                                 adequate_support=all(p["adequate_support"] for p in parts),
                                 **{k:float(np.mean([p[k] for p in parts])) for k in metrics}))
            for name in (("MC","MM","G","S","T","raw_entropy","raw_margin") if scope in ("all","conflict") else ()):
                for c in grid:
                    parts = [r["risk"] for r in curves if r["scope"]==scope and r["method"]==name and r["fold"] in ("0","1","2") and r["coverage"]==float(c)]
                    if parts:
                        curves.append(dict(role=pool["role"],scope=scope,fold="mean",method=name,coverage=float(c),risk=float(np.mean(parts))))
    return rows,curves


def intervals(pool,predictions,spec,device,update):
    seeds = spec["permutation"]["seeds"]
    methods = ["MC","MM","G","S","T"]+[f"sham{s}_T" for s in seeds]
    all_rows=[]
    metrics = list(KEYS)+["precision_at_0.9"]
    pairs = spec["statistics"]["primary_contrasts"]+[["T","MC"],["MM","MC"],["G","MM"],["S","G"]]
    pairs += [["T",f"sham{s}_T"] for s in seeds]
    folds=folds_for(pool,spec)
    for scope in ("all","conflict"):
        mask=scopes(pool)[scope]
        per_fold=[];points=[]
        for fold,(_,held) in enumerate(folds):
            ix=held[mask[held]]
            images=np.unique(pool["image"][held])
            groups=np.searchsorted(images,pool["image"][ix])
            if not len(ix):
                raise ValueError("Empty fixed fold scope; report unsupported rather than change folds")
            update("bootstrap",scope=scope,fold=fold)
            _,samples=paired_image_bootstrap_gpu(pool["error"][ix],{m:-predictions[m][ix] for m in methods},
                pool["sampling_weight"][ix],groups,replicates=spec["statistics"]["bootstrap_replicates"],
                seed=spec["statistics"]["bootstrap_seed"]+fold,baseline="MC",coverages=(.9,),
                device=device,return_samples=True,all_images=len(images))
            per_fold.append(samples)
            points.append({m:Ranking(pool["error"][ix],-predictions[m][ix]).evaluate(pool["sampling_weight"][ix],(.9,)) for m in methods})
        aggregate={m:np.mean([f[m] for f in per_fold],axis=0) for m in methods}
        point={m:np.array([np.mean([f[m][key] for f in points]) for key in metrics]) for m in methods}
        aggregate["sham_mean"]=np.mean([aggregate[f"sham{s}_T"] for s in seeds],axis=0)
        point["sham_mean"]=np.mean([point[f"sham{s}_T"] for s in seeds],axis=0)
        for a,b in pairs:
            for j,metric in enumerate(metrics):
                confidence=spec["statistics"]["primary_confidence"] if scope=="conflict" and metric=="error_ap" and [a,b] in spec["statistics"]["primary_contrasts"] else .95
                differences=aggregate[a][:,j]-aggregate[b][:,j]
                finite=differences[np.isfinite(differences)]
                if len(finite) < .99*len(differences):
                    raise ValueError("Too many undefined bootstrap replicates")
                lo,hi=np.quantile(finite,[(1-confidence)/2,1-(1-confidence)/2])
                all_rows.append(dict(role="D_rel",scope=scope,estimator="within_fold_mean",method=a,comparator=b,
                    metric=metric,delta=float(point[a][j]-point[b][j]),confidence=confidence,ci_low=float(lo),ci_high=float(hi),
                    valid_replicates=len(finite),images=len(pool["image_ids"])))
    return all_rows


def support_tables(pool,predictions,spec):
    n=len(pool["error"]);w=pool["sampling_weight"];e=pool["error"].astype(bool)
    conflict=pool["strict_conflict"].astype(bool)
    proto_correct=pool["prototype_label"]==pool["GT"]
    teacher_correct=~e
    folds=folds_for(pool,spec) if pool["role"]=="D_rel" else []
    groups=[("overall",0,np.ones(n,bool),{})]
    mb=np.searchsorted(spec["strata"]["mc_edges"],pool["mc"],side="right")-1
    db=np.searchsorted(spec["strata"]["margin_edges"],pool["margin"],side="right")-1
    nm,nd=len(spec["strata"]["mc_edges"])-1,len(spec["strata"]["margin_edges"])-1
    if np.any((mb<0)|(mb>=nm)|(db<0)|(db>=nd)):
        raise ValueError("Values outside fixed strata")
    sb={k:np.searchsorted(spec["support"]["bands"],pool["count_"+k],side="right")-1 for k in ("a","b","c")}
    for a in range(nm):
        for b in range(nd):
            mask=(mb==a)&(db==b)
            groups.append(("MC_margin",a*nd+b,mask,dict(mc_bin=a,margin_bin=b)))
            for c in range(3):
                groups.append(("MC_margin_c_support",(a*nd+b)*3+c,mask&(sb["c"]==c),dict(mc_bin=a,margin_bin=b,support_bin=c)))
    for c in range(spec["strata"]["classes"]):
        for boundary in (0,1):
            groups.append(("class_boundary",c*2+boundary,(pool["pseudo_label"]==c)&(pool["boundary"]==boundary),dict(predicted_class=c,boundary=boundary)))
        groups.append(("prototype_class",c,pool["prototype_label"]==c,dict(prototype_class=c,prototype_images=int(pool["prototype_counts"][c]))))
    for k in ("a","b","c"):
        for band in range(3):
            groups.append(("support_"+k,band,sb[k]==band,dict(support_bin=band)))
    rows=[]; ranks=[]
    selected_methods=[m for m in ("MC","MM","G","S","T") if m in predictions]
    for kind,group,mask,extra in groups:
        for scope,restriction in (("all",np.ones(n,bool)),("conflict",conflict)):
            ix=np.flatnonzero(mask&restriction)
            info=support_indices(pool,ix,spec["strata"]["minimum_support"])
            row=dict(role=pool["role"],scope=scope,stratum=kind,group=group,**extra,**info)
            if len(ix):
                mass=float(w[ix].sum())
                row.update(estimated_pixels=mass,error_rate=float(np.average(e[ix],weights=w[ix])),
                    teacher_wrong_proto_correct=float(np.average(e[ix]&proto_correct[ix],weights=w[ix])),
                    teacher_correct_proto_wrong=float(np.average(teacher_correct[ix]&~proto_correct[ix],weights=w[ix])),
                    both_wrong=float(np.average(e[ix]&~proto_correct[ix],weights=w[ix])),
                    agreeing_wrong=float(np.average(e[ix]&(pool["pseudo_label"][ix]==pool["prototype_label"][ix]),weights=w[ix])))
                for k in ("mc","margin","p2","count_a","count_b","count_c"):
                    row[k+"_mean"]=float(np.average(pool[k][ix],weights=w[ix]))
                for j,k in ((3,"e"),(4,"top2_objection")):
                    row[k+"_mean"]=float(np.average(pool["anchor__real"][ix,j],weights=w[ix]))
                for method in selected_methods:
                    # Within-fold means are descriptive for sparse strata: retain
                    # the number of contributing folds rather than hiding gaps.
                    parts=[]
                    for _,held in folds or [(None,np.arange(n))]:
                        jj=held[(mask&restriction)[held]]
                        if len(jj):
                            parts.append(basic_metrics(pool,jj,-predictions[method],spec))
                    ranks.append(dict(role=pool["role"],scope=scope,stratum=kind,group=group,method=method,
                        adequate_support=info["adequate_support"],folds=len(parts),
                        **{k:float(np.mean([p[k] for p in parts])) for k in ("error_ap","auroc","precision_at_0.9","coverage_at_precision_0.95")}))
            rows.append(row)
    return rows,ranks


def decide_gate(stages,spec):
    checks=[]
    def add(name,passed,**data):
        checks.append(dict(check=name,passed=bool(passed),**data))
    def interval(stage,scope,a,b):
        return next(r for r in stages[stage]["intervals"] if r["scope"]==scope and r["method"]==a and r["comparator"]==b and r["metric"]=="error_ap")
    for a,b in spec["statistics"]["primary_contrasts"]:
        late=interval("late","conflict",a,b)
        add(f"late_CI_{a}_minus_{b}",late["ci_low"]>0,**late)
        values=[interval(stage,"conflict",a,b)["delta"] for stage in spec["stages"]]
        add(f"stage_consistency_{a}_minus_{b}",sum(v>0 for v in values)>=spec["gate"]["positive_stages_per_contrast"],deltas=values)
    delta=interval("late","conflict","T","MM")["delta"]
    add("late_practical_AP_gain",delta>=spec["gate"]["late_T_minus_MM_min_ap"],delta=delta)
    for seed in spec["permutation"]["seeds"]:
        row=interval("late","conflict","T",f"sham{seed}_T")
        add(f"late_beats_shuffle_{seed}",row["delta"]>0,delta=row["delta"])
    for stage in spec["stages"]:
        for base in ("MM","MC"):
            row=interval(stage,"all","T",base)
            add(f"{stage}_overall_noninferiority_{base}",row["ci_low"]>-spec["gate"]["all_pixel_AP_noninferiority_margin"],ci_low=row["ci_low"])
    table=stages["late"]["ranking"]
    def ranking(method,fold):
        return next(r for r in table if r["role"]=="D_rel" and r["scope"]=="conflict" and r["method"]==method and r["fold"]==fold)
    for metric in ("precision_at_0.9","coverage_at_precision_0.95"):
        difference=ranking("T","mean")[metric]-ranking("MM","mean")[metric]
        add("late_workpoint_"+metric,difference>=-1e-12,delta=difference)
    for fold in ("0","1","2"):
        row=ranking("T",fold)
        add("late_support_fold"+fold,row["adequate_support"],error_images=row["error_images"],correct_images=row["correct_images"])
    return dict(passed=all(c["passed"] for c in checks),checks=checks,teacher_training_authorized=False,
                scope="fixed development resource decision; validation excluded; old-BN single teacher",
                decision="eligible_to_propose_one_BN_confirmation" if all(c["passed"] for c in checks) else "stop_this_fixed_support_conditioned_scorer")
