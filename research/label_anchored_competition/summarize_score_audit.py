"""Summarize a completed bounded audit; never fit or choose a model."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from .prepare_data import digest
from .run_score_audit import key, write_csv, write_json
from .score_audit_gate import decide


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def combine(source, spec, output):
    ranking=[]; intervals=[]; stability=[]; primary=[]; curves=[]; distributions=[]
    feature_shift=[]; support=[]; selections=[]; joint_deltas=[]; precision=[]
    completions=[]; checks=[]
    pairs=spec["statistics"]["primary_contrasts"]+[["e","MC"]]
    primary_seed=spec["statistics"]["primary_seed"]
    primary_budget=spec["statistics"]["primary_budget"]
    for stage in spec["stages"]:
        folder=source/('seed42_'+stage)
        complete=json.loads((folder/'COMPLETE.json').read_text())
        if complete["state"]!="complete" or complete["signature"]["stage"]!=stage:
            raise ValueError("Missing or mismatched completed stage")
        for name,meta in complete["files"].items():
            if name.endswith('.npz'):
                continue  # Large per-pixel outputs stay on the server; transport receipt verifies them.
            file=folder/name
            if not file.exists() or file.stat().st_size!=meta["bytes"] or digest(file)!=meta["sha256"]:
                raise ValueError("Completed small artifact hash mismatch: "+str(file))
        completions.append(complete)
        local_ranking=read_csv(folder/'ranking.csv')
        lookup={(r["role"],r["scope"],r["method"]):r for r in local_ranking}
        ranking.extend(dict(stage=stage,**r) for r in local_ranking)
        intervals.extend(dict(stage=stage,role="D_rel",**r) for r in read_csv(folder/'D_rel_intervals.csv'))
        selections.extend(dict(stage=stage,**r) for r in read_csv(folder/'selection.csv'))
        curves.extend(dict(stage=stage,**r) for r in read_csv(folder/'oof_curves.csv'))
        distributions.extend(dict(stage=stage,**r) for r in read_csv(folder/'distributions.csv'))
        for row in read_csv(folder/'precision_targets.csv'):
            if row['method']=='MC' or row['method'].startswith(f's{primary_seed}_u{primary_budget}_'):
                precision.append(dict(stage=stage,**row))
        for role in ("D_rel","validation"):
            for scope in ("all","conflict","high_mc","high_mc_conflict"):
                for name in ["MC",*spec["feature_sets"]]:
                    model="MC" if name=="MC" else key(primary_seed,primary_budget,name)
                    primary.append(dict(stage=stage,feature_set=name,**lookup[role,scope,model]))
                for seed in spec["correction"]["fit_seeds"]:
                    for budget in spec["correction"]["budgets"]:
                        for a,b in pairs:
                            left=lookup[role,scope,key(seed,budget,a)]
                            right=lookup[role,scope,"MC" if b=="MC" else key(seed,budget,b)]
                            stability.append(dict(stage=stage,role=role,scope=scope,seed=seed,budget=budget,method=a,comparator=b,
                                delta_error_ap=float(left["error_ap"])-float(right["error_ap"]),
                                delta_auroc=float(left["auroc"])-float(right["auroc"]),
                                delta_nll=float(left["nll"])-float(right["nll"]),
                                delta_precision_at_0_9=float(left["precision_at_0.9"])-float(right["precision_at_0.9"])))
        real=json.loads((folder/'real_cache_checks.json').read_text())
        checks.append(dict(stage=stage,roles={role:{k:v for k,v in r.items() if k!='feature_moments'} for role,r in real.items()}))
        for train,test in zip(real['D_rel']['feature_moments'],real['validation']['feature_moments']):
            if train['predicted_class']!=test['predicted_class']:
                raise ValueError('Feature class mismatch')
            a,b=np.asarray(train['feature_mean']),np.asarray(test['feature_mean'])
            v=(np.asarray(train['feature_variance'])+np.asarray(test['feature_variance']))/2
            valid=train['estimated_pixels']>0 and test['estimated_pixels']>0
            feature_shift.append(dict(stage=stage,predicted_class=train['predicted_class'],
                D_rel_estimated_pixels=train['estimated_pixels'],validation_estimated_pixels=test['estimated_pixels'],
                prototype_support_images='' if train['predicted_class']<0 else real['D_rel']['prototype_support_images'][train['predicted_class']],
                mean_vector_L2=float(np.linalg.norm(a-b)) if valid else None,
                mean_vector_cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))) if valid else None,
                dimensionwise_SMD_RMS=float(np.sqrt(np.mean((a-b)**2/(v+1e-12)))) if valid else None))
        supports=read_csv(folder/'joint_support.csv')
        for role in ('D_rel','validation'):
            for kind in ('MC_prototype','class_boundary','four_way'):
                rows=[r for r in supports if r['role']==role and r['stratum']==kind]
                support.append(dict(stage=stage,role=role,stratum=kind,cells=len(rows),
                    nonempty=sum(int(r['sampled_pixels'])>0 for r in rows),
                    adequate=sum(r['adequate_support']=='True' for r in rows)))
        grouped={}
        for row in read_csv(folder/'joint_conditional.csv'):
            grouped.setdefault((row['role'],row['stratum'],row['group']),{})[row['method']]=row
        for (role,kind,group),rows in grouped.items():
            for a,b in pairs:
                left=rows[key(primary_seed,primary_budget,a)]
                right=rows['MC' if b=='MC' else key(primary_seed,primary_budget,b)]
                joint_deltas.append(dict(stage=stage,role=role,stratum=kind,group=group,
                    mc_bin=left.get('mc_bin',''),prototype_margin_bin=left.get('prototype_margin_bin',''),
                    predicted_class=left.get('predicted_class',''),boundary=left.get('boundary',''),
                    method=a,comparator=b,adequate_support=left['adequate_support'],
                    sampled_pixels=left['sampled_pixels'],errors=left['errors'],error_images=left['error_images'],
                    delta_error_ap=float(left['error_ap'])-float(right['error_ap']),
                    delta_precision_at_0_9=float(left['precision_at_0.9'])-float(right['precision_at_0.9'])))
    source_commits={c['signature']['source_commit'] for c in completions}
    protocol_hashes={c['signature']['protocol_sha256'] for c in completions}
    if len(source_commits)!=1 or len(protocol_hashes)!=1 or next(iter(protocol_hashes))!=digest(Path(__file__).with_name('score_audit_protocol.json')):
        raise ValueError('Inconsistent execution source or fixed protocol')
    gate=decide(spec,intervals,[r for r in stability if r['role']=='D_rel'])
    outputs=dict(ranking_primary=primary,primary_intervals=intervals,stability=stability,
        selections=selections,oof_curves=curves,distributions=distributions,feature_shift=feature_shift,
        support_summary=support,joint_deltas=joint_deltas,precision_targets_primary=precision)
    output.mkdir(parents=True,exist_ok=False)
    for name,rows in outputs.items():
        write_csv(output/(name+'.csv'),rows)
    write_json(output/'gate.json',gate)
    write_json(output/'cache_checks_summary.json',checks)
    write_json(output/'summary.json',dict(source_commits=sorted(source_commits),protocol_sha256=next(iter(protocol_hashes)),
        completed_stages=spec['stages'],elapsed_seconds_by_stage={c['stage']:c['elapsed_seconds'] for c in completions},
        gate=gate['eligible_for_one_bn_confirmation_teacher'],validation_role='previously inspected development description only',
        number_of_primary_ranking_rows=len(primary),limitation='One old-BN teacher trajectory; sampler seeds are not independent teachers'))
    return outputs,gate


def figures(output,rows,spec):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(MC='#30343b',prototype='#2980b9',product='#8e44ad',uniform='#d68910',e='#c0392b',controls='#138d75',controls_e='#21618c')
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,(role,scope) in zip(axs.flat,[(r,s) for r in ('D_rel','validation') for s in ('all','conflict')]):
        for name in colors:
            values=[next(float(r['error_ap'])*100 for r in rows['ranking_primary'] if r['stage']==st and r['role']==role and r['scope']==scope and r['feature_set']==name) for st in spec['stages']]
            ax.plot(spec['stages'],values,'o-',label=name,color=colors[name],lw=1.6)
        ax.set(title=f'{role}: {scope}',ylabel='Error AP (%)')
        ax.grid(alpha=.2)
    axs[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('Fixed MC-offset probes | seed 2027, 2,000 updates\nD_rel nested OOF; validation is development description')
    fig.savefig(output/'ranking_overview.png',dpi=170)
    fig.savefig(output/'ranking_overview.svg')
    plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,form in enumerate(('linear','additive')):
        for j,stage in enumerate(spec['stages']):
            ax=axs[i,j]
            for name in spec['feature_sets']:
                r=sorted((r for r in rows['oof_curves'] if r['stage']==stage and r['form']==form and int(r['seed'])==2027),key=lambda r:int(r['update']))
                r=[a for a in r if a['method']==name]
                ax.plot([int(a['update']) for a in r],[float(a['oof_nll']) for a in r],label=name,color=colors[name])
            ax.set(title=f'{stage} / {form}',xlabel='Updates',ylabel='Candidate image-OOF NLL')
            ax.axvline(500,color='gray',ls=':',lw=.8);ax.axvline(2000,color='gray',ls=':',lw=.8)
            ax.grid(alpha=.2)
    axs[0,0].legend(ncol=2,fontsize=8)
    fig.suptitle('Training path diagnostics | fixed candidates; no new early stopping search')
    fig.savefig(output/'oof_curves.png',dpi=170);fig.savefig(output/'oof_curves.svg')
    plt.close(fig)
    for svg in output.glob('*.svg'):
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    spec=json.loads(Path(__file__).with_name('score_audit_protocol.json').read_text())
    rows,gate=combine(args.source,spec,args.output)
    figures(args.output,rows,spec)
    write_json(args.output/'MANIFEST.json',{f.name:dict(bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(args.output.iterdir()) if f.name!='MANIFEST.json'})
    print(json.dumps(dict(gate=gate['eligible_for_one_bn_confirmation_teacher'],output=str(args.output))))


if __name__=='__main__':
    main()
