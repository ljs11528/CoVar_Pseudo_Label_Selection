"""Post-hoc read-only replay, fold ranking and coefficient-direction diagnostics.

Never fits a coefficient, changes a selection, or enters the resource gate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .bounded_probe import apply_probe, mc_logits, sigmoid
from .prepare_data import digest
from .ranking_metrics import Ranking
from .run_score_audit import write_csv, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    complete=json.loads((args.run/'COMPLETE.json').read_text())
    if digest(args.run/'D_rel_predictions.npz')!=complete['files']['D_rel_predictions.npz']['sha256']:
        raise ValueError('Prediction archive mismatch')
    manifest=json.loads((args.export/'manifest.json').read_text())
    if digest(args.export/'manifest.json')!=complete['signature']['export_manifest_sha256']:
        raise ValueError('Export manifest mismatch')
    models=json.loads((args.run/'outer_models.json').read_text())
    folds=json.loads((args.run/'folds.json').read_text())
    rows=[];directions=[];replay=[]
    with np.load(args.run/'D_rel_predictions.npz',allow_pickle=False) as archive:
        ids=archive['image_ids'].tolist(); im=archive['image_index']; error=archive['error']; w=archive['sampling_weight']
        index=archive['pixel_indices']
        by_id={e['image_id']:e for e in manifest['files'] if e['role']=='D_rel'}
        evidence=[];lm=[];conflicts=[]
        for i,image_id in enumerate(ids):
            entry=by_id[image_id];path=args.export/entry['file']
            if digest(path)!=entry['sha256']:
                raise ValueError('Changed source cache')
            with np.load(path,allow_pickle=False) as cache:
                selected=np.flatnonzero(im==i);n=len(selected)
                np.testing.assert_array_equal(index[selected],cache['pixel_indices'][:n])
                np.testing.assert_array_equal(error[selected],~cache['correct'][:n])
                evidence.append(cache['score__competitor_evidence'][:n]);lm.append(cache['score__logit_mc'][:n])
                conflicts.append(cache['score__strict_conflict'][:n])
        e=np.concatenate(evidence); lm=np.concatenate(lm); conflict=np.concatenate(conflicts).astype(bool)
        methods=['MC','prototype','product','uniform','e','controls','controls_e']
        for fold,item in enumerate(folds):
            held=np.isin(im,[ids.index(s) for s in item['heldout_images']]);train=~held
            for seed in [2027,2028]:
                model=models[f'fold{fold}_s{seed}_u2000_e']
                restored=apply_probe(model,e[held,None],lm[held])
                stored=archive[f'correctness_logit__s{seed}_u2000_e'][held]
                maxdiff=float(np.max(np.abs(restored-stored)))
                if maxdiff>1e-12:raise ValueError('OOF model replay mismatch')
                replay.append(dict(fold=fold,seed=seed,max_abs_diff=maxdiff))
                center=np.average(e[train],weights=w[train]);scale=np.sqrt(np.average((e[train]-center)**2,weights=w[train]))
                z=(e-center)/scale
                base=mc_logits(model['mc'],lm)
                for role,mask in [('outer_train',train),('outer_heldout',held)]:
                    directions.append(dict(fold=fold,seed=seed,role=role,chosen=model['form'],
                        coefficient=model.get('coefficients',[0.])[0],
                        zero_gradient=float(np.average((sigmoid(base[mask])-(1-error[mask]))*z[mask],weights=w[mask])),
                        error_prevalence=float(np.average(error[mask],weights=w[mask])),
                        images=len(np.unique(im[mask]))))
            for scope,mask in [('all',held),('conflict',held & conflict)]:
                for name in methods:
                    key='MC' if name=='MC' else 's2027_u2000_'+name
                    result=Ranking(error[mask],-archive['correctness_logit__'+key][mask]).evaluate(w[mask],())
                    rows.append(dict(fold=fold,scope=scope,method=name,images=len(np.unique(im[mask])),
                        errors=int(error[mask].sum()),**result))
    args.output.mkdir(parents=True)
    write_csv(args.output/'fold_ranking.csv',rows)
    write_csv(args.output/'e_direction.csv',directions)
    write_json(args.output/'receipt.json',dict(stage=complete['stage'],source_sha256=digest(Path(__file__)),
        scope='Post-hoc read-only diagnostic; no new fitting, selection or gate changes',replay=replay,
        original_run_signature=complete['signature'],interpretation='Positive correctness-logit coefficient rewards larger e at fixed MC; negative zero gradient favors a positive correction locally.'))
    print(json.dumps(dict(stage=complete['stage'],output=str(args.output),replays=len(replay),max_abs_diff=max(r['max_abs_diff'] for r in replay))))


if __name__=='__main__':
    main()
