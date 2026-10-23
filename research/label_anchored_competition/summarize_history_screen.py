"""Independently reconstruct history-screen aggregates and bootstrap from CSV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def verify(input_path):
    snapshot = json.loads((input_path/'snapshot_receipt.json').read_text())
    for name, meta in snapshot['files'].items():
        path = input_path/name
        if path.stat().st_size != meta['bytes'] or digest(path) != meta['sha256']:
            raise ValueError('Snapshot file changed: '+name)
    spec = json.loads((input_path/'effective_protocol.json').read_text())
    complete = json.loads((input_path/'COMPLETE.json').read_text())
    if complete['signature']['synthetic'] or complete['new_fits'] != 0 or complete['teacher_updates'] != 0:
        raise ValueError('Expected formal no-fitting screen')
    checks = dict(transport_files_verified=len(snapshot['files']), candidate_rows=0, interval_rows=0,
                  maximum_aggregate_difference=0., maximum_interval_difference=0., temporal_partition_rows=0,
                  fold_mass_reconciliations=0, both_roles_full_alignment=True)
    def close(actual, expected, key):
        if not np.isclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True):
            raise ValueError(f'Independent reconstruction differs: {key}: {actual} / {expected}')
        return abs(actual-expected) if np.isfinite(actual) else 0.
    for role in spec['role_counts']:
        alignment = json.loads((input_path/(role+'_alignment.json')).read_text())
        if alignment['images'] != spec['role_counts'][role] or not alignment['all_full_pixel_arrays_equal']:
            raise ValueError('Incomplete alignment')
        if alignment['shards_verified'] != 3*alignment['images']:
            raise ValueError('Missing shard validation')
        for pair in spec['pairs']:
            folder = input_path/pair['name']
            sufficient = read_csv(folder/(role+'_image_sufficient.csv'))
            aggregates = read_csv(folder/(role+'_candidate_recovery.csv'))
            scopes = sorted(set(r['scope'] for r in sufficient))
            for r in sufficient:
                total = sum(float(r[k]) for k in ['past_correct_current_error_mass', 'past_error_current_correct_mass',
                    'both_correct_mass', 'both_wrong_same_mass', 'both_wrong_different_mass'])
                close(total, float(r['mass']), 'temporal mass partition')
                checks['temporal_partition_rows'] += 1
            for row in aggregates:
                selected = [r for r in sufficient if r['scope'] == row['scope']]
                method = row['method']
                error = sum(float(r['error_mass']) for r in selected)
                mass = sum(float(r['mass']) for r in selected)
                hit = sum(float(r[method+'__recovered_mass']) for r in selected)
                gained = sum(float(r[method+'__gained_vs_now_mass']) for r in selected)
                lost = sum(float(r[method+'__lost_vs_now_mass']) for r in selected)
                now = sum(float(r['now__recovered_mass']) for r in selected)
                close(hit-now, gained-lost, 'candidate gains minus losses')
                values = dict(error_recovery=hit/error if error else np.nan,
                    set_truth_coverage=(mass-error+hit)/mass if mass else np.nan,
                    gain_vs_now=gained/error if error else np.nan, loss_vs_now=lost/error if error else np.nan,
                    error_rate=error/mass if mass else np.nan)
                for key, value in values.items():
                    difference = close(value, float(row[key]), key)
                    checks['maximum_aggregate_difference'] = max(checks['maximum_aggregate_difference'], difference)
                checks['candidate_rows'] += 1
            if role == 'D_rel':
                folds = read_csv(folder/(role+'_candidate_recovery_folds.csv'))
                for row in aggregates:
                    parts = [r for r in folds if r['scope'] == row['scope'] and r['method'] == row['method']]
                    if len(parts) != 3:
                        raise ValueError('Missing fold description')
                    for key in ('mass', 'error_mass', 'recovered_mass', 'image_pool'):
                        close(sum(float(r[key]) for r in parts), float(row[key]), 'fold '+key)
                    checks['fold_mass_reconciliations'] += 1
                intervals = read_csv(folder/(role+'_recovery_intervals.csv'))
                for scope in scopes:
                    selected = sorted([r for r in sufficient if r['scope'] == scope], key=lambda r:r['image_id'])
                    n = len(selected)
                    if n != spec['role_counts'][role]:
                        raise ValueError('Bootstrap omitted zero-scope image')
                    rng = np.random.default_rng(spec['screen']['bootstrap_seed'])
                    # Direct indexing of resampled image arrays is independent
                    # of the runner's count-vector matrix multiplication.
                    draws = rng.integers(n, size=(spec['screen']['bootstrap_replicates'], n))
                    den = np.array([float(r['error_mass']) for r in selected])
                    method_names = ['now', 'history', 'arithmetic', 'geometric']+[f'sham{s}' for s in spec['permutation']['seeds']]
                    values = {}; point = {}
                    for method in method_names:
                        hits = np.array([float(r[method+'__recovered_mass']) for r in selected])
                        with np.errstate(divide='ignore', invalid='ignore'):
                            values[method] = hits[draws].sum(1)/den[draws].sum(1)
                        point[method] = hits.sum()/den.sum() if den.sum() else np.nan
                    names = [f'sham{s}' for s in spec['permutation']['seeds']]
                    values['sham_mean'] = np.mean([values[m] for m in names], axis=0)
                    point['sham_mean'] = np.mean([point[m] for m in names])
                    for row in [r for r in intervals if r['scope'] == scope]:
                        differences = values['history']-values[row['comparator']]
                        finite = differences[np.isfinite(differences)]
                        confidence = float(row['confidence'])
                        bounds = np.quantile(finite, [(1-confidence)/2, (1+confidence)/2]) if len(finite) else [np.nan]*2
                        for key, actual in zip(('ci_low', 'ci_high'), bounds):
                            difference = close(actual, float(row[key]), key)
                            checks['maximum_interval_difference'] = max(checks['maximum_interval_difference'], difference)
                        close(point['history']-point[row['comparator']], float(row['delta']), 'point delta')
                        if len(finite) != int(row['valid_replicates']):
                            raise ValueError('Valid bootstrap count differs')
                        checks['interval_rows'] += 1
    checks.update(state='passed', source_commit=complete['signature']['source_commit'], new_fits=0,
                  independent_bootstrap='direct array indexing, same registered image draws')
    return checks, spec, complete


def summarize(input_path, output):
    if output.exists():
        raise FileExistsError(output)
    checks, spec, complete = verify(input_path)
    output.mkdir(parents=True)
    (output/'independent_verification.json').write_text(json.dumps(checks, indent=2)+'\n')
    candidates, ranks, intervals, alignment, transitions = [], [], [], [], []
    for role in spec['role_counts']:
        a = json.loads((input_path/(role+'_alignment.json')).read_text())
        alignment.append({k:a[k] for k in ('role', 'images', 'shards_verified', 'full_pixel_positions', 'analyzed_pixel_positions', 'all_full_pixel_arrays_equal')})
        for pair in spec['pairs']:
            folder = input_path/pair['name']
            all_candidates = read_csv(folder/(role+'_candidate_recovery.csv'))
            candidates.extend(r for r in all_candidates if r['scope'] in ('all', 'high_mc', 'alternative_changes', 'past_disagrees'))
            for r in all_candidates:
                if r['method'] != 'history':
                    continue
                den = float(r['error_mass']); mass = float(r['mass'])
                keys = ['past_correct_current_error_mass','both_wrong_same_mass','both_wrong_different_mass']
                transitions.append(dict(pair=pair['name'], role=role, scope=r['scope'],
                    **{k.removesuffix('_mass')+'_error_fraction':float(r[k])/den if den else np.nan for k in keys},
                    past_error_current_correct_all_fraction=float(r['past_error_current_correct_mass'])/mass if mass else np.nan))
            fold = 'mean' if role == 'D_rel' else 'pooled'
            ranks.extend(r for r in read_csv(folder/(role+'_ranking.csv')) if r['fold'] == fold and r['scope'] in ('all', 'high_mc'))
            if role == 'D_rel':
                intervals.extend(r for r in read_csv(folder/(role+'_recovery_intervals.csv')) if r['scope'] in ('all', 'high_mc') and r['comparator'] in spec['screen']['comparators'])
    for name, rows in [('candidate_summary.csv',candidates), ('ranking_summary.csv',ranks), ('recovery_primary_intervals.csv',intervals),
                       ('alignment_summary.csv',alignment), ('temporal_transitions.csv',transitions)]:
        write_csv(output/name, rows)
    gate = json.loads((input_path/'screen_gate.json').read_text())
    summary = dict(source_commit=complete['signature']['source_commit'], elapsed_seconds=complete['elapsed_seconds'],
        gate_passed=gate['passed'], failed_checks=[r['check'] for r in gate['checks'] if not r['passed']],
        decision=gate['decision'], no_new_fits=complete['new_fits'] == 0, teacher_updates=complete['teacher_updates'],
        validation_is_previously_inspected=True, alignment=alignment, independent_verification=checks)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(input_path, output, spec)
    files={p.name:dict(bytes=p.stat().st_size, sha256=digest(p)) for p in sorted(output.iterdir()) if p.is_file()}
    (output/'SUMMARY_COMPLETE.json').write_text(json.dumps(dict(state='complete', input_receipt_sha256=digest(input_path/'snapshot_receipt.json'), files=files), indent=2)+'\n')
    print(json.dumps(summary))


def plot(input_path, output, spec):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    labels = ['Current top-2','Past competitor','Arithmetic','Geometric','Shuffled mean']
    colors = ['#394b59','#bd563d','#277f8e','#779c53','#b6b6b6']
    for ax,pair in zip(axes,spec['pairs']):
        rows = read_csv(input_path/pair['name']/'D_rel_candidate_recovery.csv')
        scores = {r['method']:100*float(r['error_recovery']) for r in rows if r['scope'] == 'high_mc'}
        values = [scores[m] for m in ['now','history','arithmetic','geometric']]+[np.mean([scores[f'sham{s}'] for s in spec['permutation']['seeds']])]
        ax.bar(np.arange(5), values, color=colors)
        for x,v in enumerate(values):ax.text(x,v+.8,f'{v:.1f}',ha='center',fontsize=9)
        ax.set_xticks(np.arange(5),labels,rotation=22,ha='right')
        ax.set_title(pair['name'].replace('_to_',' → ').replace('_',' '))
        ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Current errors covered by alternative (%)')
    fig.suptitle('D_rel, current MC ≥ 0.95; fixed two-label sets, no model fitting')
    fig.tight_layout();fig.savefig(output/'high_mc_candidate_recovery.png',dpi=180);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5,4), sharey=True)
    methods = {'MM':('MC + margin','#394b59'),'arithmetic_current_a':('Arithmetic','#277f8e'),
               'geometric_current_a':('Geometric','#779c53'),'past_b_vs_a':('Past logit(b) - logit(a)','#bd563d')}
    for ax,pair in zip(axes,spec['pairs']):
        curves = read_csv(input_path/pair['name']/'D_rel_risk_coverage.csv')
        for method,(label,color) in methods.items():
            points = sorted([r for r in curves if r['method']==method and r['fold']=='mean' and r['scope']=='high_mc'],key=lambda r:float(r['coverage']))
            ax.plot([100*float(r['coverage']) for r in points],[100*float(r['risk']) for r in points],label=label,color=color)
        ax.set_title(pair['name'].replace('_to_',' → '));ax.set_xlabel('Coverage within high-MC pixels (%)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Error among retained pixels (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Frozen error rankings; equal mean of three image folds, descriptive')
    fig.tight_layout();fig.savefig(output/'frozen_high_mc_risk_coverage.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.input, args.output)
