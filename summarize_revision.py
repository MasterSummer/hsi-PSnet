"""Summarize completed run_revision outputs without treating folds as replicates."""
import argparse
from pathlib import Path
import json
import pandas as pd
from run_controlled_ablations import verify_predictions
from reviewer_experiments.metrics import plant_predictions, classification_metrics


def summarize(root, output):
    root=Path(root);output=Path(output)
    if output.exists():
        raise ValueError('Use a new summary output directory')
    manifests=sorted(root.glob('seed_*/experiment_manifest.json'))
    if not manifests:
        raise ValueError('No per-seed manifests found')
    records=[];expected=None;seen_seeds=set()
    for path in manifests:
        manifest=json.loads(path.read_text());identity=manifest['identity']
        seeds=identity['seeds']
        if len(seeds)!=1 or seeds[0] in seen_seeds:
            raise ValueError('Expected exactly one unique seed per directory')
        seed=seeds[0];seen_seeds.add(seed)
        common={k:identity[k] for k in ['models','tasks','source_sha256','training','band_spec','top_k_bands']}
        common['task_sha256']=manifest['task_sha256']
        if expected is None:expected=common
        elif common!=expected:raise ValueError('Cannot pool seeds with different configurations, source or task tables')
        paths=[]
        for model in identity['models']:
            for task in identity['tasks']:
                paths += [path.parent/'runs'/model/task/f'seed_{seed}'/f'fold_{fold}'/'predictions.csv' for fold in range(1,6)]
        if not all(p.is_file() for p in paths):
            raise ValueError(f'Incomplete preset for seed {seed}')
        verify_predictions(paths,path.parent/'task_snapshot')
        plants=plant_predictions(pd.concat([pd.read_csv(p) for p in paths],ignore_index=True))
        for (model,task,run_seed),frame in plants.groupby(['model','task','seed']):
            metrics=classification_metrics(frame)
            records.append(dict(model=model,task=task,seed=int(run_seed),plants=len(frame),
                **{k:v for k,v in metrics.items() if k in ['accuracy','balanced_accuracy','auroc','f1','precision','sensitivity','specificity']}))
    points=pd.DataFrame(records)
    metric_cols=[c for c in points if c not in ['model','task','seed','plants']]
    rows=[]
    for (model,task),group in points.groupby(['model','task']):
        row=dict(model=model,task=task,n_seeds=len(group),seeds=','.join(map(str,sorted(group.seed))))
        for metric in metric_cols:
            row[metric+'_mean']=group[metric].mean()
            row[metric+'_sd']=group[metric].std(ddof=1)
        rows.append(row)
    output.mkdir(parents=True)
    points.to_csv(output/'metrics_by_seed.csv',index=False)
    pd.DataFrame(rows).to_csv(output/'summary.csv',index=False)
    print(f'Summarized {len(points)} complete model/task/seed groups; {len(seen_seeds)} seeds. Single-seed SD is unavailable.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();summarize(a.runs_root,a.output)
