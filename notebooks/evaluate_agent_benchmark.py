"""Dataset-configurable evaluation reusing the MALT notebook's actual code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, StratifiedKFold
from xgboost import XGBClassifier

NOTEBOOK = Path(__file__).with_name('agent_outcome_hpo.ipynb')


def notebook_helpers():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    namespace = {'PREDICTION_THRESHOLD': 0.5, 'Path': Path}
    setup = next(''.join(c['source']) for c in notebook['cells']
                 if ''.join(c['source']).startswith('import json\nimport math'))
    exec(setup, namespace)
    wanted = {'read_feature_run', 'classification_metrics', 'clean_json', 'suggest_params', 'resolve_results_path'}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<MALT notebook helper>', 'exec'), namespace)
    if not wanted.issubset(namespace):
        raise ValueError('Notebook helper contract changed')
    # Reuse fixed XGB parameters without executing parameter/output cells.
    namespace['RANDOM_STATE'] = 42
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'XGB_BASE_PARAMS' for t in node.targets):
                    exec(compile(ast.Module(body=[node], type_ignores=[]), '<MALT notebook helper>', 'exec'), namespace)
    return namespace, notebook


def binary_metrics(helpers, y, scores, threshold=0.5):
    result = helpers['classification_metrics'](y, scores, threshold)
    result.update(macro_f1=float(f1_score(y, scores >= threshold, average='macro', labels=[0, 1], zero_division=0)),
                  abnormal_catch_rate=result['recall_sensitivity'], normal_false_alarm_rate=result['false_positive_rate'],
                  rows=len(y), abnormal_rows=int(np.sum(y)), normal_rows=int(len(y) - np.sum(y)))
    return result


def inner_splits(x, normal, groups, seed):
    splitter = (StratifiedGroupKFold(3, shuffle=True, random_state=seed) if groups is not None
                else StratifiedKFold(3, shuffle=True, random_state=seed))
    splits = list(splitter.split(x, normal, groups))
    for train, valid in splits:
        if normal.iloc[train].nunique() != 2 or normal.iloc[valid].nunique() != 2:
            raise ValueError('Inner fold lacks a class')
        if groups is not None and set(groups.iloc[train]) & set(groups.iloc[valid]):
            raise ValueError('Inner task leakage')
    return splits


def select_threshold(y, scores, target):
    candidates = np.unique(np.r_[0.0, np.linspace(.05, .95, 19), np.quantile(scores, np.linspace(.05, .95, 19))])
    eligible = [float(t) for t in candidates if np.mean(scores[np.asarray(y) == 1] >= t) >= target]
    return max(eligible)


def outer_splits(x, y, meta, dataset, group):
    groups = None
    if group:
        if group not in meta or meta[group].isna().any():
            raise ValueError('Missing requested task group; refusing fallback')
        groups = meta[group].astype(str)
        outer = list(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(x, y, groups))
    else:
        if dataset != 'trace':
            raise ValueError('Ungrouped CV is only authorized for TRACE')
        if 'transcript_sha256' not in meta or meta.transcript_sha256.isna().any():
            raise ValueError('TRACE requires transcript hashes to prevent duplicate leakage')
        groups = meta.transcript_sha256.astype(str)
        if pd.DataFrame({'group': groups, 'label': y}).groupby('group').label.nunique().max() > 1:
            raise ValueError('Identical TRACE transcripts have conflicting labels')
        outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(x, y, groups))
    return groups, outer


def evaluate(args):
    h, notebook = notebook_helpers()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    manifest = json.loads((args.features/'manifest.json').read_text())
    for shard in manifest['shards']:
        path = args.features/shard['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard['sha256']:
            raise ValueError(f'Feature shard checksum mismatch: {path}')
    frame = h['read_feature_run'](args.features).reset_index(drop=True)
    if len(frame) != sum(s['rows'] for s in manifest['shards']):
        raise ValueError('Feature row count mismatch')
    audit_path = args.features/'source_audit.json'
    if audit_path.exists() and len(frame) != json.loads(audit_path.read_text())['rows']:
        raise ValueError('Source-to-feature row count mismatch')
    meta = pd.DataFrame(frame.metadata_json.map(json.loads).tolist())
    if args.label not in meta or meta[args.label].isna().any() or not set(meta[args.label]) <= {0, 1}:
        raise ValueError('Missing or nonbinary labels')
    y = meta[args.label].astype(int)
    features = [c for c in frame if c.startswith(('zones__', 'layer_distribution__'))]
    if not features:
        raise ValueError('No MALT activation columns')
    x = frame[features].astype(float)
    if not (x.nunique() > 1).any():
        raise ValueError("All activation features are constant; inspect prompt coverage before evaluating")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Nonfinite activation features')
    identifier = 'example_id' if 'example_id' in frame else 'id'
    if identifier not in frame or frame[identifier].duplicated().any():
        raise ValueError('Missing or duplicate trajectory IDs')
    groups, outer = outer_splits(x, y, meta, args.dataset, args.group)
    out.mkdir(parents=True, exist_ok=True)
    predictions, operating, sweeps, fold_metrics, assignments = [], [], [], [], []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for fold, (train, test) in enumerate(outer):
        if groups is not None and set(groups.iloc[train]) & set(groups.iloc[test]):
            raise ValueError('Outer task leakage')
        xt, normal = x.iloc[train], 1 - y.iloc[train]
        gt = groups.iloc[train] if groups is not None else None
        splits = inner_splits(xt, normal, gt, 42)
        def objective(trial):
            params = {**h['XGB_BASE_PARAMS'], **h['suggest_params'](trial)}
            scores = []
            for a, b in splits:
                model = XGBClassifier(**params).fit(xt.iloc[a], normal.iloc[a])
                scores.append(h['classification_metrics'](normal.iloc[b], model.predict_proba(xt.iloc[b])[:, 1])['f1'])
            return float(np.mean(scores))
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=args.trials)
        params = {**h['XGB_BASE_PARAMS'], **study.best_params}
        oof = np.full(len(train), np.nan)
        for a, b in splits:
            model = XGBClassifier(**params).fit(xt.iloc[a], normal.iloc[a])
            oof[b] = 1 - model.predict_proba(xt.iloc[b])[:, 1]
        model = XGBClassifier(**params).fit(xt, normal)
        scores = 1 - model.predict_proba(x.iloc[test])[:, 1]
        metrics = binary_metrics(h, y.iloc[test], scores)
        fold_metrics.append({'fold': fold, 'metrics': metrics, 'best_params': study.best_params,
                             'inner_normal_f1': study.best_value, 'train_rows': len(train),
                             'train_tasks': None if gt is None else gt.nunique(),
                             'test_tasks': None if groups is None else groups.iloc[test].nunique()})
        pred = meta.iloc[test].copy()
        pred['trajectory_id'] = frame.iloc[test][identifier].to_numpy()
        pred['fold'], pred['abnormal_score'], pred['predicted_abnormal'] = fold, scores, scores >= .5
        for target in [.95, .97, .99]:
            threshold = select_threshold(y.iloc[train], oof, target)
            pred[f'prediction_catch_{target}'] = scores >= threshold
            operating.append({'fold': fold, 'target_abnormal_catch': target, 'threshold': threshold,
                              'selection': 'training_out_of_fold_only',
                              **binary_metrics(h, y.iloc[test], scores, threshold)})
        predictions.append(pred)
        for threshold in np.linspace(.05, .95, 19):
            sweeps.append({'fold': fold, 'threshold': threshold, **binary_metrics(h, y.iloc[test], scores, threshold)})
        for role, indices in [('train', train), ('test', test)]:
            assignments.extend({'fold': fold, 'role': role, 'trajectory_id': frame.iloc[i][identifier],
                                'group': None if groups is None else groups.iloc[i]} for i in indices)
        model.save_model(out / f'model-fold-{fold}.json')
        study.trials_dataframe().to_csv(out / f'hpo-fold-{fold}.csv', index=False)
        pd.DataFrame({'trajectory_id': frame.iloc[train][identifier].to_numpy(),
                      'abnormal': y.iloc[train].to_numpy(), 'abnormal_score': oof}).to_csv(out / f'train-oof-fold-{fold}.csv', index=False)
        print(f'{args.dataset} fold {fold}: AUC={metrics["auc"]:.4f}, macro F1={metrics["macro_f1"]:.4f}', flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    breakdown = []
    if 'model' in pred:
        for value, subset in pred.groupby('model'):
            breakdown.append({'kind': 'model', 'value': value, **binary_metrics(h, subset[args.label], subset.abnormal_score.to_numpy())})
    if 'categories' in pred:
        for category in sorted({v for cats in pred.categories for v in cats}):
            mask = (pred[args.label] == 0) | pred.categories.map(lambda cats: category in cats)
            subset = pred[mask]
            breakdown.append({'kind': 'category_vs_all_benign', 'value': category,
                              **binary_metrics(h, subset[args.label], subset.abnormal_score.to_numpy())})
    if 'classification' in pred:
        subset = pred[(pred[args.label] == 1) | (pred.classification == 'baseline')]
        breakdown.append({'kind': 'baseline_population', 'value': 'hacks_vs_baselines_only',
                          **binary_metrics(h, subset[args.label], subset.abnormal_score.to_numpy())})
    payload = {'dataset': args.dataset, 'positive_class': 'abnormal', 'label_column': args.label,
               'group_column': args.group or 'transcript_sha256', 'split': 'task-held-out GroupShuffleSplit' if args.group else 'stratified 5-fold CV grouped by exact transcript; NOT task-held-out',
               'hpo': '8 trials, normal-class F1, training-only inner CV' if args.trials == 8 else f'{args.trials} trials',
               'inner_split': 'StratifiedGroupKFold' if args.group else 'StratifiedKFold',
               'metrics': binary_metrics(h, pred[args.label], pred.abnormal_score.to_numpy()), 'folds': fold_metrics,
               'input_rows': len(frame), 'class_counts': y.value_counts().to_dict(), 'feature_count': len(features),
               'notebook_sha256': hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
               'feature_columns': features, 'source_audit': json.loads((args.features / 'source_audit.json').read_text()) if (args.features / 'source_audit.json').exists() else None}
    (out / 'metrics.json').write_text(json.dumps(h['clean_json'](payload), indent=2), encoding='utf-8')
    pred.to_csv(out / 'predictions.csv', index=False)
    pd.DataFrame(breakdown).to_csv(out / 'breakdown.csv', index=False)
    pd.DataFrame(operating).to_csv(out / 'operating-points.csv', index=False)
    pd.DataFrame(sweeps).to_csv(out / 'threshold-sweep.csv', index=False)
    pd.DataFrame(assignments).to_csv(out / 'split-assignments.csv', index=False)
    latency = next(''.join(c['source']) for c in notebook['cells'] if 'LATENCY_WARMUP = 10' in ''.join(c['source']))
    h.update(x_test_t=x.iloc[test], task_holdout_model=model, SELECTED_FEATURES=features, OUTPUT_DIR=out)
    exec(latency, h)
    return payload


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--label', default='abnormal')
    parser.add_argument('--group')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trials', type=int, default=8)
    evaluate(parser.parse_args())
