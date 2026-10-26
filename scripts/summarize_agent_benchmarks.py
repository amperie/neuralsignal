"""Summarize completed benchmark outputs without selecting thresholds on test data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, average_precision_score

ROOT = Path('notebooks/runs/notebook-agent-outcome-hpo')
NAMES = ['terminal-wrench-full', 'terminal-wrench-stripped', 'trace-full']


def main():
    rows, operations = [], []
    for name in NAMES:
        directory = ROOT/name
        payload = json.loads((directory/'metrics.json').read_text())
        pred = pd.read_csv(directory/'predictions.csv')
        splits = pd.read_csv(directory/'split-assignments.csv')
        if pred.trajectory_id.duplicated().any():
            raise ValueError('Repeated held-out predictions')
        for _, fold in splits.groupby('fold'):
            train, test = fold[fold.role == 'train'], fold[fold.role == 'test']
            if set(train.trajectory_id) & set(test.trajectory_id):
                raise ValueError('Outer trajectory overlap')
            if payload['group_column'] and set(train.group) & set(test.group):
                raise ValueError('Outer task overlap')
        y = pred.abnormal.astype(int)
        scores = pred.abnormal_score.to_numpy()
        checks = {'auc': roc_auc_score(y, scores), 'auprc': average_precision_score(y, scores),
                  'macro_f1': f1_score(y, scores >= .5, average='macro', labels=[0, 1])}
        for key, value in checks.items():
            if not np.isclose(value, payload['metrics'][key]):
                raise ValueError(f'Metric/prediction mismatch: {name}, {key}')
        fold_summary = {key: {'mean': float(np.mean([f['metrics'][key] for f in payload['folds']])),
                              'std': float(np.std([f['metrics'][key] for f in payload['folds']]))} for key in checks}
        payload['fold_summary'] = fold_summary
        dataset_ops = []
        for target in [.95, .97, .99]:
            predictions = pred[f'prediction_catch_{target}'].astype(bool)
            tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
            row = {'dataset': name, 'training_target_catch': target, 'test_catch': tp/(tp+fn),
                   'test_false_alarm': fp/(fp+tn), 'held_out_rows': len(y)}
            dataset_ops.append(row)
            operations.append(row)
        payload['held_out_operating_summary'] = dataset_ops
        (directory/'metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        pd.DataFrame(dataset_ops).to_csv(directory/'operating-summary.csv', index=False)
        rows.append({'dataset': name, 'split': payload['split'], 'input_rows': payload['input_rows'],
                     'test_rows': len(y), **{k: payload['metrics'][k] for k in ['auc', 'auprc', 'macro_f1', 'abnormal_catch_rate', 'normal_false_alarm_rate']}})
    original = pd.read_csv(ROOT/'terminal-wrench-full'/'split-assignments.csv')
    stripped = pd.read_csv(ROOT/'terminal-wrench-stripped'/'split-assignments.csv')
    pd.testing.assert_frame_equal(original, stripped)
    results = pd.DataFrame(rows)
    results.to_csv(ROOT/'benchmark-comparison.csv', index=False)
    pd.DataFrame(operations).to_csv(ROOT/'benchmark-operating-comparison.csv', index=False)
    report = ['**Superseded diagnostic run — not a full-trajectory benchmark.** These artifacts used only the first 512 tokens. TRACE also contains an exact duplicate pair (`trajectory_0516`, `trajectory_0517`) that crossed the original CV folds. Do not use these scores as validated full-trajectory results. See [the full-context rerun status](agent_full_context_status.md).', '', '# NeuralSignal public agent benchmark results', '',
              '**Terminal Wrench exposes a context-window failure in the unchanged MALT configuration.** '
              'Every original and stripped trajectory has the same first 512 tokens, entirely before the first assistant turn. '
              'Its results measure a detector receiving no distinguishing behavior evidence.', '',
              '| Dataset | Split | Evaluated rows | AUC | PR-AUC (AP) | Macro F1 | Published baseline |',
              '|---|---|---:|---:|---:|---:|---|']
    baselines = ['GPT-5.4 AUC 0.9679', 'GPT-5.4 AUC 0.9168', 'GPT-5.2 macro F1 0.45']
    for row, baseline in zip(rows, baselines):
        report.append(f'| {row["dataset"]} | {row["split"]} | {row["test_rows"]} | {row["auc"]:.4f} | {row["auprc"]:.4f} | {row["macro_f1"]:.4f} | {baseline} |')
    report += ['', '## Operating points', '',
               'Thresholds were selected using training-only out-of-fold predictions. Achieved test catch rates need not meet the training targets.', '',
               '| Dataset | Training catch target | Held-out catch | Held-out false alarm |', '|---|---:|---:|---:|']
    for row in operations:
        report.append(f'| {row["dataset"]} | {row["training_target_catch"]:.0%} | {row["test_catch"]:.2%} | {row["test_false_alarm"]:.2%} |')
    report += ['', '## Breakdowns', '']
    for name in NAMES:
        report += [f'### {name}', '', '| Slice | Rows | Hacks | AUC | PR-AUC (AP) | Macro F1 |', '|---|---:|---:|---:|---:|---:|']
        breakdown = pd.read_csv(ROOT/name/'breakdown.csv')
        if name.startswith('terminal-wrench'):
            breakdown = breakdown[breakdown.kind.isin(['model', 'baseline_population'])]
        for _, row in breakdown.iterrows():
            report.append(f'| {row["value"]} | {row["rows"]} | {row["abnormal_rows"]} | {row["auc"]:.4f} | {row["auprc"]:.4f} | {row["macro_f1"]:.4f} |')
        report += ['']
    report += ['## TRACE fold variation', '', '| Fold | Rows | AUC | Macro F1 |', '|---|---:|---:|---:|']
    trace_payload = json.loads((ROOT/'trace-full'/'metrics.json').read_text())
    for fold in trace_payload['folds']:
        m = fold['metrics']
        report.append(f'| {fold["fold"]} | {m["rows"]} | {m["auc"]:.4f} | {m["macro_f1"]:.4f} |')
    report += ['', '## XGBoost latency', '',
               'Notebook microbenchmark: 10 warmups, 100 timed calls per batch size, using DataFrame inputs on the local Xeon E5-2695 v4. '
               'Evaluation jobs overlapped, so these are indicative timings rather than a controlled hardware comparison. Feature extraction is excluded.', '',
               '| Dataset | Batch size | Median ms per call | P95 ms per call |', '|---|---:|---:|---:|']
    for name in NAMES:
        for _, row in pd.read_csv(ROOT/name/'agent_outcome_xgb_latency.csv').iterrows():
            report.append(f'| {name} | {int(row["batch_size"])} | {row["ms_p50"]:.2f} | {row["ms_p95"]:.2f} |')
    report += ['', '## Interpretation and limits', '',
               '- Terminal Wrench: 7,200 included rows per variant, 3,632 hacks and 3,568 normals across 331 tasks. '
               'Normals comprise 2,352 baselines and 1,216 legitimate attacker solves; 1,441 no-reward attempts are excluded. '
               'The original and stripped variants share identical task assignments.',
               '- TRACE: 517 rows, 268 hacks and 249 benign trajectories. No task/scenario field exists in the release. '
               'Five-fold trajectory CV cannot establish generalization to unseen tasks. Category slices compare each code against all benign trajectories and overlap for multi-label rows.',
               '- All feature columns match MALT, including decoder features: 7,128 total, 4,320 decoder. '
               'Batch size 1 and the 512-token limit are unchanged. These are not full-context behavioral scores.',
               '- Terminal Wrench input reuse was allowed only after 24 representatives across source models, outcome classes, and variants yielded bit-identical features. '
               'All trajectories and statistical weights remain in the feature datasets.',
               '- The published Terminal Wrench baseline uses hacks versus baselines only and evaluates a full-corpus LLM judge. '
               'Our main negative population additionally includes attacker legitimate solves; the baseline-only slice still evaluates held-out tasks only.',
               '- TRACE compares supervised cross-validation against a zero-shot judge baseline. A numerical difference is not evidence of superiority under an equivalent protocol.',
               '- Operating points use training-only predictions. The original MALT notebook is untouched; its test-selected operating points are not reproduced as deployment estimates.',
               '- XGBoost latency CSVs measure prediction on existing features, excluding feature extraction. '
               'Zero-shot MALT transfer and separate no-reward scoring were not run.', '',
               '## Sources and reproducibility', '',
               '- [Terminal Wrench release and monitor baseline](https://github.com/few-sh/terminal-wrench)',
               '- [TRACE dataset](https://huggingface.co/datasets/PatronusAI/trace-dataset)',
               '- [TRACE paper](https://arxiv.org/html/2601.20103v1)',
               '- [Protocol, revisions, changed components, and rerun commands](agent_benchmark_runs.md)',
               '- Complete metrics, predictions, folds, operating points, breakdowns, HPO trials, models, and latency outputs: `notebooks/runs/notebook-agent-outcome-hpo/<dataset>/`.', '']
    Path('docs/agent_benchmark_results.md').write_text('\n'.join(report), encoding='utf-8')
    print(results.to_string(index=False))


if __name__ == '__main__':
    main()
