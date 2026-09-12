# S1 evaluation behavior

Evaluation currently happens inside [train_s1](../neuralsignal/training/s1.py).
There is no evaluation-only CLI, saved-model evaluation loader, validation set,
threshold search, calibration, or promotion automation.

## Split and predictions

The trainer uses a stratified 75/25 train/test split with random state 42 by
default. MALT rows are pooled into runs first, keeping a run out of both partitions
at once. Other datasets split by row and do not enforce arbitrary group isolation.
XGBoost predicts class-1 probability; threshold 0.5 yields flags.

## Reported metrics

| Scope | Keys |
| --- | --- |
| Global | `auroc`, `auprc`, `f1`, `precision`, `recall` |
| Counts | `train_rows`, `test_rows`, `feature_count` |
| When `run_source` exists | `<condition>_test_runs`, `<condition>_auroc`, `<condition>_f1` |

AUPRC uses sklearn average precision. Metrics that raise `ValueError` become NaN;
single-class slice AUROC can be undefined. Counts are returned as floats.
There are no automatically reported accuracy, false-positive/negative rates,
latency, duration, confusion matrix, or arbitrary task/model slices.

Tests validate mechanics with small fixtures, not MALT detector performance.
No successful post-fix GPU experiment or quality benchmark is established by
those tests. See [current verification](../docs/current-state.md) and
[MLflow output](local-s1-mlflow.md).
