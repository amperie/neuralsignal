# S1 evaluation behavior

Evaluation currently happens inside [train_s1](../neuralsignal/training/s1.py).
There is no evaluation-only CLI, saved-model evaluation loader, validation set,
threshold search, calibration, or promotion automation.

## Split and predictions

The trainer uses a stratified 75/25 train/test split with random state 42 by
default. MALT rows are pooled into runs first, keeping a run out of both partitions
at once. Other datasets split by row and do not enforce arbitrary group isolation.
XGBoost predicts class-1 probability; threshold 0.5 yields flags.

## Logged test metrics

- `auc` and `auroc`: ROC AUC computed from class-1 probabilities.
- `auprc`: average precision.
- `accuracy`, `precision`, `recall`, `f1`, `specificity`.
- `false_positive_rate`, `false_negative_rate`, `negative_predictive_value`.
- `mcc`, `log_loss`, `brier_score`.
- `tn`, `fp`, `fn`, `tp`, `positive_support`, `negative_support`.
- `train_rows`, `test_rows`, `feature_count`.

When `run_source` exists, each condition receives the same classification metrics
with a `<condition>_` prefix, plus `<condition>_test_runs`. Precision/recall/F1 use
zero for undefined divisions; undefined AUC and rates are NaN in the returned
results. MLflow scalar logging omits nonfinite values and `metrics.json` records
them as null rather than reporting a misleading zero.

## Artifacts

Every logged run includes `metrics.json`, `classification_report.json` (per-class
precision/recall/F1/support plus averages), and `confusion_matrix.json`, `.csv`,
and `.png`. Matrix rows are actual labels, columns predicted labels, both ordered
[0, 1]. The test split and prediction threshold 0.5 are logged as parameters.
Selected feature names and the XGBoost model are also saved.

There is no validation-set threshold search, calibration procedure, timing report,
or arbitrary task/model slice evaluation. Local integration tests verify artifact
contents and model reload. Actual writes to `http://z440.lan:5000` remain unverified
while that hostname is unresolved from this machine.
