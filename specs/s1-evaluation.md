# S1 Evaluation

## Goal

S1 evaluation must make model performance comparable across tests, datasets,
feature sets, and slices. A single global score is not enough.

## Required Inputs

- feature dataset URI;
- feature dataset manifest hash;
- detector name;
- feature column selection;
- label column;
- train/validation/test split definition;
- model config.

## Metrics

Global metrics:

```text
AUROC
AUPRC
F1
precision
recall
false_positive_rate
false_negative_rate
accuracy
```

Operational metrics:

```text
train_rows
validation_rows
test_rows
feature_count
training_seconds
predict_seconds_per_1000_rows
```

## Slice Metrics

Evaluate metrics by:

- dataset split;
- source dataset;
- label type;
- task family;
- source model;
- prompt template version;
- feature-set selection.

## Threshold Selection

Thresholds are selected on validation data only. Test data is used only for final
reporting.

Store:

- selected threshold;
- selection metric;
- validation metric at threshold;
- test metric at threshold.

## Baselines

Each S1 experiment should compare against:

- majority-class baseline;
- simple logistic regression baseline when practical;
- previous promoted S1 model for the same detector when available.

## Promotion Criteria

A model can be promoted only if:

- test metrics beat baseline;
- key slice metrics do not regress materially;
- calibration is acceptable for the intended threshold;
- feature dataset manifest is available and immutable;
- training config and code SHA are logged.

## MLflow Artifacts

Log:

- trained model;
- training config;
- resolved feature column list;
- feature dataset manifest;
- metrics JSON;
- slice metrics table;
- confusion matrix;
- threshold report;
- calibration data.
