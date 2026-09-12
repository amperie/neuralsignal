from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class S1TrainingResult:
    model: Any
    metrics: dict[str, float]
    feature_columns: list[str]


def select_feature_columns(
    columns: list[str],
    include_sets: list[str] | None = None,
    include_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> list[str]:
    include_sets = include_sets or []
    include_columns = include_columns or []
    exclude = set(exclude_columns or [])
    selected: list[str] = []

    for column in columns:
        if include_sets and any(column.startswith(f"{name}__") for name in include_sets):
            selected.append(column)
        if column in include_columns:
            selected.append(column)

    if not include_sets and not include_columns:
        selected = [column for column in columns if "__" in column]

    deduped = []
    for column in selected:
        if column not in exclude and column not in deduped:
            deduped.append(column)
    return deduped


def train_s1(
    dataset_path: str | Path,
    label_column: str,
    feature_config: dict[str, Any] | None = None,
    mlflow_config: dict[str, Any] | None = None,
    random_state: int = 42,
) -> S1TrainingResult:
    data = _read_features(dataset_path)
    features = select_feature_columns(
        list(data.columns),
        include_sets=(feature_config or {}).get("include_sets"),
        include_columns=(feature_config or {}).get("include_columns"),
        exclude_columns=(feature_config or {}).get("exclude_columns"),
    )
    if not features:
        raise ValueError("No feature columns selected")
    if "metadata_json" in data.columns:
        metadata = [json.loads(value) for value in data["metadata_json"]]
        if any(meta.get("source") == "metr-evals/malt-public" for meta in metadata):
            from neuralsignal.features.malt_runs import aggregate_malt_runs
            data = aggregate_malt_runs(data, features, label_column)
    if label_column not in data.columns:
        raise ValueError(f"Missing label column: {label_column}")

    x = data[features].astype(float)
    y = data[label_column].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state, stratify=y)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x_train, y_train)

    scores = model.predict_proba(x_test)[:, 1]
    preds = (scores >= 0.5).astype(int)
    metrics = {
        "auroc": _safe_metric(roc_auc_score, y_test, scores),
        "auprc": _safe_metric(average_precision_score, y_test, scores),
        "f1": _safe_metric(f1_score, y_test, preds),
        "precision": _safe_metric(precision_score, y_test, preds, zero_division=0),
        "recall": _safe_metric(recall_score, y_test, preds, zero_division=0),
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "feature_count": float(len(features)),
    }
    if "run_source" in data.columns:
        conditions = data.loc[x_test.index, "run_source"]
        for condition in conditions.unique():
            mask = (conditions == condition).to_numpy()
            metrics[f"{condition}_test_runs"] = float(mask.sum())
            metrics[f"{condition}_auroc"] = _safe_metric(roc_auc_score, y_test.iloc[mask], scores[mask])
            metrics[f"{condition}_f1"] = _safe_metric(f1_score, y_test.iloc[mask], preds[mask], zero_division=0)
    _log_mlflow(model, metrics, features, dataset_path, mlflow_config or {})
    return S1TrainingResult(model=model, metrics=metrics, feature_columns=features)


def _read_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        frames = [pd.read_parquet(item) for item in sorted((path / "features").glob("part-*.parquet"))]
        if not frames:
            raise ValueError(f"No feature shards found under {path / 'features'}")
        return pd.concat(frames, ignore_index=True)
    return pd.read_parquet(path)


def _safe_metric(fn, y_true, y_score, **kwargs) -> float:
    try:
        return float(fn(y_true, y_score, **kwargs))
    except ValueError:
        return float("nan")


def _log_mlflow(model, metrics: dict[str, float], features: list[str], dataset_path: str | Path, config: dict[str, Any]) -> None:
    if not config:
        return

    import mlflow

    if config.get("tracking_uri"):
        mlflow.set_tracking_uri(config["tracking_uri"])
    if config.get("experiment_name"):
        mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=config.get("run_name")):
        params = {
            "dataset_path": str(dataset_path),
            "feature_count": len(features),
            "model_type": "logistic_regression",
        }
        params.update(config.get("extra_params") or {})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected_features.json"
            path.write_text(json.dumps(features, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(path))
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=config.get("registered_model_name"))