from __future__ import annotations

import json
import math
import logging
import os
from datetime import datetime, timezone
from uuid import uuid4
from contextlib import contextmanager
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from neuralsignal.storage.manifests import local_shard_path, sha256_file

from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score, accuracy_score, confusion_matrix, classification_report,
                             log_loss, brier_score_loss, matthews_corrcoef)
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class S1TrainingResult:
    model: Any
    metrics: dict[str, float]
    feature_columns: list[str]
    output_dir: str | None = None


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
    model_config: dict[str, Any] | None = None,
    output_root: str | Path = "runs/s1",
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
    labels = pd.to_numeric(data[label_column], errors="raise")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Binary labels must contain only 0 and 1")
    y = labels.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state, stratify=y)
    model_config = model_config or {}
    if model_config.get("type", "xgboost") != "xgboost":
        raise ValueError("S1 model.type must be xgboost")
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": 4,
        "random_state": random_state,
        "eval_metric": "logloss",
        **(model_config.get("params") or {}),
    }
    if params.get("objective", "binary:logistic") != "binary:logistic":
        raise ValueError("S1 requires objective binary:logistic")
    params["objective"] = "binary:logistic"
    model = XGBClassifier(**params)
    model.fit(x_train, y_train)

    scores = model.predict_proba(x_test)[:, 1]
    metrics = _classification_metrics(y_test, scores)
    metrics.update({
        "train_rows": float(len(x_train)),
        "test_rows": float(len(x_test)),
        "feature_count": float(len(features)),
    })
    if "run_source" in data.columns:
        conditions = data.loc[x_test.index, "run_source"]
        for condition in conditions.unique():
            mask = (conditions == condition).to_numpy()
            metrics[f"{condition}_test_runs"] = float(mask.sum())
            metrics.update({f"{condition}_{key}": value for key, value in
                            _classification_metrics(y_test.iloc[mask], scores[mask]).items()})
    output = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    model.save_model(output / "model.ubj")
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "selected_features.json", features)
    _write_json(output / "training.json", {
        "dataset_path": str(Path(dataset_path).resolve()), "label_column": label_column,
        "random_state": random_state, "test_size": 0.25, "threshold": 0.5,
        "model_params": model.get_params(), "features": feature_config or {},
    })
    preds = (scores >= 0.5).astype(int)
    _write_json(output / "confusion_matrix.json", {
        "labels": [0, 1], "rows": "actual", "columns": "predicted",
        "threshold": 0.5, "matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
    })
    _write_json(output / "classification_report.json", classification_report(
        y_test, preds, labels=[0, 1], output_dict=True, zero_division=0))
    pd.DataFrame({"row_index": x_test.index, "label": y_test.to_numpy(),
                  "score": scores, "prediction": preds}).to_csv(output / "predictions.csv", index=False)
    logging.getLogger(__name__).info("S1 outputs saved to %s", output)
    try:
        with _mlflow_request_limits():
            _log_mlflow(model, metrics, features, dataset_path, mlflow_config or {}, y_test, scores)
    except Exception as error:
        logging.getLogger(__name__).warning("MLflow reporting failed: %s. Training completed; outputs saved to %s", error, output)
    return S1TrainingResult(model=model, metrics=metrics, feature_columns=features, output_dir=str(output))


def _classification_metrics(y_true, scores) -> dict[str, float]:
    preds = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    ratio = lambda numerator, denominator: float(numerator / denominator) if denominator else float("nan")
    auc = _safe_metric(roc_auc_score, y_true, scores) if len(set(y_true)) == 2 else float("nan")
    return {
        "auc": auc,
        "auroc": auc,
        "auprc": _safe_metric(average_precision_score, y_true, scores) if tp + fn else float("nan"),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "specificity": ratio(tn, tn + fp),
        "false_positive_rate": ratio(fp, tn + fp),
        "false_negative_rate": ratio(fn, tp + fn),
        "negative_predictive_value": ratio(tn, tn + fn),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
        "positive_support": float(tp + fn), "negative_support": float(tn + fp),
    }


def _read_features(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("state", "completed") != "completed":
                raise ValueError("Feature run is not completed")
            frames = []
            seen = set()
            for shard in manifest.get("shards", []):
                item = local_shard_path(path, shard["path"])
                if item in seen:
                    raise ValueError(f"Duplicate shard path: {shard['path']}")
                seen.add(item)
                if sha256_file(item) != shard["sha256"]:
                    raise ValueError(f"Shard checksum mismatch: {shard['path']}")
                frame = pd.read_parquet(item)
                if len(frame) != shard["rows"]:
                    raise ValueError(f"Shard row count mismatch: {shard['path']}")
                frames.append(frame)
        else:
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


def _log_mlflow(model, metrics: dict[str, float], features: list[str], dataset_path: str | Path, config: dict[str, Any], y_true, scores) -> None:
    if config.get("enabled", True) is False:
        return

    config = {"tracking_uri": os.environ.get("MLFLOW_TRACKING_URI", "http://z440.lan:5000"),
              "experiment_name": "neuralsignal-s1", **config}
    import mlflow
    import mlflow.xgboost

    if config.get("tracking_uri"):
        mlflow.set_tracking_uri(config["tracking_uri"])
    if config.get("experiment_name"):
        mlflow.set_experiment(config["experiment_name"])

    with mlflow.start_run(run_name=config.get("run_name")):
        params = {
            "dataset_path": str(dataset_path),
            "feature_count": len(features),
            "model_type": "xgboost",
            "evaluation_split": "test",
            "prediction_threshold": 0.5,
            **{f"xgboost.{key}": value for key, value in model.get_params().items() if value is not None},
        }
        params.update(config.get("extra_params") or {})
        mlflow.log_params(params)
        mlflow.log_metrics({key: value for key, value in metrics.items() if math.isfinite(value)})
        mlflow.log_dict({key: value if math.isfinite(value) else None for key, value in metrics.items()}, "metrics.json")
        preds = (scores >= 0.5).astype(int)
        matrix = confusion_matrix(y_true, preds, labels=[0, 1])
        mlflow.log_dict({"labels": [0, 1], "rows": "actual", "columns": "predicted",
                         "threshold": 0.5, "matrix": matrix.tolist()}, "confusion_matrix.json")
        mlflow.log_dict(classification_report(y_true, preds, labels=[0, 1],
                       output_dict=True, zero_division=0), "classification_report.json")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected_features.json"
            path.write_text(json.dumps(features, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(path))
            pd.DataFrame(matrix, index=["actual_0", "actual_1"],
                         columns=["predicted_0", "predicted_1"]).to_csv(Path(tmp) / "confusion_matrix.csv")
            mlflow.log_artifact(str(Path(tmp) / "confusion_matrix.csv"))
            from matplotlib.figure import Figure
            from sklearn.metrics import ConfusionMatrixDisplay
            figure = Figure(figsize=(5, 4))
            ax = figure.subplots()
            ConfusionMatrixDisplay(matrix, display_labels=[0, 1]).plot(ax=ax, colorbar=False)
            ax.set_title("Held-out test set (threshold 0.5)")
            figure.tight_layout()
            figure.savefig(Path(tmp) / "confusion_matrix.png", dpi=150)
            mlflow.log_artifact(str(Path(tmp) / "confusion_matrix.png"))
        mlflow.xgboost.log_model(model, artifact_path="model", registered_model_name=config.get("registered_model_name"))


def _write_json(path: Path, value) -> None:
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, list):
            return [clean(val) for val in item]
        return item
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


@contextmanager
def _mlflow_request_limits():
    # Avoid MLflow's long default retries when the tracking service is offline.
    defaults = {"MLFLOW_HTTP_REQUEST_TIMEOUT": "5", "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0"}
    added = [key for key in defaults if key not in os.environ]
    for key in added:
        os.environ[key] = defaults[key]
    try:
        yield
    finally:
        for key in added:
            os.environ.pop(key, None)
