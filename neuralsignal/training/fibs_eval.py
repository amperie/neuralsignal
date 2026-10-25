"""FIBS evaluation: episode-grouped splits, fixed-FPR thresholds and simple baselines.

Thresholds for "recall at X% FPR" are chosen on a validation portion carved out of
the training data (never on the test set) and then applied unchanged to test rows,
so the reported test FPR can differ from the target.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from neuralsignal.datasets.sources.fibs import FIBS_SPLITS, resolve_fibs_splits

DEFAULT_FPR_TARGETS = (0.01, 0.05)
BASELINE_COLUMNS = {"transcript_chars": "_ns_fibs_transcript_chars", "step_position": "_ns_fibs_step_position"}
BASELINES = {"length": ["transcript_chars"], "step": ["step_position"], "length_step": ["transcript_chars", "step_position"]}


class FibsEvaluationError(ValueError):
    pass


def fibs_options(evaluation_config: dict[str, Any] | None) -> dict[str, Any]:
    options = dict(((evaluation_config or {}).get("fibs")) or {})
    mode = options.get("mode", "grouped")
    if mode not in {"grouped", "cross_split"}:
        raise FibsEvaluationError("evaluation.fibs.mode must be grouped or cross_split")
    for key, default in (("test_size", 0.25), ("validation_size", 0.2)):
        value = float(options.get(key, default))
        if not 0 < value < 1:
            raise FibsEvaluationError(f"evaluation.fibs.{key} must be between 0 and 1")
        options[key] = value
    targets = [float(value) for value in options.get("fpr_targets", DEFAULT_FPR_TARGETS)]
    if not targets or any(not 0 < value < 1 for value in targets):
        raise FibsEvaluationError("evaluation.fibs.fpr_targets must contain rates between 0 and 1")
    options["fpr_targets"] = targets
    options["mode"] = mode
    if mode == "cross_split":
        options["train_splits"] = resolve_fibs_splits(options.get("train_splits") or [])
        options["test_splits"] = resolve_fibs_splits(options.get("test_splits") or [])
        if set(options["train_splits"]) & set(options["test_splits"]):
            raise FibsEvaluationError("evaluation.fibs train_splits and test_splits must not overlap")
    options["baselines"] = bool(options.get("baselines", True))
    return options


def episode_groups(group_keys: list[list[str]], fallback: list[str]) -> list[str]:
    """Union rows that share any group key (content md5 or pair id) into one episode."""
    parent: dict[str, str] = {}

    def find(item):
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    nodes = []
    for keys, default in zip(group_keys, fallback):
        keys = list(keys or []) or [f"row:{default}"]
        root = find(keys[0])
        for key in keys[1:]:
            parent[find(key)] = root
        nodes.append(keys[0])
    return [find(node) for node in nodes]


def split_fibs(data: pd.DataFrame, target: str, options: dict[str, Any], random_state: int):
    """Return (train, validation, test) index arrays with no episode in two partitions."""
    groups = data["group_id"].to_numpy()
    info: dict[str, Any] = {"mode": options["mode"]}
    splitter = GroupShuffleSplit(n_splits=1, test_size=options["test_size"], random_state=random_state)
    pool, test = (np.asarray(part) for part in next(splitter.split(data, groups=groups)))
    if options["mode"] == "cross_split":
        # Episodes are partitioned first, then each side keeps only its splits, so a
        # transcript shared across quadrants never informs both training and test.
        split_names = data["fibs_split"].to_numpy()
        train_pool = pool[np.isin(split_names[pool], options["train_splits"])]
        test = test[np.isin(split_names[test], options["test_splits"])]
        info.update(train_splits=options["train_splits"], test_splits=options["test_splits"])
    else:
        train_pool = pool
    if len(train_pool) == 0 or len(test) == 0:
        raise FibsEvaluationError("FIBS split produced an empty train or test partition")
    splitter = GroupShuffleSplit(n_splits=1, test_size=options["validation_size"], random_state=random_state)
    fit, val = next(splitter.split(train_pool, groups=groups[train_pool]))
    train, validation = train_pool[fit], train_pool[val]
    y = data[target].to_numpy()
    for name, index in (("train", train), ("validation", validation), ("test", test)):
        counts = np.bincount(y[index].astype(int), minlength=2)
        info[f"{name}_rows"], info[f"{name}_positive"] = int(len(index)), int(counts[1])
        if counts.min() == 0:
            raise FibsEvaluationError(f"FIBS {name} partition needs both honest and deceptive rows "
                                      f"({counts[0]} honest, {counts[1]} deceptive); collect more data.")
    assert not (set(groups[train]) | set(groups[validation])) & set(groups[test])
    assert not set(groups[train]) & set(groups[validation])
    return train, validation, test, info


def threshold_at_fpr(negative_scores, fpr: float) -> float:
    """Largest-recall threshold whose validation FPR (score > threshold) is at most ``fpr``."""
    scores = np.sort(np.asarray(negative_scores, dtype=float))[::-1]
    if len(scores) == 0:
        return float("nan")
    allowed = int(math.floor(fpr * len(scores)))
    return float(scores[allowed]) if allowed < len(scores) else float("-inf")


def fixed_fpr_metrics(y_val, val_scores, y_test, test_scores, fpr_targets) -> dict[str, float]:
    y_val, y_test = np.asarray(y_val).astype(int), np.asarray(y_test).astype(int)
    val_scores, test_scores = np.asarray(val_scores, float), np.asarray(test_scores, float)
    metrics = {}
    for target in fpr_targets:
        tag = _pct(target)
        threshold = threshold_at_fpr(val_scores[y_val == 0], target)
        flags = test_scores > threshold
        pos, neg = y_test == 1, y_test == 0
        metrics[f"recall_at_{tag}_fpr"] = float(flags[pos].mean()) if pos.any() else float("nan")
        metrics[f"test_fpr_at_{tag}_fpr_threshold"] = float(flags[neg].mean()) if neg.any() else float("nan")
        metrics[f"threshold_at_{tag}_fpr"] = threshold
    return metrics


def ranking_metrics(y_true, scores) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    if len(set(y_true)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {"auroc": float(roc_auc_score(y_true, scores)), "auprc": float(average_precision_score(y_true, scores))}


def per_split_metrics(data, test, y_test, test_scores, prefix="") -> dict[str, float]:
    metrics = {}
    splits = data.iloc[test]["fibs_split"].to_numpy()
    for split in FIBS_SPLITS:
        mask = splits == split
        if not mask.any():
            continue
        metrics[f"{prefix}split_{split}_test_rows"] = float(mask.sum())
        metrics.update({f"{prefix}split_{split}_{key}": value
                        for key, value in ranking_metrics(np.asarray(y_test)[mask], test_scores[mask]).items()})
    return metrics


def baseline_metrics(data, target, train, validation, test, fpr_targets) -> dict[str, float]:
    """Logistic-regression baselines on transcript length and step position only."""
    metrics = {}
    y = data[target].to_numpy().astype(int)
    for name, columns in BASELINES.items():
        x = np.column_stack([_baseline_feature(data, column) for column in columns])
        mean, std = x[train].mean(axis=0), x[train].std(axis=0)
        x = (x - mean) / np.where(std > 0, std, 1.0)
        model = LogisticRegression(max_iter=1000).fit(x[train], y[train])
        val_scores, test_scores = model.predict_proba(x[validation])[:, 1], model.predict_proba(x[test])[:, 1]
        prefix = f"baseline_{name}_"
        metrics.update({prefix + key: value for key, value in ranking_metrics(y[test], test_scores).items()})
        metrics.update({prefix + key: value for key, value in
                        fixed_fpr_metrics(y[validation], val_scores, y[test], test_scores, fpr_targets).items()})
        metrics.update(per_split_metrics(data, test, y[test], test_scores, prefix=prefix))
    return metrics


def _baseline_feature(data, column):
    values = data[BASELINE_COLUMNS[column]].astype(float).to_numpy()
    return np.log1p(values) if column == "transcript_chars" else values


def _pct(rate: float) -> str:
    text = f"{rate * 100:g}".replace(".", "p")
    return f"{text}pct"
