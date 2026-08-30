"""Evaluation helpers for imbalanced binary classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _positive_scores(model: Any, X: Any) -> Any:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def evaluate_binary_classifier(model: Any, X_test: Any, y_test: Any, *, threshold: float = 0.5) -> dict[str, Any]:
    """Evaluate a binary classifier with failure class as the positive class."""

    y_score = _positive_scores(model, X_test)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "failure_precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "failure_recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "failure_f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }

    return metrics


def save_metrics(metrics_by_model: dict[str, dict[str, Any]], output_path: str | Path) -> Path:
    """Save model metrics as JSON and a flat CSV next to it."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_by_model, handle, indent=2)

    flat_rows = []
    for model_name, metrics in metrics_by_model.items():
        flat = {key: value for key, value in metrics.items() if key != "confusion_matrix"}
        flat["model"] = model_name
        flat_rows.append(flat)

    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(flat_rows).set_index("model").to_csv(csv_path)
    return output_path
