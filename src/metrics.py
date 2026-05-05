"""Student-implemented metrics.

Implements the contract from basile-desjuzeur/ml-poc-project:
    compute_metrics(y_true, y_pred) -> dict[str, float]
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Binary classification metrics for direction_correct in {0, 1}.

    y_pred is expected to be hard predictions (0/1) since Basile's main.py
    calls model.predict(). Probability-based ROC-AUC will be computed in
    train.py / app.py where we have access to predict_proba.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
    }
