"""Tests for src/metrics.py."""

from __future__ import annotations

import numpy as np

from metrics import compute_metrics


def test_compute_metrics_returns_dict_with_expected_keys() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert isinstance(metrics, dict)
    expected = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(metrics.keys()) == expected


def test_compute_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_metrics_all_values_are_floats() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0])
    metrics = compute_metrics(y_true, y_pred)
    for k, v in metrics.items():
        assert isinstance(v, float), f"{k} is {type(v)}, expected float"


def test_compute_metrics_handles_single_class_pred_without_crash() -> None:
    """If all predictions are 0, precision/recall could divide by zero."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.0  # zero_division=0
