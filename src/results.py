"""Helpers for saving evaluation results."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from config import MODEL_METRICS_FILE


def write_metrics(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Write model metrics to ``results/model_metrics.csv`` and return a DataFrame."""

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(MODEL_METRICS_FILE, index=False)
    return metrics_df
