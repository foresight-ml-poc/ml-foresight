"""Student-implemented dataset loading.

Implements the contract from basile-desjuzeur/ml-poc-project:
    load_dataset_split() -> tuple[X_train, X_test, y_train, y_test]
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR,
    EXCLUDED_FROM_FEATURES,
    MODELS_DIR,
    SEED,
    TARGET_COLUMN,
)

CRITICAL_FEATURES = [
    "freshness_factor", "source_weight", "confirmation_factor",
    "liquidity_factor", "spread_penalty", "time_to_resolution_factor",
    "impact_strength", "llm_confidence",
]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows and cast types.

    Drops rows where:
      - direction_correct is NULL (missing label)
      - any critical numeric feature is NULL

    Casts:
      - direction_correct to int64

    Raises:
      ValueError if fewer than 2 classes remain after cleaning.
    """
    df = df.dropna(subset=[TARGET_COLUMN])
    df = df.dropna(subset=CRITICAL_FEATURES)
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int64")

    n_classes = df[TARGET_COLUMN].nunique()
    if n_classes < 2:
        raise ValueError(
            f"Need at least 2 classes after cleaning, got {n_classes}. "
            f"Class distribution: {df[TARGET_COLUMN].value_counts().to_dict()}"
        )
    return df


def _feature_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and label vector y. Implemented in Task 13."""
    raise NotImplementedError("_feature_engineer is implemented in Task 13.")


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return (X_train, X_test, y_train, y_test). Implemented in Task 14."""
    raise NotImplementedError("load_dataset_split is implemented in Task 14.")
