"""Student-implemented dataset loading.

Implements the contract from basile-desjuzeur/ml-poc-project:
    load_dataset_split() -> tuple[X_train, X_test, y_train, y_test]
"""

from __future__ import annotations

import ast
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
    """Build the feature matrix X and label vector y.

    Adds derived features:
      - tier_1_count, tier_2_count, tier_3_count from source_tier_mix
        (Python dict literal as string — parse with ast.literal_eval)
      - is_buy_yes (binary) from direction
      - market_price_centered = abs(market_price_at_signal - 0.5)
      - hour_of_day from created_at if present (already pre-computed in our sample CSV)

    One-hot encodes:
      - bucket (drop_first=True)

    Drops:
      - the target column itself
      - all columns in EXCLUDED_FROM_FEATURES (anti-leak)
      - direction, bucket, source_tier_mix, market_price_at_signal (replaced by derived)
      - created_at, signal_id (if present)
      - heuristic_score (anti-leak baseline)
      - move_t24h_pct (anti-leak future-derived)
    """
    df = df.copy()

    def _parse_tier(blob: Any, tier: str) -> int:
        """Return the int count for a given tier from the source_tier_mix blob.

        Handles both real dicts and string-serialized Python dict literals
        (e.g. "{'tier_1': 2.0}"). Casts to int (the values are float in CSV).
        """
        if pd.isna(blob):
            return 0
        if isinstance(blob, str):
            try:
                blob = ast.literal_eval(blob)
            except (ValueError, SyntaxError):
                return 0
        if isinstance(blob, dict):
            val = blob.get(tier, 0)
            try:
                return int(val)
            except (TypeError, ValueError):
                return 0
        return 0

    # Tier counts
    df["tier_1_count"] = df["source_tier_mix"].apply(lambda b: _parse_tier(b, "tier_1"))
    df["tier_2_count"] = df["source_tier_mix"].apply(lambda b: _parse_tier(b, "tier_2"))
    df["tier_3_count"] = df["source_tier_mix"].apply(lambda b: _parse_tier(b, "tier_3"))

    # is_buy_yes
    df["is_buy_yes"] = (df["direction"].astype(str).str.upper() == "BUY_YES").astype(int)

    # market_price_centered
    df["market_price_centered"] = (df["market_price_at_signal"] - 0.5).abs()

    # hour_of_day from created_at if present and not already computed
    if "created_at" in df.columns and "hour_of_day" not in df.columns:
        df["hour_of_day"] = pd.to_datetime(df["created_at"]).dt.hour

    # one-hot bucket
    bucket_dummies = pd.get_dummies(df["bucket"], prefix="bucket",
                                     drop_first=True, dtype=int)
    df = pd.concat([df, bucket_dummies], axis=1)

    # Extract y, build drop list
    y = df[TARGET_COLUMN]

    drop_cols: list[str] = [
        TARGET_COLUMN, "direction", "bucket", "source_tier_mix",
        "market_price_at_signal", "heuristic_score", "move_t24h_pct",
    ]
    if "created_at" in df.columns:
        drop_cols.append("created_at")
    if "signal_id" in df.columns:
        drop_cols.append("signal_id")
    drop_cols += [c for c in EXCLUDED_FROM_FEATURES if c in df.columns]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Some columns may be NULL (LEFT JOIN in export); impute with median
    nullable_cols = ["specificity_score", "cosine_score"]
    needs_copy = any(
        col in X.columns and X[col].isnull().any() for col in nullable_cols
    )
    if needs_copy:
        X = X.copy()
    for col in nullable_cols:
        if col in X.columns and X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())

    # Sanity: no nulls remain
    if X.isnull().any().any():
        bad = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Unexpected NaN in features after engineering: {bad}")

    return X, y


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return (X_train, X_test, y_train, y_test). Implemented in Task 14."""
    raise NotImplementedError("load_dataset_split is implemented in Task 14.")
