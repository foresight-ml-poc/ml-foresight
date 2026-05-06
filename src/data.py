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
    "impact_strength", "llm_confidence", "ambiguity_score",
    "market_price_at_signal", "cosine_score",
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
    """Return (X_train, X_test, y_train, y_test) — scaled.

    Loading priority:
      1. The latest dated export `data/raw/signals_export_<DATE>.csv`
      2. Fallback: `data/raw/signals_export_sample.csv`

    Pipeline:
      load → _clean → _feature_engineer
      → train_test_split (stratified, 80/20, seed=42)
      → fit StandardScaler on X_train only → transform train + test
      → save scaler.pkl + feature_order.pkl in models/

    Returns numpy arrays (per Basile's contract). Feature column order is
    persisted to models/feature_order.pkl for the backend repo.
    """
    raw_dir = DATA_DIR / "raw"
    dated = sorted(raw_dir.glob("signals_export_20*.csv"))
    candidates = [p for p in dated if p.name != "signals_export_sample.csv"]

    if candidates:
        csv_path = candidates[-1]
    elif (raw_dir / "signals_export_sample.csv").exists():
        csv_path = raw_dir / "signals_export_sample.csv"
    else:
        raise FileNotFoundError(
            f"No CSV in {raw_dir}. Run scripts/export_from_foresight.py first."
        )

    print(f"[data] Loading: {csv_path.name}")
    df = pd.read_csv(csv_path)

    cleaned = _clean(df)
    print(f"[data] After clean: {len(cleaned)} rows, "
          f"target balance = {cleaned[TARGET_COLUMN].value_counts().to_dict()}")

    X, y = _feature_engineer(cleaned)
    print(f"[data] After feature engineering: X shape = {X.shape}")

    # 80/20 split, stratified on y. With ~411 samples → 328 train / 83 test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"[data] Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Save heuristic_score for the test set so train.py can compute the
    # baseline directly from the persisted column (no recomputation needed).
    if "heuristic_score" in cleaned.columns:
        # Re-split the cleaned df with the same indices to get test heuristic
        train_idx = X_train.index.tolist()
        test_idx = X_test.index.tolist()
        cleaned_aligned = cleaned.loc[X.index]
        test_heuristic = cleaned_aligned.loc[test_idx, "heuristic_score"].to_numpy()
        joblib.dump(test_heuristic, MODELS_DIR / "test_heuristic_scores.pkl")
        print(f"[data] Saved test_heuristic_scores.pkl ({len(test_heuristic)} values)")

    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(list(X.columns), MODELS_DIR / "feature_order.pkl")
    print(f"[data] Saved scaler.pkl and feature_order.pkl in {MODELS_DIR}")

    return X_train_arr, X_test_arr, y_train.to_numpy(), y_test.to_numpy()
