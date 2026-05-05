"""Tests for src/data.py cleaning logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import _clean, _feature_engineer


def test_clean_drops_null_labels(raw_df: pd.DataFrame) -> None:
    df = raw_df.copy()
    df.loc[0, "direction_correct"] = None
    cleaned = _clean(df)
    assert len(cleaned) == 3
    assert cleaned["direction_correct"].notnull().all()


def test_clean_drops_null_critical_features(raw_df: pd.DataFrame) -> None:
    df = raw_df.copy()
    df.loc[0, "freshness_factor"] = None
    df.loc[1, "liquidity_factor"] = None
    cleaned = _clean(df)
    assert len(cleaned) == 2


def test_clean_casts_label_to_int(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    assert cleaned["direction_correct"].dtype == np.dtype("int64")
    assert set(cleaned["direction_correct"].unique()) <= {0, 1}


def test_clean_raises_if_only_one_class(raw_df: pd.DataFrame) -> None:
    df = raw_df.copy()
    df["direction_correct"] = 1
    with pytest.raises(ValueError, match="2 classes"):
        _clean(df)


def test_feature_engineer_returns_X_y(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, y = _feature_engineer(cleaned)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y) == len(cleaned)


def test_feature_engineer_excludes_leak_columns(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    forbidden = {
        "move_t24h_pct", "outcome_label", "price_t24h",
        "signal_score", "signal_strength", "trade_quality",
        "heuristic_score",
    }
    assert forbidden.isdisjoint(X.columns), \
        f"Leak columns leaked into X: {forbidden & set(X.columns)}"


def test_feature_engineer_creates_tier_counts(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    assert "tier_1_count" in X.columns
    assert "tier_2_count" in X.columns
    assert "tier_3_count" in X.columns


def test_feature_engineer_creates_is_buy_yes(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    assert "is_buy_yes" in X.columns
    assert set(X["is_buy_yes"].unique()) <= {0, 1}


def test_feature_engineer_creates_market_price_centered(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    assert "market_price_centered" in X.columns
    assert (X["market_price_centered"] >= 0).all()
    assert (X["market_price_centered"] <= 0.5).all()


def test_feature_engineer_one_hot_encodes_bucket(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    bucket_cols = [c for c in X.columns if c.startswith("bucket_")]
    # The fixture has 3 unique buckets (geopolitics, politics, sports);
    # drop_first=True → expect 2 columns
    assert len(bucket_cols) >= 1
    assert "bucket" not in X.columns


def test_feature_engineer_no_nulls_in_output(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    assert not X.isnull().any().any(), "X contains NaN values"


def test_feature_engineer_keeps_cosine_score(raw_df: pd.DataFrame) -> None:
    """cosine_score is a useful retrieval-quality feature, must be kept."""
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    assert "cosine_score" in X.columns


def test_feature_engineer_keeps_heuristic_factors(raw_df: pd.DataFrame) -> None:
    """The 6 heuristic factor features must be passed through unchanged."""
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    for f in ["freshness_factor", "source_weight", "confirmation_factor",
              "liquidity_factor", "spread_penalty", "time_to_resolution_factor"]:
        assert f in X.columns, f"Missing heuristic factor: {f}"


from data import load_dataset_split


def test_load_dataset_split_with_sample() -> None:
    """End-to-end: load_dataset_split should run on the committed sample."""
    from config import DATA_DIR
    if not (DATA_DIR / "raw" / "signals_export_sample.csv").exists():
        pytest.skip("Sample CSV not present.")

    X_train, X_test, y_train, y_test = load_dataset_split()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert set(np.unique(y_train)) <= {0, 1}
    assert set(np.unique(y_test)) <= {0, 1}
    # With 40 rows and 80/20 → 32/8
    assert len(X_train) + len(X_test) == 40
