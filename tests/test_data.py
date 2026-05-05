"""Tests for src/data.py cleaning logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data import _clean


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
