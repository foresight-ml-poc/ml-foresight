"""Shared fixtures for tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src to path so tests can import data, metrics, etc.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Mini raw DataFrame matching the schema of data/raw/signals_export_sample.csv."""
    return pd.DataFrame({
        "direction": ["BUY_YES", "BUY_NO", "BUY_YES", "BUY_NO"],
        "market_price_at_signal": [0.42, 0.78, 0.55, 0.18],
        "source_tier_mix": [
            "{'tier_1': 2, 'tier_2': 1}",
            "{'tier_1': 1}",
            "{'tier_2': 3, 'tier_3': 1}",
            "{'tier_1': 4, 'tier_2': 2}",
        ],
        "cosine_score": [0.62, 0.55, 0.71, 0.48],
        "freshness_factor": [0.95, 0.7, 0.4, 0.85],
        "source_weight": [0.8, 0.5, 0.6, 0.9],
        "confirmation_factor": [1.0, 0.5, 0.7, 1.0],
        "liquidity_factor": [0.6, 0.3, 0.4, 0.7],
        "spread_penalty": [0.9, 0.5, 0.6, 0.85],
        "time_to_resolution_factor": [0.5, 0.3, 0.6, 0.7],
        "impact_strength": [0.7, 0.4, 0.5, 0.8],
        "llm_confidence": [0.85, 0.6, 0.7, 0.9],
        "ambiguity_score": [0.2, 0.5, 0.4, 0.15],
        "specificity_score": [0.7, 0.5, 0.6, 0.8],
        "bucket": ["geopolitics", "politics", "geopolitics", "sports"],
        "articles_count": [4, 2, 3, 5],
        "unique_sources_count": [3, 2, 2, 4],
        "move_t24h_pct": [12.5, -3.0, 8.2, 15.1],
        "direction_correct": [1, 0, 1, 0],
        "heuristic_score": [78, 52, 65, 82],
        "hour_of_day": [14, 9, 21, 16],
    })
