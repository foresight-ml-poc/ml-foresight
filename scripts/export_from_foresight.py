"""Export historical signals + outcomes + features from Foresight DB to CSV.

Real schema (verified 2026-05-06):
  - The `direction_correct` column on signal_outcomes is sparse (only 30/438
    rows have it filled). We compute our own label from `move_t24h_pct` +
    `direction` instead, which gives ~401 labelled signals.
  - The `signals.below_threshold` column referenced in BLUEPRINT does not
    exist; the filter is dropped.
  - We previously joined `event_market_features` for the 6 heuristic factors,
    but that table was only persisted from 2026-04-27 onward (a known bug
    documented in app/scoring/event_market_features_writer.py). Joining it
    capped the dataset to 40 samples.
  - **v1.1.0 change**: drop the `event_market_features` join entirely. We
    train on whatever features are available across all 401 signals (LLM
    features from `event_market_analysis`, signal-time metadata, event
    context). The heuristic baseline uses the already-computed
    `signals.signal_score` column directly (no recomputation needed).

Usage:
    Set FORESIGHT_DB_DSN in .env or .env.local, then run:
    python scripts/export_from_foresight.py

Output:
    data/raw/signals_export_<YYYY-MM-DD>.csv
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local")

DSN = os.environ.get("FORESIGHT_DB_DSN")
if not DSN:
    print("ERROR: set FORESIGHT_DB_DSN in .env or .env.local (see .env.example).",
          file=sys.stderr)
    sys.exit(1)

QUERY = """
SELECT
    s.id              AS signal_id,
    s.created_at,
    s.direction,
    s.market_price_at_signal,
    s.source_tier_mix,
    s.cosine_score,
    -- LLM features (always populated for all 401 analyzed signals)
    ema.impact_strength,
    ema.llm_confidence,
    ema.ambiguity_score,
    ema.specificity_score,
    -- Event context
    e.bucket,
    e.articles_count,
    e.unique_sources_count,
    -- Outcome (training label source)
    so.move_t24h_pct,
    -- Computed label: direction-correctness at T+24h
    CASE
        WHEN s.direction IN ('BUY_YES', 'YES', 'UP')
            THEN (so.move_t24h_pct > 0)::int
        WHEN s.direction IN ('BUY_NO', 'NO', 'DOWN')
            THEN (so.move_t24h_pct < 0)::int
        ELSE NULL
    END AS direction_correct,
    -- Heuristic baseline (used by train.py for vs-heuristic comparison,
    -- NEVER fed to ML as a feature — anti-leak).
    s.signal_score AS heuristic_score
FROM signals s
JOIN signal_outcomes so
    ON so.signal_id = s.id
JOIN event_market_analysis ema
    ON ema.event_id = s.event_id AND ema.market_id = s.market_id
JOIN events e
    ON e.id = s.event_id
WHERE so.move_t24h_pct IS NOT NULL
  AND s.direction IN ('BUY_YES', 'BUY_NO', 'YES', 'NO', 'UP', 'DOWN')
  AND ema.impact_strength IS NOT NULL
ORDER BY s.created_at;
"""


def main() -> None:
    print("Connecting to Foresight DB...")
    with psycopg2.connect(DSN) as conn:
        df = pd.read_sql_query(QUERY, conn)

    print(f"Fetched {len(df)} rows.")

    if df["direction_correct"].isnull().any():
        n_null = df["direction_correct"].isnull().sum()
        print(f"WARNING: {n_null} rows have NULL direction_correct after compute "
              f"(unexpected direction values?). Dropping them.")
        df = df.dropna(subset=["direction_correct"])

    df["direction_correct"] = df["direction_correct"].astype(int)

    if len(df) < 40:
        print(f"WARNING: only {len(df)} usable rows. Statistics will be very noisy.")

    out_dir = PROJECT_ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"signals_export_{date.today().isoformat()}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(f"Class balance: {df['direction_correct'].value_counts().to_dict()}")
    print(f"Date range: {df['created_at'].min()} → {df['created_at'].max()}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    main()
