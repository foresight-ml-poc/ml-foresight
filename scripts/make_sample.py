"""Generate an anonymized sample CSV from the latest signals export.

Strategy:
- Drop signal_id (privacy)
- Replace created_at with hour_of_day (kept as feature, exact timestamp removed)
- Use all rows when dataset is small (≤80), else stratified sample of 50

Usage:
    python scripts/make_sample.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_OUT = RAW_DIR / "signals_export_sample.csv"

MAX_SAMPLE_ROWS = 50
SEED = 42


def main() -> None:
    csvs = sorted(
        c for c in glob.glob(str(RAW_DIR / "signals_export_*.csv"))
        if "sample" not in c
    )
    if not csvs:
        raise SystemExit(
            "No signals_export_*.csv found. "
            "Run scripts/export_from_foresight.py first."
        )

    df = pd.read_csv(csvs[-1])
    print(f"Source: {csvs[-1]}, {len(df)} rows.")

    # Anonymize: derive hour_of_day, drop identifying columns
    df["hour_of_day"] = pd.to_datetime(df["created_at"]).dt.hour
    df = df.drop(columns=["signal_id", "created_at"])

    if len(df) <= MAX_SAMPLE_ROWS * 1.6:
        # Small dataset → keep everything (still anonymized)
        sample = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"Dataset is small ({len(df)} rows) — keeping all rows in the sample.")
    else:
        # Stratified sample on direction_correct
        wins = df[df["direction_correct"] == 1]
        losses = df[df["direction_correct"] == 0]
        n_per_class = MAX_SAMPLE_ROWS // 2
        sample = pd.concat([
            wins.sample(n=min(n_per_class, len(wins)), random_state=SEED),
            losses.sample(n=min(n_per_class, len(losses)), random_state=SEED),
        ]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"Stratified sample: {len(sample)} rows.")

    sample.to_csv(SAMPLE_OUT, index=False)
    print(f"Wrote anonymized sample: {SAMPLE_OUT}")
    print(f"Class balance: {sample['direction_correct'].value_counts().to_dict()}")
    print(f"Columns ({len(sample.columns)}): {list(sample.columns)}")


if __name__ == "__main__":
    main()
