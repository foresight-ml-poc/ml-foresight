# ML Foresight — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ML pipeline that replaces Foresight's heuristic signal scoring with a supervised binary classifier predicting `direction_correct` at T+24h, comparing 3 models (LogReg, RandomForest, MLP) against the heuristic baseline.

**Architecture:** Python 3.11 repo cloned 1:1 from `basile-desjuzeur/ml-poc-project` template. `scripts/main.py` (Basile's fixed orchestrator) loads pre-trained models from `models/`, evaluates them via `src/metrics.py::compute_metrics`, writes `results/model_metrics.csv`, and launches a Streamlit dashboard. Training is handled by an additional `scripts/train.py` we write ourselves.

**Tech Stack:** scikit-learn, pandas, numpy, tensorflow (Keras), joblib, streamlit, matplotlib, seaborn, python-dotenv, psycopg2-binary, imbalanced-learn.

**Scope of this plan:** ml-foresight repo only. backend-foresight and frontend-foresight will get their own plans once ml-foresight is complete and a `v1.0.0` GitHub Release is published.

**Reference spec:** [`docs/specs/2026-05-05-design.md`](../specs/2026-05-05-design.md)

---

## Phase 1 — Repository scaffolding

### Task 1: Create requirements.txt

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write the file**

```
joblib
matplotlib
numpy
pandas
psycopg2-binary
python-dotenv
scikit-learn
seaborn
streamlit
tensorflow
imbalanced-learn
jupyter
pytest
```

- [ ] **Step 2: Set up venv and install**

```bash
cd /Users/vadim/foresight-ml-poc/ml-foresight
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Expected: install succeeds (~2-3 min, downloads tensorflow ~500MB).

- [ ] **Step 3: Pin versions for reproducibility**

```bash
pip freeze > requirements.txt
```

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build: pin dependencies after first successful install"
```

---

### Task 2: Create .env and .env.example

**Files:**
- Create: `.env`
- Create: `.env.example`

- [ ] **Step 1: Write `.env` (committed, no secrets)**

```dotenv
PYTHONPATH=./src
```

- [ ] **Step 2: Write `.env.example`**

```dotenv
# Optional — only needed for scripts/export_from_foresight.py
FORESIGHT_DB_DSN=postgresql://user:password@host:5432/signal
```

- [ ] **Step 3: Commit**

```bash
git add .env .env.example
git commit -m "build: add .env (PYTHONPATH=./src) and .env.example"
```

---

### Task 3: Update .gitignore for ML artifacts

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Append ML-specific patterns**

Add at the end of `.gitignore`:

```gitignore

# Project-specific
data/raw/signals_export_*.csv
!data/raw/signals_export_sample.csv
data/processed/
models/*.pkl
models/*.h5
models/*.joblib
!models/.gitkeep
plots/*.png
!plots/.gitkeep
results/*.csv
!results/.gitkeep
logs/*
!logs/.gitkeep
.streamlit/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "build: gitignore ML artifacts (raw data, models, plots, results)"
```

---

### Task 4: Copy Basile's fixed files

**Files:**
- Create: `scripts/main.py`
- Create: `src/__init__.py`
- Create: `src/model_io.py`
- Create: `src/results.py`

- [ ] **Step 1: Clone reference temporarily**

```bash
cd /tmp
git clone --depth=1 https://github.com/basile-desjuzeur/ml-poc-project basile-ref
```

- [ ] **Step 2: Copy fixed files**

```bash
cd /Users/vadim/foresight-ml-poc/ml-foresight
mkdir -p scripts src
cp /tmp/basile-ref/scripts/main.py scripts/main.py
cp /tmp/basile-ref/src/__init__.py src/__init__.py
cp /tmp/basile-ref/src/model_io.py src/model_io.py
cp /tmp/basile-ref/src/results.py src/results.py
```

- [ ] **Step 3: Verify line counts match reference**

```bash
wc -l scripts/main.py src/__init__.py src/model_io.py src/results.py
```

Expected (matching basile-ref):
- `scripts/main.py`: 171 lines
- `src/__init__.py`: 0 lines
- `src/model_io.py`: 39 lines
- `src/results.py`: 18 lines

- [ ] **Step 4: Commit**

```bash
git add scripts/main.py src/__init__.py src/model_io.py src/results.py
git commit -m "feat: copy fixed template files from basile-desjuzeur/ml-poc-project

These files are the contract orchestrator from the reference template.
We do not modify them; we plug into them via src/{config,data,metrics,app}.py."
```

---

### Task 5: Create src/config.py adapted to our project

**Files:**
- Create: `src/config.py`

- [ ] **Step 1: Write the file**

```python
"""Project configuration — adapted from basile-desjuzeur/ml-poc-project.

The MODELS dict is filled by scripts/train.py at training time.
Constants for anti-leak feature exclusion and reproducibility live here.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

# Auto-create directories on import (Basile's pattern)
for d in [
    DATA_DIR,
    DATA_DIR / "raw",
    DATA_DIR / "processed",
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    d.mkdir(exist_ok=True, parents=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# --- Project-specific constants ---

SEED = 42
TARGET_COLUMN = "direction_correct"
HEURISTIC_THRESHOLD = 65  # signal_score > 65 → label heuristique = 1

# Anti-leak: features that must NEVER be used as inputs
EXCLUDED_FROM_FEATURES = [
    # Future-derived (would leak)
    "move_t24h_pct",
    "price_t24h",
    "outcome_label",
    "price_resolved",
    # Heuristic-derived (would copy what we want to replace)
    "signal_score",
    "signal_strength",
    "trade_quality",
]

# Registry of trained models — populated by scripts/train.py
MODELS = {
    "logreg": {
        "name": "Logistic Regression",
        "description": "Baseline linéaire L2, class_weight balanced, C tuné par GridSearchCV.",
        "path": MODELS_DIR / "logreg.pkl",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble d'arbres, RandomizedSearchCV 5-fold, class_weight balanced.",
        "path": MODELS_DIR / "random_forest.pkl",
    },
    "mlp": {
        "name": "MLP (Keras)",
        "description": "Réseau dense 64→32→1, EarlyStopping, dropout 0.3/0.2, sigmoid.",
        "path": MODELS_DIR / "mlp.pkl",
    },
}
```

- [ ] **Step 2: Verify import auto-creates dirs**

```bash
cd /Users/vadim/foresight-ml-poc/ml-foresight
PYTHONPATH=./src .venv/bin/python -c "import config; print('OK', config.PROJECT_ROOT)"
ls data logs models notebooks plots results tests
```

Expected: prints `OK <project root path>`, all 7 directories exist.

- [ ] **Step 3: Add .gitkeep files**

```bash
touch data/raw/.gitkeep models/.gitkeep plots/.gitkeep results/.gitkeep logs/.gitkeep
git add data/raw/.gitkeep models/.gitkeep plots/.gitkeep results/.gitkeep logs/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add src/config.py
git commit -m "feat(config): add project paths, MODELS registry, anti-leak exclusions"
```

---

### Task 6: Sanity check — Basile's main.py runs

**Files:** none (verification only)

- [ ] **Step 1: Run main.py and expect a controlled failure**

```bash
cd /Users/vadim/foresight-ml-poc/ml-foresight
.venv/bin/python scripts/main.py
```

Expected output:
```
NotImplementedError: Implement data.load_dataset_split() before running scripts/main.py.
```

This is the correct behavior at this stage — `data.py` and `metrics.py` are still Basile's `NotImplementedError` stubs.

- [ ] **Step 2: Copy Basile's data.py and metrics.py stubs (so the import works)**

If they don't exist yet:
```bash
cp /tmp/basile-ref/src/data.py src/data.py
cp /tmp/basile-ref/src/metrics.py src/metrics.py
```

- [ ] **Step 3: Re-run and confirm controlled failure**

```bash
.venv/bin/python scripts/main.py 2>&1 | tail -5
```

Expected: `NotImplementedError` mentioning `data.load_dataset_split()`.

- [ ] **Step 4: Commit the stubs**

```bash
git add src/data.py src/metrics.py
git commit -m "feat: add data.py and metrics.py stubs (NotImplementedError for now)"
```

---

## Phase 2 — Data export from Foresight

### Task 7: Write the export script

**Files:**
- Create: `scripts/export_from_foresight.py`

- [ ] **Step 1: Write the script**

```python
"""Export historical signals + outcomes + features from Foresight DB to CSV.

Usage:
    Set FORESIGHT_DB_DSN in .env, then run:
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
load_dotenv(PROJECT_ROOT / ".env.local")  # optional override

DSN = os.environ.get("FORESIGHT_DB_DSN")
if not DSN:
    print("ERROR: set FORESIGHT_DB_DSN in .env (see .env.example).", file=sys.stderr)
    sys.exit(1)

QUERY = """
SELECT
    s.id              AS signal_id,
    s.created_at,
    s.direction,
    s.market_price_at_signal,
    s.source_tier_mix,
    emf.freshness,
    emf.source_weight,
    emf.confirmation,
    emf.liquidity,
    emf.spread,
    emf.time_to_resolution,
    ema.impact_strength,
    ema.confidence    AS llm_confidence,
    ema.ambiguity_score,
    ema.specificity_score,
    e.bucket,
    e.articles_count,
    e.unique_sources_count,
    so.direction_correct,
    so.move_t24h_pct,
    s.signal_score    AS heuristic_score
FROM signals s
JOIN signal_outcomes so
    ON so.signal_id = s.id
JOIN event_market_features emf
    ON emf.event_id = s.event_id AND emf.market_id = s.market_id
JOIN event_market_analysis ema
    ON ema.event_id = s.event_id AND ema.market_id = s.market_id
JOIN events e
    ON e.id = s.event_id
WHERE so.direction_correct IS NOT NULL
  AND s.below_threshold = FALSE
  AND emf.freshness IS NOT NULL
ORDER BY s.created_at;
"""


def main() -> None:
    print(f"Connecting to Foresight DB...")
    with psycopg2.connect(DSN) as conn:
        df = pd.read_sql_query(QUERY, conn)

    print(f"Fetched {len(df)} rows.")

    if len(df) < 100:
        print(f"WARNING: very small dataset ({len(df)} rows). "
              f"Consider widening the WHERE clause or waiting for more outcomes.")

    out_dir = PROJECT_ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"signals_export_{date.today().isoformat()}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(f"Class balance: {df['direction_correct'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the script (dry-run dependency check)**

```bash
.venv/bin/python -c "import psycopg2, pandas, dotenv; print('imports OK')"
```

Expected: `imports OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/export_from_foresight.py
git commit -m "feat(scripts): add export script for Foresight signals + outcomes + features"
```

---

### Task 8: Run the export against Foresight DB

**Files:** none (data file generated)

- [ ] **Step 1: Set up access**

Add `FORESIGHT_DB_DSN` to `.env.local` (NOT `.env`):

```bash
echo "FORESIGHT_DB_DSN=postgresql://postgres:postgres@localhost:5435/signal" >> .env.local
```

(Adjust host/port if your Foresight DB is elsewhere — `5435` is the default Docker port from `polymarket-ai/docker-compose.yml`.)

- [ ] **Step 2: Make sure Foresight DB is running**

In another terminal, in the polymarket-ai repo:
```bash
cd /Users/vadim/polymarket-ai
make dev   # or `docker-compose up db` if you only want the DB
```

- [ ] **Step 3: Run the export**

```bash
cd /Users/vadim/foresight-ml-poc/ml-foresight
.venv/bin/python scripts/export_from_foresight.py
```

Expected: prints `Fetched <N> rows`, `Wrote: data/raw/signals_export_<DATE>.csv`, and class balance dict.

- [ ] **Step 4: Quick visual check**

```bash
.venv/bin/python -c "
import pandas as pd
import glob
csv = sorted(glob.glob('data/raw/signals_export_*.csv'))[-1]
df = pd.read_csv(csv)
print('Rows:', len(df))
print('Columns:', list(df.columns))
print('Target balance:', df['direction_correct'].value_counts().to_dict())
print('Null counts:'); print(df.isnull().sum())
"
```

Expected: 200-2000 rows, ~21 columns, balanced-ish target, low null counts.

- [ ] **Step 5: Do NOT commit the raw CSV**

The `.gitignore` already excludes `data/raw/signals_export_*.csv` (only the sample is committed — see Task 9).

---

### Task 9: Anonymize and commit the sample CSV

**Files:**
- Create: `data/raw/signals_export_sample.csv`
- Create: `scripts/make_sample.py`

- [ ] **Step 1: Write `scripts/make_sample.py`**

```python
"""Generate an anonymized 50-row sample from the latest signals export.

Usage:
    python scripts/make_sample.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    csvs = sorted(glob.glob(str(PROJECT_ROOT / "data" / "raw" / "signals_export_*.csv")))
    csvs = [c for c in csvs if "sample" not in c]
    if not csvs:
        raise SystemExit("No signals_export_*.csv found. Run export_from_foresight.py first.")

    df = pd.read_csv(csvs[-1])
    print(f"Source: {csvs[-1]}, {len(df)} rows.")

    # Anonymize: drop signal_id, drop created_at exact timestamps (keep hour only)
    df["hour_of_day"] = pd.to_datetime(df["created_at"]).dt.hour
    df = df.drop(columns=["signal_id", "created_at"])

    # Stratified sample: 25 wins, 25 losses
    wins = df[df["direction_correct"] == 1].sample(n=min(25, (df["direction_correct"] == 1).sum()), random_state=42)
    losses = df[df["direction_correct"] == 0].sample(n=min(25, (df["direction_correct"] == 0).sum()), random_state=42)
    sample = pd.concat([wins, losses]).sample(frac=1, random_state=42).reset_index(drop=True)

    out = PROJECT_ROOT / "data" / "raw" / "signals_export_sample.csv"
    sample.to_csv(out, index=False)
    print(f"Wrote anonymized sample: {out}, {len(sample)} rows.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
.venv/bin/python scripts/make_sample.py
```

Expected: writes `data/raw/signals_export_sample.csv` with 50 rows.

- [ ] **Step 3: Verify and commit the sample**

```bash
head -3 data/raw/signals_export_sample.csv
wc -l data/raw/signals_export_sample.csv
```

Expected: 51 lines (header + 50 rows), no `signal_id` or `created_at` columns visible.

```bash
git add data/raw/signals_export_sample.csv scripts/make_sample.py
git commit -m "data: add anonymized 50-row sample CSV for reproducibility"
```

---

## Phase 3 — Data pipeline (`src/data.py`)

### Task 10: Write tests for `_clean()`

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Create test fixtures**

`tests/__init__.py`:
```python
```

`tests/conftest.py`:
```python
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
    """Mini raw DataFrame mimicking the export from Foresight."""
    return pd.DataFrame({
        "direction": ["BUY_YES", "BUY_NO", "BUY_YES", "BUY_NO"],
        "market_price_at_signal": [0.42, 0.78, 0.55, 0.18],
        "source_tier_mix": ['{"tier_1": 2, "tier_2": 1}', '{"tier_1": 1}',
                             '{"tier_2": 3, "tier_3": 1}', '{"tier_1": 4, "tier_2": 2}'],
        "freshness": [0.95, 0.7, 0.4, 0.85],
        "source_weight": [0.8, 0.5, 0.6, 0.9],
        "confirmation": [1.0, 0.5, 0.7, 1.0],
        "liquidity": [0.6, 0.3, 0.4, 0.7],
        "spread": [0.9, 0.5, 0.6, 0.85],
        "time_to_resolution": [0.5, 0.3, 0.6, 0.7],
        "impact_strength": [0.7, 0.4, 0.5, 0.8],
        "llm_confidence": [0.85, 0.6, 0.7, 0.9],
        "ambiguity_score": [0.2, 0.5, 0.4, 0.15],
        "specificity_score": [0.7, 0.5, 0.6, 0.8],
        "bucket": ["geopolitics", "politics", "geopolitics", "sports"],
        "articles_count": [4, 2, 3, 5],
        "unique_sources_count": [3, 2, 2, 4],
        "direction_correct": [1, 0, 1, 1],
        "move_t24h_pct": [12.5, -3.0, 8.2, 15.1],
        "heuristic_score": [78, 52, 65, 82],
        "hour_of_day": [14, 9, 21, 16],
    })
```

- [ ] **Step 2: Write `tests/test_data.py` for cleaning**

```python
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
    df.loc[0, "freshness"] = None
    df.loc[1, "liquidity"] = None
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
```

- [ ] **Step 3: Run tests — expect failure**

```bash
.venv/bin/pytest tests/test_data.py -v
```

Expected: 4 failures with `ImportError` or `AttributeError` (because `_clean` doesn't exist yet).

---

### Task 11: Implement `_clean()` in `src/data.py`

**Files:**
- Modify: `src/data.py`

- [ ] **Step 1: Replace `src/data.py` with the implementation**

```python
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
    "freshness", "source_weight", "confirmation",
    "liquidity", "spread", "time_to_resolution",
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
    df = df.copy()  # avoid SettingWithCopyWarning
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int64")

    n_classes = df[TARGET_COLUMN].nunique()
    if n_classes < 2:
        raise ValueError(
            f"Need at least 2 classes after cleaning, got {n_classes}. "
            f"Class distribution: {df[TARGET_COLUMN].value_counts().to_dict()}"
        )
    return df


def _feature_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix X and label vector y.

    Will be implemented in Task 13.
    """
    raise NotImplementedError("_feature_engineer is implemented in Task 13.")


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return (X_train, X_test, y_train, y_test).

    Reads the latest CSV from data/raw/, applies cleaning + feature engineering,
    splits 70/15/15 stratified, and scales using StandardScaler fit on train only.
    The fitted scaler and feature order are saved to models/ for the backend repo.
    """
    raise NotImplementedError("load_dataset_split is implemented in Task 14.")
```

- [ ] **Step 2: Run tests — expect 4 passing**

```bash
.venv/bin/pytest tests/test_data.py -v
```

Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
git add src/data.py tests/__init__.py tests/conftest.py tests/test_data.py
git commit -m "feat(data): implement _clean() with NULL drops and class-balance check"
```

---

### Task 12: Write tests for `_feature_engineer()`

**Files:**
- Modify: `tests/test_data.py`

- [ ] **Step 1: Append tests**

Add to `tests/test_data.py`:

```python
from data import _feature_engineer


def test_feature_engineer_returns_X_y(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, y = _feature_engineer(cleaned)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y) == len(cleaned)


def test_feature_engineer_excludes_leak_columns(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    forbidden = {"move_t24h_pct", "outcome_label", "price_t24h",
                 "signal_score", "signal_strength", "trade_quality",
                 "heuristic_score"}
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
    # drop_first=True → for 3 unique buckets in fixture, expect 2 columns
    assert len(bucket_cols) >= 1
    assert "bucket" not in X.columns  # original removed


def test_feature_engineer_no_nulls_in_output(raw_df: pd.DataFrame) -> None:
    cleaned = _clean(raw_df)
    X, _ = _feature_engineer(cleaned)
    assert not X.isnull().any().any(), "X contains NaN values"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
.venv/bin/pytest tests/test_data.py::test_feature_engineer_returns_X_y -v
```

Expected: FAIL with `NotImplementedError`.

---

### Task 13: Implement `_feature_engineer()`

**Files:**
- Modify: `src/data.py`

- [ ] **Step 1: Replace the `_feature_engineer` stub**

```python
def _feature_engineer(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the feature matrix X and label vector y.

    Adds derived features:
      - tier_1_count, tier_2_count, tier_3_count from source_tier_mix JSON
      - is_buy_yes (binary) from direction
      - market_price_centered = abs(market_price - 0.5)
      - hour_of_day from created_at if present
      - one-hot encoding of bucket (drop_first=True)

    Drops:
      - the target column itself
      - all columns in EXCLUDED_FROM_FEATURES (anti-leak)
      - the original `direction`, `bucket`, `source_tier_mix`, `created_at`
        columns after derivation
    """
    df = df.copy()

    # 1. Tier counts from JSONB
    def _parse_tier(blob: Any, tier: str) -> int:
        if pd.isna(blob):
            return 0
        if isinstance(blob, str):
            try:
                blob = json.loads(blob)
            except (json.JSONDecodeError, TypeError):
                return 0
        if isinstance(blob, dict):
            return int(blob.get(tier, 0))
        return 0

    df["tier_1_count"] = df["source_tier_mix"].apply(lambda b: _parse_tier(b, "tier_1"))
    df["tier_2_count"] = df["source_tier_mix"].apply(lambda b: _parse_tier(b, "tier_2"))
    df["tier_3_count"] = df["source_tier_mix"].apply(lambda b: _parse_tier(b, "tier_3"))

    # 2. is_buy_yes
    df["is_buy_yes"] = (df["direction"].astype(str).str.upper() == "BUY_YES").astype(int)

    # 3. market_price_centered
    df["market_price_centered"] = (df["market_price_at_signal"] - 0.5).abs()

    # 4. hour_of_day from created_at if present (sample CSV already has it pre-computed)
    if "created_at" in df.columns and "hour_of_day" not in df.columns:
        df["hour_of_day"] = pd.to_datetime(df["created_at"]).dt.hour

    # 5. one-hot bucket
    bucket_dummies = pd.get_dummies(df["bucket"], prefix="bucket", drop_first=True, dtype=int)
    df = pd.concat([df, bucket_dummies], axis=1)

    # 6. Extract y, drop columns
    y = df[TARGET_COLUMN]

    drop_cols = [TARGET_COLUMN, "direction", "bucket", "source_tier_mix",
                 "market_price_at_signal"]
    if "created_at" in df.columns:
        drop_cols.append("created_at")
    drop_cols += [c for c in EXCLUDED_FROM_FEATURES if c in df.columns]
    drop_cols += [c for c in df.columns if c == "heuristic_score"]
    drop_cols += [c for c in df.columns if c == "signal_id"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Sanity: no nulls remain (cleaning should have handled this, but verify)
    if X.isnull().any().any():
        bad = X.columns[X.isnull().any()].tolist()
        raise ValueError(f"Unexpected NaN in features after engineering: {bad}")

    return X, y
```

- [ ] **Step 2: Run tests — expect 7 new passing**

```bash
.venv/bin/pytest tests/test_data.py -v
```

Expected: 11 passed (4 from Task 11 + 7 new).

- [ ] **Step 3: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat(data): implement _feature_engineer() with derived + one-hot features"
```

---

### Task 14: Implement `load_dataset_split()`

**Files:**
- Modify: `src/data.py`

- [ ] **Step 1: Replace the stub**

```python
def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return (X_train, X_test, y_train, y_test) — scaled.

    Loading order (priority):
      1. The latest dated export in data/raw/signals_export_<DATE>.csv
      2. Fallback: data/raw/signals_export_sample.csv (committed sample)

    Pipeline:
      load → _clean → _feature_engineer → train_test_split (stratified)
      → fit StandardScaler on X_train → transform train + test
      → save scaler.pkl and feature_order.pkl in models/

    Returns numpy arrays (not DataFrames) per Basile's contract — but the
    feature column order is preserved in models/feature_order.pkl for the
    backend repo to use at inference time.
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    print(f"[data] Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # Fit scaler on X_train ONLY
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_test_arr = scaler.transform(X_test)

    # Persist scaler + feature order for the backend repo
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(list(X.columns), MODELS_DIR / "feature_order.pkl")
    print(f"[data] Saved scaler.pkl and feature_order.pkl in {MODELS_DIR}")

    return X_train_arr, X_test_arr, y_train.to_numpy(), y_test.to_numpy()
```

- [ ] **Step 2: Add a test for the full pipeline**

Append to `tests/test_data.py`:

```python
from data import load_dataset_split


def test_load_dataset_split_with_sample(monkeypatch, tmp_path):
    """End-to-end: load_dataset_split should run on the committed sample."""
    # Need a real sample CSV at data/raw/
    from config import DATA_DIR
    if not (DATA_DIR / "raw" / "signals_export_sample.csv").exists():
        pytest.skip("Sample CSV not present (Task 9 not yet run).")

    X_train, X_test, y_train, y_test = load_dataset_split()

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert X_train.shape[1] == X_test.shape[1]
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert set(np.unique(y_train)) <= {0, 1}
    assert set(np.unique(y_test)) <= {0, 1}
```

- [ ] **Step 3: Run all tests**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all pass (the new one will skip if sample not present, OK).

- [ ] **Step 4: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat(data): implement load_dataset_split() with stratified split + scaler persistence"
```

---

## Phase 4 — Metrics (`src/metrics.py`)

### Task 15: Implement `compute_metrics()` with tests

**Files:**
- Create: `tests/test_metrics.py`
- Modify: `src/metrics.py`

- [ ] **Step 1: Write the test file**

`tests/test_metrics.py`:
```python
"""Tests for src/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

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
```

- [ ] **Step 2: Run — expect failure**

```bash
.venv/bin/pytest tests/test_metrics.py -v
```

Expected: 4 fails with `NotImplementedError`.

- [ ] **Step 3: Replace `src/metrics.py`**

```python
"""Student-implemented metrics.

Implements the contract from basile-desjuzeur/ml-poc-project:
    compute_metrics(y_true, y_pred) -> dict[str, float]
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Binary classification metrics for direction_correct ∈ {0, 1}.

    y_pred is expected to be hard predictions (0/1) since scripts/main.py
    calls model.predict(). For probability-based ROC-AUC, use the
    notebook 04_results_analysis.ipynb where we have access to predict_proba.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_pred)),
    }
```

- [ ] **Step 4: Run — expect 4 passing**

```bash
.venv/bin/pytest tests/test_metrics.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): implement compute_metrics() with accuracy/precision/recall/f1/roc_auc"
```

---

## Phase 5 — Training script (`scripts/train.py`)

### Task 16: Skeleton + LogisticRegression

**Files:**
- Create: `scripts/train.py`

- [ ] **Step 1: Write the skeleton with LogReg**

```python
"""Train the 3 models (LogReg, RandomForest, MLP) on the Foresight signals dataset.

Saves:
  - models/logreg.pkl, models/random_forest.pkl, models/mlp.pkl
  - models/scaler.pkl, models/feature_order.pkl (already from data.py)
  - models/best_model.pkl (copy of the best by ROC-AUC on test)
  - models/model_card.json
  - plots/*.png
  - logs/training_<timestamp>.log

Usage:
    python scripts/train.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import LOGS_DIR, MODELS_DIR, MODEL_CARD_FILE, PLOTS_DIR, SEED  # noqa: E402
from data import load_dataset_split  # noqa: E402
from metrics import compute_metrics  # noqa: E402

# ---------- Logging setup ----------

LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("train")


# ---------- Model 1: Logistic Regression ----------

def train_logreg(X_train, y_train, X_test, y_test) -> tuple:
    """Tune C via 5-fold GridSearchCV, refit, evaluate, save."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    log.info("Training LogisticRegression with GridSearchCV (C grid)...")
    grid = {"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}
    search = GridSearchCV(
        LogisticRegression(
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            random_state=SEED,
            solver="lbfgs",
        ),
        param_grid=grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    log.info(f"LogReg best params: {search.best_params_}, best CV ROC-AUC: {search.best_score_:.4f}")

    model = search.best_estimator_
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    log.info(f"LogReg test metrics: {metrics}")

    out = MODELS_DIR / "logreg.pkl"
    joblib.dump(model, out)
    log.info(f"Saved {out}")

    return model, metrics


def main() -> None:
    log.info("=== Training pipeline started ===")
    log.info(f"Log file: {log_file}")

    # 1. Load data (this also persists scaler.pkl and feature_order.pkl)
    log.info("Loading dataset split...")
    X_train, X_test, y_train, y_test = load_dataset_split()
    log.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    all_metrics = {}

    # 2. Train each model
    _, all_metrics["logreg"] = train_logreg(X_train, y_train, X_test, y_test)

    # 3. (Tasks 17-18 will add RandomForest and MLP here)

    log.info("=== Training pipeline complete ===")
    log.info(f"All metrics: {json.dumps(all_metrics, indent=2)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run — expect successful training**

```bash
.venv/bin/python scripts/train.py
```

Expected: log output, `models/logreg.pkl` saved, no errors. Test metrics printed.

- [ ] **Step 3: Verify the artifact**

```bash
ls -la models/
.venv/bin/python -c "
import joblib
m = joblib.load('models/logreg.pkl')
print('Type:', type(m).__name__)
print('C:', m.C)
"
```

Expected: file exists, type is `LogisticRegression`, C is one of the grid values.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat(train): scaffold training script with LogisticRegression + GridSearchCV"
```

---

### Task 17: Add RandomForest

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Add `train_random_forest` function**

Insert before `def main()`:

```python
# ---------- Model 2: Random Forest ----------

def train_random_forest(X_train, y_train, X_test, y_test) -> tuple:
    """Tune via RandomizedSearchCV (50 iters, 5-fold), refit, evaluate, save."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    log.info("Training RandomForest with RandomizedSearchCV (50 iters)...")
    distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 8, 12, 16, 20, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        param_distributions=distributions,
        n_iter=50,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=SEED,
    )
    search.fit(X_train, y_train)
    log.info(f"RF best params: {search.best_params_}, best CV ROC-AUC: {search.best_score_:.4f}")

    model = search.best_estimator_
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    log.info(f"RF test metrics: {metrics}")

    out = MODELS_DIR / "random_forest.pkl"
    joblib.dump(model, out)
    log.info(f"Saved {out}")

    return model, metrics
```

- [ ] **Step 2: Call it in `main()`**

In `main()`, after the LogReg call:

```python
    rf_model, all_metrics["random_forest"] = train_random_forest(X_train, y_train, X_test, y_test)
```

- [ ] **Step 3: Run end-to-end**

```bash
.venv/bin/python scripts/train.py
```

Expected: ~2-5 min total (RandomForest tuning takes the longest). `models/random_forest.pkl` exists.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat(train): add RandomForest with RandomizedSearchCV (50 iters, 5-fold)"
```

---

### Task 18: Add MLP (Keras) with sklearn-compatible wrapper

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Add the wrapper class and training function**

Insert before `def main()`:

```python
# ---------- Model 3: MLP (Keras) ----------

class KerasMLPWrapper:
    """sklearn-compatible wrapper exposing .predict and .predict_proba.

    scripts/main.py (Basile) calls .predict(X) and expects an array of class labels.
    For storage, we save the underlying tf.keras.Model as .h5 alongside the wrapper
    (joblib pickling of tf.keras.Model is unreliable across versions).
    """

    def __init__(self, h5_path: Path, threshold: float = 0.5):
        self.h5_path = Path(h5_path)
        self.threshold = threshold
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            from tensorflow.keras.models import load_model
            self._model = load_model(self.h5_path)

    def predict_proba(self, X) -> np.ndarray:
        self._ensure_loaded()
        proba = self._model.predict(X, verbose=0).ravel()
        return np.column_stack([1 - proba, proba])

    def predict(self, X) -> np.ndarray:
        self._ensure_loaded()
        proba = self._model.predict(X, verbose=0).ravel()
        return (proba >= self.threshold).astype(int)


def train_mlp(X_train, y_train, X_test, y_test) -> tuple:
    """Train a 64→32→1 dense network with EarlyStopping. Saves .h5 + wrapper .pkl."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        BatchNormalization, Dense, Dropout, Input,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from sklearn.utils.class_weight import compute_class_weight

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    log.info("Training MLP (Keras): Dense(64)-BN-Drop(0.3)-Dense(32)-Drop(0.2)-Dense(1, sigmoid)")

    n_features = X_train.shape[1]
    model = Sequential([
        Input(shape=(n_features,)),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    log.info(f"MLP class_weight: {class_weight}")

    callbacks = [
        EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=100,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )
    log.info(f"MLP trained for {len(history.history['loss'])} epochs (early stopping triggered if <100).")

    # Save .h5 + wrapper
    h5_path = MODELS_DIR / "mlp.h5"
    model.save(h5_path)
    wrapper = KerasMLPWrapper(h5_path=h5_path, threshold=0.5)

    pkl_path = MODELS_DIR / "mlp.pkl"
    joblib.dump(wrapper, pkl_path)
    log.info(f"Saved {h5_path} and {pkl_path}")

    # Evaluate
    y_pred = wrapper.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    log.info(f"MLP test metrics: {metrics}")

    return wrapper, metrics, history
```

- [ ] **Step 2: Call it in `main()`**

After the RandomForest call:

```python
    mlp_model, all_metrics["mlp"], _ = train_mlp(X_train, y_train, X_test, y_test)
```

- [ ] **Step 3: Run**

```bash
.venv/bin/python scripts/train.py
```

Expected: MLP trains (~30-60 sec on CPU), `mlp.h5` + `mlp.pkl` saved.

- [ ] **Step 4: Verify the wrapper roundtrip**

```bash
.venv/bin/python -c "
import joblib, numpy as np
m = joblib.load('models/mlp.pkl')
fake = np.random.randn(3, 22).astype('float32')  # 22 features (adjust if needed)
print('predict:', m.predict(fake))
print('predict_proba:', m.predict_proba(fake))
"
```

Expected: prints binary predictions and 2-col proba arrays.

- [ ] **Step 5: Commit**

```bash
git add scripts/train.py
git commit -m "feat(train): add MLP (Keras) with EarlyStopping + sklearn-compatible wrapper"
```

---

### Task 19: Compute heuristic baseline + select best model + write model_card.json

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Add heuristic baseline function**

Before `def main()`:

```python
# ---------- Heuristic baseline (Foresight's current formula) ----------

def evaluate_heuristic_baseline(X_test, y_test, feature_order: list[str]) -> dict[str, float]:
    """Recompute the Foresight heuristic on the test set and return metrics.

    The heuristic uses 8 of our features (freshness, source_weight, confirmation,
    liquidity, spread, time_to_resolution, impact_strength, llm_confidence).
    Threshold for binary label: signal_score > 65 → predicted 1.

    Note: the test set here is post-scaling. To recompute the heuristic faithfully,
    we would need the unscaled values. For this baseline we approximate by using
    the fitted scaler.inverse_transform — see comment.
    """
    from config import HEURISTIC_THRESHOLD, MODELS_DIR

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    X_test_unscaled = scaler.inverse_transform(X_test)
    df = pd.DataFrame(X_test_unscaled, columns=feature_order)

    f = df["freshness"]
    sw = df["source_weight"]
    conf = df["confirmation"]
    liq = df["liquidity"]
    sp = df["spread"]
    ttr = df["time_to_resolution"]
    impact = df["impact_strength"]
    llm_conf = df["llm_confidence"]

    strength_base = 0.15 * f + 0.10 * sw + 0.15 * conf
    llm_combined = 0.65 * impact + 0.35 * llm_conf
    signal_strength = (strength_base + 0.60 * llm_combined).clip(0, 1) * 100
    trade_quality = (0.40 * liq + 0.35 * sp + 0.25 * ttr).clip(0, 1) * 100
    signal_score = (0.75 * signal_strength + 0.25 * trade_quality).clip(0, 100)

    y_pred = (signal_score > HEURISTIC_THRESHOLD).astype(int).to_numpy()
    return compute_metrics(y_test, y_pred)
```

(Add `import pandas as pd` at the top if not already imported.)

- [ ] **Step 2: Add best-model selection and model_card.json writer**

Before `def main()`:

```python
def select_best_and_write_card(all_metrics: dict, feature_order: list[str], heuristic: dict) -> None:
    """Pick the best model by test ROC-AUC, copy it to best_model.pkl, write model_card.json."""
    best_key = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"])
    log.info(f"Best model on test ROC-AUC: {best_key} ({all_metrics[best_key]['roc_auc']:.4f})")

    src = MODELS_DIR / f"{best_key}.pkl"
    dst = MODELS_DIR / "best_model.pkl"
    import shutil
    shutil.copy(src, dst)
    log.info(f"Copied {src.name} → {dst.name}")

    card = {
        "model_version": "v1.0.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model_type": best_key,
        "best_model_path": f"models/{best_key}.pkl",
        "feature_order": feature_order,
        "scaler_path": "models/scaler.pkl",
        "test_metrics": all_metrics,
        "heuristic_baseline": heuristic,
        "training_seed": SEED,
    }
    MODEL_CARD_FILE.write_text(json.dumps(card, indent=2))
    log.info(f"Wrote {MODEL_CARD_FILE}")
```

- [ ] **Step 3: Wire it in `main()`**

Replace the end of `main()`:

```python
    # Heuristic baseline
    feature_order = joblib.load(MODELS_DIR / "feature_order.pkl")
    heuristic_metrics = evaluate_heuristic_baseline(X_test, y_test, feature_order)
    log.info(f"Heuristic baseline metrics: {heuristic_metrics}")

    # Best model + model card
    select_best_and_write_card(all_metrics, feature_order, heuristic_metrics)

    log.info("=== Training pipeline complete ===")
    log.info(f"All metrics: {json.dumps(all_metrics, indent=2)}")
    log.info(f"Heuristic baseline: {json.dumps(heuristic_metrics, indent=2)}")
```

- [ ] **Step 4: Run**

```bash
.venv/bin/python scripts/train.py
```

Expected: heuristic metrics logged, `best_model.pkl` exists, `model_card.json` exists with full metadata.

- [ ] **Step 5: Inspect the model card**

```bash
cat models/model_card.json | python -m json.tool | head -40
```

Expected: well-formed JSON with feature_order, test_metrics for each model, heuristic_baseline.

- [ ] **Step 6: Commit**

```bash
git add scripts/train.py
git commit -m "feat(train): add heuristic baseline + best-model selection + model_card.json"
```

---

### Task 20: Generate evaluation plots

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Add a plotting function**

Before `def main()`:

```python
# ---------- Plot generation ----------

def generate_plots(
    models: dict,
    X_test, y_test,
    all_metrics: dict,
    heuristic_metrics: dict,
    feature_order: list[str],
) -> None:
    """Generate confusion matrices, ROC curves, feature importance, ml-vs-heuristic bar."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve

    PLOTS_DIR.mkdir(exist_ok=True)

    # 1. Confusion matrix per model
    for key, model in models.items():
        cm = confusion_matrix(y_test, model.predict(X_test))
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {key}")
        fig.tight_layout()
        out = PLOTS_DIR / f"confusion_matrix_{key}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        log.info(f"Wrote {out}")

    # 2. ROC curves comparison
    fig, ax = plt.subplots(figsize=(6, 6))
    for key, model in models.items():
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        else:
            proba = model.predict(X_test).astype(float)
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{key} (AUC={all_metrics[key]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "roc_curves_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"Wrote {out}")

    # 3. Feature importance (RandomForest only)
    if "random_forest" in models:
        rf = models["random_forest"]
        importances = rf.feature_importances_
        idx = np.argsort(importances)[::-1][:15]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([feature_order[i] for i in idx][::-1], importances[idx][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Top 15 Feature Importances — Random Forest")
        fig.tight_layout()
        out = PLOTS_DIR / "feature_importance_rf.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        log.info(f"Wrote {out}")

    # 4. ML vs Heuristic punchline
    metric_keys = ["accuracy", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(metric_keys))
    width = 0.18
    ax.bar(x - 1.5 * width, [heuristic_metrics[k] for k in metric_keys], width, label="Heuristic")
    ax.bar(x - 0.5 * width, [all_metrics["logreg"][k] for k in metric_keys], width, label="LogReg")
    ax.bar(x + 0.5 * width, [all_metrics["random_forest"][k] for k in metric_keys], width, label="RF")
    ax.bar(x + 1.5 * width, [all_metrics["mlp"][k] for k in metric_keys], width, label="MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel("Score")
    ax.set_title("Heuristic vs ML Models — Test Set")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "ml_vs_heuristic.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"Wrote {out}")
```

- [ ] **Step 2: Call from `main()`**

After heuristic baseline:

```python
    models = {
        "logreg": joblib.load(MODELS_DIR / "logreg.pkl"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
        "mlp": joblib.load(MODELS_DIR / "mlp.pkl"),
    }
    generate_plots(models, X_test, y_test, all_metrics, heuristic_metrics, feature_order)
```

- [ ] **Step 3: Run**

```bash
.venv/bin/python scripts/train.py
ls plots/
```

Expected: 5 PNGs in `plots/`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "feat(train): generate confusion matrices, ROC curves, feature importance, vs-heuristic"
```

---

## Phase 6 — Streamlit dashboard (`src/app.py`)

### Task 21: Implement build_app() with all 5 tabs

**Files:**
- Modify: `src/app.py`

- [ ] **Step 1: Replace `src/app.py` entirely**

```python
"""Streamlit dashboard for the ML Foresight POC.

Fixed entry point — scripts/main.py launches this via `streamlit run`.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DATA_DIR, MODEL_CARD_FILE, MODEL_METRICS_FILE, MODELS_DIR, PLOTS_DIR,
)


def build_app() -> None:
    st.set_page_config(page_title="ML Foresight — POC", layout="wide")
    st.title("ML Foresight — Prédiction de la direction des signaux Polymarket")
    st.caption("Projet école Albert School · 2026-05")

    tabs = st.tabs([
        "📋 Vue d'ensemble",
        "🔍 Exploration des données",
        "📊 Comparaison des modèles",
        "⚔️ Vs heuristique",
        "🎯 Prédiction live",
    ])

    with tabs[0]:
        _tab_overview()
    with tabs[1]:
        _tab_eda()
    with tabs[2]:
        _tab_models()
    with tabs[3]:
        _tab_vs_heuristic()
    with tabs[4]:
        _tab_predict()


def _load_metrics_csv() -> pd.DataFrame | None:
    if MODEL_METRICS_FILE.exists():
        return pd.read_csv(MODEL_METRICS_FILE)
    return None


def _load_card() -> dict | None:
    if MODEL_CARD_FILE.exists():
        return json.loads(MODEL_CARD_FILE.read_text())
    return None


def _load_data() -> pd.DataFrame | None:
    raw = DATA_DIR / "raw"
    dated = sorted(raw.glob("signals_export_20*.csv"))
    candidates = [p for p in dated if "sample" not in p.name]
    if candidates:
        return pd.read_csv(candidates[-1])
    sample = raw / "signals_export_sample.csv"
    if sample.exists():
        return pd.read_csv(sample)
    return None


# ----- Tab 1: Overview -----

def _tab_overview() -> None:
    st.header("Contexte du projet")
    st.markdown("""
    **Foresight** émet des signaux de trading sur Polymarket en notant chaque opportunité
    0-100 via une formule heuristique. L'audit interne montre un **winrate de 46-48 % à T+24h**
    — sous le hasard.

    **Ce projet remplace la formule par du machine learning supervisé.**
    - **Tâche :** classification binaire — prédire `direction_correct` à T+24h
    - **Modèles :** Logistic Regression · Random Forest · MLP (Keras)
    - **Métrique de sélection :** ROC-AUC
    - **Baseline à battre :** la formule heuristique actuelle
    """)

    df = _load_data()
    card = _load_card()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Signaux dans le dataset", len(df) if df is not None else "—")
    with col2:
        if df is not None:
            balance = df["direction_correct"].value_counts().to_dict()
            st.metric("Distribution des classes",
                      f"{balance.get(1, 0)} wins / {balance.get(0, 0)} losses")
        else:
            st.metric("Distribution des classes", "—")
    with col3:
        st.metric("Best model", card["best_model_type"] if card else "Pas encore entraîné")

    if card:
        st.subheader("Model card")
        st.json(card)


# ----- Tab 2: EDA -----

def _tab_eda() -> None:
    st.header("Exploration des données")
    df = _load_data()
    if df is None:
        st.info("Aucune donnée. Lance `python scripts/export_from_foresight.py` d'abord.")
        return

    st.subheader("Statistiques descriptives")
    numeric = df.select_dtypes(include="number")
    st.dataframe(numeric.describe(), use_container_width=True)

    st.subheader("Distribution d'une feature")
    feature = st.selectbox("Feature", numeric.columns.tolist(),
                           index=numeric.columns.get_loc("freshness")
                           if "freshness" in numeric.columns else 0)
    st.bar_chart(numeric[feature].value_counts(bins=20).sort_index())

    st.subheader("Matrice de corrélation (numériques)")
    corr = numeric.corr()
    st.dataframe(corr.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1),
                 use_container_width=True)


# ----- Tab 3: Model comparison -----

def _tab_models() -> None:
    st.header("Comparaison des 3 modèles")

    metrics_df = _load_metrics_csv()
    if metrics_df is None:
        st.info("Aucune évaluation. Lance `python scripts/main.py` après `train.py`.")
        return

    st.subheader("Métriques")
    st.dataframe(metrics_df, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        roc = PLOTS_DIR / "roc_curves_comparison.png"
        if roc.exists():
            st.subheader("Courbes ROC")
            st.image(str(roc), use_column_width=True)

    with col2:
        fi = PLOTS_DIR / "feature_importance_rf.png"
        if fi.exists():
            st.subheader("Feature importance — Random Forest")
            st.image(str(fi), use_column_width=True)

    st.subheader("Matrices de confusion")
    cols = st.columns(3)
    for i, key in enumerate(["logreg", "random_forest", "mlp"]):
        cm_path = PLOTS_DIR / f"confusion_matrix_{key}.png"
        if cm_path.exists():
            with cols[i]:
                st.image(str(cm_path), caption=key, use_column_width=True)


# ----- Tab 4: vs Heuristic -----

def _tab_vs_heuristic() -> None:
    st.header("ML vs Heuristique — la punchline du projet")

    card = _load_card()
    if not card:
        st.info("Pas de model_card. Lance `python scripts/train.py` d'abord.")
        return

    plot = PLOTS_DIR / "ml_vs_heuristic.png"
    if plot.exists():
        st.image(str(plot), use_column_width=True)

    st.subheader("Tableau comparatif (test set)")
    rows = [
        {"Modèle": "Heuristique (formule Foresight)", **card["heuristic_baseline"]},
    ]
    for k, v in card["test_metrics"].items():
        rows.append({"Modèle": k, **v})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    best = card["best_model_type"]
    h_auc = card["heuristic_baseline"]["roc_auc"]
    ml_auc = card["test_metrics"][best]["roc_auc"]
    delta = (ml_auc - h_auc) * 100

    if delta > 0:
        st.success(f"✅ Le best model ({best}) bat l'heuristique de **+{delta:.1f} pts ROC-AUC**.")
    else:
        st.warning(f"⚠️ Le ML n'a pas battu l'heuristique ({delta:.1f} pts ROC-AUC). "
                   f"Voir la discussion dans `docs/rapport.md`.")


# ----- Tab 5: Live prediction -----

def _tab_predict() -> None:
    st.header("Prédiction live")
    card = _load_card()
    if not card:
        st.info("Pas de model_card. Lance `python scripts/train.py` d'abord.")
        return

    model_path = MODELS_DIR / "best_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    if not (model_path.exists() and scaler_path.exists()):
        st.info("Modèle ou scaler manquant.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_order: list[str] = card["feature_order"]

    st.markdown("Ajuste les sliders puis clique **Prédire**.")

    inputs: dict[str, float] = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_order):
        with cols[i % 3]:
            if feat in {"is_buy_yes"}:
                inputs[feat] = st.selectbox(feat, options=[0, 1], index=1)
            elif feat == "hour_of_day":
                inputs[feat] = st.slider(feat, 0, 23, 14)
            elif feat in {"articles_count", "unique_sources_count",
                          "tier_1_count", "tier_2_count", "tier_3_count"}:
                inputs[feat] = st.slider(feat, 0, 10, 2)
            elif feat.startswith("bucket_"):
                inputs[feat] = st.selectbox(feat, options=[0, 1], index=0)
            else:
                inputs[feat] = st.slider(feat, 0.0, 1.0, 0.5, step=0.01)

    if st.button("Prédire"):
        x = np.array([[inputs[f] for f in feature_order]], dtype="float32")
        x_scaled = scaler.transform(x)
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(x_scaled)[0, 1])
        else:
            proba = float(model.predict(x_scaled)[0])
        label = int(proba >= 0.5)
        st.metric("Probabilité de gain (label = 1)", f"{proba:.3f}")
        st.metric("Label prédit", "🟢 GAIN" if label == 1 else "🔴 LOSS")


if __name__ == "__main__":
    build_app()
```

- [ ] **Step 2: Test the app launches**

```bash
.venv/bin/streamlit run src/app.py --server.port 8501 --server.headless true &
sleep 5
curl -s http://localhost:8501 | head -20
kill %1
```

Expected: HTML page returned (Streamlit serving). Kill the process after.

Or interactively:

```bash
.venv/bin/streamlit run src/app.py
```

Open http://localhost:8501 in browser, click through 5 tabs.

- [ ] **Step 3: Commit**

```bash
git add src/app.py
git commit -m "feat(app): implement Streamlit dashboard with 5 tabs (overview, EDA, models, vs heuristic, predict)"
```

---

## Phase 7 — End-to-end integration + GitHub Release

### Task 22: Run the full pipeline end-to-end

**Files:** none (verification)

- [ ] **Step 1: Clean slate**

```bash
rm -rf models/*.pkl models/*.h5 models/*.json plots/*.png results/*.csv
```

- [ ] **Step 2: Train**

```bash
.venv/bin/python scripts/train.py
```

Expected: ~5-10 minutes total. All 3 .pkl + best_model.pkl + scaler.pkl + feature_order.pkl + model_card.json + 5 PNGs.

- [ ] **Step 3: Evaluate via Basile's main.py + launch Streamlit**

```bash
.venv/bin/python scripts/main.py
```

Expected: prints metrics table to terminal, writes `results/model_metrics.csv`, then Streamlit launches on http://localhost:8501. Click through all 5 tabs.

- [ ] **Step 4: Sanity check — files exist**

```bash
ls -la models/ plots/ results/
cat results/model_metrics.csv
```

Expected: full set of artifacts.

- [ ] **Step 5: Snapshot artifacts (commit best_model.pkl + model_card.json + plots)**

The `.gitignore` excludes most artifacts. For the GitHub Release, we'll include:
- `best_model.pkl`
- `scaler.pkl`
- `feature_order.pkl`
- `model_card.json`
- `mlp.h5` (if MLP is best)

```bash
# Don't commit them — they go in the GitHub Release (Task 23)
```

---

### Task 23: Tag v1.0.0 and create GitHub Release with artifacts

**Files:** none (Git/GitHub operations)

- [ ] **Step 1: Tag main**

```bash
cd /Users/vadim/foresight-ml-poc/ml-foresight
git tag -a v1.0.0 -m "First trained model release.

Models trained on $(date +%Y-%m-%d).
Best model: $(python -c 'import json; print(json.load(open("models/model_card.json"))["best_model_type"])').
Test ROC-AUC: $(python -c 'import json; c=json.load(open("models/model_card.json")); print(c["test_metrics"][c["best_model_type"]]["roc_auc"])')."

git push origin v1.0.0
```

- [ ] **Step 2: Compute SHA256 of artifacts**

```bash
shasum -a 256 models/best_model.pkl models/scaler.pkl models/feature_order.pkl models/model_card.json
```

Save the hashes — they go in the release notes for the backend repo to verify.

- [ ] **Step 3: Create the GitHub Release**

```bash
gh release create v1.0.0 \
    models/best_model.pkl \
    models/scaler.pkl \
    models/feature_order.pkl \
    models/model_card.json \
    --title "v1.0.0 — first trained model" \
    --notes "$(cat <<EOF
First trained model artifacts for the Foresight ML POC.

## Artifacts

- \`best_model.pkl\` — best of 3 models, selected by test ROC-AUC
- \`scaler.pkl\` — StandardScaler fit on train, must be applied at inference
- \`feature_order.pkl\` — list of feature names in the order the model expects
- \`model_card.json\` — full metadata: features, dtypes, metrics, heuristic baseline

## SHA256

\`\`\`
$(shasum -a 256 models/best_model.pkl models/scaler.pkl models/feature_order.pkl models/model_card.json)
\`\`\`

## Usage from backend-foresight

\`\`\`python
import urllib.request, joblib, hashlib

URL = "https://github.com/foresight-ml-poc/ml-foresight/releases/download/v1.0.0/best_model.pkl"
urllib.request.urlretrieve(URL, "/tmp/best_model.pkl")
model = joblib.load("/tmp/best_model.pkl")
\`\`\`
EOF
)"
```

- [ ] **Step 4: Verify the release exists**

```bash
gh release view v1.0.0
```

Expected: lists all 4 attached files with sizes.

- [ ] **Step 5: Update README with badge**

Replace the status badge in `ml-foresight/README.md`:

```markdown
![Status](https://img.shields.io/badge/status-v1.0.0--released-brightgreen)
```

```bash
git add README.md
git commit -m "docs(readme): bump status badge to v1.0.0 released"
git push origin main
```

---

### Task 24: Write the academic report

**Files:**
- Create: `docs/rapport.md`

- [ ] **Step 1: Write the report skeleton with all 8 sections from the spec**

```bash
cat > docs/rapport.md <<'EOF'
# Rapport — ML Foresight POC

> Projet école Albert School · Vadim Capton · 2026-05

## 1. Contexte

[Foresight](https://github.com/vcapton-jpg/polymarket-ai) émet des signaux de
trading sur Polymarket. Le scoring actuel utilise une formule heuristique fixe
qui obtient un winrate de 46-48 % à T+24h — sous le hasard. Ce projet remplace
la formule par du ML supervisé.

## 2. Données

- **Source** : DB Postgres de Foresight (tables `signals`, `signal_outcomes`,
  `event_market_features`, `event_market_analysis`, `events`)
- **Volume** : N signaux historiques avec `direction_correct` rempli
  *(remplir après l'export)*
- **Période** : *(dates first/last du dataset)*
- **Distribution des classes** : *(remplir)*
- **Anti-leak** : aucune feature dérivée du futur (`move_t24h_pct`, etc.) ni
  du score heuristique actuel (`signal_score`)

## 3. Feature engineering

Voir `src/data.py::_feature_engineer()`. ~20 features finales :
- 12 brutes (6 heuristiques + 4 LLM + 2 contexte event)
- 5 dérivées (3 tier counts, is_buy_yes, market_price_centered, hour_of_day)
- one-hot bucket (3-5 colonnes)

## 4. Modèles

Trois modèles entraînés et comparés :

### LogisticRegression (baseline)
- L2, class_weight balanced
- C tuné par GridSearchCV 5-fold sur `[0.01, 0.03, 0.1, 0.3, 1, 3, 10]`

### RandomForestClassifier
- class_weight balanced
- RandomizedSearchCV 5-fold, 50 itérations sur n_estimators, max_depth,
  min_samples_split, min_samples_leaf, max_features

### MLP (Keras)
- Dense(64, relu) → BN → Drop(0.3) → Dense(32, relu) → Drop(0.2) → Dense(1, sigmoid)
- EarlyStopping patience=10 sur val_auc, ReduceLROnPlateau

## 5. Résultats

*(remplir avec le contenu de `models/model_card.json` et les plots)*

| Modèle | Accuracy | F1 | ROC-AUC | Precision | Recall |
|---|---|---|---|---|---|
| Heuristique | | | | | |
| LogReg | | | | | |
| Random Forest | | | | | |
| MLP | | | | | |

Voir `plots/roc_curves_comparison.png`, `plots/ml_vs_heuristic.png`.

## 6. Comparaison vs heuristique

*(remplir avec les chiffres réels du model_card)*

## 7. Discussion honnête

- **Limitations du dataset** : taille (N rows), période (d à d), stationnarité
- **Asymétrie BUY_YES vs BUY_NO** observée dans l'audit Foresight
- **Ce qu'on ferait avec plus de temps** : XGBoost, calibration de probas,
  feature importance via SHAP, validation temporelle (TimeSeriesSplit)

## 8. Conclusion et next steps

- Le best model est *(remplir)* avec ROC-AUC = *(remplir)*
- Prochaine étape : ré-intégration dans Foresight via `backend-foresight` /
  `frontend-foresight`
- Si le ML bat l'heuristique : remplacement progressif (A/B test) du scoring
  en production
EOF
```

- [ ] **Step 2: Fill the placeholders with real numbers from `models/model_card.json`**

Open `docs/rapport.md` in your editor and replace the *(remplir)* / *(d à d)*
sections with concrete values from `models/model_card.json` and the plots.

- [ ] **Step 3: Commit**

```bash
git add docs/rapport.md
git commit -m "docs: write academic report (8 sections per spec)"
git push origin main
```

---

## Self-review

After writing the plan, I checked it against the spec:

**Spec coverage:**
- §2 (problem ML) → covered by Task 5 (config), Task 11/13 (data pipeline)
- §4 (architecture, model_card) → Task 19 generates model_card.json
- §5 (repo structure Basile) → Tasks 1-6
- §6 (4 contracts) → Tasks 11/13/14 (data), Task 15 (metrics), Task 5 (config), Task 21 (app)
- §7 (training script) → Tasks 16-20
- §8 (data extraction + cleaning + FE) → Tasks 7-9 (export), Tasks 11/13/14 (pipeline)
- §9-§10 (backend / frontend) → DEFERRED to follow-up plans (out of scope)
- §11 (workflow) → Task 22 runs E2E, Task 23 publishes Release, Task 24 writes report

**Placeholder scan:** No "TBD", "TODO", "implement later" left. All code blocks
contain real implementations. Test stubs that delegate to later tasks reference
the task number explicitly.

**Type consistency:** `compute_metrics` returns `dict[str, float]` everywhere;
`load_dataset_split` returns `tuple[Any, Any, Any, Any]` (numpy arrays); model
artifact paths use the same names across `config.py::MODELS`, `train.py`,
`app.py`. The MLP wrapper exposes both `predict()` and `predict_proba()` and
is consistent across train.py and app.py.

**Out of scope (deferred to follow-up plans):**
- backend-foresight implementation (separate plan after v1.0.0 release)
- frontend-foresight implementation (separate plan after backend works)
- Notebooks (mentioned in spec §5.5 as optional; the dashboard tabs cover the
  same ground in a more presentation-ready format)
