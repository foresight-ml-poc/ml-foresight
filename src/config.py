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

# --- Anti-leak: EXPLICIT FEATURE ALLOWLIST ---
#
# We use an allowlist (not a denylist) so that enriching the export CSV with
# analysis columns (price trajectory, market microstructure, heuristic
# internals, ...) can NEVER leak into X. Only columns listed here — plus the
# one-hot `bucket_*` columns generated at runtime — become model features.
#
# Anything not in this list (labels, future-derived outcomes, heuristic
# outputs, free-text, ids, timestamps) is simply never selected.
FEATURE_BASE_COLUMNS = [
    # Raw numeric, available at signal-emission time
    "cosine_score",
    "impact_strength",
    "llm_confidence",
    "ambiguity_score",
    "specificity_score",
    "articles_count",
    "unique_sources_count",
    # Derived in _feature_engineer()
    "tier_1_count",
    "tier_2_count",
    "tier_3_count",
    "is_buy_yes",
    "market_price_centered",
    "hour_of_day",
    # + one-hot `bucket_*` columns added dynamically
]

# Kept for backwards reference / documentation of what would leak.
EXCLUDED_FROM_FEATURES = [
    "move_t24h_pct", "move_t5min_pct", "move_t15min_pct", "move_t1h_pct",
    "price_t5min", "price_t15min", "price_t1h", "price_t24h", "price_resolved",
    "outcome_label", "signal_score", "heuristic_score", "heuristic_strength",
    "heuristic_trade_quality", "signal_strength", "trade_quality",
]

# Registry of trained models — populated by scripts/train.py
MODELS = {
    "logreg": {
        "name": "Logistic Regression",
        "description": "Baseline linéaire L2, class_weight balanced, C tuné par GridSearchCV.",
        "path": MODELS_DIR / "logreg.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "description": "Ensemble d'arbres, RandomizedSearchCV 5-fold, class_weight balanced.",
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "description": "Boosting d'arbres séquentiel sklearn, RandomizedSearchCV 5-fold.",
        "path": MODELS_DIR / "gradient_boosting.joblib",
    },
    "lightgbm": {
        "name": "LightGBM",
        "description": "Boosting léger et rapide, leaf-wise tree growth, RandomizedSearchCV 5-fold.",
        "path": MODELS_DIR / "lightgbm.joblib",
    },
    "xgboost": {
        "name": "XGBoost",
        "description": "Boosting classique level-wise avec régularisation L1+L2, RandomizedSearchCV 5-fold.",
        "path": MODELS_DIR / "xgboost.joblib",
    },
    "svm": {
        "name": "SVM (RBF)",
        "description": "Support Vector Machine kernel RBF, GridSearchCV 5-fold sur C et gamma.",
        "path": MODELS_DIR / "svm.joblib",
    },
}
