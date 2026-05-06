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
