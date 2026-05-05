"""Train the 3 models (LogReg, RandomForest, GradientBoosting) on the Foresight
signals dataset.

Saves to models/:
  - logreg.pkl
  - random_forest.pkl
  - gradient_boosting.pkl
  - scaler.pkl, feature_order.pkl (already saved by load_dataset_split())

Originally the third model was a Keras MLP, but on the Apple Silicon /
TensorFlow 2.21 conda-env used to ship this project, model.fit() hangs
indefinitely on small N=32 datasets (root cause not pinpointed; bypass over
diagnose given school-project timeline). GradientBoostingClassifier is a
strong sklearn-native alternative — different model family from RF (boosting
vs bagging), trains fast, well-suited to small tabular data.

Heuristic baseline, best-model selection, model_card.json, and plots are
added in the next task (19+20).

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

from config import LOGS_DIR, MODELS_DIR, SEED  # noqa: E402
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

    log.info("Training LogisticRegression with GridSearchCV (C grid, 5-fold)...")
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


# ---------- Model 2: Random Forest ----------

def train_random_forest(X_train, y_train, X_test, y_test) -> tuple:
    """Tune via RandomizedSearchCV (20 iters, 5-fold), refit, evaluate, save.

    n_iter reduced from 50 to 20 due to small dataset size (N=40).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    log.info("Training RandomForest with RandomizedSearchCV (20 iters, 5-fold)...")
    distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 8, 12, None],  # smaller depths for small data
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5],
        "max_features": ["sqrt", "log2", 0.5],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
        param_distributions=distributions,
        n_iter=20,
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


# ---------- Model 3: Gradient Boosting ----------

def train_gradient_boosting(X_train, y_train, X_test, y_test) -> tuple:
    """Tune via RandomizedSearchCV (20 iters, 5-fold), refit, evaluate, save.

    Gradient boosting is a strong sequential ensemble (different family from RF):
    trees are added one at a time, each correcting residuals of previous ones.
    Well-suited to small tabular data — competitive with neural nets here without
    the TF environment fragility.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import RandomizedSearchCV

    log.info("Training GradientBoosting with RandomizedSearchCV (20 iters, 5-fold)...")
    distributions = {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.03, 0.1, 0.3],
        "max_depth": [2, 3, 5, 8],
        "min_samples_split": [2, 5, 10],
        "subsample": [0.7, 0.85, 1.0],
    }
    search = RandomizedSearchCV(
        GradientBoostingClassifier(random_state=SEED),
        param_distributions=distributions,
        n_iter=20,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=SEED,
    )
    search.fit(X_train, y_train)
    log.info(f"GBM best params: {search.best_params_}, best CV ROC-AUC: {search.best_score_:.4f}")

    model = search.best_estimator_
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    log.info(f"GBM test metrics: {metrics}")

    out = MODELS_DIR / "gradient_boosting.pkl"
    joblib.dump(model, out)
    log.info(f"Saved {out}")

    return model, metrics


def main() -> None:
    log.info("=== Training pipeline started ===")
    log.info(f"Log file: {log_file}")

    log.info("Loading dataset split...")
    X_train, X_test, y_train, y_test = load_dataset_split()
    log.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    all_metrics = {}

    _, all_metrics["logreg"] = train_logreg(X_train, y_train, X_test, y_test)
    _, all_metrics["random_forest"] = train_random_forest(X_train, y_train, X_test, y_test)
    _, all_metrics["gradient_boosting"] = train_gradient_boosting(X_train, y_train, X_test, y_test)

    log.info("=== Training complete (Phase 5a — 3 models trained) ===")
    log.info(f"All test metrics: {json.dumps(all_metrics, indent=2)}")


if __name__ == "__main__":
    main()
