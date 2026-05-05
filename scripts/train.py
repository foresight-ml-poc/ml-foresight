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

from config import (  # noqa: E402
    HEURISTIC_THRESHOLD,
    LOGS_DIR,
    MODEL_CARD_FILE,
    MODELS_DIR,
    PLOTS_DIR,
    SEED,
)
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

    out = MODELS_DIR / "logreg.joblib"
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

    out = MODELS_DIR / "random_forest.joblib"
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

    out = MODELS_DIR / "gradient_boosting.joblib"
    joblib.dump(model, out)
    log.info(f"Saved {out}")

    return model, metrics


# ---------- Heuristic baseline (Foresight's current formula) ----------

def evaluate_heuristic_baseline(X_test, y_test, feature_order: list) -> dict:
    """Recompute the Foresight heuristic on the test set and return metrics.

    The heuristic uses 8 features. Test data here is post-scaling, so we
    inverse_transform via the saved scaler to get original [0,1]-bounded values.
    Threshold for binary label: heuristic_score > 65 → predicted class 1.
    """
    import pandas as pd

    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    X_test_unscaled = scaler.inverse_transform(X_test)
    df = pd.DataFrame(X_test_unscaled, columns=feature_order)

    f = df["freshness_factor"]
    sw = df["source_weight"]
    conf = df["confirmation_factor"]
    liq = df["liquidity_factor"]
    sp = df["spread_penalty"]
    ttr = df["time_to_resolution_factor"]
    impact = df["impact_strength"]
    llm_conf = df["llm_confidence"]

    strength_base = 0.15 * f + 0.10 * sw + 0.15 * conf
    llm_combined = 0.65 * impact + 0.35 * llm_conf
    signal_strength = (strength_base + 0.60 * llm_combined).clip(0, 1) * 100
    trade_quality = (0.40 * liq + 0.35 * sp + 0.25 * ttr).clip(0, 1) * 100
    signal_score = (0.75 * signal_strength + 0.25 * trade_quality).clip(0, 100)

    y_pred = (signal_score > HEURISTIC_THRESHOLD).astype(int).to_numpy()
    return compute_metrics(y_test, y_pred)


# ---------- Best model selection + model card ----------

def select_best_and_write_card(all_metrics: dict, feature_order: list,
                                heuristic: dict) -> str:
    """Pick best model by test ROC-AUC, copy to best_model.pkl, write model_card.json."""
    import shutil

    best_key = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"])
    log.info(f"Best model on test ROC-AUC: {best_key} "
             f"({all_metrics[best_key]['roc_auc']:.4f})")

    src = MODELS_DIR / f"{best_key}.joblib"
    dst = MODELS_DIR / "best_model.joblib"
    shutil.copy(src, dst)
    log.info(f"Copied {src.name} → {dst.name}")

    card = {
        "model_version": "v1.0.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model_type": best_key,
        "best_model_path": f"models/{best_key}.joblib",
        "feature_order": feature_order,
        "scaler_path": "models/scaler.pkl",
        "test_metrics": all_metrics,
        "heuristic_baseline": heuristic,
        "training_seed": SEED,
        "dataset_size": {"train": 32, "test": 8, "total": 40},
        "notes": (
            "MLP was substituted with GradientBoosting because tf.keras.fit() hung "
            "on this conda env / Apple Silicon for N=32. Test metrics are noisy "
            "due to the small test set (8 samples) — re-train when more outcomes "
            "accumulate in Foresight."
        ),
    }
    MODEL_CARD_FILE.write_text(json.dumps(card, indent=2))
    log.info(f"Wrote {MODEL_CARD_FILE}")
    return best_key


# ---------- Plot generation ----------

def generate_plots(models: dict, X_test, y_test, all_metrics: dict,
                    heuristic_metrics: dict, feature_order: list) -> None:
    """Generate confusion matrices, ROC curves, feature importance, ml-vs-heuristic bar."""
    import matplotlib
    matplotlib.use("Agg")  # no GUI needed
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve

    PLOTS_DIR.mkdir(exist_ok=True)

    # 1. Confusion matrix per model
    for key, model in models.items():
        cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1])
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

    # 2. ROC curves
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
    ax.set_title("ROC Curves — All Models (test set N=8)")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "roc_curves_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"Wrote {out}")

    # 3. Feature importance (RF)
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
    ax.bar(x - 1.5 * width, [heuristic_metrics[k] for k in metric_keys],
           width, label="Heuristic")
    ax.bar(x - 0.5 * width, [all_metrics["logreg"][k] for k in metric_keys],
           width, label="LogReg")
    ax.bar(x + 0.5 * width, [all_metrics["random_forest"][k] for k in metric_keys],
           width, label="RF")
    ax.bar(x + 1.5 * width, [all_metrics["gradient_boosting"][k] for k in metric_keys],
           width, label="GBM")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel("Score")
    ax.set_title("Heuristic vs ML Models — Test Set (N=8)")
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "ml_vs_heuristic.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"Wrote {out}")


def main() -> None:
    log.info("=== Training pipeline started ===")
    log.info(f"Log file: {log_file}")

    log.info("Loading dataset split...")
    X_train, X_test, y_train, y_test = load_dataset_split()
    log.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    all_metrics = {}
    models: dict = {}

    models["logreg"], all_metrics["logreg"] = train_logreg(
        X_train, y_train, X_test, y_test)
    models["random_forest"], all_metrics["random_forest"] = train_random_forest(
        X_train, y_train, X_test, y_test)
    models["gradient_boosting"], all_metrics["gradient_boosting"] = train_gradient_boosting(
        X_train, y_train, X_test, y_test)

    feature_order = joblib.load(MODELS_DIR / "feature_order.pkl")

    heuristic_metrics = evaluate_heuristic_baseline(X_test, y_test, feature_order)
    log.info(f"Heuristic baseline metrics: {heuristic_metrics}")

    best_key = select_best_and_write_card(all_metrics, feature_order, heuristic_metrics)
    log.info(f"Selected best model: {best_key}")

    generate_plots(models, X_test, y_test, all_metrics,
                    heuristic_metrics, feature_order)

    log.info("=== Training pipeline complete ===")
    log.info(f"All test metrics: {json.dumps(all_metrics, indent=2)}")
    log.info(f"Heuristic baseline: {json.dumps(heuristic_metrics, indent=2)}")


if __name__ == "__main__":
    main()
