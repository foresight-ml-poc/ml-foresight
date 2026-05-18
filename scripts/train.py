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


# ---------- Model 4: LightGBM ----------

def train_lightgbm(X_train, y_train, X_test, y_test) -> tuple:
    """LightGBM — fastest of the gradient boosting libs, often best on tabular."""
    import lightgbm as lgb
    from sklearn.model_selection import RandomizedSearchCV

    log.info("Training LightGBM with RandomizedSearchCV (20 iters, 5-fold)...")
    distributions = {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3],
        "num_leaves": [15, 31, 63],
        "max_depth": [-1, 4, 8, 12],
        "min_child_samples": [5, 10, 20],
        "subsample": [0.7, 0.85, 1.0],
    }
    search = RandomizedSearchCV(
        lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1),
        param_distributions=distributions,
        n_iter=20, cv=5, scoring="roc_auc", n_jobs=-1, random_state=SEED,
    )
    search.fit(X_train, y_train)
    log.info(f"LGBM best params: {search.best_params_}, best CV ROC-AUC: {search.best_score_:.4f}")

    model = search.best_estimator_
    metrics = compute_metrics(y_test, model.predict(X_test))
    log.info(f"LGBM test metrics: {metrics}")

    out = MODELS_DIR / "lightgbm.joblib"
    joblib.dump(model, out)
    log.info(f"Saved {out}")
    return model, metrics


# ---------- Model 5: XGBoost ----------

def train_xgboost(X_train, y_train, X_test, y_test) -> tuple:
    """XGBoost — the OG gradient boosting lib, typically performs similarly to LightGBM."""
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    scale_pos_weight = float(cw[1] / cw[0])  # XGB uses scale_pos_weight not class_weight

    log.info(f"Training XGBoost with RandomizedSearchCV (20 iters, 5-fold), scale_pos_weight={scale_pos_weight:.3f}...")
    distributions = {
        "n_estimators": [50, 100, 200, 300],
        "learning_rate": [0.01, 0.03, 0.1, 0.3],
        "max_depth": [3, 5, 8, 12],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
    }
    search = RandomizedSearchCV(
        xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=SEED,
            eval_metric="logloss",
            tree_method="hist",
        ),
        param_distributions=distributions,
        n_iter=20, cv=5, scoring="roc_auc", n_jobs=-1, random_state=SEED,
    )
    search.fit(X_train, y_train)
    log.info(f"XGB best params: {search.best_params_}, best CV ROC-AUC: {search.best_score_:.4f}")

    model = search.best_estimator_
    metrics = compute_metrics(y_test, model.predict(X_test))
    log.info(f"XGB test metrics: {metrics}")

    out = MODELS_DIR / "xgboost.joblib"
    joblib.dump(model, out)
    log.info(f"Saved {out}")
    return model, metrics


# ---------- Model 6: SVM (non-linear, RBF kernel) ----------

def train_svm(X_train, y_train, X_test, y_test) -> tuple:
    """SVM with RBF kernel — classic non-linear baseline different from trees."""
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    log.info("Training SVM (RBF kernel) with GridSearchCV (5-fold)...")
    grid = {
        "C": [0.1, 1.0, 3.0, 10.0],
        "gamma": ["scale", "auto", 0.01, 0.1],
    }
    search = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=SEED),
        param_grid=grid,
        cv=5, scoring="roc_auc", n_jobs=-1,
    )
    search.fit(X_train, y_train)
    log.info(f"SVM best params: {search.best_params_}, best CV ROC-AUC: {search.best_score_:.4f}")

    model = search.best_estimator_
    metrics = compute_metrics(y_test, model.predict(X_test))
    log.info(f"SVM test metrics: {metrics}")

    out = MODELS_DIR / "svm.joblib"
    joblib.dump(model, out)
    log.info(f"Saved {out}")
    return model, metrics


# ---------- Heuristic baseline (Foresight's current formula) ----------

def evaluate_heuristic_baseline(y_test) -> dict:
    """Use the persisted Foresight signal_score directly as the heuristic baseline.

    The heuristic_score column from the export holds the score Foresight
    actually emitted at signal-time (already clipped to [0, 100]). We don't
    recompute from features — that was the v1.0.0 approach which forced us
    to keep the 6 heuristic factors as features (limiting us to 40 samples).
    The persisted score is the ground truth of what Foresight predicted then.

    Threshold > HEURISTIC_THRESHOLD (=65) → predicted class 1.
    """
    test_heuristic = joblib.load(MODELS_DIR / "test_heuristic_scores.pkl")
    if len(test_heuristic) != len(y_test):
        raise ValueError(
            f"Heuristic score count ({len(test_heuristic)}) does not match "
            f"y_test count ({len(y_test)})."
        )
    y_pred = (test_heuristic > HEURISTIC_THRESHOLD).astype(int)
    return compute_metrics(y_test, y_pred)


# ---------- Best model selection + model card ----------

def select_best_and_write_card(all_metrics: dict, feature_order: list,
                                heuristic: dict, n_train: int, n_test: int) -> str:
    """Pick best model by test ROC-AUC, copy to best_model.joblib, write model_card.json."""
    import shutil

    best_key = max(all_metrics, key=lambda k: all_metrics[k]["roc_auc"])
    log.info(f"Best model on test ROC-AUC: {best_key} "
             f"({all_metrics[best_key]['roc_auc']:.4f})")

    src = MODELS_DIR / f"{best_key}.joblib"
    dst = MODELS_DIR / "best_model.joblib"
    shutil.copy(src, dst)
    log.info(f"Copied {src.name} → {dst.name}")

    card = {
        "model_version": "v1.3.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model_type": best_key,
        "best_model_path": f"models/{best_key}.joblib",
        "feature_order": feature_order,
        "scaler_path": "models/scaler.pkl",
        "test_metrics": all_metrics,
        "heuristic_baseline": heuristic,
        "training_seed": SEED,
        "dataset_size": {
            "train": n_train,
            "test": n_test,
            "total": n_train + n_test,
        },
        "notes": (
            "6 models compared (LogReg, RandomForest, GradientBoosting, "
            "LightGBM, XGBoost, SVM). MLP was dropped earlier because "
            "tf.keras.fit() hung on this conda env. Trained on the production "
            "Hetzner dataset."
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

    # 4. ML vs Heuristic punchline (handles any number of models)
    metric_keys = ["accuracy", "f1", "roc_auc"]
    series = [("Heuristic", heuristic_metrics)] + [
        (k, all_metrics[k]) for k in all_metrics.keys()
    ]
    n_series = len(series)
    fig, ax = plt.subplots(figsize=(max(8, n_series * 1.3), 5))
    x = np.arange(len(metric_keys))
    width = 0.85 / n_series
    for i, (label, m) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(x + offset, [m[k] for k in metric_keys], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylabel("Score")
    n_test = len(y_test)
    ax.set_title(f"Heuristic vs ML Models — Test Set (N={n_test})")
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(ncol=min(4, n_series), fontsize=9, loc="lower right")
    fig.tight_layout()
    out = PLOTS_DIR / "ml_vs_heuristic.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    log.info(f"Wrote {out}")

    # 5. Feature importance comparison across tree models (RF, GBM, LightGBM, XGBoost)
    tree_models = {k: m for k, m in models.items()
                   if hasattr(m, "feature_importances_")}
    if len(tree_models) >= 2:
        fig, ax = plt.subplots(figsize=(10, 7))
        # rank features by avg importance across tree models
        avg_imp = np.mean([m.feature_importances_ for m in tree_models.values()], axis=0)
        idx = np.argsort(avg_imp)[::-1][:12]
        n = len(tree_models)
        bar_w = 0.85 / n
        for i, (name, m) in enumerate(tree_models.items()):
            offset = (i - (n - 1) / 2) * bar_w
            ax.barh(np.arange(len(idx)) + offset,
                    m.feature_importances_[idx][::-1],
                    bar_w, label=name)
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels([feature_order[i] for i in idx][::-1])
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature importance — top 12 across {n} tree models")
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        out = PLOTS_DIR / "feature_importance_comparison.png"
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
    models["lightgbm"], all_metrics["lightgbm"] = train_lightgbm(
        X_train, y_train, X_test, y_test)
    models["xgboost"], all_metrics["xgboost"] = train_xgboost(
        X_train, y_train, X_test, y_test)
    models["svm"], all_metrics["svm"] = train_svm(
        X_train, y_train, X_test, y_test)

    feature_order = joblib.load(MODELS_DIR / "feature_order.pkl")

    heuristic_metrics = evaluate_heuristic_baseline(y_test)
    log.info(f"Heuristic baseline metrics: {heuristic_metrics}")

    best_key = select_best_and_write_card(
        all_metrics, feature_order, heuristic_metrics,
        n_train=len(X_train), n_test=len(X_test),
    )
    log.info(f"Selected best model: {best_key}")

    generate_plots(models, X_test, y_test, all_metrics,
                    heuristic_metrics, feature_order)

    log.info("=== Training pipeline complete ===")
    log.info(f"All test metrics: {json.dumps(all_metrics, indent=2)}")
    log.info(f"Heuristic baseline: {json.dumps(heuristic_metrics, indent=2)}")


if __name__ == "__main__":
    main()
