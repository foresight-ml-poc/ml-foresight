"""Comprehensive honesty check before the pitch deck.

(a) FULL-HISTORY model sweep: ~12 classifiers, 5-fold stratified CV on ALL
    855 signals, ROC-AUC / accuracy / F1 vs the heuristic baseline.
    Answers: "based on ALL historical signals, can any model beat the
    heuristic?" (not just the noisy 3-day slice).

(b) TEMPORAL improvement: expanding-window walk-forward. Sorted by time,
    train on history-so-far, test on the next chunk. Track ML test
    ROC-AUC and winrate as the cumulative dataset grows. Answers:
    "does performance improve as more data accumulates?"

Outputs:
  results/analysis_full.json
  plots/model_sweep_full.png
  plots/temporal_improvement.png

Usage: python scripts/analysis_full.py
"""

from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import HEURISTIC_THRESHOLD, PLOTS_DIR, RESULTS_DIR, SEED, TARGET_COLUMN  # noqa: E402
from data import _clean, _feature_engineer  # noqa: E402

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, HistGradientBoostingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb

MINT, LOSS, GREY = "#0BE0A6", "#f76d6d", "#5c6878"


def _models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=SEED),
        "AdaBoost": AdaBoostClassifier(random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1),
        "XGBoost": xgb.XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=SEED),
        "SVM_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "GaussianNB": GaussianNB(),
        "MLP_sklearn": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=SEED),
    }


def _latest_csv() -> Path:
    csvs = sorted(g for g in glob.glob(str(PROJECT_ROOT / "data" / "raw" / "signals_export_*.csv"))
                  if "sample" not in g)
    return Path(csvs[-1])


def _signed_ret(df) -> np.ndarray:
    sign = np.where(df["direction"].astype(str).str.upper().str.contains("YES|UP"), 1.0, -1.0)
    return sign * df["move_t24h_pct"].to_numpy(dtype=float)


def main() -> None:
    raw = pd.read_csv(_latest_csv())
    raw["created_at"] = pd.to_datetime(raw["created_at"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)
    clean = _clean(raw)
    X, y = _feature_engineer(clean)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    clean = clean.reset_index(drop=True)
    n = len(X)

    # Heuristic baseline on full history (its own threshold as a classifier)
    h_pred = (clean["heuristic_score"] > HEURISTIC_THRESHOLD).astype(int)
    heur = {
        "roc_auc": float(roc_auc_score(y, clean["heuristic_score"] / 100.0)),
        "accuracy": float(accuracy_score(y, h_pred)),
        "f1": float(f1_score(y, h_pred)),
    }
    print(f"[full] n={n} | heuristique ROC-AUC={heur['roc_auc']:.3f} "
          f"acc={heur['accuracy']:.3f} f1={heur['f1']:.3f}")

    # (a) Full-history 5-fold CV sweep
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sweep = {}
    for name, mdl in _models().items():
        try:
            proba = cross_val_predict(mdl, Xs, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        except Exception:
            proba = cross_val_predict(mdl, Xs, y, cv=cv, n_jobs=-1).astype(float)
        pred = (proba >= 0.5).astype(int)
        sweep[name] = {
            "roc_auc": float(roc_auc_score(y, proba)),
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred)),
        }
        print(f"[full] {name:<22} ROC-AUC={sweep[name]['roc_auc']:.3f} "
              f"acc={sweep[name]['accuracy']:.3f} f1={sweep[name]['f1']:.3f}")

    best_name = max(sweep, key=lambda k: sweep[k]["roc_auc"])
    print(f"[full] BEST = {best_name} ({sweep[best_name]['roc_auc']:.3f}) "
          f"vs heuristique {heur['roc_auc']:.3f} "
          f"=> {(sweep[best_name]['roc_auc']-heur['roc_auc'])*100:+.1f} pts")

    # (b) Expanding-window walk-forward (temporal improvement)
    steps = 8
    start = int(n * 0.35)
    idxs = np.linspace(start, n, steps + 1).astype(int)
    wf = []
    for i in range(len(idxs) - 1):
        tr_end = idxs[i]
        te_end = idxs[i + 1]
        if te_end - tr_end < 10 or tr_end < 50:
            continue
        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        Xte, yte = X.iloc[tr_end:te_end], y.iloc[tr_end:te_end]
        sc = StandardScaler().fit(Xtr)
        m = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1)
        m.fit(sc.transform(Xtr), ytr)
        p = m.predict_proba(sc.transform(Xte))[:, 1]
        if len(np.unique(yte)) < 2:
            continue
        sub = clean.iloc[tr_end:te_end]
        h_sub = (sub["heuristic_score"] > HEURISTIC_THRESHOLD)
        wf.append({
            "train_size": int(tr_end),
            "test_size": int(te_end - tr_end),
            "date_mid": str(clean.iloc[(tr_end + te_end) // 2]["created_at"])[:10],
            "ml_roc_auc": float(roc_auc_score(yte, p)),
            "ml_top_tercile_winrate": float(
                yte[p >= np.quantile(p, 2 / 3)].mean()) if (p >= np.quantile(p, 2/3)).sum() else None,
            "heuristic_winrate": float(sub.loc[h_sub, TARGET_COLUMN].mean()) if h_sub.sum() else None,
        })
        print(f"[wf] train={tr_end:<4} {wf[-1]['date_mid']} "
              f"ML ROC-AUC={wf[-1]['ml_roc_auc']:.3f}")

    summary = {
        "n_total": n,
        "heuristic": heur,
        "full_history_sweep": sweep,
        "best_model": best_name,
        "best_vs_heuristic_pts": round((sweep[best_name]["roc_auc"] - heur["roc_auc"]) * 100, 1),
        "walk_forward": wf,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "analysis_full.json").write_text(json.dumps(summary, indent=2))

    # Plot 1 — full sweep
    plt.style.use("dark_background")
    order = sorted(sweep, key=lambda k: sweep[k]["roc_auc"])
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0b0f17"); ax.set_facecolor("#11161f")
    vals = [sweep[k]["roc_auc"] for k in order]
    colors = [MINT if v >= heur["roc_auc"] else GREY for v in vals]
    ax.barh(range(len(order)), vals, color=colors)
    ax.axvline(heur["roc_auc"], color=LOSS, ls="--", lw=2,
               label=f"heuristique Foresight ({heur['roc_auc']:.3f})")
    ax.axvline(0.5, color=GREY, ls=":", lw=1, label="hasard (0.5)")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9, color="#e7edf5")
    ax.set_xlim(0.45, max(0.62, max(vals) + 0.02))
    ax.set_xlabel("ROC-AUC (5-fold CV, tout l'historique — 855 signaux)")
    ax.set_title("Tous les modèles vs heuristique — historique complet", color="#e7edf5")
    ax.legend(loc="lower right", fontsize=9); ax.grid(color="#1a212d", axis="x")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "model_sweep_full.png", dpi=130); plt.close(fig)

    # Plot 2 — temporal improvement
    if wf:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0b0f17"); ax.set_facecolor("#11161f")
        xs = [w["train_size"] for w in wf]
        ax.plot(xs, [w["ml_roc_auc"] for w in wf], marker="o", color=MINT, lw=2.5,
                label="ML ROC-AUC (out-of-sample, fenêtre suivante)")
        ax.axhline(0.5, color=GREY, ls=":", lw=1, label="hasard")
        ax.axhline(heur["roc_auc"], color=LOSS, ls="--", lw=1.5,
                   label=f"heuristique ({heur['roc_auc']:.3f})")
        # trend line
        if len(xs) >= 3:
            z = np.polyfit(xs, [w["ml_roc_auc"] for w in wf], 1)
            ax.plot(xs, np.poly1d(z)(xs), color=MINT, ls="--", lw=1, alpha=0.6,
                    label=f"tendance (pente {z[0]*1000:+.2f}/1000 signaux)")
        ax.set_xlabel("taille du train (signaux accumulés dans le temps)")
        ax.set_ylabel("ROC-AUC out-of-sample")
        ax.set_title("Performance ML quand les données s'accumulent (walk-forward)",
                     color="#e7edf5")
        ax.legend(loc="best", fontsize=9); ax.grid(color="#1a212d")
        fig.tight_layout(); fig.savefig(PLOTS_DIR / "temporal_improvement.png", dpi=130); plt.close(fig)

    print("\n=== SYNTHÈSE ===")
    print(f"Historique complet : best = {best_name} {sweep[best_name]['roc_auc']:.3f} "
          f"vs heuristique {heur['roc_auc']:.3f} ({summary['best_vs_heuristic_pts']:+.1f} pts)")
    if wf:
        first, last = wf[0]["ml_roc_auc"], wf[-1]["ml_roc_auc"]
        print(f"Walk-forward : ML ROC-AUC {first:.3f} (début) → {last:.3f} (fin) "
              f"= {(last-first)*100:+.1f} pts sur la période")


if __name__ == "__main__":
    main()
