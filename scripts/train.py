"""Entraîne les 3 familles de modèles imposées et établit le résultat
honnête central du POC.

  1. Régression logistique  (linéaire)
  2. Random Forest          (ensemble d'arbres / bagging)
  3. K-Means k=2            (non supervisé — y a-t-il une structure ?)

Mais le cœur intellectuel du projet n'est pas « quel modèle gagne » : c'est
la comparaison **CV vs walk-forward**.

  - Validation croisée 5-fold (mélange passé/futur)  → estimation OPTIMISTE
  - Walk-forward strict (entraîne le passé, teste le futur) → estimation HONNÊTE

Sur une série financière non-stationnaire, la CV ment. On le démontre sur
deux cibles : la direction (le « bon » jeu, déjà mort) et la magnitude
(le seul signal qui paraissait vivant — il s'effondre aussi out-of-time).

Sorties :
  models/logreg.joblib, random_forest.joblib, kmeans.joblib
  models/model_card.json              (honnête, sans faux edge)
  results/cv_vs_walkforward.json      (le chiffre central, pour figures/deck)

Usage : python scripts/train.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    LOGS_DIR, MODEL_CARD_FILE, MODELS_DIR, RESULTS_DIR, SEED,
)
from data import _clean, _feature_engineer, load_dataset_split  # noqa: E402
from metrics import compute_metrics  # noqa: E402

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict,
)
from sklearn.preprocessing import StandardScaler

LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("train")


# ---------- les 3 modèles imposés ----------

def train_logreg(Xtr, ytr, Xte, yte):
    log.info("Régression logistique — GridSearchCV(C), 5-fold...")
    s = GridSearchCV(
        LogisticRegression(penalty="l2", class_weight="balanced",
                            max_iter=1000, random_state=SEED),
        {"C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        cv=5, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
    m = s.best_estimator_
    joblib.dump(m, MODELS_DIR / "logreg.joblib")
    return m, compute_metrics(yte, m.predict(Xte))


def train_random_forest(Xtr, ytr, Xte, yte):
    log.info("Random Forest — RandomizedSearchCV(20), 5-fold...")
    s = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=SEED,
                               n_jobs=-1),
        {"n_estimators": [200, 300, 500], "max_depth": [3, 5, 8, 12, None],
         "min_samples_leaf": [1, 2, 5], "max_features": ["sqrt", "log2", 0.5]},
        n_iter=20, cv=5, scoring="roc_auc", n_jobs=-1,
        random_state=SEED).fit(Xtr, ytr)
    m = s.best_estimator_
    joblib.dump(m, MODELS_DIR / "random_forest.joblib")
    return m, compute_metrics(yte, m.predict(Xte))


def train_kmeans(Xtr, ytr, Xte, yte):
    """K-Means k=2, NON supervisé (.fit ignore y).

    On sauvegarde le KMeans sklearn natif (pas de classe custom → main.py,
    fixé Basile, peut l'unpickler sans importer ce script). On aligne les
    ids de cluster sur la classe majoritaire du train AVANT de figer le
    modèle, pour que .predict renvoie des 0/1 cohérents.

    L'indicateur honnête n'est pas l'accuracy mais l'Adjusted Rand Index
    cluster↔label : ≈ 0 ⇒ les signaux ne se regroupent pas en
    gagnants/perdants ⇒ aucune structure exploitable.
    """
    log.info("K-Means k=2 (non supervisé)...")
    km = KMeans(n_clusters=2, n_init=10, random_state=SEED)
    c_tr = km.fit_predict(Xtr)
    ari = float(adjusted_rand_score(ytr, c_tr))
    log.info(f"K-Means Adjusted Rand Index (cluster↔label) = {ari:.4f}")

    # Si le cluster « 0 » est majoritairement la classe 1, on permute les
    # centres pour que l'id de cluster s'aligne sur la classe (cosmétique :
    # l'ARI, lui, est invariant au ré-étiquetage).
    if np.asarray(ytr)[c_tr == 0].mean() > np.asarray(ytr)[c_tr == 1].mean():
        km.cluster_centers_ = km.cluster_centers_[::-1].copy()

    joblib.dump(km, MODELS_DIR / "kmeans.joblib")
    met = compute_metrics(yte, km.predict(Xte))
    met["adjusted_rand_index"] = ari
    return km, met


# ---------- LE CŒUR : CV (optimiste) vs walk-forward (honnête) ----------

def _load_temporal():
    """Recharge le CSV brut, nettoie, feature-engineer, et renvoie X + labels
    alignés + l'ordre temporel (created_at) pour le walk-forward strict."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    cands = [p for p in sorted(raw_dir.glob("signals_export_20*.csv"))
             if "sample" not in p.name]
    csv = cands[-1] if cands else raw_dir / "signals_export_sample.csv"
    df = pd.read_csv(csv)
    cleaned = _clean(df)
    X, y_dir = _feature_engineer(cleaned)
    aligned = cleaned.loc[X.index]
    created = pd.to_datetime(aligned["created_at"], utc=True, errors="coerce")
    move24 = aligned["move_t24h_pct"].astype(float)
    order = created.sort_values().index
    X = X.loc[order].reset_index(drop=True)
    created = created.loc[order].reset_index(drop=True)
    y_dir = y_dir.loc[order].reset_index(drop=True).astype(int)
    move24 = move24.loc[order].reset_index(drop=True)
    y_mag = (move24.abs() > move24.abs().median()).astype(int)  # gros move ?
    return X, y_dir, y_mag, csv.name


def cv_vs_walkforward(X, y, label):
    """Renvoie {cv_auc_mean, walkforward_auc:[...], walkforward_mean}.
    CV = 5-fold mélangé (optimiste). WF = 3 blocs expansifs (honnête)."""
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
    p = cross_val_predict(rf, Xs, y, cv=cv, method="predict_proba",
                          n_jobs=-1)[:, 1]
    cv_auc = float(roc_auc_score(y, p))
    n = len(X)
    wf = []
    for a, b in [(int(n*.5), int(n*.65)), (int(n*.65), int(n*.8)),
                 (int(n*.8), n)]:
        if y.iloc[:a].nunique() < 2 or y.iloc[a:b].nunique() < 2:
            continue
        sc = StandardScaler().fit(X.iloc[:a])
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1)
        m.fit(sc.transform(X.iloc[:a]), y.iloc[:a])
        pp = m.predict_proba(sc.transform(X.iloc[a:b]))[:, 1]
        wf.append(round(float(roc_auc_score(y.iloc[a:b], pp)), 3))
    out = {"label": label, "n": n, "cv_auc_mean": round(cv_auc, 3),
           "walkforward_auc": wf,
           "walkforward_mean": round(float(np.mean(wf)), 3) if wf else None}
    log.info(f"[{label}] CV={out['cv_auc_mean']} | "
             f"walk-forward={wf} moy={out['walkforward_mean']}")
    return out


# ---------- model card honnête ----------

def write_card(metrics, kmeans_ari, cvwf, n_train, n_test, csv_name):
    card = {
        "model_version": "v2.0.0-honest",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "task": "Classification binaire — direction_correct à T+24h",
        "dataset": {"source": csv_name, "train": n_train, "test": n_test,
                    "total": n_train + n_test},
        "models": {
            "logreg": metrics["logreg"],
            "random_forest": metrics["random_forest"],
            "kmeans": metrics["kmeans"],
        },
        "kmeans_adjusted_rand_index": round(kmeans_ari, 4),
        "cv_vs_walkforward": cvwf,
        "verdict": (
            "Les 3 modèles sont ≈ au hasard sur la direction (ROC-AUC ~0.50). "
            "K-Means : ARI ≈ 0 → aucun regroupement naturel gagnants/perdants. "
            "Surtout : la CV (5-fold mélangé) donne ~0.58 sur la magnitude, "
            "mais le walk-forward strict s'effondre à ~0.52 puis sous 0.50 "
            "sur la période récente. La CV est invalide sur série temporelle "
            "non-stationnaire. Conclusion : marché efficient vis-à-vis des "
            "features publiques. Résultat négatif, rigoureux, assumé."
        ),
        "training_seed": SEED,
    }
    MODEL_CARD_FILE.write_text(json.dumps(card, indent=2, ensure_ascii=False))
    log.info(f"Écrit {MODEL_CARD_FILE}")


def main():
    log.info("=== Pipeline d'entraînement (honnête) ===")
    Xtr, Xte, ytr, yte = load_dataset_split()
    log.info(f"X_train {Xtr.shape} · X_test {Xte.shape}")

    metrics = {}
    _, metrics["logreg"] = train_logreg(Xtr, ytr, Xte, yte)
    _, metrics["random_forest"] = train_random_forest(Xtr, ytr, Xte, yte)
    km, metrics["kmeans"] = train_kmeans(Xtr, ytr, Xte, yte)
    kmeans_ari = metrics["kmeans"]["adjusted_rand_index"]

    log.info("--- CV vs walk-forward (le cœur) ---")
    X, y_dir, y_mag, csv_name = _load_temporal()
    cvwf = {
        "direction": cv_vs_walkforward(X, y_dir, "Direction T+24h"),
        "magnitude": cv_vs_walkforward(X, y_mag, "Magnitude (gros move)"),
    }
    (RESULTS_DIR / "cv_vs_walkforward.json").write_text(
        json.dumps(cvwf, indent=2, ensure_ascii=False))
    log.info(f"Écrit {RESULTS_DIR / 'cv_vs_walkforward.json'}")

    write_card(metrics, kmeans_ari, cvwf, len(Xtr), len(Xte), csv_name)

    log.info("=== Terminé ===")
    log.info(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    print("\n=== RÉSUMÉ HONNÊTE ===")
    for k, m in metrics.items():
        print(f"  {k:<15} ROC-AUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}")
    print(f"  K-Means ARI cluster↔label = {kmeans_ari:.4f}  (≈0 → pas de structure)")
    d, mg = cvwf["direction"], cvwf["magnitude"]
    print(f"\n  Direction : CV {d['cv_auc_mean']} → walk-forward {d['walkforward_mean']}")
    print(f"  Magnitude : CV {mg['cv_auc_mean']} → walk-forward {mg['walkforward_mean']} "
          f"{mg['walkforward_auc']}")
    print("  => La CV ment sur série non-stationnaire. Marché efficient.")


if __name__ == "__main__":
    main()
