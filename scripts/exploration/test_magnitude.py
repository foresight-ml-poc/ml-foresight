"""The right question: can we predict the MAGNITUDE of the move
(volatility) instead of its direction?

Efficient-market theory: direction is ~unpredictable (arbitraged away),
but reaction SIZE is often predictable from news features. This is the
single most principled pivot — and product-relevant ("this news WILL
rock the market", regardless of which way).

Targets (binary), features = the anti-leak allowlist (same pipeline):
  M24_med : |move_t24h_pct| > median        (big vs small 24h move)
  M24_20  : |move_t24h_pct| > 20%           (major mover — product signal)
  M1h_med : |move_t1h_pct|  > median        (big early move)
Baseline for contrast: predicting DIRECTION (we know ≈ 0.50).

5-fold CV, ROC-AUC, several models. Honest.
Output: results/test_magnitude.json
"""

from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import RESULTS_DIR, SEED  # noqa: E402
from data import _clean, _feature_engineer  # noqa: E402

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
import lightgbm as lgb


def models():
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(class_weight="balanced", random_state=SEED, verbose=-1),
    }


def evaluate(raw: pd.DataFrame, ycol: str, label: str) -> dict:
    d = raw.dropna(subset=[ycol]).copy()
    d["_y"] = d[ycol].astype(int)
    if d["_y"].nunique() < 2 or len(d) < 100:
        return {"target": label, "n": int(len(d)), "note": "insuffisant"}
    # reuse the leak-safe pipeline by injecting the target as direction_correct
    dd = _clean(d.assign(direction_correct=d["_y"])).reset_index(drop=True)
    X, y = _feature_engineer(dd)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res = {}
    best = (None, 0.0)
    for name, m in models().items():
        p = cross_val_predict(m, Xs, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        auc = float(roc_auc_score(y, p))
        acc = float(accuracy_score(y, (p >= 0.5).astype(int)))
        res[name] = {"roc_auc": round(auc, 3), "accuracy": round(acc, 3)}
        if auc > best[1]:
            best = (name, auc)
    return {"target": label, "n": int(len(dd)),
            "class_balance": dd["direction_correct"].value_counts().to_dict(),
            "models": res, "best_model": best[0],
            "best_roc_auc": round(best[1], 3)}


def main() -> None:
    csv = sorted(g for g in glob.glob(str(PROJECT_ROOT / "data/raw/signals_export_2*.csv"))
                 if "sample" not in g)[-1]
    raw = pd.read_csv(csv)

    # magnitude targets
    raw["abs24"] = raw["move_t24h_pct"].abs()
    raw["abs1h"] = raw["move_t1h_pct"].abs()
    m24 = raw["abs24"].median()
    m1h = raw["abs1h"].median()
    raw["M24_med"] = (raw["abs24"] > m24).astype("Int64")
    raw["M24_20"] = (raw["abs24"] > 20).astype("Int64")
    raw["M1h_med"] = (raw["abs1h"] > m1h).astype("Int64")
    # direction baseline (the old, hard target)
    yes = raw["direction"].astype(str).str.upper().str.contains("YES|UP")
    raw["DIR"] = ((yes & (raw["move_t24h_pct"] > 0)) |
                  (~yes & (raw["move_t24h_pct"] < 0))).astype("Int64")

    out = {}
    for col, lab in [("DIR", "DIRECTION T+24h (l'ancienne cible)"),
                     ("M24_med", "MAGNITUDE — gros move 24h (vs médiane)"),
                     ("M24_20", "MAGNITUDE — move majeur 24h (>20%)"),
                     ("M1h_med", "MAGNITUDE — gros move 1h (vs médiane)")]:
        r = evaluate(raw, col, lab)
        out[col] = r
        if "best_roc_auc" in r:
            print(f"{lab:<42} n={r['n']:<4} best={r['best_model']:<16} "
                  f"ROC-AUC={r['best_roc_auc']:.3f}")
        else:
            print(f"{lab:<42} {r.get('note')}")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "test_magnitude.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    dir_auc = out["DIR"].get("best_roc_auc", 0.5)
    mag_best = max((out[k].get("best_roc_auc", 0)
                    for k in ["M24_med", "M24_20", "M1h_med"]))
    print(f"\n=== SYNTHÈSE ===")
    print(f"Prédire la DIRECTION   : ROC-AUC ≈ {dir_auc:.3f}  (≈ hasard, comme prévu)")
    print(f"Prédire la MAGNITUDE   : ROC-AUC ≈ {mag_best:.3f}  (meilleur des cibles volatilité)")
    if mag_best - dir_auc > 0.05:
        print("=> La magnitude EST nettement plus prédictible que la direction. "
              "Vrai levier produit + soutenance.")
    elif mag_best - dir_auc > 0.02:
        print("=> La magnitude est un peu plus prédictible — signal modeste mais réel.")
    else:
        print("=> Même la magnitude n'est pas prédictible ici — honnête à dire.")


if __name__ == "__main__":
    main()
