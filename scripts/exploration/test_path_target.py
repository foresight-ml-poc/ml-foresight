"""The user's quant insight: a single T+24h snapshot is a bad label for an
actionable signal — the price is a continuous PATH. Evaluate on the path
(Maximum Favorable Excursion), not an arbitrary endpoint.

Path points available: signed move at T+5min / T+15min / T+1h / T+24h
(signed = in the signal's predicted direction).

  MFE = max over the 4 horizons of signed_move      (best the trade reached)
  MAE = max over the 4 horizons of -signed_move, ≥0 (worst drawdown)

Targets (binary), features = anti-leak allowlist (same pipeline):
  WIN10  : MFE > 10%        — a clearly tradeable window existed
  WIN05  : MFE > 5%         — a modest window existed
  NETUSE : MFE > MAE        — favorable beat adverse on the path
                              (directionally net-useful, timing aside)
Baseline for contrast: DIR = direction correct at the T+24h snapshot.

5-fold CV (15 seeds for robustness) + strict walk-forward. Honest.
Caveat reported: "a window existed" assumes you timed the exit.
Output: results/test_path_target.json
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
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def _signed_path(raw: pd.DataFrame) -> pd.DataFrame:
    s = np.where(raw["direction"].astype(str).str.upper().str.contains("YES|UP"),
                 1.0, -1.0)
    cols = ["move_t5min_pct", "move_t15min_pct", "move_t1h_pct", "move_t24h_pct"]
    P = pd.DataFrame({c: s * raw[c].astype(float) for c in cols})
    raw = raw.copy()
    raw["MFE"] = P.max(axis=1)               # best favorable point
    raw["MAE"] = (-P).clip(lower=0).max(axis=1)  # worst adverse drawdown
    raw["DIR"] = (s * raw["move_t24h_pct"].astype(float) > 0).astype("Int64")
    raw["WIN10"] = (raw["MFE"] > 10).astype("Int64")
    raw["WIN05"] = (raw["MFE"] > 5).astype("Int64")
    raw["NETUSE"] = (raw["MFE"] > raw["MAE"]).astype("Int64")
    return raw


def _auc_cv(X, y, seeds=15):
    Xs = StandardScaler().fit_transform(X)
    out = []
    for sd in range(seeds):
        cv = StratifiedKFold(5, shuffle=True, random_state=sd)
        p = cross_val_predict(
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1),
            Xs, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        out.append(roc_auc_score(y, p))
    return np.array(out)


def _walkforward(X, y):
    n = len(X)
    res = []
    for a, b in [(int(n*.5), int(n*.65)), (int(n*.65), int(n*.8)), (int(n*.8), n)]:
        if y.iloc[:a].nunique() < 2 or y.iloc[a:b].nunique() < 2:
            continue
        sc = StandardScaler().fit(X.iloc[:a])
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=SEED, n_jobs=-1)
        m.fit(sc.transform(X.iloc[:a]), y.iloc[:a])
        pp = m.predict_proba(sc.transform(X.iloc[a:b]))[:, 1]
        res.append(round(roc_auc_score(y.iloc[a:b], pp), 3))
    return res


def main() -> None:
    csv = sorted(g for g in glob.glob(str(PROJECT_ROOT / "data/raw/signals_export_2*.csv"))
                 if "sample" not in g)[-1]
    raw = pd.read_csv(csv)
    raw["created_at"] = pd.to_datetime(raw["created_at"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["created_at", "move_t24h_pct"]).sort_values(
        "created_at").reset_index(drop=True)
    raw = _signed_path(raw)

    targets = [("DIR", "DIRECTION @ T+24h (snapshot — l'ancienne)"),
               ("NETUSE", "CHEMIN — favorable > adverse (net-utile)"),
               ("WIN05", "CHEMIN — fenêtre exploitable >5% a existé"),
               ("WIN10", "CHEMIN — fenêtre exploitable >10% a existé")]
    out = {}
    for col, lab in targets:
        d = raw.dropna(subset=[col]).copy()
        d[col] = d[col].astype(int)
        if d[col].nunique() < 2:
            out[col] = {"label": lab, "note": "une seule classe"}
            print(f"{lab:<46} une seule classe"); continue
        dd = _clean(d.assign(direction_correct=d[col])).reset_index(drop=True)
        X, y = _feature_engineer(dd)
        X = X.reset_index(drop=True); y = y.reset_index(drop=True).astype(int)
        aucs = _auc_cv(X, y)
        wf = _walkforward(X, y)
        bal = dd["direction_correct"].value_counts().to_dict()
        out[col] = {
            "label": lab, "n": int(len(dd)), "class_balance": bal,
            "cv_auc_mean": round(float(aucs.mean()), 3),
            "cv_auc_min": round(float(aucs.min()), 3),
            "cv_auc_max": round(float(aucs.max()), 3),
            "cv_auc_std": round(float(aucs.std()), 3),
            "walkforward_auc": wf,
            "walkforward_mean": round(float(np.mean(wf)), 3) if wf else None,
        }
        print(f"{lab:<46} n={len(dd):<4} CV={aucs.mean():.3f}±{aucs.std():.3f} "
              f"(min {aucs.min():.3f}) | walk-fwd {wf} moy "
              f"{np.mean(wf):.3f}" if wf else "")

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "test_path_target.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    dir_cv = out["DIR"]["cv_auc_mean"]
    print("\n=== SYNTHÈSE (honnête) ===")
    print(f"DIRECTION snapshot  : CV {dir_cv:.3f}  (≈ hasard — le mauvais label)")
    for k in ["NETUSE", "WIN05", "WIN10"]:
        r = out.get(k, {})
        if "cv_auc_mean" in r:
            wf = r.get("walkforward_mean")
            print(f"{r['label']:<42}: CV {r['cv_auc_mean']:.3f} "
                  f"| out-of-time {wf}")
    print("\nCaveat assumé : « une fenêtre a existé » suppose un timing de "
          "sortie correct — ce n'est pas un profit garanti, c'est "
          "« le signal a créé une opportunité ».")


if __name__ == "__main__":
    main()
