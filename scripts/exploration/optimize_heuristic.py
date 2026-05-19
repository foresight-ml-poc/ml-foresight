"""Answer the user's question: should we OPTIMIZE the heuristic's weights
instead of training a separate black-box classifier?

Mathematically, "learning the best weights for the heuristic formula" =
a logistic regression on the heuristic's own 8 inputs. This is fully
interpretable: we can print "hand-picked weight" vs "data-learned weight"
side by side.

Heuristic (from Foresight prod):
  signal_strength = 0.15*fresh + 0.10*srcw + 0.15*conf
                  + 0.60*(0.65*impact + 0.35*llm)
  trade_quality   = 0.40*liq + 0.35*spread + 0.25*ttr
  signal_score    = 0.75*signal_strength + 0.25*trade_quality

=> effective hand weight of each input in signal_score:
  freshness            0.75*0.15            = 0.1125
  source_weight        0.75*0.10            = 0.0750
  confirmation         0.75*0.15            = 0.1125
  impact_strength      0.75*0.60*0.65       = 0.2925
  llm_confidence       0.75*0.60*0.35       = 0.1575
  liquidity            0.25*0.40            = 0.1000
  spread               0.25*0.35            = 0.0875
  time_to_resolution   0.25*0.25            = 0.0625      (sum = 1.0)

We compare, on the 530 signals that have the real factors:
  - ROC-AUC of the hand-tuned heuristic
  - ROC-AUC of logistic regression (learned weights), 5-fold CV
  - the learned weights vs the hand weights, normalised

Output: results/optimize_heuristic.json + console table.
"""

from __future__ import annotations

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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score

FEATURES = ["freshness_factor", "source_weight", "confirmation_factor",
            "impact_strength", "llm_confidence", "liquidity_factor",
            "spread_penalty", "time_to_resolution_factor"]
HAND = {  # effective hand-picked weight in signal_score
    "freshness_factor": 0.1125, "source_weight": 0.0750,
    "confirmation_factor": 0.1125, "impact_strength": 0.2925,
    "llm_confidence": 0.1575, "liquidity_factor": 0.1000,
    "spread_penalty": 0.0875, "time_to_resolution_factor": 0.0625,
}


def main() -> None:
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "heuristic_factors.csv")
    df = df.dropna(subset=["direction_correct"] + FEATURES).copy()
    df["direction_correct"] = df["direction_correct"].astype(int)
    y = df["direction_correct"].to_numpy()
    X = df[FEATURES].to_numpy(dtype=float)
    n = len(df)
    print(f"[opt] {n} signaux avec les 8 facteurs · winrate global "
          f"{y.mean():.1%}")

    # 1. Hand-tuned heuristic, as a scorer (its own signal_score / 100)
    heur_auc = roc_auc_score(y, df["heuristic_score"] / 100.0)
    heur_acc = accuracy_score(y, (df["heuristic_score"] > 65).astype(int))

    # 2. Logistic regression = learned weights, 5-fold CV (honest, no leak)
    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(
        LogisticRegression(max_iter=1000, class_weight="balanced",
                           random_state=SEED),
        Xs, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    lr_auc = roc_auc_score(y, proba)
    lr_acc = accuracy_score(y, (proba >= 0.5).astype(int))

    # 3. Learned weights (fit on all data, just to read coefficients)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                            random_state=SEED).fit(Xs, y)
    coef = lr.coef_[0]
    # importance = |standardised coefficient|, normalised to sum 1
    learned = np.abs(coef)
    learned = learned / learned.sum()
    # direction of effect (sign): does more of this feature help (+) or hurt (-)?
    sign = np.sign(coef)

    rows = []
    print(f"\n{'feature':<24}{'poids main':>12}{'poids appris':>14}{'effet':>8}")
    print("-" * 58)
    for i, f in enumerate(FEATURES):
        hw = HAND[f]
        lw = learned[i]
        eff = "↑ aide" if sign[i] > 0 else "↓ nuit"
        rows.append({"feature": f, "hand_weight": round(hw, 4),
                     "learned_weight": round(float(lw), 4),
                     "effect": "helps" if sign[i] > 0 else "hurts"})
        print(f"{f:<24}{hw:>12.3f}{lw:>14.3f}{eff:>10}")

    summary = {
        "n_signals": n,
        "winrate": round(float(y.mean()), 4),
        "heuristic_hand_tuned": {"roc_auc": round(heur_auc, 4),
                                 "accuracy_at_65": round(heur_acc, 4)},
        "logreg_learned_weights": {"roc_auc": round(lr_auc, 4),
                                   "accuracy": round(lr_acc, 4)},
        "delta_roc_auc_pts": round((lr_auc - heur_auc) * 100, 2),
        "weights": rows,
        "verdict": (
            "La régression apprend de meilleurs poids"
            if lr_auc - heur_auc > 0.02 else
            "Poids appris ≈ poids main : la formule manuelle est déjà "
            "bien calibrée (ou le signal est trop faible pour départager)"
        ),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "optimize_heuristic.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nHeuristique main  ROC-AUC = {heur_auc:.3f}")
    print(f"Poids appris (LR) ROC-AUC = {lr_auc:.3f}  "
          f"({(lr_auc-heur_auc)*100:+.1f} pts)")
    print(f"\n=> {summary['verdict']}")
    # most over/under-weighted by the hand formula
    diff = [(f, learned[i] - HAND[f]) for i, f in enumerate(FEATURES)]
    diff.sort(key=lambda x: -x[1])
    print(f"\nLa data voudrait MONTER : {diff[0][0]} ({diff[0][1]:+.2f})")
    print(f"La data voudrait BAISSER : {diff[-1][0]} ({diff[-1][1]:+.2f})")


if __name__ == "__main__":
    main()
