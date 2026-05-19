"""Can ML help HERE? Not by predicting direction (dead 6 ways) but by
TRIAGE: at signal time, rank signals so we only trade the ones likely to
work, dropping the ~58% that never go favorable.

Honest test on the CORRECTED dense labels (dense_metrics_fixed.csv):
  - Target = win under a SINGLE pre-committed rule, fixed a priori
    (TP10 / H60, net of 3pp spread). No per-signal best-rule (look-ahead).
  - Features = the signal-time features available in signals.csv
    (heuristic internals + news structure). No leakage: nothing from the
    future path is a feature.
  - STRICT walk-forward only (train past -> predict future). No CV
    optimism for an economic claim.
  - The decisive test: if we trade ONLY the top-K% by model score, is the
    net RTP positive OUT-OF-TIME, and better than trading everything?
  - Multiple models; honest verdict; multiple-testing acknowledged.

Output: results/ml_signal_triage.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "dense_cache"
RES = ROOT / "results"

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

RULE = "net_tp10_h60"          # pre-committed, fixed a priori
FEATURES = ["heuristic_score", "impact_strength", "llm_confidence",
            "ambiguity_score", "specificity_score", "cosine_score",
            "articles_count", "unique_sources_count"]


def models():
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced",
                                     random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=400,
                                               class_weight="balanced",
                                               random_state=42, n_jobs=-1),
        "GradBoost": GradientBoostingClassifier(random_state=42),
    }


def main() -> None:
    sig = pd.read_csv(CACHE / "signals.csv")
    met = pd.read_csv(CACHE / "dense_metrics_fixed.csv")
    df = met.merge(sig[["id", "bucket"] + FEATURES], on="id", how="left")
    df = df.dropna(subset=[RULE] + FEATURES).sort_values("ts").reset_index(drop=True)
    df["win"] = (df[RULE] > 0).astype(int)

    n = len(df)
    base_rate = df["win"].mean()
    base_net = df[RULE].mean()
    # add one-hot bucket (signal-time, leak-safe)
    X_all = pd.concat([df[FEATURES],
                       pd.get_dummies(df["bucket"], prefix="b")], axis=1)
    y_all = df["win"].to_numpy()
    net = df[RULE].to_numpy()

    print(f"[triage] n={n}  rule={RULE}  base winrate={base_rate:.1%}  "
          f"base net/sig={base_net:+.2f}%  (trade-everything baseline)")

    # strict walk-forward: expanding train, predict the next block
    folds = [(int(n*.5), int(n*.65)), (int(n*.65), int(n*.8)), (int(n*.8), n)]
    out = {"n": n, "rule": RULE, "base_winrate": round(float(base_rate), 4),
           "base_net_rtp_pct": round(float(base_net), 3),
           "models": {}}

    for mname, mk in models().items():
        oof_p = np.full(n, np.nan)
        for a, b in folds:
            if pd.Series(y_all[:a]).nunique() < 2:
                continue
            sc = StandardScaler().fit(X_all.iloc[:a])
            m = mk
            m.fit(sc.transform(X_all.iloc[:a]), y_all[:a])
            oof_p[a:b] = m.predict_proba(sc.transform(X_all.iloc[a:b]))[:, 1]

        mask = ~np.isnan(oof_p)
        yv, pv, nv = y_all[mask], oof_p[mask], net[mask]
        auc = float(roc_auc_score(yv, pv)) if pd.Series(yv).nunique() > 1 else float("nan")

        # economics of selective execution, OUT-OF-TIME (walk-forward preds)
        sel = {}
        for k in (0.10, 0.20, 0.30, 0.50):
            thr = np.quantile(pv, 1 - k)
            take = pv >= thr
            if take.sum() < 10:
                continue
            sel[f"top_{int(k*100)}pct"] = {
                "n_trades": int(take.sum()),
                "winrate": round(float(yv[take].mean()), 3),
                "net_rtp_pct": round(float(nv[take].mean()), 2),
                "vs_trade_all_pp": round(float(nv[take].mean() - nv.mean()), 2),
            }
        out["models"][mname] = {
            "walkforward_auc": round(auc, 3),
            "oot_n": int(mask.sum()),
            "selective": sel,
        }
        print(f"\n  {mname}: walk-forward AUC={auc:.3f}  (out-of-time n={mask.sum()})")
        for k, r in sel.items():
            flag = "  <-- POSITIF" if r["net_rtp_pct"] > 0 else ""
            print(f"    {k:<11} n={r['n_trades']:>3}  winrate={r['winrate']:.0%}"
                  f"  net/sig={r['net_rtp_pct']:+.2f}%"
                  f"  (vs tout {r['vs_trade_all_pp']:+.2f}pp){flag}")

    # honest verdict: any model+top-K with positive OOT net AND beating all?
    wins = []
    for mn, mr in out["models"].items():
        for k, r in mr["selective"].items():
            if r["net_rtp_pct"] > 0 and r["vs_trade_all_pp"] > 0:
                wins.append(f"{mn}/{k} ({r['net_rtp_pct']:+.2f}%)")
    out["verdict"] = {
        "positive_oot_selections": wins,
        "honest": (
            "Aucune sélection ML ne devient nette positive out-of-time : "
            "même en triant, le ML ne sauve pas un edge directionnel "
            "inexistant. Le levier honnête reste la MAGNITUDE, pas la "
            "sélection directionnelle."
            if not wins else
            "Au moins une sélection ML est nette positive out-of-time ET "
            "bat le trade-tout : piste de triage à creuser/valider "
            "(attention multi-tests : " + str(len(wins)) + " combinaisons "
            "positives sur 12 testées)."
        ),
    }
    RES.mkdir(exist_ok=True)
    (RES / "ml_signal_triage.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== VERDICT (ML triage, honnête) ===\n" + out["verdict"]["honest"])


if __name__ == "__main__":
    main()
