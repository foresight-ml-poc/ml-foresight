"""Does the MAGNITUDE signal MONETIZE? Honest, no-look-ahead, sign-agnostic.

Direction is dead (7x). Magnitude (~0.60 AUC) is the only live signal.
The user picked: test it to the end -- does "big move predicted" make money,
WITHOUT ever betting on the heuristic's direction.

A straddle is structurally impossible on a binary market (buy YES+NO = pay
~1.00 to get 1.00 back, you just lose the spread). So the only honest,
backtestable, sign-agnostic monetization candidate is OVERREACTION REVERSION:
news-driven markets often overshoot then partially revert. The rule reacts
to the REALIZED move (observable), not to any predicted direction.

PART A -- informational value of the magnitude model (product metric):
  walk-forward; in the predicted-top-decile, what share are real major
  movers (|24h move| > 20%) vs base rate? (lift, out-of-time).

PART B -- the money question (overreaction fade):
  At minute T, observe realized signed move of the YES price since signal.
  If |move| >= m, the market reacted big -> bet on PARTIAL REVERSION
  (enter AGAINST the realized move; this is sign-agnostic -- it goes against
  whatever way the spike went, never uses the heuristic's call). Exit at the
  FIRST minute the fade gains r*|spike| (pre-committed take-profit) else at
  T+H. Return uses the TRUE traded token (NO = 1-YES). Net of 3pp round trip.
  Tested UNCONDITIONAL and CONDITIONED on the magnitude model's top-quantile.
  Pre-registered small grid; full grid + expected false positives reported;
  out-of-time positivity REQUIRED before any positive claim.

Zero new API calls (reuses data/dense_cache/<id>.json).
Output: results/magnitude_monetize.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "dense_cache"
RES = ROOT / "results"

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SPREAD = 3.0          # pp round trip (fade crosses the spread twice)
DAY = 24 * 3600
FEATURES = ["heuristic_score", "impact_strength", "llm_confidence",
            "ambiguity_score", "specificity_score", "cosine_score",
            "articles_count", "unique_sources_count"]
# pre-registered, intentionally small grid
GRID_M = [8.0, 15.0]      # spike trigger: |realized move| since signal (%)
GRID_T = [15, 30]         # observation minute (enter the fade here)
GRID_H = [60, 120]        # max hold after entry (min)
FADE_TP = 0.5             # take profit when fade recoups 50% of the spike


def load_paths():
    sig = pd.read_csv(CACHE / "signals.csv")
    out = {}
    for _, s in sig.iterrows():
        sid = int(s["id"])
        f = CACHE / f"{sid}.json"
        if not f.exists():
            continue
        try:
            h = json.loads(f.read_text())
        except Exception:
            continue
        if not h or len(h) < 5:
            continue
        tm = np.array([x["t"] for x in h], dtype=float)
        yes = np.array([x["p"] for x in h], dtype=float)
        rel = (tm - tm[0]) / 60.0
        keep = (tm - tm[0]) <= DAY
        yes, rel = yes[keep], rel[keep]
        if len(yes) < 5 or not (0.02 <= yes[0] <= 0.98):
            continue
        out[sid] = (rel, yes)
    return sig, out


def fade_trade(rel, yes, m, T, H):
    """Sign-agnostic overreaction fade. Returns net % (of token traded) or None."""
    y0 = yes[0]
    # price at observation minute T (last quote at or before T)
    iT = np.where(rel <= T)[0]
    if len(iT) == 0:
        return None
    iT = iT[-1]
    pT = yes[iT]
    spike = (pT - y0) / y0 * 100.0          # realized YES move since signal
    if abs(spike) < m:
        return None                          # no big reaction -> no trade
    # bet AGAINST the spike (sign-agnostic): spike up -> buy NO; down -> buy YES
    if spike > 0:
        base = 1.0 - yes                     # NO token path
    else:
        base = yes                           # YES token path
    b0 = base[iT]
    if b0 <= 0:
        return None
    win = (rel > T) & (rel <= T + H)
    if not win.any():
        return None
    seg = base[win]
    ret = (seg - b0) / b0 * 100.0
    target = FADE_TP * abs(spike)            # take-profit on the fade
    hit = np.where(ret >= target)[0]
    realized = float(ret[hit[0]]) if len(hit) else float(ret[-1])
    return realized - SPREAD


def main() -> None:
    sig, paths = load_paths()
    keep_ids = set(pd.read_csv(CACHE / "dense_metrics_fixed.csv")["id"])
    feat = sig[sig["id"].isin(keep_ids)].dropna(subset=FEATURES).copy()
    feat = feat[feat["id"].isin(paths)].sort_values("ts").reset_index(drop=True)

    # ---- PART A: informational value of the magnitude model (walk-forward)
    raw = pd.read_csv(CACHE / "signals.csv")
    mag = raw[raw["id"].isin(set(feat["id"]))].copy()
    mag = mag.sort_values("ts").reset_index(drop=True)
    mag["abs24"] = mag["move24"].abs()
    mag["major"] = (mag["abs24"] > 20).astype(int)        # real major mover
    mag["bigmed"] = (mag["abs24"] > mag["abs24"].median()).astype(int)
    Xa = pd.concat([mag[FEATURES],
                    pd.get_dummies(mag["bucket"], prefix="b")], axis=1)
    n = len(mag)
    a = int(n * 0.6)
    partA = {"n": n, "base_rate_major_pct": round(float(mag["major"].mean()) * 100, 2)}
    if mag["bigmed"].iloc[:a].nunique() > 1 and n - a > 30:
        sc = StandardScaler().fit(Xa.iloc[:a])
        gb = GradientBoostingClassifier(random_state=42).fit(
            sc.transform(Xa.iloc[:a]), mag["bigmed"].iloc[:a])
        pv = gb.predict_proba(sc.transform(Xa.iloc[a:]))[:, 1]
        yv_major = mag["major"].iloc[a:].to_numpy()
        yv_big = mag["bigmed"].iloc[a:].to_numpy()
        auc = float(roc_auc_score(yv_big, pv)) if pd.Series(yv_big).nunique() > 1 else None
        order = np.argsort(-pv)
        for k in (0.10, 0.20, 0.30):
            top = order[:max(1, int(len(order) * k))]
            partA[f"top_{int(k*100)}pct_major_rate_pct"] = round(
                float(yv_major[top].mean()) * 100, 2)
        partA["walkforward_auc_bigmove"] = round(auc, 3) if auc else None
        partA["lift_top10"] = (round(partA["top_10pct_major_rate_pct"]
                               / partA["base_rate_major_pct"], 2)
                               if partA["base_rate_major_pct"] > 0 else None)

    # ---- PART B: the money question -- overreaction fade
    # magnitude filter = walk-forward predicted-big top-50% (regime filter)
    mag_score = np.full(n, np.nan)
    if mag["bigmed"].iloc[:a].nunique() > 1:
        sc = StandardScaler().fit(Xa.iloc[:a])
        gb = GradientBoostingClassifier(random_state=42).fit(
            sc.transform(Xa.iloc[:a]), mag["bigmed"].iloc[:a])
        mag_score[a:] = gb.predict_proba(sc.transform(Xa.iloc[a:]))[:, 1]
    id2score = dict(zip(mag["id"], mag_score))
    id2ts = dict(zip(feat["id"], feat["ts"]))

    cut_ts = feat["ts"].quantile(0.7)
    rules = {}
    for m in GRID_M:
        for T in GRID_T:
            for H in GRID_H:
                rows = []
                for sid in feat["id"]:
                    r = fade_trade(*paths[sid], m, T, H)
                    if r is None:
                        continue
                    rows.append((sid, id2ts[sid], r, id2score.get(sid, np.nan)))
                if len(rows) < 25:
                    continue
                D = pd.DataFrame(rows, columns=["id", "ts", "net", "mscore"])
                v = D["net"]
                t, p = stats.ttest_1samp(v, 0.0)
                late = D[D["ts"] >= cut_ts]["net"]
                # conditioned on magnitude regime (predicted-big half), OOT only
                oot = D[~D["mscore"].isna()]
                cond = oot[oot["mscore"] >= oot["mscore"].median()]["net"] \
                    if len(oot) else pd.Series(dtype=float)
                rules[f"m{int(m)}_T{T}_H{H}"] = {
                    "n_trades": int(len(v)),
                    "winrate": round(float((v > 0).mean()), 3),
                    "net_pct": round(float(v.mean()), 2),
                    "gross_pct": round(float(v.mean()) + SPREAD, 2),
                    "t": round(float(t), 2), "p": round(float(p), 4),
                    "net_out_of_time_pct": round(float(late.mean()), 2)
                        if len(late) else None,
                    "net_mag_filtered_oot_pct": round(float(cond.mean()), 2)
                        if len(cond) else None,
                    "significant_positive": bool(p < 0.05 and t > 0),
                }

    n_tests = len(rules)
    winners = [k for k, r in rules.items()
               if r["significant_positive"]
               and (r["net_out_of_time_pct"] or -9) > 0]
    out = {
        "part_A_informational": partA,
        "part_B_overreaction_fade": {
            "spread_pp_roundtrip": SPREAD,
            "grid": {"m": GRID_M, "T": GRID_T, "H": GRID_H, "fade_tp": FADE_TP},
            "n_rules_tested": n_tests,
            "expected_false_pos_at_p05": round(n_tests * 0.05, 1),
            "rules": rules,
            "positive_and_oot_confirmed": winners,
        },
        "verdict": (
            "L'amplitude n'a PAS de monétisation directe honnête sur "
            "Polymarket : le fade d'overréaction ne survit pas net de spread "
            "out-of-time. Sa valeur est INFORMATIONNELLE (priorisation : "
            f"lift x{partA.get('lift_top10', '?')} sur les gros movers), "
            "pas un edge de trading autonome."
            if not winners else
            f"{len(winners)} règle(s) de fade nette(s) positive(s) ET "
            "confirmée(s) out-of-time (sur "
            f"{n_tests} testées, ~{round(n_tests*0.05,1)} faux positifs "
            "attendus) : piste de monétisation volatilité à valider en live "
            "avant tout claim."
        ),
    }
    RES.mkdir(exist_ok=True)
    (RES / "magnitude_monetize.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=str))

    print("=== PART A — valeur informationnelle de l'amplitude (out-of-time) ===")
    print(f"  base rate gros mover (|24h|>20%) : {partA['base_rate_major_pct']}%")
    for k in ("top_10pct_major_rate_pct", "top_20pct_major_rate_pct",
              "top_30pct_major_rate_pct"):
        if k in partA:
            print(f"  {k:<32} {partA[k]}%")
    print(f"  AUC walk-forward (gros move)      : {partA.get('walkforward_auc_bigmove')}")
    print(f"  LIFT top-10%                      : x{partA.get('lift_top10')}")
    print("\n=== PART B — fade d'overréaction (la question argent) ===")
    print(f"{'règle':<16}{'n':>5}{'winrate':>9}{'net':>9}{'brut':>8}"
          f"{'t':>7}{'p':>7}{'oot':>8}{'oot+filtre':>12}")
    for k, r in rules.items():
        flag = "  <== +" if (r["significant_positive"]
                             and (r["net_out_of_time_pct"] or -9) > 0) else ""
        print(f"{k:<16}{r['n_trades']:>5}{r['winrate']:>8.0%}"
              f"{r['net_pct']:>+8.2f}%{r['gross_pct']:>+7.2f}%{r['t']:>7.2f}"
              f"{r['p']:>7.3f}{(r['net_out_of_time_pct'] or 0):>+7.2f}%"
              f"{(r['net_mag_filtered_oot_pct'] or 0):>+11.2f}%{flag}")
    print(f"\n(tests={n_tests} · faux positifs attendus p<0.05 ≈ "
          f"{round(n_tests*0.05,1)})")
    print("\n=== VERDICT ===\n" + out["verdict"])


if __name__ == "__main__":
    main()
