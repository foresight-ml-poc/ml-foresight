"""Exhaustive, honest search for ANY ML edge.

The user's question: 'is there a single honest way the ML wins?'
We test, with proper statistics (t-test on per-signal return, time-based
out-of-sample), every reasonable angle and report the truth — including
multiple-testing discipline (if we test K angles, ~K*0.05 false positives
are expected at p<0.05, so a lone p<0.05 among many is NOT a real win).

Angles:
  T1. Alternative horizons: direction-correct at T+5m / T+15m / T+1h / T+24h
  T2. Alternative target: outcome_label (final market resolution)
  S1. ML-selective strategy at several proba thresholds (gross + net of 3pp)
  S2. Subgroups: per bucket, per direction, high vs low liquidity
  S3. Heuristic+ML weighted blend vs heuristic alone
All ML evaluated via expanding-window walk-forward (no leakage).

Output: results/exhaustive_search.json + console table.
"""

from __future__ import annotations

import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import HEURISTIC_THRESHOLD, RESULTS_DIR, SEED  # noqa: E402
from data import _clean, _feature_engineer  # noqa: E402

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

SPREAD_PP = 3.0  # realistic Polymarket round-trip cost, percentage points


def _csv() -> Path:
    g = sorted(c for c in glob.glob(str(PROJECT_ROOT / "data/raw/signals_export_*.csv"))
               if "sample" not in c)
    return Path(g[-1])


def _signed(df, move_col) -> np.ndarray:
    s = np.where(df["direction"].astype(str).str.upper().str.contains("YES|UP"), 1.0, -1.0)
    return s * df[move_col].to_numpy(dtype=float)


def _tstat(x: np.ndarray):
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return None, None
    t, p = stats.ttest_1samp(x, 0.0)
    return float(t), float(p)


def _walkforward_proba(X, y, clean, steps=6, start_frac=0.4):
    """Expanding-window OOS proba for every test row (concatenated)."""
    n = len(X)
    idxs = np.linspace(int(n * start_frac), n, steps + 1).astype(int)
    proba = np.full(n, np.nan)
    for i in range(len(idxs) - 1):
        a, b = idxs[i], idxs[i + 1]
        if a < 40 or b - a < 8:
            continue
        if y.iloc[:a].nunique() < 2:
            continue
        sc = StandardScaler().fit(X.iloc[:a])
        m = ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=SEED, n_jobs=-1)
        m.fit(sc.transform(X.iloc[:a]), y.iloc[:a])
        proba[a:b] = m.predict_proba(sc.transform(X.iloc[a:b]))[:, 1]
    return proba


def main() -> None:
    raw = pd.read_csv(_csv())
    raw["created_at"] = pd.to_datetime(raw["created_at"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["created_at"]).sort_values("created_at").reset_index(drop=True)

    results = {}
    K = 0  # count of hypothesis tests for multiple-comparison context

    # ---- T1/T2: alternative targets, ML discriminative power (OOS ROC-AUC) ----
    targets = {
        "dir_t5min": ("move_t5min_pct", lambda d: (
            (d["direction"].str.upper().str.contains("YES|UP") & (d["move_t5min_pct"] > 0)) |
            (~d["direction"].str.upper().str.contains("YES|UP") & (d["move_t5min_pct"] < 0))).astype(int)),
        "dir_t15min": ("move_t15min_pct", lambda d: (
            (d["direction"].str.upper().str.contains("YES|UP") & (d["move_t15min_pct"] > 0)) |
            (~d["direction"].str.upper().str.contains("YES|UP") & (d["move_t15min_pct"] < 0))).astype(int)),
        "dir_t1h": ("move_t1h_pct", lambda d: (
            (d["direction"].str.upper().str.contains("YES|UP") & (d["move_t1h_pct"] > 0)) |
            (~d["direction"].str.upper().str.contains("YES|UP") & (d["move_t1h_pct"] < 0))).astype(int)),
        "dir_t24h": ("move_t24h_pct", lambda d: (
            (d["direction"].str.upper().str.contains("YES|UP") & (d["move_t24h_pct"] > 0)) |
            (~d["direction"].str.upper().str.contains("YES|UP") & (d["move_t24h_pct"] < 0))).astype(int)),
        "outcome_label": ("move_t24h_pct", lambda d: d["outcome_label"]),
    }
    horizon = {}
    for tname, (mcol, fn) in targets.items():
        d = raw.dropna(subset=[mcol]).copy()
        d["_y"] = fn(d)
        d = d.dropna(subset=["_y"])
        d["_y"] = d["_y"].astype(int)
        if d["_y"].nunique() < 2 or len(d) < 80:
            horizon[tname] = {"n": int(len(d)), "note": "trop peu / une classe"}
            continue
        dclean = _clean(d.assign(direction_correct=d["_y"])).reset_index(drop=True)
        X, yv = _feature_engineer(dclean)
        X = X.reset_index(drop=True)
        yv = yv.reset_index(drop=True).astype(int)
        p = _walkforward_proba(X, yv, dclean)
        mask = ~np.isnan(p)
        K += 1
        if mask.sum() > 30 and yv[mask].nunique() == 2:
            auc = roc_auc_score(yv[mask].to_numpy(), p[mask])
        else:
            auc = None
        horizon[tname] = {"n": int(len(d)), "oos_n": int(mask.sum()),
                          "oos_roc_auc": None if auc is None else round(auc, 3)}
        print(f"[T] {tname:<14} n={len(d):<4} OOS ROC-AUC="
              f"{'NA' if auc is None else round(auc,3)}")
    results["targets_oos_auc"] = horizon

    # ---- S1: ML-selective strategy on T+24h, thresholds, gross + net ----
    d = raw.dropna(subset=["move_t24h_pct"]).copy()
    d["_y"] = ((d["direction"].str.upper().str.contains("YES|UP") & (d["move_t24h_pct"] > 0)) |
               (~d["direction"].str.upper().str.contains("YES|UP") & (d["move_t24h_pct"] < 0))).astype(int)
    dd = _clean(d.assign(direction_correct=d["_y"])).reset_index(drop=True)
    X, yv = _feature_engineer(dd)
    X = X.reset_index(drop=True)
    yv = yv.reset_index(drop=True).astype(int)
    dd["_y"] = yv
    p = _walkforward_proba(X, yv, dd)
    dd["p"] = p
    oos = dd[~dd["p"].isna()].copy()
    oos["ret_gross"] = _signed(oos, "move_t24h_pct")
    oos["ret_net"] = oos["ret_gross"] - SPREAD_PP
    sel = {}
    for q, lab in [(0.0, "tous (proba≥0)"), (0.5, "proba≥0.5"),
                   (0.6, "proba≥0.6"), (0.7, "proba≥0.7"),
                   ("top10", "top 10% proba"), ("top5", "top 5% proba")]:
        if q == "top10":
            sub = oos[oos["p"] >= oos["p"].quantile(0.90)]
        elif q == "top5":
            sub = oos[oos["p"] >= oos["p"].quantile(0.95)]
        else:
            sub = oos[oos["p"] >= q]
        if len(sub) < 10:
            sel[lab] = {"n": int(len(sub)), "note": "n<10"}
            continue
        tg, pg = _tstat(sub["ret_gross"].to_numpy())
        tn, pn = _tstat(sub["ret_net"].to_numpy())
        K += 2
        sel[lab] = {
            "n": int(len(sub)),
            "winrate": round(float(sub["_y"].mean()), 3),
            "rtp_gross_pct": round(float(sub["ret_gross"].mean()), 2),
            "t_gross": round(tg, 2), "p_gross": round(pg, 3),
            "rtp_net_pct": round(float(sub["ret_net"].mean()), 2),
            "t_net": round(tn, 2), "p_net": round(pn, 3),
            "significant_gross": bool(pg < 0.05 and tg > 0),
            "significant_net_positive": bool(pn < 0.05 and tn > 0),
        }
        print(f"[S1] {lab:<16} n={sel[lab]['n']:<4} wr={sel[lab]['winrate']:.0%} "
              f"gross={sel[lab]['rtp_gross_pct']:+.1f}%(t={tg:+.2f}) "
              f"net={sel[lab]['rtp_net_pct']:+.1f}%(t={tn:+.2f})")
    results["ml_selective_t24h"] = sel

    # ---- S2: subgroups (gross edge per bucket / direction / liquidity) ----
    sub_res = {}
    for col, groups in [
        ("bucket", oos["bucket"].dropna().unique().tolist()),
        ("direction", ["BUY_YES", "BUY_NO"]),
    ]:
        for gname in groups:
            g = oos[oos[col].astype(str).str.upper().str.contains(str(gname).upper(), na=False)] \
                if col == "direction" else oos[oos[col] == gname]
            g = g[g["p"] >= 0.5]
            if len(g) < 15:
                continue
            tg, pg = _tstat(g["ret_gross"].to_numpy())
            K += 1
            sub_res[f"{col}={gname} & ml≥0.5"] = {
                "n": int(len(g)), "winrate": round(float(g["_y"].mean()), 3),
                "rtp_gross_pct": round(float(g["ret_gross"].mean()), 2),
                "t": round(tg, 2), "p": round(pg, 3),
                "significant": bool(pg < 0.05 and tg > 0)}
    # high liquidity
    if "market_liquidity" in oos.columns:
        hi = oos[(oos["market_liquidity"] > oos["market_liquidity"].median()) & (oos["p"] >= 0.5)]
        if len(hi) >= 15:
            tg, pg = _tstat(hi["ret_gross"].to_numpy())
            K += 1
            sub_res["haute liquidité & ml≥0.5"] = {
                "n": int(len(hi)), "winrate": round(float(hi["_y"].mean()), 3),
                "rtp_gross_pct": round(float(hi["ret_gross"].mean()), 2),
                "t": round(tg, 2), "p": round(pg, 3),
                "significant": bool(pg < 0.05 and tg > 0)}
    results["subgroups"] = sub_res
    for k, v in sub_res.items():
        print(f"[S2] {k:<32} n={v['n']:<3} wr={v['winrate']:.0%} "
              f"gross={v['rtp_gross_pct']:+.1f}%(p={v['p']}) sig={v['significant']}")

    # ---- S3: heuristic+ML blend vs heuristic alone (gross) ----
    blend = {}
    H = oos["heuristic_score"] / 100.0
    for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
        score = (1 - w) * H + w * oos["p"]
        thr = score.quantile(0.5)
        sub = oos[score >= thr]
        if len(sub) < 15:
            continue
        tg, pg = _tstat(sub["ret_gross"].to_numpy())
        K += 1
        blend[f"w_ml={w}"] = {
            "n": int(len(sub)), "winrate": round(float(sub["_y"].mean()), 3),
            "rtp_gross_pct": round(float(sub["ret_gross"].mean()), 2),
            "t": round(tg, 2), "p": round(pg, 3)}
        print(f"[S3] blend w_ml={w:<4} n={blend[f'w_ml={w}']['n']:<3} "
              f"wr={blend[f'w_ml={w}']['winrate']:.0%} "
              f"gross={blend[f'w_ml={w}']['rtp_gross_pct']:+.1f}%(t={tg:+.2f})")
    results["heuristic_ml_blend"] = blend

    # ---- verdict with multiple-testing context ----
    sig_gross = []
    for sec in ["ml_selective_t24h", "subgroups"]:
        for k, v in results[sec].items():
            if v.get("significant") or v.get("significant_gross"):
                sig_gross.append(f"{sec}:{k}")
    results["multiple_testing"] = {
        "n_hypotheses_tested": K,
        "expected_false_positives_at_p05": round(K * 0.05, 1),
        "significant_findings": sig_gross,
        "verdict": (
            "AUCUN edge net-de-coûts significatif. "
            f"{len(sig_gross)} test(s) gross significatif(s) sur ~{K} "
            f"(attendu par hasard ≈ {round(K*0.05,1)}) => "
            "non distinguable du bruit de multiple-testing."
            if not any(results['ml_selective_t24h'].get(k, {}).get('significant_net_positive')
                       for k in results['ml_selective_t24h'])
            else "Un edge NET significatif trouvé — à investiguer."
        ),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "exhaustive_search.json").write_text(json.dumps(results, indent=2))
    print("\n=== VERDICT ===")
    print(results["multiple_testing"]["verdict"])


if __name__ == "__main__":
    main()
