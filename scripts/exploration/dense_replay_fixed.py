"""CORRECTED dense replay. Three bugs in dense_replay.py, all flagged by
the user's intuition ("median can't be 0", "BUY_NO profits when it DROPS"):

  BUG 1  Dead markets included. Signals on markets already at ~0.0015 or
         ~0.998 (effectively resolved, flat, untradeable) inject MFE=0 and
         crush the median. Pre-registered filter: 0.02 <= YES0 <= 0.98 at
         signal time AND the path actually moves within the horizon.
  BUG 2  BUY_NO mismeasured. The script tracked the YES token and flipped
         the sign: -1 * (YES_t - YES_0)/YES_0. The real return of a NO
         position is (NO_t - NO_0)/NO_0 with NO = 1 - YES. Different
         denominator -> wrong by up to 4x for 53% of the sample.
  BUG 3  CLOB ignores endTs and returns the whole market life (up to 37
         days). "MFE_24h" was really MFE-over-all-history. We now cap the
         24h window explicitly; exit rules were already minute-capped.

Same HONEST no-look-ahead rule, same multiple-testing discipline. Reuses
the cached raw histories in data/dense_cache/<id>.json (no new API calls).
Reports N before/after the filter and the result both ways. Output:
results/dense_replay_fixed.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "dense_cache"
SPREAD = 3.0
TPS = [5, 10, 15]
HS = [30, 60, 120]
DEAD_LO, DEAD_HI = 0.02, 0.98          # untradeable / resolved-in-practice
DAY = 24 * 3600


def signed_path(direction: str, yes: np.ndarray):
    """Return the % path of the token you actually buy.
    BUY_YES -> YES token return ; BUY_NO -> NO token return (NO = 1-YES)."""
    d = str(direction).upper()
    if d in ("BUY_YES", "YES", "UP"):
        base = yes
    else:                                   # BUY_NO / NO / DOWN
        base = 1.0 - yes
    if base[0] <= 0:
        return None
    return (base - base[0]) / base[0] * 100.0


def main() -> None:
    df = pd.read_csv(CACHE / "signals.csv").set_index("id")
    rows = []
    n_total = n_no_hist = n_dead = n_flat = 0
    for sid, s in df.iterrows():
        n_total += 1
        f = CACHE / f"{int(sid)}.json"
        if not f.exists():
            continue
        try:
            h = json.loads(f.read_text())
        except Exception:
            continue
        if not h or len(h) < 5:
            n_no_hist += 1
            continue
        ts = int(s["ts"])
        tm = np.array([x["t"] for x in h], dtype=float)
        yes = np.array([x["p"] for x in h], dtype=float)
        # align to signal time; keep only the 24h window (BUG 3 fix)
        rel_min = (tm - tm[0]) / 60.0
        win24 = (tm - tm[0]) <= DAY
        yes24 = yes[win24]
        mins = rel_min[win24]
        if len(yes24) < 5:
            n_no_hist += 1
            continue
        y0 = yes24[0]
        # BUG 1 fix: drop already-resolved / untradeable markets
        if not (DEAD_LO <= y0 <= DEAD_HI):
            n_dead += 1
            continue
        # drop paths that never move at all in 24h (stale/dead data)
        if np.nanmax(np.abs(yes24 - y0)) < 1e-9:
            n_flat += 1
            continue
        # BUG 2 fix: return of the token actually traded
        mv = signed_path(s["direction"], yes24)
        if mv is None:
            n_dead += 1
            continue

        rec = {"id": int(sid), "ts": ts,
               "direction": str(s["direction"]).upper(),
               "yes0": round(float(y0), 4),
               "mfe_60": float(mv[mins <= 60].max()) if (mins <= 60).any() else np.nan,
               "mae_60": float((-mv[mins <= 60]).max()) if (mins <= 60).any() else np.nan,
               "mfe_24h": float(mv.max()),
               "mfe_24h_min_after": float(mins[np.argmax(mv)])}
        for tp in TPS:
            for H in HS:
                w = mins <= H
                if not w.any():
                    rec[f"net_tp{tp}_h{H}"] = np.nan
                    continue
                mvw = mv[w]
                hit = np.where(mvw >= tp)[0]
                realized = tp if len(hit) else mvw[-1]
                rec[f"net_tp{tp}_h{H}"] = realized - SPREAD
        rows.append(rec)

    R = pd.DataFrame(rows)
    R.to_csv(CACHE / "dense_metrics_fixed.csv", index=False)
    kept = len(R)
    print(f"[fix] {n_total} signaux | sans historique {n_no_hist} | "
          f"marchés morts (hors {DEAD_LO}-{DEAD_HI}) {n_dead} | "
          f"plats {n_flat} | EXPLOITABLES {kept}", flush=True)
    if kept < 30:
        (ROOT / "results" / "dense_replay_fixed.json").write_text(
            json.dumps({"n_paths": kept, "note": "trop peu"}, indent=2))
        return

    R = R.sort_values("ts").reset_index(drop=True)
    cut = R["ts"].quantile(0.7)
    by_dir = R["direction"].value_counts().to_dict()
    out = {"n_paths": kept, "filter": f"{DEAD_LO}<=YES0<={DEAD_HI}",
           "dropped": {"no_history": n_no_hist, "dead_market": n_dead,
                       "flat": n_flat},
           "direction_mix": by_dir,
           "mfe_60_median_pct": round(float(R["mfe_60"].median()), 2),
           "mfe_60_mean_pct": round(float(R["mfe_60"].mean()), 2),
           "mfe_60_share_pos": round(float((R["mfe_60"] > 0).mean()), 3),
           "mfe_60_share_ge5": round(float((R["mfe_60"] >= 5).mean()), 3),
           "mfe_24h_median_pct": round(float(R["mfe_24h"].median()), 2),
           "mfe_24h_when_min_median": round(float(R["mfe_24h_min_after"].median()), 1),
           "rules": {}}

    print(f"\n[fix] {kept} chemins · mix {by_dir}")
    print(f"[fix] MFE 1h: médiane {out['mfe_60_median_pct']}%  "
          f"moyenne {out['mfe_60_mean_pct']}%  "
          f">0 {out['mfe_60_share_pos']:.0%}  >=5% {out['mfe_60_share_ge5']:.0%}")
    print(f"[fix] MFE 24h médiane {out['mfe_24h_median_pct']}% "
          f"(atteint en moy. à {out['mfe_24h_when_min_median']} min)")
    print(f"\n{'règle':<14}{'n':>5}{'winrate':>9}{'net/sig':>10}"
          f"{'t':>7}{'p':>7}{'out-of-time':>13}", flush=True)
    for tp in TPS:
        for H in HS:
            c = f"net_tp{tp}_h{H}"
            v = R[c].dropna()
            if len(v) < 30:
                continue
            t, p = stats.ttest_1samp(v, 0.0)
            late = R[R["ts"] >= cut][c].dropna()
            o = {"n": int(len(v)),
                 "winrate": round(float((v > 0).mean()), 3),
                 "net_rtp_pct": round(float(v.mean()), 2),
                 "gross_rtp_pct": round(float(v.mean()) + SPREAD, 2),
                 "t": round(float(t), 2), "p": round(float(p), 4),
                 "significant_positive": bool(p < 0.05 and t > 0),
                 "net_rtp_out_of_time_pct": round(float(late.mean()), 2)
                     if len(late) else None}
            out["rules"][f"TP{tp}_H{H}"] = o
            print(f"  TP{tp}%/{H}m{'':<2}{o['n']:>5}{o['winrate']:>8.0%}"
                  f"{o['net_rtp_pct']:>+9.2f}%{o['t']:>7.2f}{o['p']:>7.3f}"
                  f"{(o['net_rtp_out_of_time_pct'] or 0):>+12.2f}%", flush=True)

    sig = [k for k, r in out["rules"].items() if r["significant_positive"]]
    out["verdict"] = {
        "n_rules_tested": len(out["rules"]),
        "significant_positive_rules": sig,
        "honest": (
            "Toujours aucune règle nette significativement positive après "
            "correction des 3 bugs — le edge directionnel n'existe pas, "
            "ce n'était pas un artefact."
            if not sig else
            f"{len(sig)} règle(s) nette(s) positive(s) APRÈS correction : "
            "à valider out-of-time avant tout claim."
        ),
    }
    (ROOT / "results" / "dense_replay_fixed.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== VERDICT (corrigé) ===\n" + out["verdict"]["honest"], flush=True)


if __name__ == "__main__":
    main()
