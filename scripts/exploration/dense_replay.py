"""Reconstruct minute-by-minute price paths from Polymarket and run an
HONEST, no-look-ahead exit-rule backtest.

The user's (correct) point: a signal's value lives somewhere in the path,
not at a fixed T+24h snapshot. We only stored 4 coarse points. Polymarket
keeps minute history; we have the condition_id for every signal.

Pipeline (cached + resumable):
  1. condition_id -> Gamma API -> clobTokenIds (YES token)
  2. CLOB prices-history fidelity=1 over [ts, ts+24h] -> dense path
  3. signed path = move in the signal's predicted direction
  4. NO-LOOK-AHEAD rule, pre-committed per (TP, H):
       enter at signal price; exit at the FIRST minute signed return >= TP
       within H minutes; else exit at minute H. net = realized - SPREAD.
  5. winrate + mean net RTP + t-stat ; split early/late (out-of-time)

Raw histories cached to data/dense_cache/<id>.json so reruns are free.
Output: results/dense_replay.json
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "dense_cache"
CACHE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}
SPREAD = 3.0  # pp round-trip cost
TPS = [5, 10, 15]
HS = [30, 60, 120]


def _get(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(1.5 * (k + 1)); continue
            return None
        except Exception:
            time.sleep(1.0); continue
    return None


def dense_path(cid: str, token_cache: dict, sid: int, ts: int):
    """Return signed-ready (times, prices) or None. Cached per signal."""
    cf = CACHE / f"{sid}.json"
    if cf.exists():
        try:
            h = json.loads(cf.read_text())
            return h if h else None
        except Exception:
            pass
    if cid not in token_cache:
        g = _get("https://gamma-api.polymarket.com/markets?condition_ids=" + cid)
        tok = None
        if g and isinstance(g, list) and g:
            try:
                tok = json.loads(g[0].get("clobTokenIds") or "[]")
            except Exception:
                tok = None
        token_cache[cid] = tok[0] if tok else None
    yes = token_cache[cid]
    if not yes:
        cf.write_text("[]"); return None
    u = (f"https://clob.polymarket.com/prices-history?market={yes}"
         f"&startTs={ts}&endTs={ts + 24 * 3600}&fidelity=1")
    h = _get(u)
    hist = (h or {}).get("history") if isinstance(h, dict) else None
    cf.write_text(json.dumps(hist or []))
    time.sleep(0.12)
    return hist or None


def main() -> None:
    df = pd.read_csv(CACHE / "signals.csv")
    n = len(df)
    print(f"[dense] {n} signaux à rejouer", flush=True)
    token_cache: dict = {}
    rows = []
    done = 0
    for _, s in df.iterrows():
        ts = int(s["ts"])
        hist = dense_path(s["market_id"], token_cache, int(s["id"]), ts)
        done += 1
        if done % 50 == 0:
            print(f"[dense] {done}/{n} (cache {len(token_cache)} marchés)", flush=True)
        if not hist or len(hist) < 5:
            continue
        pr = np.array([x["p"] for x in hist], dtype=float)
        tm = np.array([x["t"] for x in hist], dtype=float)
        if pr[0] <= 0:
            continue
        sign = 1.0 if str(s["direction"]).upper() in (
            "BUY_YES", "YES", "UP") else -1.0
        mv = sign * (pr - pr[0]) / pr[0] * 100.0      # signed % path
        mins = (tm - tm[0]) / 60.0
        rec = {"id": int(s["id"]), "ts": ts, "bucket": s["bucket"],
               "heuristic_score": s["heuristic_score"],
               "impact_strength": s["impact_strength"],
               "llm_confidence": s["llm_confidence"],
               "ambiguity_score": s["ambiguity_score"],
               "specificity_score": s["specificity_score"],
               "cosine_score": s["cosine_score"],
               "articles_count": s["articles_count"],
               "unique_sources_count": s["unique_sources_count"],
               "mfe_60": float(mv[mins <= 60].max()) if (mins <= 60).any() else np.nan,
               "mfe_24h": float(mv.max())}
        # no-look-ahead pre-committed exit rules
        for tp in TPS:
            for H in HS:
                w = mins <= H
                if not w.any():
                    rec[f"net_tp{tp}_h{H}"] = np.nan
                    continue
                mvw = mv[w]
                hit = np.where(mvw >= tp)[0]
                realized = tp if len(hit) else mvw[-1]  # exit at TP or at H
                rec[f"net_tp{tp}_h{H}"] = realized - SPREAD
        rows.append(rec)

    R = pd.DataFrame(rows)
    R.to_csv(CACHE / "dense_metrics.csv", index=False)
    if len(R) < 30:
        print(f"[dense] seulement {len(R)} chemins exploitables — stop", flush=True)
        (ROOT / "results" / "dense_replay.json").write_text(
            json.dumps({"n_paths": int(len(R)),
                        "note": "trop peu de chemins denses récupérés"}, indent=2))
        return

    R = R.sort_values("ts").reset_index(drop=True)
    cut = R["ts"].quantile(0.7)
    out = {"n_paths": int(len(R)),
           "mfe_60_median_pct": round(float(R["mfe_60"].median()), 2),
           "mfe_24h_median_pct": round(float(R["mfe_24h"].median()), 2),
           "rules": {}}
    print(f"\n[dense] {len(R)} chemins denses · MFE médian 1h="
          f"{R['mfe_60'].median():.1f}% · 24h={R['mfe_24h'].median():.1f}%",
          flush=True)
    print(f"\n{'règle (TP/H)':<16}{'n':>5}{'winrate':>9}{'net/sig':>10}"
          f"{'t':>7}{'p':>7}{'net out-of-time':>16}", flush=True)
    for tp in TPS:
        for H in HS:
            c = f"net_tp{tp}_h{H}"
            v = R[c].dropna()
            if len(v) < 30:
                continue
            t, p = stats.ttest_1samp(v, 0.0)
            late = R[R["ts"] >= cut][c].dropna()
            o = {
                "n": int(len(v)),
                "winrate": round(float((v > 0).mean()), 3),
                "net_rtp_pct": round(float(v.mean()), 2),
                "t": round(float(t), 2), "p": round(float(p), 3),
                "significant_positive": bool(p < 0.05 and t > 0),
                "net_rtp_out_of_time_pct": round(float(late.mean()), 2)
                    if len(late) else None,
            }
            out["rules"][f"TP{tp}_H{H}"] = o
            print(f"  TP{tp}% / {H}min{'':<3}{o['n']:>5}{o['winrate']:>8.0%}"
                  f"{o['net_rtp_pct']:>+9.2f}%{o['t']:>7.2f}{o['p']:>7.3f}"
                  f"{(o['net_rtp_out_of_time_pct'] or 0):>+15.2f}%", flush=True)

    sig = [k for k, r in out["rules"].items() if r["significant_positive"]]
    n_tests = len(out["rules"])
    out["verdict"] = {
        "n_rules_tested": n_tests,
        "expected_false_pos_at_p05": round(n_tests * 0.05, 1),
        "significant_positive_rules": sig,
        "honest": (
            "Aucune règle nette significativement positive — pas d'edge "
            "exploitable même avec le chemin dense."
            if not sig else
            f"{len(sig)} règle(s) nette(s) significativement positive(s) : "
            "à valider out-of-time (cf. colonne) avant tout claim."
        ),
    }
    (ROOT / "results" / "dense_replay.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("\n=== VERDICT ===\n" + out["verdict"]["honest"], flush=True)


if __name__ == "__main__":
    main()
