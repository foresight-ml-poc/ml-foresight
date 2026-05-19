"""Honest time-based backtest on the last N days of signals.

Methodology (this also fixes the earlier 'no TimeSeriesSplit' limitation):
  1. cutoff = max(created_at) - N days
  2. train  = signals strictly BEFORE cutoff   (model never sees the test window)
  3. test   = signals in the last N days        (true out-of-sample)
  4. Fit the scaler on train only, train the best model type on train.
  5. On the test window, compare two strategies:
       A. "Foresight actuel"  — act on every emitted signal (status quo)
       B. "Filtre ML"         — act only on signals the model is confident about
  6. Metrics per strategy:
       - winrate   = mean(direction_correct) on acted signals
       - RTP proxy = mean 24h P&L of a unit stake taken in the predicted
                     direction (signed move_t24h_pct, fees excluded)
       - cumulative return curve over the window

Outputs:
  results/backtest_last3d.json
  plots/backtest_last3d.png

Usage:
    python scripts/backtest_last3d.py [--days 3] [--proba 0.5]
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    HEURISTIC_THRESHOLD, MODELS_DIR, PLOTS_DIR, RESULTS_DIR, SEED, TARGET_COLUMN,
)
from data import _clean, _feature_engineer  # noqa: E402

MINT, LOSS, GREY = "#0BE0A6", "#f76d6d", "#5c6878"


def _latest_csv() -> Path:
    csvs = sorted(g for g in glob.glob(str(PROJECT_ROOT / "data" / "raw" / "signals_export_*.csv"))
                  if "sample" not in g)
    if not csvs:
        raise SystemExit("No export CSV found. Run export_from_foresight.py.")
    return Path(csvs[-1])


def _signed_return_pct(df: pd.DataFrame) -> np.ndarray:
    """24h P&L (%) of a unit stake taken in the predicted direction.

    BUY_YES profits when the YES price rises (+move_t24h_pct).
    BUY_NO  profits when the YES price falls (-move_t24h_pct).
    Proxy, fees excluded.
    """
    sign = np.where(
        df["direction"].astype(str).str.upper().str.contains("YES|UP"), 1.0, -1.0
    )
    return sign * df["move_t24h_pct"].to_numpy(dtype=float)


def _strategy_stats(sub: pd.DataFrame) -> dict:
    if len(sub) == 0:
        return {"n": 0, "winrate": None, "mean_return_pct": None,
                "total_return_pct": None}
    ret = _signed_return_pct(sub)
    return {
        "n": int(len(sub)),
        "winrate": float(sub[TARGET_COLUMN].mean()),
        "mean_return_pct": float(np.mean(ret)),
        "total_return_pct": float(np.sum(ret)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--proba", type=float, default=0.5,
                    help="ML probability threshold to act on a signal")
    args = ap.parse_args()

    csv = _latest_csv()
    raw = pd.read_csv(csv)
    raw["created_at"] = pd.to_datetime(raw["created_at"], errors="coerce", utc=True)
    raw = raw.dropna(subset=["created_at"])

    cutoff = raw["created_at"].max() - pd.Timedelta(days=args.days)
    train_raw = raw[raw["created_at"] < cutoff].copy()
    test_raw = raw[raw["created_at"] >= cutoff].copy()
    print(f"[backtest] cutoff = {cutoff}")
    print(f"[backtest] train = {len(train_raw)} signals (avant) | "
          f"test = {len(test_raw)} signals (3 derniers jours)")

    if len(test_raw) < 15:
        print(f"[backtest] WARNING: only {len(test_raw)} signals in the window — "
              f"results will be noisy.")

    # Build features with the same pipeline (allowlist => leak-safe)
    Xtr, ytr = _feature_engineer(_clean(train_raw))
    test_clean = _clean(test_raw)
    Xte, yte = _feature_engineer(test_clean)

    # Align columns (a bucket dummy may be missing in one split)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # Train the same family as the current best model
    card = json.loads((MODELS_DIR / "model_card.json").read_text())
    best_type = card.get("best_model_type", "random_forest")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    if best_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=SEED)
    else:
        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=SEED, n_jobs=-1)
    model.fit(Xtr_s, ytr)
    proba = model.predict_proba(Xte_s)[:, 1]

    # Attach predictions back to the cleaned test rows
    test_clean = test_clean.reset_index(drop=True)
    test_clean["ml_proba"] = proba

    # Strategy A — Foresight actuel : on agit sur tous les signaux émis
    stratA = _strategy_stats(test_clean)
    # (variante : seulement heuristic_score > seuil)
    stratA_thr = _strategy_stats(
        test_clean[test_clean["heuristic_score"] > HEURISTIC_THRESHOLD])
    # Strategy B — Filtre ML : on agit seulement si proba >= seuil
    stratB = _strategy_stats(test_clean[test_clean["ml_proba"] >= args.proba])
    # Strategy B' — top tercile par proba ML (sélectif)
    if len(test_clean) >= 6:
        top_thr = test_clean["ml_proba"].quantile(2 / 3)
        stratB_top = _strategy_stats(
            test_clean[test_clean["ml_proba"] >= top_thr])
    else:
        stratB_top = {"n": 0, "winrate": None, "mean_return_pct": None,
                      "total_return_pct": None}

    summary = {
        "window_days": args.days,
        "cutoff": str(cutoff),
        "n_train": int(len(train_raw)),
        "n_test_window": int(len(test_clean)),
        "ml_model": best_type,
        "ml_proba_threshold": args.proba,
        "strategies": {
            "foresight_all_emitted": stratA,
            "foresight_score_gt_threshold": stratA_thr,
            "ml_filter_proba_05": stratB,
            "ml_filter_top_tercile": stratB_top,
        },
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "backtest_last3d.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["strategies"], indent=2))

    # ---- Cumulative return plot ----
    tc = test_clean.sort_values("created_at").reset_index(drop=True)
    tc["ret_all"] = _signed_return_pct(tc)
    tc["cum_all"] = tc["ret_all"].cumsum()
    ml_mask = tc["ml_proba"] >= args.proba
    tc["ret_ml"] = np.where(ml_mask, tc["ret_all"], 0.0)  # flat when ML skips
    tc["cum_ml"] = tc["ret_ml"].cumsum()

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0b0f17")
    ax.set_facecolor("#11161f")
    ax.plot(range(len(tc)), tc["cum_all"], color=GREY, lw=2,
            label=f"Foresight actuel — agir sur tout (n={stratA['n']}, "
                  f"winrate {stratA['winrate']:.0%})")
    ax.plot(range(len(tc)), tc["cum_ml"], color=MINT, lw=2.5,
            label=f"Filtre ML — agir si proba≥{args.proba} (n={stratB['n']}, "
                  f"winrate {stratB['winrate']:.0%})" if stratB["n"] else "Filtre ML")
    ax.axhline(0, color=GREY, ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("signaux des 3 derniers jours (ordre chronologique)")
    ax.set_ylabel("rendement 24h cumulé (%, hors frais)")
    ax.set_title("Backtest 3 derniers jours — rendement cumulé par stratégie",
                 color="#e7edf5")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(color="#1a212d")
    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / "backtest_last3d.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[backtest] wrote {out}")


if __name__ == "__main__":
    main()
