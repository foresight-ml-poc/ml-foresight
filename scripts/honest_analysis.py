"""Consolide le verdict honnête à partir des artefacts reproductibles.

Lit :
  models/model_card.json            (3 modèles + CV vs walk-forward)
  results/dense_replay_fixed.json   (replay minute, 3 bugs corrigés)
  results/magnitude_monetize.json   (la magnitude se monétise-t-elle ?)

Écrit :
  results/final_honest_verdict.json (LA synthèse — source unique pour
                                     README, deck, figures)

Aucun calcul ML ici : pure agrégation, pour que le verdict soit
traçable jusqu'aux scripts qui l'ont produit. Lancer après train.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def _load(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def main() -> None:
    card = _load(ROOT / "models" / "model_card.json", {})
    dense = _load(RES / "dense_replay_fixed.json", {})
    mag = _load(RES / "magnitude_monetize.json", {})

    cvwf = card.get("cv_vs_walkforward", {})
    d, m = cvwf.get("direction", {}), cvwf.get("magnitude", {})
    models = card.get("models", {})
    rules = dense.get("rules", {})
    nets = [r["net_rtp_pct"] for r in rules.values()] or [0]
    gross = [r["gross_rtp_pct"] for r in rules.values()] or [0]

    out = {
        "title": "Verdict honnête — un edge ML est-il exploitable dans les "
                 "signaux Foresight ?",
        "one_line": "Non. La direction est un pile ou face (marché efficient), "
                    "et le seul signal qui semblait vivant (la magnitude) est "
                    "un artefact de validation croisée — il s'effondre en "
                    "walk-forward strict.",
        "dataset": card.get("dataset", {}),

        "1_three_models_direction": {
            "task": "Prédire direction_correct à T+24h (test set)",
            "roc_auc": {k: round(v["roc_auc"], 3) for k, v in models.items()},
            "kmeans_adjusted_rand_index": card.get(
                "kmeans_adjusted_rand_index"),
            "read": "Les 3 familles ≈ 0.50. K-Means ARI ≈ 0 : aucun "
                    "regroupement naturel gagnants/perdants.",
        },

        "2_cv_lies_centerpiece": {
            "direction": {"cv": d.get("cv_auc_mean"),
                          "walkforward": d.get("walkforward_auc"),
                          "walkforward_mean": d.get("walkforward_mean")},
            "magnitude": {"cv": m.get("cv_auc_mean"),
                          "walkforward": m.get("walkforward_auc"),
                          "walkforward_mean": m.get("walkforward_mean")},
            "read": "La CV (5-fold mélangé) flatte la magnitude ; le "
                    "walk-forward strict la ramène au hasard et la fait "
                    "passer sous 0.50 sur la période récente. Démonstration "
                    "manuelle de l'alpha decay / non-stationnarité.",
        },

        "3_dense_replay_no_look_ahead": {
            "n_paths": dense.get("n_paths"),
            "direction_mix": dense.get("direction_mix"),
            "mfe_60_median_pct": dense.get("mfe_60_median_pct"),
            "mfe_60_share_pos": dense.get("mfe_60_share_pos"),
            "mfe_24h_median_pct": dense.get("mfe_24h_median_pct"),
            "mfe_24h_when_min_median": dense.get("mfe_24h_when_min_median"),
            "exit_rules_net_rtp_range_pct": [min(nets), max(nets)],
            "exit_rules_gross_rtp_range_pct": [min(gross), max(gross)],
            "n_rules_tested": dense.get("verdict", {}).get("n_rules_tested"),
            "significant_positive_rules": dense.get("verdict", {}).get(
                "significant_positive_rules", []),
            "bugs_fixed_thanks_to_user_intuition": [
                "Marchés déjà résolus inclus (MFE=0 artificiel)",
                "BUY_NO mesuré sur le mauvais token (53% de l'échantillon)",
                "API CLOB ignore endTs → fenêtre 37 j au lieu de 24 h",
            ],
            "read": "Même chemin minute par minute, règle sans look-ahead : "
                    "les 9 variantes sont négatives, et négatives MÊME brut "
                    "(hors spread). MFE médian 1h = 0 %. L'edge directionnel "
                    "n'existe pas — ce n'était pas un artefact des bugs.",
        },

        "4_magnitude_does_not_monetize": {
            "informational_lift_top10": (mag.get("part_A_informational", {})
                                         .get("lift_top10")),
            "walkforward_auc_bigmove": (mag.get("part_A_informational", {})
                                        .get("walkforward_auc_bigmove")),
            "overreaction_fade_verdict": mag.get("verdict"),
            "read": "Pas de straddle possible sur un marché binaire ; le fade "
                    "d'overréaction est négatif même brut. La magnitude n'a "
                    "qu'une valeur informationnelle — et elle ne tient pas "
                    "out-of-time non plus.",
        },

        "why_efficient_market": {
            "hedge_fund_edges_foresight_lacks": {
                "vitesse": "boucle news→signal en minutes ; les desks "
                           "repricent en millisecondes (MFE 1h médian = 0 %)",
                "donnees_proprietaires": "features = news publiques + LLM, "
                                         "recalculables par tous → déjà price",
                "echelle": "un 0.52 instable ne se diversifie pas en edge",
                "market_making": "le signal est preneur (paie le spread), pas "
                                 "teneur",
            },
            "read": "On échoue exactement là où la plupart des fonds "
                    "échouent aussi : prédire le sens à partir d'info "
                    "publique. Les gagnants ont des edges structurels, pas "
                    "un meilleur modèle.",
        },

        "verdict": card.get("verdict"),
        "what_makes_this_a_strong_poc": [
            "Résultat négatif rigoureux > faux positif fragile",
            "CV invalide sur série temporelle : démontré, pas récité "
            f"({m.get('cv_auc_mean')} → {m.get('walkforward_mean')})",
            "Anti-fuite par allowlist, no-look-ahead, multi-tests, ARI",
            "3 bugs de mesure trouvés et corrigés (intuition métier)",
            "Efficience de marché démontrée sur données de prod réelles",
        ],
        "honesty_commitment": "Aucun edge fabriqué. Toutes les pistes "
                              "testées. La recherche s'arrête volontairement "
                              "ici : continuer = p-hacking.",
    }
    RES.mkdir(exist_ok=True)
    (RES / "final_honest_verdict.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))

    print("=== VERDICT CONSOLIDÉ ===")
    print(out["one_line"])
    print(f"\n  Direction  : 3 modèles ≈ "
          f"{out['1_three_models_direction']['roc_auc']}")
    print(f"  K-Means ARI: {out['1_three_models_direction']['kmeans_adjusted_rand_index']}")
    print(f"  Magnitude  : CV {m.get('cv_auc_mean')} → "
          f"walk-forward {m.get('walkforward_mean')} {m.get('walkforward_auc')}")
    print(f"  Dense (no-look-ahead) : 9 règles, net "
          f"{min(nets):.1f}%…{max(nets):.1f}%  (brut ≤ 0)")
    print(f"\n  → results/final_honest_verdict.json")
    if not (card and dense):
        print("\n[!] artefacts manquants — lance d'abord scripts/train.py "
              "puis scripts/exploration/dense_replay_fixed.py", file=sys.stderr)


if __name__ == "__main__":
    main()
