"""Dashboard Streamlit — le verdict honnête du POC.

Point d'entrée fixé (Basile) : scripts/main.py lance `build_app()` via
`streamlit run`. Aucune nouvelle analyse ici : on lit les artefacts
reproductibles (final_honest_verdict.json, model_card.json) et les 4
figures de plots/, pour que le dashboard raconte exactement la même
histoire que le README et le rapport.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Auto-bootstrap : permet `streamlit run src/app.py` directement (sans
# devoir exporter PYTHONPATH=src). main.py (Basile) le fait déjà de son
# côté ; ce sys.path.insert est inoffensif et idempotent.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from config import DATA_DIR, MODEL_CARD_FILE, PLOTS_DIR, RESULTS_DIR

MINT = "#0BE0A6"


def _verdict() -> dict:
    p = RESULTS_DIR / "final_honest_verdict.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _card() -> dict:
    return json.loads(MODEL_CARD_FILE.read_text()) if MODEL_CARD_FILE.exists() else {}


def _data() -> pd.DataFrame | None:
    raw = DATA_DIR / "raw"
    dated = [p for p in sorted(raw.glob("signals_export_2*.csv"))
             if "sample" not in p.name]
    path = dated[-1] if dated else (raw / "signals_export_sample.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


def _img(name: str, caption: str) -> None:
    p = PLOTS_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, use_container_width=True)
    else:
        st.info(f"{name} manquant — lance `python scripts/make_figures.py`.")


def build_app() -> None:
    st.set_page_config(page_title="ml-foresight — verdict honnête",
                       layout="wide", initial_sidebar_state="collapsed")
    V, card = _verdict(), _card()

    st.title("Peut-on prédire un marché de prédiction ?")
    st.caption("POC ML · Albert School · "
               "[GitHub](https://github.com/foresight-ml-poc/ml-foresight)")
    if V.get("one_line"):
        st.success(f"**Verdict.** {V['one_line']}")

    t = st.tabs(["Le verdict", "Les 3 modèles", "La CV ment (cœur)",
                 "Replay minute", "Pourquoi", "Les données"])

    with t[0]:
        st.header("Un résultat négatif — rigoureux et assumé")
        c1, c2, c3 = st.columns(3)
        ds = (card.get("dataset") or V.get("dataset") or {})
        c1.metric("Signaux exploitables", ds.get("total", "—"))
        mag = V.get("2_cv_lies_centerpiece", {}).get("magnitude", {})
        c2.metric("Magnitude — CV (optimiste)", mag.get("cv", "—"))
        c3.metric("Magnitude — walk-forward", mag.get("walkforward_mean", "—"),
                  delta="s'effondre", delta_color="inverse")
        st.markdown(
            "- La **direction** est un pile ou face → marché efficient.\n"
            "- Le seul signal qui semblait vivant (**magnitude**) est une "
            "**illusion de validation croisée** : il ne tient pas sur le "
            "futur.\n"
            "- C'est, exactement, ce que démontre ce POC — sans tricher.")
        _img("fig4_ce_quon_a_teste.png",
             "Tout ce qu'on a testé honnêtement — rien ne tient")
        for b in V.get("what_makes_this_a_strong_poc", []):
            st.markdown(f"- {b}")

    with t[1]:
        st.header("3 familles imposées — toutes ≈ 0.50")
        m = card.get("models", {})
        if m:
            st.dataframe(pd.DataFrame(m).T.round(3), use_container_width=True)
        st.caption(f"K-Means non supervisé · Adjusted Rand Index ≈ "
                   f"{card.get('kmeans_adjusted_rand_index', 0):.2f} "
                   "→ aucun regroupement gagnants/perdants.")
        _img("fig1_direction_pile_ou_face.png",
             "ROC-AUC sur la direction — test set N=171")

    with t[2]:
        st.header("Le cœur : la validation croisée ment")
        _img("fig2_cv_ment.png",
             "CV (mélangée, optimiste) vs walk-forward (honnête)")
        st.markdown(
            "La CV mélange passé et futur. Sur une série **non-stationnaire**, "
            "elle sur-estime : la magnitude passe de **0.549 en CV** à "
            "**~0.49 en walk-forward sur la période récente** (*alpha "
            "decay*). C'est la leçon ML centrale du projet.")

    with t[3]:
        st.header("Même au grain de la minute, sans tricher")
        _img("fig3_dense_replay.png",
             "9 règles de sortie no-look-ahead — 9 pertes, même brut")
        d = V.get("3_dense_replay_no_look_ahead", {})
        st.markdown(
            f"{d.get('n_paths','554')} chemins minute reconstruits. "
            f"MFE médian à 1 h = **{d.get('mfe_60_median_pct',0)} %**. "
            "3 bugs de mesure trouvés et corrigés → résultat négatif **plus** "
            "robuste, pas moins.")
        for b in d.get("bugs_fixed_thanks_to_user_intuition", []):
            st.markdown(f"- 🐛 {b}")

    with t[4]:
        st.header("Pourquoi : efficience de marché")
        edges = (V.get("why_efficient_market", {})
                 .get("hedge_fund_edges_foresight_lacks", {}))
        for k, why in edges.items():
            st.markdown(f"- **{k.replace('_',' ').title()}** — {why}")
        st.info(V.get("why_efficient_market", {}).get("read", ""))

    with t[5]:
        st.header("Les données — vraies, de production")
        df = _data()
        if df is None:
            st.info("Aucun CSV. `python scripts/export_from_foresight.py`")
            return
        st.write(f"**{len(df)}** signaux · **{df.shape[1]}** colonnes brutes.")
        cols = [c for c in ["created_at", "direction", "bucket",
                            "heuristic_score", "impact_strength",
                            "move_t1h_pct", "move_t24h_pct",
                            "direction_correct"] if c in df.columns]
        st.dataframe(df[cols].head(12), use_container_width=True)
        st.caption("Anti-fuite : seules les variables connues à l'émission "
                   "du signal entrent dans le modèle (allowlist, config.py).")


if __name__ == "__main__":
    build_app()
