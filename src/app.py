"""Streamlit dashboard for the ML Foresight POC.

Fixed entry point — scripts/main.py launches this via `streamlit run`.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DATA_DIR, MODEL_CARD_FILE, MODEL_METRICS_FILE, MODELS_DIR, PLOTS_DIR,
)


def build_app() -> None:
    st.set_page_config(page_title="ML Foresight — POC", layout="wide")
    st.title("ML Foresight — Prédiction de la direction des signaux Polymarket")
    st.caption("Projet école Albert School · 2026-05 · "
               "[GitHub](https://github.com/foresight-ml-poc/ml-foresight)")

    tabs = st.tabs([
        "📋 Vue d'ensemble",
        "🔍 Exploration des données",
        "📊 Comparaison des modèles",
        "⚔️ Vs heuristique",
        "🎯 Prédiction live",
    ])

    with tabs[0]:
        _tab_overview()
    with tabs[1]:
        _tab_eda()
    with tabs[2]:
        _tab_models()
    with tabs[3]:
        _tab_vs_heuristic()
    with tabs[4]:
        _tab_predict()


def _load_metrics_csv() -> pd.DataFrame | None:
    if MODEL_METRICS_FILE.exists():
        return pd.read_csv(MODEL_METRICS_FILE)
    return None


def _load_card() -> dict | None:
    if MODEL_CARD_FILE.exists():
        return json.loads(MODEL_CARD_FILE.read_text())
    return None


def _load_data() -> pd.DataFrame | None:
    raw = DATA_DIR / "raw"
    dated = sorted(raw.glob("signals_export_20*.csv"))
    candidates = [p for p in dated if "sample" not in p.name]
    if candidates:
        return pd.read_csv(candidates[-1])
    sample = raw / "signals_export_sample.csv"
    if sample.exists():
        return pd.read_csv(sample)
    return None


# ----- Tab 1: Overview -----

def _tab_overview() -> None:
    st.header("Contexte du projet")
    st.markdown("""
    **Foresight** émet des signaux de trading sur Polymarket en notant chaque
    opportunité 0-100 via une formule heuristique. L'audit interne montre un
    **winrate de 46-48 % à T+24h** — sous le hasard.

    **Ce projet remplace la formule par du machine learning supervisé.**
    - **Tâche** : classification binaire — prédire `direction_correct` à T+24h
    - **Modèles** : Logistic Regression · Random Forest · Gradient Boosting
    - **Métrique de sélection** : ROC-AUC
    - **Baseline à battre** : la formule heuristique actuelle
    """)

    df = _load_data()
    card = _load_card()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Signaux dans le dataset", len(df) if df is not None else "—")
    with col2:
        if df is not None and "direction_correct" in df.columns:
            balance = df["direction_correct"].value_counts().to_dict()
            st.metric("Distribution des classes",
                      f"{balance.get(1, 0)} wins / {balance.get(0, 0)} losses")
        else:
            st.metric("Distribution des classes", "—")
    with col3:
        st.metric("Best model", card["best_model_type"] if card else "Pas encore entraîné")

    if card:
        st.subheader("Model card")
        display_card = {k: v for k, v in card.items() if k != "feature_order"}
        st.json(display_card)
        with st.expander(f"feature_order ({len(card['feature_order'])} features)"):
            st.write(card["feature_order"])


# ----- Tab 2: EDA -----

def _tab_eda() -> None:
    st.header("Exploration des données")
    df = _load_data()
    if df is None:
        st.info("Aucune donnée. Lance `python scripts/export_from_foresight.py` d'abord.")
        return

    st.subheader("Aperçu")
    st.write(f"**{len(df)} signaux**, {len(df.columns)} colonnes")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistiques descriptives")
    numeric = df.select_dtypes(include="number")
    st.dataframe(numeric.describe(), use_container_width=True)

    st.subheader("Distribution d'une feature")
    feat_options = numeric.columns.tolist()
    default_idx = feat_options.index("freshness_factor") if "freshness_factor" in feat_options else 0
    feature = st.selectbox("Feature", feat_options, index=default_idx)
    fig_data = numeric[feature].dropna()
    st.bar_chart(fig_data.value_counts(bins=15).sort_index())

    st.subheader("Matrice de corrélation")
    if len(numeric.columns) > 1:
        corr = numeric.corr()
        st.dataframe(
            corr.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1),
            use_container_width=True,
        )


# ----- Tab 3: Model comparison -----

def _tab_models() -> None:
    st.header("Comparaison des 3 modèles")

    metrics_df = _load_metrics_csv()
    if metrics_df is None:
        st.info(
            "Aucune évaluation. Lance `python scripts/train.py` puis "
            "`python scripts/main.py`."
        )
        return

    st.subheader("Métriques (sur le test set, N=8 samples)")
    st.dataframe(metrics_df, use_container_width=True)
    st.caption(
        "⚠️ Avec 8 samples au test, les métriques sont très bruitées. "
        "La pipeline scalera quand Foresight aura accumulé plus d'outcomes."
    )

    col1, col2 = st.columns(2)
    with col1:
        roc = PLOTS_DIR / "roc_curves_comparison.png"
        if roc.exists():
            st.subheader("Courbes ROC")
            st.image(str(roc), use_container_width=True)

    with col2:
        fi = PLOTS_DIR / "feature_importance_rf.png"
        if fi.exists():
            st.subheader("Feature importance — Random Forest")
            st.image(str(fi), use_container_width=True)

    st.subheader("Matrices de confusion")
    cols = st.columns(3)
    for i, key in enumerate(["logreg", "random_forest", "gradient_boosting"]):
        cm_path = PLOTS_DIR / f"confusion_matrix_{key}.png"
        if cm_path.exists():
            with cols[i]:
                st.image(str(cm_path), caption=key, use_container_width=True)


# ----- Tab 4: vs Heuristic -----

def _tab_vs_heuristic() -> None:
    st.header("ML vs Heuristique — la punchline du projet")

    card = _load_card()
    if not card:
        st.info("Pas de model_card. Lance `python scripts/train.py` d'abord.")
        return

    plot = PLOTS_DIR / "ml_vs_heuristic.png"
    if plot.exists():
        st.image(str(plot), use_container_width=True)

    st.subheader("Tableau comparatif (test set N=8)")
    rows = [
        {"Modèle": "Heuristique (formule Foresight)", **card["heuristic_baseline"]},
    ]
    for k, v in card["test_metrics"].items():
        rows.append({"Modèle": k, **v})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    best = card["best_model_type"]
    h_auc = card["heuristic_baseline"]["roc_auc"]
    ml_auc = card["test_metrics"][best]["roc_auc"]
    delta = (ml_auc - h_auc) * 100

    st.subheader("Verdict")
    if delta > 1:
        st.success(
            f"✅ Le best model ({best}) bat l'heuristique de "
            f"**+{delta:.1f} pts ROC-AUC** sur ce test set."
        )
    elif delta < -1:
        st.warning(
            f"⚠️ Le ML ({best}) n'a pas battu l'heuristique "
            f"({delta:.1f} pts ROC-AUC). À discuter dans le rapport."
        )
    else:
        st.info(
            f"🤝 Le ML ({best}) et l'heuristique sont à égalité "
            f"({delta:+.1f} pts ROC-AUC). Le test set est trop petit pour "
            f"conclure — re-train avec plus de samples."
        )


# ----- Tab 5: Live prediction -----

def _tab_predict() -> None:
    st.header("Prédiction live")
    card = _load_card()
    if not card:
        st.info("Pas de model_card. Lance `python scripts/train.py` d'abord.")
        return

    model_path = MODELS_DIR / "best_model.joblib"
    scaler_path = MODELS_DIR / "scaler.pkl"  # joblib.dump uses .pkl by convention here
    if not (model_path.exists() and scaler_path.exists()):
        st.info("Modèle ou scaler manquant.")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_order: list[str] = card["feature_order"]

    st.markdown(
        f"Modèle utilisé : **{card['best_model_type']}** "
        f"(version {card['model_version']}). "
        f"Ajuste les sliders puis clique **Prédire**."
    )

    inputs: dict[str, float] = {}
    cols = st.columns(3)
    for i, feat in enumerate(feature_order):
        with cols[i % 3]:
            if feat == "is_buy_yes":
                inputs[feat] = st.selectbox(feat, options=[0, 1], index=1)
            elif feat == "hour_of_day":
                inputs[feat] = st.slider(feat, 0, 23, 14)
            elif feat in {"articles_count", "unique_sources_count",
                          "tier_1_count", "tier_2_count", "tier_3_count"}:
                inputs[feat] = st.slider(feat, 0, 10, 2)
            elif feat.startswith("bucket_"):
                inputs[feat] = st.selectbox(feat, options=[0, 1], index=0)
            else:
                inputs[feat] = st.slider(feat, 0.0, 1.0, 0.5, step=0.01)

    if st.button("Prédire", type="primary"):
        x = np.array([[inputs[f] for f in feature_order]], dtype="float32")
        x_scaled = scaler.transform(x)
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(x_scaled)[0, 1])
        else:
            proba = float(model.predict(x_scaled)[0])
        label = int(proba >= 0.5)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Probabilité de gain (label = 1)", f"{proba:.3f}")
        with col_b:
            st.metric("Label prédit",
                      "🟢 GAIN" if label == 1 else "🔴 LOSS")


if __name__ == "__main__":
    build_app()
