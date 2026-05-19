"""Streamlit dashboard for the ML Foresight POC.

Fixed entry point — scripts/main.py launches this via `streamlit run`.
Reads the enriched export CSV (39 cols: full price trajectory, heuristic
internals, market microstructure) for rich EDA, plus the model_card /
metrics for the model section.
"""

from __future__ import annotations

import glob
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from config import (
    DATA_DIR,
    HEURISTIC_THRESHOLD,
    MODEL_CARD_FILE,
    MODEL_METRICS_FILE,
    MODELS_DIR,
    PLOTS_DIR,
    TARGET_COLUMN,
)

MINT = "#0BE0A6"
LOSS = "#f76d6d"
GREY = "#5c6878"
sns.set_theme(style="darkgrid", rc={
    "figure.facecolor": "#0b0f17", "axes.facecolor": "#11161f",
    "axes.edgecolor": "#252e3d", "grid.color": "#1a212d",
    "text.color": "#e7edf5", "axes.labelcolor": "#9aa7b8",
    "xtick.color": "#9aa7b8", "ytick.color": "#9aa7b8",
})


def build_app() -> None:
    st.set_page_config(page_title="ML Foresight", layout="wide",
                       initial_sidebar_state="collapsed")
    st.title("ML Foresight — analyse des signaux Polymarket")
    st.caption(
        "Peut-on battre l'heuristique de scoring de Foresight avec du ML ? "
        "· [GitHub](https://github.com/foresight-ml-poc/ml-foresight)"
    )

    df = _load_data()
    card = _load_card()

    tabs = st.tabs([
        "Vue d'ensemble",
        "Exploration des signaux",
        "Trajectoire des prix",
        "Modèles",
        "Vs heuristique",
        "Prédiction live",
    ])
    with tabs[0]:
        _tab_overview(df, card)
    with tabs[1]:
        _tab_explore(df)
    with tabs[2]:
        _tab_trajectory(df)
    with tabs[3]:
        _tab_models(card)
    with tabs[4]:
        _tab_vs_heuristic(df, card)
    with tabs[5]:
        _tab_predict(card)


# ---------- loaders ----------

def _load_data() -> pd.DataFrame | None:
    raw = DATA_DIR / "raw"
    dated = sorted(p for p in raw.glob("signals_export_2*.csv"))
    candidates = [p for p in dated if "sample" not in p.name]
    path = candidates[-1] if candidates else (raw / "signals_export_sample.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    return df


def _load_card() -> dict | None:
    return json.loads(MODEL_CARD_FILE.read_text()) if MODEL_CARD_FILE.exists() else None


def _load_metrics_csv() -> pd.DataFrame | None:
    return pd.read_csv(MODEL_METRICS_FILE) if MODEL_METRICS_FILE.exists() else None


def _winrate_bar(series_groupby, title, xlabel):
    """Helper: horizontal bar of winrate by category with n labels."""
    g = series_groupby
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.5 * len(g))))
    order = g["mean"].sort_values().index
    colors = [MINT if v >= 0.5 else LOSS for v in g.loc[order, "mean"]]
    ax.barh(range(len(order)), g.loc[order, "mean"], color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.axvline(0.5, color=GREY, ls="--", lw=1)
    for i, idx in enumerate(order):
        ax.text(g.loc[idx, "mean"] + 0.01, i,
                f"{g.loc[idx, 'mean']:.0%}  (n={int(g.loc[idx, 'count'])})",
                va="center", fontsize=9, color="#e7edf5")
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_title(title, color="#e7edf5")
    fig.tight_layout()
    return fig


# ---------- Tab 1: Overview ----------

def _tab_overview(df, card):
    st.header("Vue d'ensemble")
    if df is None:
        st.info("Aucune donnée. Lance `python scripts/export_from_foresight.py`.")
        return

    wr = df[TARGET_COLUMN].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signaux", len(df))
    c2.metric("Winrate global", f"{wr:.1%}")
    if "created_at" in df:
        span = (df["created_at"].max() - df["created_at"].min()).days
        c3.metric("Période couverte", f"{span} jours")
    c4.metric("Best model", card["best_model_type"] if card else "—")

    st.markdown(
        f"""
        **Le problème.** Foresight émet des signaux notés 0–100 par une formule
        heuristique. Sur ces {len(df)} signaux, le winrate réel à T+24h est de
        **{wr:.1%}** — proche du hasard. On entraîne 6 modèles ML pour prédire
        si un signal sera correct, et on compare honnêtement à l'heuristique.
        """
    )

    if card:
        best = card["best_model_type"]
        m = card["test_metrics"][best]
        h = card["heuristic_baseline"]
        delta = (m["roc_auc"] - h["roc_auc"]) * 100
        st.subheader("Résultat")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric(f"ROC-AUC — {best}", f"{m['roc_auc']:.3f}")
        cc2.metric("ROC-AUC — heuristique", f"{h['roc_auc']:.3f}")
        cc3.metric("Écart", f"{delta:+.1f} pts",
                   delta=f"{delta:+.1f}", delta_color="normal")
        st.caption(
            "L'écart est modeste et instable selon le refresh des données "
            "(le best model a changé entre versions) — c'est le résultat "
            "honnête : le ML égale l'heuristique sans la dominer."
        )

    st.subheader("Volume de signaux dans le temps")
    if "created_at" in df:
        daily = (df.set_index("created_at")
                   .resample("D")[TARGET_COLUMN]
                   .agg(["count", "mean"]))
        st.bar_chart(daily["count"], color=MINT, height=200)
        st.caption("Nombre de signaux émis par jour (prod Hetzner).")


# ---------- Tab 2: Explore ----------

def _tab_explore(df):
    st.header("Exploration des signaux")
    if df is None:
        st.info("Aucune donnée.")
        return

    st.subheader("Winrate par direction — l'asymétrie BUY_YES / BUY_NO")
    g = df.groupby("direction")[TARGET_COLUMN].agg(["mean", "count"])
    st.pyplot(_winrate_bar(g, "Winrate par direction", "winrate T+24h"))
    st.caption(
        "L'audit Foresight avait relevé une forte asymétrie : les BUY_NO "
        "gagnent plus souvent que les BUY_YES. On le vérifie ici sur la donnée."
    )

    st.subheader("Winrate par bucket (thème de l'événement)")
    g = df.groupby("bucket")[TARGET_COLUMN].agg(["mean", "count"])
    g = g[g["count"] >= 5]
    st.pyplot(_winrate_bar(g, "Winrate par bucket (n≥5)", "winrate T+24h"))

    st.subheader("L'heuristique est-elle calibrée ?")
    st.caption(
        "Si l'heuristique est bonne, un score plus haut → winrate plus élevé. "
        "Voilà le winrate réel par tranche de heuristic_score."
    )
    df2 = df.copy()
    df2["score_band"] = pd.cut(df2["heuristic_score"],
                                bins=[0, 50, 60, 65, 70, 80, 100],
                                labels=["<50", "50-60", "60-65", "65-70", "70-80", "80+"])
    g = df2.groupby("score_band", observed=True)[TARGET_COLUMN].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(g.index.astype(str), g["mean"],
           color=[MINT if v >= 0.5 else LOSS for v in g["mean"]])
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    for i, (idx, row) in enumerate(g.iterrows()):
        ax.text(i, row["mean"] + 0.01, f"{row['mean']:.0%}\nn={int(row['count'])}",
                ha="center", fontsize=9, color="#e7edf5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("winrate réel T+24h")
    ax.set_xlabel("tranche de heuristic_score")
    ax.axvspan(2.5, 5.5, alpha=0.07, color=MINT)
    ax.set_title("Winrate réel vs score heuristique", color="#e7edf5")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"Le seuil d'émission de Foresight est {HEURISTIC_THRESHOLD}. "
        "Si les barres ne montent pas avec le score, l'heuristique ne "
        "discrimine pas bien les winners — ce qui justifie le projet ML."
    )

    st.subheader("Distribution d'une feature : winners vs losers")
    numcols = [c for c in [
        "cosine_score", "impact_strength", "llm_confidence", "ambiguity_score",
        "specificity_score", "market_price_at_signal", "articles_count",
        "unique_sources_count", "heuristic_score",
    ] if c in df.columns]
    feat = st.selectbox("Feature", numcols)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for label, color, name in [(1, MINT, "win"), (0, LOSS, "loss")]:
        sub = df[df[TARGET_COLUMN] == label][feat].dropna()
        ax.hist(sub, bins=30, alpha=0.55, color=color, label=name, density=True)
    ax.legend()
    ax.set_xlabel(feat)
    ax.set_ylabel("densité")
    ax.set_title(f"{feat} — distribution win vs loss", color="#e7edf5")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Si les deux distributions se superposent presque, la feature ne "
        "sépare pas les classes — c'est le cas pour la plupart ici, d'où "
        "un signal ML faible."
    )

    st.subheader("Corrélation des features numériques")
    corr = df[numcols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="mako", ax=ax,
                cbar=False, annot_kws={"size": 8})
    ax.set_title("Matrice de corrélation", color="#e7edf5")
    fig.tight_layout()
    st.pyplot(fig)


# ---------- Tab 3: Price trajectory ----------

def _tab_trajectory(df):
    st.header("Trajectoire des prix — la vie d'un signal")
    if df is None or "move_t5min_pct" not in df.columns:
        st.info("Données de trajectoire indisponibles dans ce CSV.")
        return

    st.caption(
        "Mouvement de prix moyen (en %) après l'émission du signal, pour les "
        "signaux gagnants vs perdants. Montre **quand** un signal « paie »."
    )
    horizons = ["move_t5min_pct", "move_t15min_pct", "move_t1h_pct", "move_t24h_pct"]
    xlabels = ["T+5min", "T+15min", "T+1h", "T+24h"]
    have = [h for h in horizons if h in df.columns]

    # signed move in the predicted direction (so wins are positive)
    sign = np.where(df["direction"].str.upper().str.contains("YES|UP"), 1, -1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, color, name in [(1, MINT, "winners"), (0, LOSS, "losers")]:
        mask = df[TARGET_COLUMN] == label
        means = [(df.loc[mask, h] * sign[mask]).mean() for h in have]
        ax.plot(range(len(have)), means, marker="o", color=color,
                label=name, lw=2)
    ax.axhline(0, color=GREY, ls="--", lw=1)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([xlabels[horizons.index(h)] for h in have])
    ax.set_ylabel("mouvement signé moyen (%)")
    ax.set_title("Mouvement du marché dans la direction prédite", color="#e7edf5")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Pour les winners le prix bouge dans le bon sens et s'amplifie "
        "jusqu'à T+24h. Pour les losers il part dans le mauvais sens dès "
        "T+5min. Le signal se révèle vite."
    )

    st.subheader("Résolution finale des marchés")
    if "outcome_label" in df.columns:
        res = df["outcome_label"].value_counts(dropna=False)
        st.write(
            f"Marchés résolus YES : **{int(res.get(1, 0))}** · "
            f"résolus NO : **{int(res.get(0, 0))}** · "
            f"non résolus : **{int(df['outcome_label'].isna().sum())}**"
        )


# ---------- Tab 4: Models ----------

def _tab_models(card):
    st.header("Comparaison des 6 modèles")
    mdf = _load_metrics_csv()
    if mdf is not None:
        st.dataframe(mdf.set_index("model_key").round(3), use_container_width=True)
    if card:
        st.caption(
            f"Test set N={card['dataset_size']['test']} · "
            f"best = {card['best_model_type']} · {card['model_version']}"
        )

    c1, c2 = st.columns(2)
    with c1:
        p = PLOTS_DIR / "roc_curves_comparison.png"
        if p.exists():
            st.image(str(p), caption="Courbes ROC", use_container_width=True)
    with c2:
        p = PLOTS_DIR / "feature_importance_comparison.png"
        if p.exists():
            st.image(str(p), caption="Feature importance (tree models)",
                     use_container_width=True)

    p = PLOTS_DIR / "ml_vs_heuristic.png"
    if p.exists():
        st.image(str(p), caption="Heuristique vs 6 modèles ML",
                 use_container_width=True)

    st.subheader("Matrices de confusion")
    cols = st.columns(3)
    for i, k in enumerate(["random_forest", "gradient_boosting", "xgboost",
                            "lightgbm", "logreg", "svm"]):
        p = PLOTS_DIR / f"confusion_matrix_{k}.png"
        if p.exists():
            cols[i % 3].image(str(p), caption=k, use_container_width=True)


# ---------- Tab 5: vs heuristic + calibration ----------

def _tab_vs_heuristic(df, card):
    st.header("ML vs heuristique")
    if not card:
        st.info("Pas de model_card.")
        return
    best = card["best_model_type"]
    rows = [{"Modèle": "Heuristique", **card["heuristic_baseline"]}]
    for k, v in card["test_metrics"].items():
        rows.append({"Modèle": k, **v})
    st.dataframe(pd.DataFrame(rows).set_index("Modèle").round(3),
                 use_container_width=True)

    h = card["heuristic_baseline"]["roc_auc"]
    ml = card["test_metrics"][best]["roc_auc"]
    d = (ml - h) * 100
    if d > 1:
        st.success(f"Le best model ({best}) bat l'heuristique de +{d:.1f} pts ROC-AUC.")
    elif d < -1:
        st.warning(f"Le ML ne bat pas l'heuristique ({d:.1f} pts).")
    else:
        st.info(f"Égalité statistique ({d:+.1f} pts) — résultat honnête.")

    st.subheader("Courbe de calibration de l'heuristique")
    st.caption(
        "Un modèle bien calibré : quand il prédit X%, le winrate réel ≈ X%. "
        "On teste l'heuristique (signal_score/100) contre le winrate réel."
    )
    if df is not None and "heuristic_score" in df.columns:
        d2 = df.dropna(subset=["heuristic_score"]).copy()
        d2["pred"] = (d2["heuristic_score"] / 100).clip(0, 1)
        d2["bin"] = pd.cut(d2["pred"], bins=np.linspace(0, 1, 11))
        cal = d2.groupby("bin", observed=True).agg(
            pred=("pred", "mean"), real=(TARGET_COLUMN, "mean"),
            n=(TARGET_COLUMN, "count"))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], ls="--", color=GREY, label="calibration parfaite")
        ax.plot(cal["pred"], cal["real"], marker="o", color=MINT,
                label="heuristique")
        ax.set_xlabel("score heuristique prédit")
        ax.set_ylabel("winrate réel")
        ax.set_title("Calibration de l'heuristique", color="#e7edf5")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        st.caption(
            "Si la courbe mint est loin de la diagonale, l'heuristique est "
            "mal calibrée — un argument fort pour la remplacer par du ML."
        )


# ---------- Tab 6: Live prediction ----------

def _tab_predict(card):
    st.header("Prédiction live")
    if not card:
        st.info("Pas de model_card. Lance `python scripts/train.py`.")
        return
    mp = MODELS_DIR / "best_model.joblib"
    sp = MODELS_DIR / "scaler.pkl"
    if not (mp.exists() and sp.exists()):
        st.info("Modèle ou scaler manquant.")
        return
    model = joblib.load(mp)
    scaler = joblib.load(sp)
    order = card["feature_order"]
    st.caption(f"modèle **{card['best_model_type']}** · {card['model_version']}")

    inputs = {}
    cols = st.columns(3)
    for i, f in enumerate(order):
        with cols[i % 3]:
            if f == "is_buy_yes":
                inputs[f] = st.selectbox(f, [0, 1], index=1)
            elif f == "hour_of_day":
                inputs[f] = st.slider(f, 0, 23, 14)
            elif f in {"articles_count", "unique_sources_count",
                       "tier_1_count", "tier_2_count", "tier_3_count"}:
                inputs[f] = st.slider(f, 0, 10, 2)
            elif f.startswith("bucket_"):
                inputs[f] = st.selectbox(f, [0, 1], index=0)
            else:
                inputs[f] = st.slider(f, 0.0, 1.0, 0.5, 0.01)

    if st.button("Prédire", type="primary"):
        x = np.array([[inputs[f] for f in order]], dtype="float32")
        xs = scaler.transform(x)
        proba = (float(model.predict_proba(xs)[0, 1])
                 if hasattr(model, "predict_proba")
                 else float(model.predict(xs)[0]))
        c1, c2 = st.columns(2)
        c1.metric("Probabilité de gain", f"{proba:.1%}")
        c2.metric("Verdict", "GAIN" if proba >= 0.5 else "LOSS")


if __name__ == "__main__":
    build_app()
