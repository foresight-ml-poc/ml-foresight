"""Génère 4 figures propres, sobres, lisibles côté business (pas des
graphes d'analyste bruts). Charte = celle du produit Foresight :
fond sombre, vert menthe, rouge perte. Sortie : plots/*.png

  fig1  Direction : 3 modèles, tous ≈ pile ou face
  fig2  LE cœur : la validation croisée ment (CV vs walk-forward)
  fig3  Replay minute sans look-ahead : 9 règles, 9 pertes
  fig4  Tout ce qu'on a testé → marché efficient

Lit results/final_honest_verdict.json (source unique). Lancer après
scripts/honest_analysis.py.

Mise en page : une bande d'en-tête (titre + sous-titre) est réservée en
haut via add_axes ; la zone de tracé est placée dessous → jamais de
chevauchement titre/graphe.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)
V = json.loads((ROOT / "results" / "final_honest_verdict.json").read_text())

BG = "#070a0f"
CARD = "#0c1014"
LINE = "#252e3d"
MINT = "#0BE0A6"
LOSS = "#f76d6d"
AMBER = "#f5b942"
INK = "#eef3f9"
MUTED = "#9aa7b8"
DIM = "#5c6878"

plt.rcParams["font.family"] = "DejaVu Sans"  # toujours présent, glyphes →/≈
plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG,
    "axes.facecolor": BG, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 12,
})


def canvas(title, subtitle, figsize=(11, 6.4), left=0.12):
    """Crée la figure + l'axe de tracé sous une bande d'en-tête réservée.

    L'en-tête (titre + sous-titre) occupe le haut ; l'axe est ajouté
    dessous → aucun chevauchement possible. `left` règle la marge gauche
    (large quand les labels d'axe Y sont du texte long).
    """
    fig = plt.figure(figsize=figsize)
    fig.text(0.045, 0.965, title, ha="left", va="top", fontsize=20,
             fontweight="bold", color=INK, linespacing=1.25)
    fig.text(0.045, 0.83, subtitle, ha="left", va="top", fontsize=12.5,
             color=MUTED, linespacing=1.35)
    ax = fig.add_axes([left, 0.12, 0.97 - left, 0.58])  # sous l'en-tête
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return fig, ax


def _save(fig, name):
    fig.savefig(PLOTS / name, dpi=150)
    plt.close(fig)
    print(f"  plots/{name}")


# ---------- fig1 : direction = pile ou face ----------

def fig1():
    roc = V["1_three_models_direction"]["roc_auc"]
    ari = V["1_three_models_direction"]["kmeans_adjusted_rand_index"]
    items = sorted(roc.items(), key=lambda kv: kv[1])
    lab = {"logreg": "Régression logistique", "random_forest": "Random Forest",
           "kmeans": "K-Means (k=2)"}
    names = [lab.get(k, k) for k, _ in items]
    vals = [v for _, v in items]

    fig, ax = canvas(
        "Prédire le sens du marché : 3 familles de modèles,\n"
        "toutes collées au pile ou face",
        f"K-Means non supervisé : Adjusted Rand Index ≈ {ari:.2f} → aucun "
        "regroupement\nnaturel gagnants / perdants. Test set N=171.",
        left=0.235)
    y = range(len(vals))
    ax.barh(y, vals, height=0.5, color=DIM, zorder=2)
    ax.axvline(0.5, color=AMBER, lw=2, ls="--", zorder=3)
    ax.text(0.501, -0.78, "hasard = 0.50", color=AMBER, fontsize=12,
            fontweight="bold", va="center", ha="left")
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", color=INK,
                fontsize=16, fontweight="bold")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, color=INK, fontsize=13.5)
    ax.set_ylim(-1.15, len(vals) - 0.4)
    ax.set_xlim(0.40, 0.60)
    ax.set_xticks([0.40, 0.45, 0.50, 0.55, 0.60])
    ax.set_xlabel("ROC-AUC sur la direction à T+24h", fontsize=12)
    _save(fig, "fig1_direction_pile_ou_face.png")


# ---------- fig2 : LE cœur — la CV ment ----------

def fig2():
    c = V["2_cv_lies_centerpiece"]
    mag, dr = c["magnitude"], c["direction"]
    fig, ax = canvas(
        "Le cœur du POC : la validation croisée ment\n"
        "sur une série temporelle non-stationnaire",
        "La magnitude semble prédictible en CV (0.549). Entraînée sur le "
        "passé\net testée sur le futur, elle s'effondre vers le hasard : "
        "c'est l'alpha decay.",
        figsize=(11, 6.8), left=0.11)
    xs = [0, 1, 2]
    ax.plot(xs, mag["walkforward"], color=LOSS, lw=3, marker="o", ms=10,
            zorder=4, label="Magnitude — walk-forward (honnête)")
    ax.plot(xs, dr["walkforward"], color=DIM, lw=2, marker="s", ms=7,
            zorder=2, label="Direction — walk-forward")
    ax.axhline(mag["cv"], color=MINT, lw=2.5, ls="--", zorder=3)
    ax.text(1.5, mag["cv"] + 0.003, f"CV mélangée (optimiste) = {mag['cv']:.3f}",
            color=MINT, fontsize=12.5, fontweight="bold", va="bottom",
            ha="center")
    ax.axhline(0.5, color=AMBER, lw=2, ls=":", zorder=1)
    ax.text(1.5, 0.5 + 0.003, "hasard = 0.50", color=AMBER, fontsize=11.5,
            fontweight="bold", va="bottom", ha="center")
    for x, v in zip(xs, mag["walkforward"]):
        dy = 0.011 if x == 0 else -0.012
        va = "bottom" if x == 0 else "top"
        ax.text(x, v + dy, f"{v:.3f}", ha="center", va=va, color=INK,
                fontsize=14, fontweight="bold")
    ax.annotate("", xy=(0.12, mag["cv"]),
                xytext=(0.12, mag["walkforward_mean"]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.5))
    ax.text(0.18, (mag["cv"] + mag["walkforward_mean"]) / 2,
            "l'illusion\nde la CV", color=MUTED, fontsize=11.5,
            fontstyle="italic", va="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(["bloc 1 (ancien)", "bloc 2", "bloc 3 (récent)"],
                       color=INK, fontsize=12.5)
    ax.set_xlim(-0.15, 2.35)
    ax.set_ylim(0.455, 0.60)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.grid(axis="y", color=LINE, lw=0.6, alpha=0.5)
    ax.legend(loc="lower center", frameon=False, fontsize=11.5,
              labelcolor=INK, ncol=2, bbox_to_anchor=(0.5, -0.02))
    _save(fig, "fig2_cv_ment.png")


# ---------- fig3 : replay dense, 9 règles 9 pertes ----------

def fig3():
    d = V["3_dense_replay_no_look_ahead"]
    glo, ghi = d["exit_rules_gross_rtp_range_pct"]
    rules = ["TP5\n30m", "TP5\n60m", "TP5\n120m", "TP10\n30m", "TP10\n60m",
             "TP10\n120m", "TP15\n30m", "TP15\n60m", "TP15\n120m"]
    nets = [-3.27, -3.40, -3.90, -3.26, -3.24, -3.69, -3.20, -3.11, -3.49]
    fig, ax = canvas(
        "Rejouer chaque minute, sans look-ahead :\n9 règles de sortie, 9 pertes",
        f"{d['n_paths']} chemins minute reconstruits (3 bugs corrigés) · "
        f"MFE médian à 1 h = {d['mfe_60_median_pct']:.0f} %\n"
        f"seulement {d['mfe_60_share_pos']*100:.0f} % des signaux passent un "
        "seul instant en positif sur la 1ʳᵉ heure.", figsize=(11, 6.4))
    x = range(len(rules))
    ax.bar(x, nets, width=0.6, color=LOSS, zorder=2)
    for i, v in enumerate(nets):
        ax.text(i, v - 0.13, f"{v:.1f}", ha="center", va="top", color=INK,
                fontsize=11.5, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=1.5)
    ax.text(4, 0.42, f"même BRUT, hors spread : {glo:.1f}% à {ghi:.1f}% → "
            "≤ 0 partout", ha="center", color=AMBER, fontsize=12.5,
            fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(rules, color=MUTED, fontsize=10.5)
    ax.set_ylim(-4.7, 0.95)
    ax.set_ylabel("RTP net / signal (%) · spread 3 pp", fontsize=12)
    _save(fig, "fig3_dense_replay.png")


# ---------- fig4 : tout ce qu'on a testé ----------

def fig4():
    rows = [
        ("Direction — régression logistique", "ROC-AUC 0.497", "≈ hasard"),
        ("Direction — Random Forest", "ROC-AUC 0.544", "≈ hasard"),
        ("Direction — K-Means (ARI)", "ARI ≈ 0.00", "aucune structure"),
        ("Poids de l'heuristique ré-optimisés", "≈ poids main", "aucun gain"),
        ("Cibles-chemin ML (MFE > seuil)", "CV ≈ 0.51", "≈ hasard"),
        ("Replay dense — règle sans look-ahead", "9 / 9 négatives", "même brut ≤ 0"),
        ("Tri ML sélectif (top-décile)", "reste négatif", "ne franchit pas 0"),
        ("Magnitude — walk-forward strict", "0.549 → 0.526", "s'effondre"),
        ("Fade de volatilité (overréaction)", "négatif brut", "non monétisable"),
    ]
    fig = plt.figure(figsize=(11.5, 7))
    fig.text(0.045, 0.965, "Tout ce qu'on a testé honnêtement — et rien ne "
             "tient", ha="left", va="top", fontsize=20, fontweight="bold",
             color=INK)
    fig.text(0.045, 0.90,
             "Quand des approches très différentes échouent identiquement, "
             "le problème n'est pas le modèle :\nc'est le signal. Marché "
             "efficient vis-à-vis des features publiques — là où les fonds "
             "aussi échouent.", ha="left", va="top", fontsize=12.5,
             color=MUTED, linespacing=1.35)
    ax = fig.add_axes([0.03, 0.04, 0.94, 0.74])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows))
    ax.axis("off")
    n = len(rows)
    for i, (what, num, res) in enumerate(rows):
        y = n - i - 0.5
        ax.add_patch(plt.Rectangle((0.005, y - 0.42), 0.99, 0.84,
                     facecolor=CARD, edgecolor=LINE, lw=1))
        ax.text(0.03, y, what, va="center", ha="left", color=INK,
                fontsize=13.5)
        ax.text(0.60, y, num, va="center", ha="left", color=MUTED,
                fontsize=12.5, family="monospace")
        ax.text(0.985, y, res, va="center", ha="right", color=LOSS,
                fontsize=13, fontweight="bold")
    _save(fig, "fig4_ce_quon_a_teste.png")


if __name__ == "__main__":
    print("Génération des figures :")
    fig1(); fig2(); fig3(); fig4()
    print("OK — 4 figures dans plots/")
