"""Configuration du projet — calquée sur basile-desjuzeur/ml-poc-project.

Le POC répond à UNE question : peut-on prédire, par ML, si un signal
Foresight va dans le bon sens (`direction_correct`) ? La réponse honnête est
non — et c'est ce résultat, rigoureusement établi, qui fait le projet.

Trois familles de modèles imposées par le cours sont entraînées :
  - Régression logistique (linéaire)
  - Random Forest (ensemble d'arbres / bagging)
  - K-Means (non supervisé — teste si les signaux se regroupent
    naturellement en gagnants / perdants ; spoiler : non)
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

# Auto-création des dossiers à l'import (pattern Basile)
for d in [
    DATA_DIR,
    DATA_DIR / "raw",
    DATA_DIR / "processed",
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    d.mkdir(exist_ok=True, parents=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# --- Constantes projet ---

SEED = 42
TARGET_COLUMN = "direction_correct"   # 1 = le marché a bougé dans le sens prédit à T+24h
HEURISTIC_THRESHOLD = 65              # signal_score > 65 → label heuristique = 1

# --- Anti-fuite : ALLOWLIST EXPLICITE de features ---
#
# On utilise une allowlist (et non une denylist) : enrichir le CSV d'export
# avec des colonnes d'analyse (trajectoire de prix, microstructure, internes
# de l'heuristique…) ne peut JAMAIS faire fuiter une variable future dans X.
# Seules les colonnes listées ici — plus les one-hot `bucket_*` générés au
# runtime — deviennent des features. Tout le reste (labels, sorties dérivées
# du futur, score heuristique, texte libre, ids, timestamps) n'est jamais
# sélectionné.
FEATURE_BASE_COLUMNS = [
    # Numériques brutes, disponibles à l'émission du signal
    "cosine_score",
    "impact_strength",
    "llm_confidence",
    "ambiguity_score",
    "specificity_score",
    "articles_count",
    "unique_sources_count",
    # Dérivées dans _feature_engineer()
    "tier_1_count",
    "tier_2_count",
    "tier_3_count",
    "is_buy_yes",
    "market_price_centered",
    "hour_of_day",
    # + colonnes one-hot `bucket_*` ajoutées dynamiquement
]

# Documenté pour mémoire : ce qui fuiterait si on l'autorisait.
EXCLUDED_FROM_FEATURES = [
    "move_t24h_pct", "move_t5min_pct", "move_t15min_pct", "move_t1h_pct",
    "price_t5min", "price_t15min", "price_t1h", "price_t24h", "price_resolved",
    "outcome_label", "signal_score", "heuristic_score", "heuristic_strength",
    "heuristic_trade_quality", "signal_strength", "trade_quality",
]

# Registre des 3 modèles imposés — peuplé par scripts/train.py.
# main.py (fixé Basile) charge chaque .joblib, appelle .predict() et
# compute_metrics() sur le test set.
MODELS = {
    "logreg": {
        "name": "Régression logistique",
        "family": "linéaire",
        "description": (
            "Baseline linéaire L2, class_weight balanced, C réglé par "
            "GridSearchCV 5-fold. Interprétable : ses poids = la version "
            "« apprise » de la formule heuristique."
        ),
        "path": MODELS_DIR / "logreg.joblib",
    },
    "random_forest": {
        "name": "Random Forest",
        "family": "ensemble d'arbres (bagging)",
        "description": (
            "300 arbres, class_weight balanced, RandomizedSearchCV 5-fold. "
            "Capture les interactions non linéaires — s'il y avait une "
            "structure, il la trouverait."
        ),
        "path": MODELS_DIR / "random_forest.joblib",
    },
    "kmeans": {
        "name": "K-Means (k=2)",
        "family": "non supervisé",
        "description": (
            "Regroupe les signaux en 2 clusters SANS voir le label. Si les "
            "gagnants et les perdants formaient des groupes naturels, les "
            "clusters s'aligneraient sur direction_correct. Ils ne s'alignent "
            "pas (ARI ≈ 0) — preuve directe d'absence de structure."
        ),
        "path": MODELS_DIR / "kmeans.joblib",
    },
}
