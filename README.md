# ml-foresight

> Machine learning POC pour le scoring des signaux Foresight.
> Projet école Albert School · 2026-05.

![Status](https://img.shields.io/badge/status-design--phase-orange) · Python 3.11 · Calqué sur [basile-desjuzeur/ml-poc-project](https://github.com/basile-desjuzeur/ml-poc-project)

## Contexte

[Foresight](https://github.com/vcapton-jpg/polymarket-ai) émet des signaux de
trading sur Polymarket en notant chaque opportunité 0-100 via une **formule
heuristique fixe**. L'audit interne montre un winrate de 46-48 % à T+24h —
sous le hasard.

**Ce projet remplace la formule heuristique par du machine learning supervisé.**

- **Tâche :** classification binaire — prédire `direction_correct` à T+24h
- **Modèles :** Logistic Regression · Random Forest · MLP (Keras)
- **Métrique de sélection :** ROC-AUC
- **Baseline à battre :** la formule heuristique actuelle de Foresight

## Architecture (3 repos)

Ce projet est splitté en **3 repos séparés** dans
l'organisation [`foresight-ml-poc`](https://github.com/foresight-ml-poc) :

| Repo | Rôle |
|---|---|
| **ml-foresight** *(ici)* | Pipeline ML : data, train, evaluate, Streamlit dashboard |
| [backend-foresight](https://github.com/foresight-ml-poc/backend-foresight) | API FastAPI qui charge le best model et expose `/predict` |
| [frontend-foresight](https://github.com/foresight-ml-poc/frontend-foresight) | Demo React qui compare prédictions ML vs heuristique |

## Documentation

- 📄 **[Design document complet](./docs/specs/2026-05-05-design.md)** —
  problème, architecture, données, features, modèles, workflow, livrables

## Quickstart (à venir)

```bash
# 1. Setup
pip install -r requirements.txt

# 2. Export des données depuis Foresight (besoin de FORESIGHT_DB_DSN dans .env)
python scripts/export_from_foresight.py

# 3. Entraînement des 3 modèles
python scripts/train.py

# 4. Évaluation + lancement Streamlit
python scripts/main.py
# → http://localhost:8501
```

## Structure

Calquée sur [Basile](https://github.com/basile-desjuzeur/ml-poc-project) :

```
ml-foresight/
├── scripts/
│   ├── main.py                     # ★ FIXÉ Basile — eval + Streamlit
│   ├── train.py                    # ☆ Notre code — entraîne les 3 modèles
│   └── export_from_foresight.py    # ☆ Notre code — SQL → CSV
└── src/
    ├── __init__.py                 # ★ FIXÉ
    ├── config.py                   # ☆ Adapté — MODELS dict + paths
    ├── data.py                     # ☆ Implémenté — load_dataset_split()
    ├── metrics.py                  # ☆ Implémenté — compute_metrics()
    ├── model_io.py                 # ★ FIXÉ Basile — load_model()
    ├── results.py                  # ★ FIXÉ Basile — write_metrics()
    └── app.py                      # ☆ Implémenté — Streamlit dashboard
```

★ = fichier identique à la référence · ☆ = adapté ou écrit pour ce projet.

Voir le [design doc](./docs/specs/2026-05-05-design.md) pour le détail complet.

## Status

🚧 **Phase de design.** Code à venir. Voir
[`docs/specs/2026-05-05-design.md`](./docs/specs/2026-05-05-design.md) pour
le plan complet.

## License

MIT
