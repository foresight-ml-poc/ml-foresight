# ml-foresight

POC de ML pour [Foresight](https://github.com/vcapton-jpg/polymarket-ai), mon projet de signaux de trading sur Polymarket. L'idée : remplacer la formule de scoring heuristique actuelle par un modèle entraîné sur les signaux historiques.

Calqué sur la structure de [basile-desjuzeur/ml-poc-project](https://github.com/basile-desjuzeur/ml-poc-project).

## Le problème

Foresight émet des signaux notés 0–100 par une formule fixe. Sur 2 mois de prod, le winrate à T+24h est à 46–48 %, sous le hasard. On veut tester si un modèle ML peut faire mieux.

**Tâche** : classification binaire — étant donné un signal qui vient d'être émis, prédire si le marché va effectivement bouger dans la direction prédite à T+24h (`direction_correct`).

## Résultats

411 signaux entraînés (2026-04-12 → 2026-05-04), split 80/20, test set N=78.

![ML vs Heuristique](plots/ml_vs_heuristic.png)

| Modèle | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Heuristique Foresight | 0.526 | 0.575 | 0.533 |
| Logistic Regression | 0.385 | 0.385 | 0.386 |
| Random Forest | 0.538 | 0.438 | 0.531 |
| **Gradient Boosting** | **0.577** | **0.492** | **0.570** |

GBM bat l'heuristique de **+3.7 pts ROC-AUC**. RF est à égalité, LogReg sous-performe (relation non-linéaire).

![ROC curves](plots/roc_curves_comparison.png)

![Feature importance RF](plots/feature_importance_rf.png)

## Quickstart

```bash
# Setup
conda create -n ml-foresight python=3.11 -y
conda activate ml-foresight
pip install -r requirements.txt

# Si tu as accès à la DB Foresight, exporte les données
echo "FORESIGHT_DB_DSN=postgresql://..." > .env.local
python scripts/export_from_foresight.py

# Sinon, le sample anonymisé de 50 lignes dans data/raw/ suffit pour tester

# Entraîne les 3 modèles
python scripts/train.py

# Lance l'évaluation + dashboard Streamlit
python scripts/main.py
# → http://localhost:8501
```

## Structure

```
scripts/
  main.py                       # fixé Basile : eval + lance Streamlit
  train.py                      # notre code : entraîne LogReg + RF + GBM
  export_from_foresight.py      # SQL → CSV (notre code)
src/
  config.py    data.py          # contrats Basile, adaptés au projet
  metrics.py   app.py
  model_io.py  results.py       # fixés Basile
  __init__.py
plots/                          # générés par train.py, commités
docs/
  rapport.md                    # rapport académique
  specs/                        # design doc
```

## Architecture (3 repos)

- [ml-foresight](https://github.com/foresight-ml-poc/ml-foresight) (ici) — pipeline ML
- [backend-foresight](https://github.com/foresight-ml-poc/backend-foresight) — FastAPI qui sert le modèle
- [frontend-foresight](https://github.com/foresight-ml-poc/frontend-foresight) — démo React

Le modèle entraîné + scaler + model_card.json sont publiés en [GitHub Release v1.1.0](https://github.com/foresight-ml-poc/ml-foresight/releases/tag/v1.1.0). Le backend les télécharge au startup.

## Notes

- Le 3e modèle prévu était un MLP Keras mais TensorFlow se bloquait sur cet env (Apple Silicon, TF 2.21). Substitué par GradientBoosting — détails dans [`docs/rapport.md`](docs/rapport.md).
- v1.0.0 était capée à 40 samples car on joignait `event_market_features` (table populée seulement depuis 2026-04-27). v1.1.0 lève cette contrainte en utilisant directement le `signal_score` de Foresight comme baseline.

Voir [`docs/rapport.md`](docs/rapport.md) pour les détails et limitations.
