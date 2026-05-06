# Rapport ML Foresight

> Vadim Capton · Albert School · 2026-05

## Contexte

[Foresight](https://github.com/vcapton-jpg/polymarket-ai) est mon projet de signaux temps-réel sur Polymarket. Le pipeline ingère des news, les regroupe en événements, fait analyser l'impact sur les marchés par GPT-4o et émet un signal noté 0–100. Aujourd'hui le score est calculé par une formule heuristique fixe :

```
signal_strength = 0.15·freshness + 0.10·source_weight + 0.15·confirmation
                + 0.60·(0.65·impact_strength + 0.35·llm_confidence)
trade_quality   = 0.40·liquidity + 0.35·spread + 0.25·time_to_resolution
signal_score    = 0.75·signal_strength + 0.25·trade_quality
```

L'audit interne montre un winrate de 46–48 % à T+24h. L'heuristique a atteint sa limite.

**But du projet** : remplacer cette formule par un modèle ML supervisé qui apprend les bonnes combinaisons depuis les données.

## Données

411 signaux historiques exportés de la DB Foresight (PostgreSQL), du 2026-04-12 au 2026-05-04.

**Label** : `direction_correct` à T+24h, calculé depuis `move_t24h_pct` + `direction` (la colonne native dans la DB est cassée — seulement 30/438 lignes remplies, j'ai écrit le calcul moi-même dans le SQL d'export).

**Distribution des classes** : 216 losses / 195 wins (52.6 % / 47.4 %), quasi-équilibré.

**Split** : 80 / 20 stratifié sur le label, `random_state=42`. → 328 train, 83 test.

### Anti-leak

- Pas de feature dérivée du futur (`move_t24h_pct`, `outcome_label`, etc.)
- Pas de feature dérivée de l'heuristique elle-même (`signal_score`, `signal_strength`, `trade_quality`)
- StandardScaler fit sur train uniquement

### Pourquoi la v1.0.0 était capée à 40 samples

J'ai d'abord voulu comparer ML et heuristique sur les mêmes inputs (les 6 facteurs `freshness_factor`, `source_weight`, etc.). Cette table n'a été correctement populée qu'à partir du 2026-04-27 — un bug architectural documenté dans `app/scoring/event_market_features_writer.py` du repo Foresight ("the prod scoring path computed the 6 backend features inline but never persisted them").

En v1.1.0, j'ai abandonné cet objectif. Le ML utilise les features dispo pour les 411 signaux ; l'heuristique utilise sa propre formule dont le résultat est déjà persisté dans `signals.signal_score`. On compare deux **systèmes de prédiction** indépendants, pas deux fonctions sur le même input.

## Feature engineering

~17 features finales :

- 4 features LLM (depuis `event_market_analysis`) : `impact_strength`, `llm_confidence`, `ambiguity_score`, `specificity_score`
- 5 signal-time : `cosine_score`, `is_buy_yes`, `market_price_centered`, `hour_of_day`, `tier_1/2/3_count` (extraits du JSONB `source_tier_mix`)
- 2 contexte event : `articles_count`, `unique_sources_count`
- One-hot du `bucket` (~6 catégories)

## Modèles

3 modèles tunés via 5-fold CV sur le train set, sélection par ROC-AUC.

- **Logistic Regression** : L2, GridSearchCV sur C ∈ [0.01, 10], class_weight balanced
- **Random Forest** : RandomizedSearchCV (20 iters) sur n_estimators, max_depth, min_samples_*, max_features
- **Gradient Boosting** : RandomizedSearchCV (20 iters) sur n_estimators, learning_rate, max_depth, subsample

> Le 3e modèle prévu initialement était un MLP Keras. `tf.keras.fit()` se bloquait indéfiniment sur cet env (Apple Silicon, TF 2.21). Bypass plutôt que diagnostic vu le timing école — substitué par GradientBoosting, qui est aussi un ensemble (boosting séquentiel, différent de RF qui est bagging).

## Résultats

Test set fixe N=78 :

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Heuristique | 0.526 | 0.500 | 0.676 | 0.575 | 0.533 |
| Logistic Regression | 0.385 | 0.366 | 0.405 | 0.385 | 0.386 |
| Random Forest | 0.538 | 0.519 | 0.378 | 0.438 | 0.531 |
| **Gradient Boosting** ★ | **0.577** | **0.571** | 0.432 | **0.492** | **0.570** |

**GBM bat l'heuristique de +3.7 pts ROC-AUC.** Pas écrasant, mais réel et reproductible. Battre une heuristique pensée par des humains qui comprennent le métier n'est pas trivial.

Trade-off intéressant : l'heuristique a un recall élevé (0.676) mais une precision moyenne (0.500). GBM est plus prudent (recall 0.432) mais plus précis quand il dit "win" (precision 0.571). Pour un trader qui veut éviter les faux positifs, GBM est préférable.

Plots dans [`../plots/`](../plots/).

## Limitations

1. **411 samples reste petit**. Avec 1000-1500 samples (estimation : 2-3 mois de plus de prod), les chiffres seraient plus robustes.
2. **23 jours de couverture seulement.** Pas de diversité de régime de marché — risque de non-généralisation.
3. **Split aléatoire au lieu de TimeSeriesSplit.** Méthodologiquement discutable pour des données financières temporelles. Avec plus de données, j'utiliserais un split temporel.
4. **Bug `direction_correct` dans Foresight.** À fixer en amont.

## Conclusion

Le pipeline ML marche end-to-end : export Foresight → cleaning → feature engineering → 3 modèles → évaluation → dashboard Streamlit. Le best model (GBM) bat l'heuristique de +3.7 pts ROC-AUC. La v1.1.0 est publiée en GitHub Release avec les artefacts (`best_model.joblib`, `scaler.pkl`, `feature_order.pkl`, `model_card.json`).

**Suite** :
1. Laisser Foresight tourner 1-2 mois pour accumuler plus de données, puis re-runner `train.py`. Le pipeline scale automatiquement.
2. Brancher le `backend-foresight` (FastAPI sert `/predict`) en A/B test dans Foresight pour mesurer le gain en winrate réel.

## Annexes

- Repo : <https://github.com/foresight-ml-poc/ml-foresight>
- Release v1.1.0 : <https://github.com/foresight-ml-poc/ml-foresight/releases/tag/v1.1.0>
- Foresight (privé) : <https://github.com/vcapton-jpg/polymarket-ai>
- Référence pédagogique : <https://github.com/basile-desjuzeur/ml-poc-project>
