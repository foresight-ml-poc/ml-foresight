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

6 modèles tunés via 5-fold CV sur le train set, sélection par ROC-AUC.

- **Logistic Regression** : L2, GridSearchCV sur C ∈ [0.01, 10], class_weight balanced
- **Random Forest** : RandomizedSearchCV (20 iters) sur n_estimators, max_depth, min_samples_*, max_features
- **Gradient Boosting** (sklearn) : RandomizedSearchCV (20 iters)
- **LightGBM** : leaf-wise boosting, RandomizedSearchCV (20 iters)
- **XGBoost** : level-wise boosting régularisé, RandomizedSearchCV (20 iters)
- **SVM** : kernel RBF, GridSearchCV sur C et gamma

> Le 3e modèle prévu initialement était un MLP Keras. `tf.keras.fit()` se bloquait indéfiniment sur cet env (Apple Silicon, TF 2.21). Bypass plutôt que diagnostic — substitué par GradientBoosting, puis on a ajouté LightGBM/XGBoost/SVM pour une comparaison robuste.

## Résultats — v1.3.0 (données prod au 2026-05-18, test N=163)

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Heuristique | 0.534 | 0.519 | 0.675 | 0.587 | 0.536 |
| **Gradient Boosting** ★ | 0.540 | 0.532 | 0.525 | 0.528 | **0.540** |
| XGBoost | 0.528 | 0.515 | 0.625 | 0.565 | 0.529 |
| Random Forest | 0.515 | 0.506 | 0.550 | 0.527 | 0.516 |
| LightGBM | 0.503 | 0.494 | 0.538 | 0.515 | 0.504 |
| Logistic Regression | 0.497 | 0.486 | 0.425 | 0.453 | 0.496 |
| SVM (RBF) | 0.454 | 0.442 | 0.425 | 0.433 | 0.453 |

### Le finding central — et la vraie leçon du projet

En **v1.2.0** (411 samples, test N=78), le GBM battait l'heuristique de **+3.7 pts ROC-AUC** (0.570 vs 0.533). J'ai trouvé ça encourageant.

En **v1.3.0**, j'ai ré-entraîné sur la prod Hetzner qui avait accumulé **2× plus de données** (814 samples, test N=163). Résultat : le GBM ne bat plus l'heuristique que de **+0.3 pts** (0.540 vs 0.536) — une **quasi-égalité statistique**.

**Conclusion honnête : le "+3.7 pts" était essentiellement du bruit dû au petit test set (N=78).** Avec un test set 2× plus grand, l'estimateur converge et l'avantage disparaît. C'est exactement l'illustration de pourquoi on ne fait pas confiance à des métriques sur un petit échantillon — et c'est plus précieux pour un jury qu'un faux résultat flatteur.

Ce qui reste vrai :
- Les modèles d'arbres (GBM, XGBoost, RF) sont au coude-à-coude avec l'heuristique (~0.52-0.54)
- LogReg et SVM sont sous l'heuristique → la relation est non-linéaire mais le signal est faible
- L'heuristique de Foresight, pensée par des humains, est **étonnamment dure à battre**

Trade-off : l'heuristique garde un recall élevé (0.675) mais une precision moyenne (0.519). Le GBM équilibre mieux (precision 0.532, recall 0.525). Pour un trader qui veut limiter les faux positifs, le GBM reste légèrement préférable malgré la ROC-AUC quasi identique.

Plots dans [`../plots/`](../plots/).

## Limitations

1. **Le signal est faible.** Toutes les ROC-AUC sont entre 0.45 et 0.54 — proche du hasard. Soit les features disponibles ne capturent pas assez d'information, soit prédire `direction_correct` à T+24h est intrinsèquement très dur (marchés quasi-efficients).
2. **814 samples reste modeste** pour un signal aussi faible. Il faudrait peut-être 5000+ samples pour distinguer proprement les modèles.
3. **26 jours de couverture, split aléatoire.** Un `TimeSeriesSplit` serait méthodologiquement plus correct pour des données financières.
4. **Bug `direction_correct` dans Foresight.** Label calculé manuellement depuis `move_t24h_pct` — à fixer en amont.

## Conclusion

Le pipeline ML marche end-to-end : export Foresight (prod Hetzner) → cleaning → feature engineering → 6 modèles → évaluation → dashboard Streamlit, le tout scalable automatiquement avec le volume.

Le résultat final est **honnête et nuancé** : sur 814 samples, aucun modèle ML ne bat clairement l'heuristique de Foresight (GBM +0.3 pts ROC-AUC, dans le bruit). Le projet a sa vraie valeur dans la **démarche** : avoir détecté que le résultat prometteur de v1.2.0 (+3.7 pts) était un artefact du petit test set, et l'avoir corrigé en ré-entraînant sur plus de données dès qu'elles étaient disponibles. C'est ça, faire du ML rigoureux.

**Suite** :
1. Le signal est faible — soit enrichir les features (ajouter du contexte marché, historique du trader), soit accepter que prédire un marché quasi-efficient à 24h est intrinsèquement dur.
2. Laisser Foresight accumuler encore (objectif 5000+ samples) et re-runner `train.py` — le pipeline scale tout seul.
3. Si un jour un modèle bat l'heuristique de façon stable, le brancher en A/B test via `backend-foresight`.

## Annexes

- Repo : <https://github.com/foresight-ml-poc/ml-foresight>
- Release v1.3.0 : <https://github.com/foresight-ml-poc/ml-foresight/releases/tag/v1.3.0>
- Foresight (privé) : <https://github.com/vcapton-jpg/polymarket-ai>
- Référence pédagogique : <https://github.com/basile-desjuzeur/ml-poc-project>
