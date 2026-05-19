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

889 signaux historiques exportés de la **prod Hetzner** (via Tailscale), du 2026-04-12 au 2026-05-19 (35 jours). Après cleaning : 855 exploitables.

**Label** : `direction_correct` à T+24h, calculé depuis `move_t24h_pct` + `direction` (la colonne native dans la DB est cassée — ~30/438 lignes remplies, j'ai écrit le calcul moi-même dans le SQL d'export).

**Distribution des classes** : 431 losses / 424 wins (~50/50), quasi-équilibré. Winrate global 49.9 %.

**Split** : 80 / 20 stratifié sur le label, `random_state=42`. → 684 train, 171 test.

**Export enrichi** : 39 colonnes (trajectoire de prix T+5m→T+24h, microstructure marché, internals heuristiques) pour alimenter le dashboard d'analyse — mais le ML n'en consomme que 19 (voir anti-leak).

### Anti-leak — allowlist explicite

Le feature engineering utilise un **allowlist** (`FEATURE_BASE_COLUMNS` dans `src/config.py`), pas un denylist. X = uniquement les 19 features autorisées + les one-hot `bucket_*`. Toute autre colonne du CSV enrichi (prix futurs, outcome, internals heuristiques, free-text) n'est jamais sélectionnée — impossible de leaker même en enrichissant l'export.

- Aucune feature dérivée du futur (`move_*`, `price_t*`, `outcome_label`)
- Aucune feature dérivée de l'heuristique (`signal_score`, `signal_strength`, `trade_quality`)
- StandardScaler fit sur train uniquement

### Pourquoi la v1.0.0 était capée à 40 samples

J'ai d'abord voulu comparer ML et heuristique sur les mêmes inputs (les 6 facteurs `freshness_factor`, `source_weight`, etc.). Cette table n'a été correctement populée qu'à partir du 2026-04-27 — un bug architectural documenté dans `app/scoring/event_market_features_writer.py` du repo Foresight ("the prod scoring path computed the 6 backend features inline but never persisted them").

En v1.1.0, j'ai abandonné cet objectif. Le ML utilise les features dispo pour les 411 signaux ; l'heuristique utilise sa propre formule dont le résultat est déjà persisté dans `signals.signal_score`. On compare deux **systèmes de prédiction** indépendants, pas deux fonctions sur le même input.

## Feature engineering

19 features finales (allowlist strict) :

- 4 features LLM (depuis `event_market_analysis`) : `impact_strength`, `llm_confidence`, `ambiguity_score`, `specificity_score`
- 6 signal-time : `cosine_score`, `is_buy_yes`, `market_price_centered`, `hour_of_day`, `tier_1/2/3_count` (extraits du JSONB `source_tier_mix`)
- 2 contexte event : `articles_count`, `unique_sources_count`
- One-hot du `bucket` (~6 catégories → bucket_economics/geopolitics/other/politics/science/sports)

## Modèles

6 modèles tunés via 5-fold CV sur le train set, sélection par ROC-AUC.

- **Logistic Regression** : L2, GridSearchCV sur C ∈ [0.01, 10], class_weight balanced
- **Random Forest** : RandomizedSearchCV (20 iters) sur n_estimators, max_depth, min_samples_*, max_features
- **Gradient Boosting** (sklearn) : RandomizedSearchCV (20 iters)
- **LightGBM** : leaf-wise boosting, RandomizedSearchCV (20 iters)
- **XGBoost** : level-wise boosting régularisé, RandomizedSearchCV (20 iters)
- **SVM** : kernel RBF, GridSearchCV sur C et gamma

> Le 3e modèle prévu initialement était un MLP Keras. `tf.keras.fit()` se bloquait indéfiniment sur cet env (Apple Silicon, TF 2.21). Bypass plutôt que diagnostic — substitué par GradientBoosting, puis on a ajouté LightGBM/XGBoost/SVM pour une comparaison robuste.

## Résultats — v1.4.0 (données prod au 2026-05-19, test N=171)

855 signaux exploitables, 35 jours de prod. Export enrichi (39 colonnes pour
l'analyse) mais ML sur un **allowlist strict de 19 features** (anti-leak).

| Modèle | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Heuristique | 0.544 | 0.602 | 0.545 |
| **Random Forest** ★ | 0.573 | 0.568 | **0.573** |
| XGBoost | 0.538 | 0.573 | 0.539 |
| LightGBM | 0.532 | 0.556 | 0.532 |
| Gradient Boosting | 0.509 | 0.553 | 0.509 |
| SVM (RBF) | 0.503 | 0.525 | 0.503 |
| Logistic Regression | 0.497 | 0.488 | 0.497 |

### Les deux findings honnêtes — la vraie leçon du projet

**Finding 1 — l'écart ML/heuristique est modeste et bruité.** Selon le refresh :

| Version | Dataset | Test | Best model | Écart vs heuristique |
|---|---|---|---|---|
| v1.2.0 | 411 | N=78 | GradientBoosting | +3.7 pts |
| v1.3.0 | 814 | N=163 | GradientBoosting | +0.3 pts |
| v1.4.0 | 855 | N=171 | **Random Forest** | +2.8 pts |

Le "+3.7 pts" de v1.2.0 était surtout du bruit (petit test set). Avec plus
de données l'écart oscille entre 0 et 3 pts. Le ML égale l'heuristique sans
la dominer franchement.

**Finding 2 — le best model est instable.** GradientBoosting gagnait en
v1.2/v1.3, Random Forest gagne en v1.4 sur quasiment les mêmes données. Quand
le signal est aussi faible (toutes les ROC-AUC entre 0.50 et 0.57), le
classement des modèles change d'un dataset à l'autre. Conclusion : ne pas
sur-interpréter "tel modèle est le meilleur" près du hasard. C'est une leçon
ML aussi importante que les chiffres eux-mêmes.

Ce qui reste vrai :
- Les modèles d'arbres (RF, XGBoost, LightGBM) sont au coude-à-coude avec
  l'heuristique (~0.53-0.57)
- LogReg et SVM sont sous l'heuristique → relation non-linéaire, signal faible
- L'heuristique de Foresight, pensée par des humains, est **dure à battre**
- Le winrate global est de **49.9 %** : prédire un marché quasi-efficient à
  T+24h est intrinsèquement difficile

Trade-off : l'heuristique garde un recall élevé (0.694) mais une precision moyenne (0.532). Le Random Forest équilibre mieux. Pour un trader qui veut limiter les faux positifs, le RF est légèrement préférable.

Plots dans [`../plots/`](../plots/) — et dashboard Streamlit interactif (`make app`).

## Limitations

1. **Le signal est faible.** Toutes les ROC-AUC sont entre 0.50 et 0.57 — proche du hasard. Soit les features disponibles ne capturent pas assez d'information, soit prédire `direction_correct` à T+24h est intrinsèquement très dur (marchés quasi-efficients).
2. **855 samples reste modeste** pour un signal aussi faible. Il faudrait peut-être 5000+ samples pour distinguer proprement les modèles (et stabiliser le best model).
3. **35 jours de couverture, split aléatoire.** Un `TimeSeriesSplit` serait méthodologiquement plus correct pour des données financières.
4. **Bug `direction_correct` dans Foresight.** Label calculé manuellement depuis `move_t24h_pct` — à fixer en amont.

## Conclusion

Le pipeline ML marche end-to-end : export Foresight (prod Hetzner) → cleaning → feature engineering → 6 modèles → évaluation → dashboard Streamlit riche (winrate par bucket/direction, calibration, trajectoire de prix), le tout scalable automatiquement avec le volume.

Le résultat final est **honnête et nuancé** : sur 855 samples, le ML (RandomForest) bat l'heuristique de +2.8 pts ROC-AUC, mais l'écart oscille entre 0 et 3.7 pts selon le refresh et le best model change (GBM→RF). Le projet a sa vraie valeur dans la **démarche** : pipeline reproductible, anti-leak par allowlist explicite, et honnêteté sur l'instabilité du signal. C'est ça, faire du ML rigoureux.

**Suite** :
1. Le signal est faible — soit enrichir les features (ajouter du contexte marché, historique du trader), soit accepter que prédire un marché quasi-efficient à 24h est intrinsèquement dur.
2. Laisser Foresight accumuler encore (objectif 5000+ samples) et re-runner `train.py` — le pipeline scale tout seul.
3. Si un jour un modèle bat l'heuristique de façon stable, le brancher en A/B test via `backend-foresight`.

## Annexes

- Repo : <https://github.com/foresight-ml-poc/ml-foresight>
- Release v1.4.0 : <https://github.com/foresight-ml-poc/ml-foresight/releases/tag/v1.4.0>
- Foresight (privé) : <https://github.com/vcapton-jpg/polymarket-ai>
- Référence pédagogique : <https://github.com/basile-desjuzeur/ml-poc-project>
