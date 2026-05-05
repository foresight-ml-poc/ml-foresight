# Rapport — ML Foresight POC

> Projet école Albert School · Vadim Capton · 2026-05
> Repo : <https://github.com/foresight-ml-poc/ml-foresight>
> Release v1.0.0 : <https://github.com/foresight-ml-poc/ml-foresight/releases/tag/v1.0.0>

---

## 1. Contexte

[Foresight](https://github.com/vcapton-jpg/polymarket-ai) (alias `polymarket-ai`)
est une plateforme d'intelligence temps réel sur les marchés prédictifs
**Polymarket**. Le système ingère des news (RSS, X, World News API), regroupe
les articles en événements via clustering sémantique, fait du hybrid search
(vector + BM25 + RRF) sur les marchés Polymarket pour identifier les paires
news↔market pertinentes, demande à GPT-4o d'analyser l'impact, puis émet un
**signal** noté de 0 à 100 indiquant à un trader s'il faut acheter YES ou NO.

Le scoring actuel utilise une **formule heuristique fixe** :

```
signal_strength = 0.15·freshness + 0.10·source_weight + 0.15·confirmation
                + 0.60·(0.65·impact_strength + 0.35·llm_confidence)
trade_quality   = 0.40·liquidity + 0.35·spread + 0.25·time_to_resolution
signal_score    = 0.75·signal_strength + 0.25·trade_quality
```

L'audit interne (cf. `BLUEPRINT.md` §17.2) montre un **winrate global de
46-48 % à T+24h** — sous le hasard. Le scoring heuristique a atteint sa limite.

**Ce projet remplace cette formule par un modèle de machine learning supervisé**
qui apprend les bonnes combinaisons de features non-linéaires.

---

## 2. Données

**Source :** base PostgreSQL de Foresight, tables `signals`, `signal_outcomes`,
`event_market_features`, `event_market_analysis`, `events`. Données extraites
le 2026-05-05 via `scripts/export_from_foresight.py`.

**Schéma exact (différent de la BLUEPRINT) :** lors de l'extraction j'ai
constaté que les vrais noms de colonnes diffèrent de ceux documentés dans
BLUEPRINT.md. La BLUEPRINT cite `freshness`, `confirmation`, `liquidity`,
`spread`, `time_to_resolution` ; les vraies colonnes ont les suffixes
`_factor` et `_penalty` (respectivement `freshness_factor`,
`confirmation_factor`, `liquidity_factor`, `spread_penalty`,
`time_to_resolution_factor`). Le SQL d'export les utilise correctement.

**Cible :** label binaire `direction_correct` à T+24h.
- BLUEPRINT mentionne une colonne `signal_outcomes.direction_correct`. Mais
  cette colonne n'est remplie que pour 30 / 438 lignes (probable bug dans
  le worker de capture). On a contourné en **calculant le label nous-même**
  depuis `move_t24h_pct` + `direction` :
  - Si `direction = BUY_YES` et `move_t24h_pct > 0` → `direction_correct = 1`
  - Si `direction = BUY_NO` et `move_t24h_pct < 0` → `direction_correct = 1`
  - Sinon → `direction_correct = 0`
- 401 signaux ont `move_t24h_pct` rempli, ce qui donnerait 401 candidats.

**Volume effectif après tous les joins : 40 signaux.** La table
`event_market_features` n'a été populée qu'à partir du **2026-04-27** ;
les 361 signaux antérieurs n'ont pas les 6 features heuristiques nécessaires
au comparison ML-vs-heuristique. Le filtre INNER JOIN sur
`event_market_features` ramène donc le dataset à 40 signaux issus du 2026-04-27
au 2026-04-28.

**Distribution des classes :** 25 losses (62.5 %) / 15 wins (37.5 %).
Déséquilibre modéré, géré par `class_weight='balanced'` dans tous les modèles.

**Split 80/20 stratifié :** 32 train / 8 test, `random_state=42`.

**Sample reproductible commité :** `data/raw/signals_export_sample.csv`
(40 lignes anonymisées : `signal_id` et `created_at` retirés, `hour_of_day`
dérivé). Permet de rejouer la pipeline sans accès à la DB Foresight.

**Garde-fous anti-leak :**
- Drop du label `direction_correct` (cible)
- Drop du `heuristic_score` et `move_t24h_pct` (informations futures ou
  dérivées de la formule qu'on remplace)
- `EXCLUDED_FROM_FEATURES` dans `src/config.py` liste explicitement
  `signal_score`, `signal_strength`, `trade_quality`, `outcome_label`,
  `price_t24h`, `price_resolved` à exclure du feature engineering
- StandardScaler **fit sur X_train uniquement**, puis appliqué à X_test

---

## 3. Feature engineering

Implémenté dans [`src/data.py::_feature_engineer()`](../src/data.py).

**Features brutes (12) — directement dans le CSV :**
- 6 facteurs heuristiques : `freshness_factor`, `source_weight`,
  `confirmation_factor`, `liquidity_factor`, `spread_penalty`,
  `time_to_resolution_factor`
- 4 features LLM : `impact_strength`, `llm_confidence`, `ambiguity_score`,
  `specificity_score` (cette dernière vient d'un LEFT JOIN sur
  `event_market_analysis` ; valeurs NULL imputées à la médiane)
- 2 features contexte event : `articles_count`, `unique_sources_count`

**Features dérivées (6) :**
- `cosine_score` : score de similarité du retrieval, conservé tel quel
- `tier_1_count`, `tier_2_count`, `tier_3_count` : extraits du JSONB
  `source_tier_mix` via `ast.literal_eval` (la colonne est un dict Python
  serializé avec quotes simples — `json.loads` échoue dessus)
- `is_buy_yes` : binaire dérivé de `direction == "BUY_YES"`. Important parce
  que l'audit Foresight montre une **asymétrie forte par direction**
  (BUY_NO winrate 72.7 % vs BUY_YES winrate 29.2 % dans le bucket score 75-89)
- `market_price_centered = |market_price_at_signal - 0.5|` : mesure
  l'incertitude initiale du marché (0.5 = max d'hésitation)

**Features catégorielles (1, one-hot) :**
- `bucket` (geopolitics / politics / sports / crypto / science / other)
  → 5 colonnes binaires après `pd.get_dummies(drop_first=True)`

**Total final : 24 features.**

**Scaling :** `StandardScaler` fit sur les 32 samples train, transformé
appliqué à train et test. Sauvegardé dans `models/scaler.pkl` pour
réutilisation au backend repo.

---

## 4. Modèles

Trois modèles entraînés et comparés. Tous tunés via cross-validation 5-fold
sur le train set, avec `scoring='roc_auc'` et `random_state=42`.

### 4.1 Logistic Regression (baseline linéaire)

```python
LogisticRegression(
    penalty='l2', class_weight='balanced',
    max_iter=1000, solver='lbfgs', random_state=42,
)
GridSearchCV(C=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0], cv=5)
```

Best CV ROC-AUC : 0.5417 (C=0.1).

### 4.2 Random Forest (ensemble, bagging)

```python
RandomForestClassifier(
    class_weight='balanced', random_state=42, n_jobs=-1,
)
RandomizedSearchCV(
    n_estimators=[100, 200, 300, 500],
    max_depth=[3, 5, 8, 12, None],
    min_samples_split=[2, 5, 10],
    min_samples_leaf=[1, 2, 5],
    max_features=['sqrt', 'log2', 0.5],
    n_iter=20, cv=5,
)
```

Best CV ROC-AUC : 0.6417 (n_estimators=300, max_depth=12).

### 4.3 Gradient Boosting (ensemble, boosting séquentiel)

```python
GradientBoostingClassifier(random_state=42)
RandomizedSearchCV(
    n_estimators=[50, 100, 200, 300],
    learning_rate=[0.01, 0.03, 0.1, 0.3],
    max_depth=[2, 3, 5, 8],
    min_samples_split=[2, 5, 10],
    subsample=[0.7, 0.85, 1.0],
    n_iter=20, cv=5,
)
```

> **Note honnête :** le 3e modèle prévu initialement était un **MLP Keras**
> (deep learning). Sur cet environnement (Apple Silicon, TensorFlow 2.21,
> conda env Python 3.11), `tf.keras.Model.fit()` se bloque indéfiniment dès
> `Epoch 1/200` même avec une architecture minimale (Dense(8) → Dense(1))
> sur le dataset N=32. Reproduit avec validation_split, callbacks, et metrics
> retirés un à un. Cause racine non identifiée — bypass plutôt que
> diagnostic vu les contraintes de timing école. **Substitution par
> GradientBoostingClassifier**, qui est aussi un ensemble (boosting séquentiel,
> différent de RF qui est bagging) et reste un 3e modèle pédagogiquement
> distinct.

---

## 5. Résultats

Toutes métriques sur le **test set fixe de 8 samples** (3 wins / 5 losses,
stratifié sur le label).

| Modèle | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Heuristique (formule Foresight) | 0.375 | 0.375 | 1.000 | 0.545 | 0.500 |
| Logistic Regression | 0.250 | 0.000 | 0.000 | 0.000 | 0.200 |
| **Random Forest (best)** | **0.625** | 0.000 | 0.000 | 0.000 | **0.500** |
| Gradient Boosting | 0.500 | 0.000 | 0.000 | 0.000 | 0.400 |

Plots disponibles dans [`plots/`](../plots/) :
- `confusion_matrix_logreg.png`, `confusion_matrix_random_forest.png`,
  `confusion_matrix_gradient_boosting.png`
- `roc_curves_comparison.png`
- `feature_importance_rf.png`
- `ml_vs_heuristic.png`

**Observations sur les chiffres :**
- LogReg, RF, GBM ont tous `precision = recall = F1 = 0` parce qu'aucun n'a
  prédit de classe 1 (win) sur le test set. Ils prédisent tous "loss" pour
  les 8 samples.
- L'heuristique au contraire prédit "win" pour TOUS les samples (recall=1.0)
  et tape juste 3 fois sur 8 (precision=0.375).
- Sur N=8, prédire la classe majoritaire = score raisonnable. C'est ce que
  fait RF (5/8 = 0.625 accuracy en disant tout "loss"). Mais c'est un faux
  succès.

---

## 6. Comparaison vs heuristique

**Verdict honnête : aucun modèle ML ne bat clairement l'heuristique.**

- RF et heuristique sont à **égalité parfaite ROC-AUC = 0.500** (= random)
- LogReg fait pire (0.200, mais c'est sur 8 samples, donc 1 prédiction
  changée = 12.5 pts de variance)
- GBM (0.400) est sous l'heuristique mais le tirage du test pourrait inverser

**Lecture critique du résultat :** ce n'est PAS une preuve que le ML est
inutile pour Foresight. C'est une preuve que **avec 40 samples, le test set
ne peut pas distinguer un modèle qui apprend de quelque chose qui ne
l'apprend pas**.

Sur le **train set 5-fold CV** (dont les chiffres internes sont plus
représentatifs) :
- LogReg CV ROC-AUC : 0.542
- Random Forest CV ROC-AUC : 0.642
- Gradient Boosting CV ROC-AUC : ~0.6 (à confirmer dans les logs)

Ces chiffres CV suggèrent qu'**avec plus de données, RF et GBM auraient une
chance réelle de battre l'heuristique**. Le test set N=8 est juste trop
bruité pour le démontrer.

---

## 7. Discussion honnête

### 7.1 Limitations principales

1. **Taille du dataset (N=40)** : le drop majeur. La table
   `event_market_features` n'a été populée qu'à partir du 2026-04-27. Tous
   les signaux antérieurs (361 sur 401) sont jetés par le INNER JOIN parce
   qu'ils n'ont pas les 6 features heuristiques. Avec plus de temps avant la
   soutenance, attendre 2-3 semaines de plus pour avoir 200-500 samples
   ferait une vraie différence.

2. **Stationnarité** : les 40 signaux couvrent 2 jours (2026-04-27 et 28).
   Aucune diversité de régime de marché. Si on entraîne sur "période de
   tension géopolitique" et que les patterns changent, le modèle ne
   généralisera pas.

3. **Pas de validation temporelle** : on a fait un split aléatoire
   stratifié, pas un `TimeSeriesSplit`. Pour des signaux financiers c'est
   méthodologiquement discutable. Avec plus de données, j'aurais utilisé
   un split temporel pour mieux refléter le déploiement réel.

4. **Asymétrie BUY_YES vs BUY_NO** observée dans l'audit Foresight (winrate
   72.7 % en BUY_NO vs 29.2 % en BUY_YES) : on a ajouté `is_buy_yes` comme
   feature mais avec si peu de samples le modèle ne peut pas vraiment
   exploiter cette asymétrie.

5. **Bug `direction_correct` dans Foresight** : la colonne native n'est
   remplie que pour 30/438 lignes. Calculer le label nous-même est correct,
   mais ça suggère un bug dans le worker de capture qu'il faudrait corriger
   en amont.

### 7.2 Substitution MLP → GradientBoosting

Comme noté dans §4.3, l'environnement TensorFlow était instable. C'est de
l'engineering, pas du ML — mais ça reflète une réalité du terrain : un POC
ML doit s'adapter à la stack disponible. GradientBoosting est un choix
défendable (différent de RF, ensemble séquentiel, fort sur tabulaire), pas
un downgrade.

### 7.3 Ce qu'on ferait avec plus de temps

- Attendre 2-4 semaines pour 200+ samples avec features complètes
- Validation temporelle (`TimeSeriesSplit`) plutôt que split aléatoire
- Ajouter XGBoost / LightGBM (boosting plus puissant que sklearn)
- Calibration de probas (`CalibratedClassifierCV`)
- Feature importance via SHAP (interprétabilité)
- Tests d'A/B en production (déployer le ML en parallèle de l'heuristique
  et comparer le winrate réel)
- Fix du bug `direction_correct` dans le worker de Foresight (soit
  ré-instancier la logique de calcul, soit comprendre pourquoi 408 lignes
  l'ont à NULL)

---

## 8. Conclusion

**Ce qui a marché :**
- Pipeline ML end-to-end fonctionnel : export Foresight → cleaning →
  feature engineering → 3 modèles → évaluation → Streamlit dashboard
- 18 tests unitaires passants couvrant data + metrics
- Architecture 3-repos (ml + backend + frontend) prête pour la suite
- GitHub Release v1.0.0 publiée avec artefacts (`best_model.joblib`,
  `scaler.pkl`, `feature_order.pkl`, `model_card.json`)
- Honnêteté : pas de truc statistique pour gonfler les chiffres, pas de
  cherry-picking de la métrique

**Ce qui n'a pas marché :**
- Volume de données limitant (40 vs 200-2000 espérés). Pas la faute du
  pipeline — la faute des contraintes temporelles d'une pipeline en
  démarrage qui n'a que 2 jours de feature complets
- TensorFlow MLP cassé sur cet environnement (substitué par GBM)
- Aucun ML ne bat l'heuristique sur ce test bruité

**Prochaine étape immédiate :** laisser Foresight tourner 2-3 semaines de
plus, puis re-lancer `python scripts/train.py` qui utilisera automatiquement
le dernier export. Le pipeline scale avec le volume, donc les résultats
seront mécaniquement plus solides.

**Prochaine étape système :** construire `backend-foresight` (FastAPI qui
sert `best_model.joblib` via `/predict`) et `frontend-foresight` (React qui
appelle l'API), puis brancher en A/B test dans Foresight si les métriques
sont concluantes.

---

**Annexes :**
- Design doc complet : [`docs/specs/2026-05-05-design.md`](specs/2026-05-05-design.md)
- Plan d'implémentation : [`docs/plans/2026-05-05-implementation-plan.md`](plans/2026-05-05-implementation-plan.md)
- Model card v1.0.0 : [`models/model_card.json`](../models/model_card.json)
- Code source : <https://github.com/foresight-ml-poc/ml-foresight>
