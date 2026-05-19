# Rapport — ml-foresight

**Auteur :** Vadim Capton · **Cours :** POC Machine Learning, Albert School
**Dépôt :** https://github.com/foresight-ml-poc/ml-foresight

---

## 1. Contexte et problème

Foresight est un produit en production : détection temps réel de news,
rattachement à un marché de prédiction Polymarket, émission d'un signal
(direction + score 0–100 par une formule heuristique). Le POC pose une
question de data science précise :

> Peut-on, par apprentissage supervisé, prédire si un signal ira dans le bon
> sens à T+24h (`direction_correct ∈ {0,1}`) mieux que l'heuristique ?

C'est une **classification binaire** sur données réelles de production
(855 signaux exploitables, ~35 jours, 39 colonnes brutes).

## 2. Méthodologie

### 2.1 Anti-fuite par allowlist explicite

Le risque n°1 d'un tel POC est la fuite de données (utiliser une variable
indisponible au moment de la prédiction). Plutôt qu'une *denylist* fragile,
`src/config.py` définit une **allowlist** : `X` ne contient que les ~19
variables connues à l'émission du signal (+ one-hot `bucket_*`). Toute
colonne future (prix T+5min…T+24h, issue, score heuristique) est
structurellement exclue. Tests unitaires : `tests/test_data.py`.

### 2.2 Les trois familles de modèles

| Modèle | Famille | Rôle |
|---|---|---|
| Régression logistique | linéaire | baseline interprétable (= heuristique « apprise ») |
| Random Forest | ensemble d'arbres / bagging | capte les interactions non linéaires |
| K-Means (k=2) | non supervisé | y a-t-il une structure latente gagnants/perdants ? |

### 2.3 Évaluation : CV **et** walk-forward

- **Validation croisée 5-fold** (mélange temporel) — la mesure usuelle.
- **Walk-forward strict** : entraîner sur le passé, tester sur le futur,
  3 blocs expansifs. C'est la *seule* mesure valide sur une série
  temporelle financière (non-stationnaire).

Discipline complémentaire : backtest **no-look-ahead** sur prix denses,
contrôle du **multiple-testing**, **Adjusted Rand Index** pour le
non-supervisé.

## 3. Résultats

### 3.1 Direction — les 3 modèles

ROC-AUC test : LogReg **0.497**, Random Forest **0.544**, K-Means **0.526**.
K-Means : **ARI ≈ 0.00** → aucun regroupement naturel. Les trois familles
sont statistiquement collées au hasard.

### 3.2 Le résultat central : la CV ment

| Cible | CV 5-fold | Walk-forward (blocs) | Moyenne WF |
|---|---|---|---|
| Direction | 0.518 | 0.467 / 0.525 / 0.468 | **0.487** |
| Magnitude | **0.549** | **0.586 / 0.494 / 0.499** | **0.526** |

La magnitude paraît prédictible en CV ; en walk-forward elle décroît vers le
hasard et passe **sous 0.50** sur la période récente. C'est une illustration
empirique de l'**alpha decay** et de la raison pour laquelle la CV standard
sur-estime la performance d'une série temporelle non-stationnaire.

### 3.3 Backtest dense, sans look-ahead

554 chemins de prix reconstruits minute par minute (API Polymarket). Règle
pré-engagée (entrée au signal ; sortie au 1ᵉʳ instant en profit, sinon à
l'horizon ; 9 couples take-profit × horizon). **9 règles sur 9 négatives, et
négatives même brut** (hors spread). MFE médian à 1 h = 0 %.

**Trois bugs corrigés en cours de route** (transparence) : marchés déjà
résolus inclus ; 53 % des signaux BUY_NO mesurés sur le token YES inversé au
lieu du vrai rendement `NO = 1 − YES` ; l'API CLOB ignorant `endTs`
(fenêtre 37 j au lieu de 24 h). Après correction le résultat négatif est
**plus** robuste — ce n'était pas un artefact de mesure.

## 4. Interprétation : efficience de marché

Un marché de prédiction intègre l'information publique en secondes ; la
boucle news→LLM→signal de Foresight prend des minutes. Les avantages des
acteurs qui gagnent (vitesse infra, données propriétaires, échelle,
market-making) sont **structurels**, pas « un meilleur modèle ». Prédire la
direction depuis de l'info publique est, par construction, proche du
non-prédictible — c'est le résultat attendu d'un marché efficient, et nos
données le confirment.

## 5. Limites

- N modéré (855) et fenêtre temporelle courte (~35 j) : variance des
  estimateurs walk-forward élevée — d'où le rapport de blocs *et* moyenne.
- `direction_correct` à T+24h est un label grossier ; on l'a complété par le
  backtest minute (§3.3), même conclusion.
- Conclusion valable **pour ces features publiques** : elle ne dit pas qu'un
  edge est impossible avec vitesse/données propriétaires (hors périmètre).
- 3ᵉ modèle « deep learning » écarté au profit de K-Means : sur N modéré
  tabulaire, K-Means donne une preuve d'absence-de-structure plus lisible
  qu'un MLP (le MLP Keras se bloquait par ailleurs sur l'environnement Apple
  Silicon / TF 2.21 — bypass assumé vu le calendrier).

## 6. Conclusion

Le POC répond honnêtement *non* à sa question, et le **démontre** par
plusieurs angles indépendants avec une méthodologie rigoureuse (anti-fuite,
walk-forward, no-look-ahead, multi-tests, non-supervisé). La contribution
n'est pas un score flatteur mais une **compétence ML mature** : savoir qu'une
validation croisée ment sur une série temporelle, l'objectiver, résister au
p-hacking, assumer un résultat négatif solide. Synthèse machine-lisible :
[`results/final_honest_verdict.json`](../results/final_honest_verdict.json).
