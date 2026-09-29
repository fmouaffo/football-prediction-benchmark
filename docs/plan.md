# Contexte projet — modélisation paris sportifs

> Fichier à joindre au modèle local avec `@file docs/plan.md` avant toute
> demande d'implémentation. Il remplace le contexte des conversations
> d'architecture, que le modèle local ne voit pas.

## Objectif

Construire un pipeline de prédiction de résultats de matchs de football,
calibré et backtesté honnêtement, afin de déterminer **s'il existe une edge
exploitable** face aux cotes de clôture du marché.

L'objectif n'est PAS de "gagner à tous les coups" — c'est de mesurer si le
modèle bat le marché, et de quelle marge.

## Stack

- Python 3.12, gestion des dépendances par `uv`
- DuckDB pour le stockage et les agrégations
- pandas / numpy pour la manipulation
- scipy pour l'optimisation (Dixon-Coles)
- scikit-learn, xgboost pour la partie ML
- matplotlib pour les diagnostics

## Données

Source : football-data.co.uk (CSV par championnat et par saison).

Table `matches` dans `data/betting.duckdb` :

| Colonne | Sens |
|---|---|
| `match_date` | date du match |
| `home_team`, `away_team` | équipes |
| `home_goals`, `away_goals` | score final |
| `result` | H / D / A |
| `psh`, `psd`, `psa` | cotes Pinnacle à l'ouverture |
| `psch`, `pscd`, `psca` | cotes Pinnacle à la **clôture** |
| `b365ch`, `b365cd`, `b365ca` | cotes Bet365 clôture |
| `maxch`, `maxcd`, `maxca` | meilleure cote du marché à la clôture |
| `avgch`, `avgcd`, `avgca` | cote moyenne du marché à la clôture |
| `season`, `division`, `league` | métadonnées |

Convention : les colonnes contenant un `c` après le préfixe bookmaker sont
les cotes de **clôture**.

## Règles métier — critiques

Ces règles priment sur toute autre considération. Une violation produit un
backtest brillant et une perte réelle.

1. **Aucune fuite temporelle.** Une feature calculée pour un match du jour J
   ne doit utiliser que des données strictement antérieures à J. Interdits :
   `groupby().transform('mean')` sur une saison complète, moyennes calculées
   sur l'ensemble du dataset, normalisations ajustées sur tout l'historique.
   Les fenêtres glissantes doivent exclure explicitement la ligne courante.

2. **Backtest sur les cotes de clôture uniquement** (`psch`, `pscd`, `psca`).
   Les cotes d'ouverture sont battables trivialement et ne reflètent pas ce
   qu'on peut réellement obtenir.

3. **Normaliser les probabilités implicites** du bookmaker avant toute
   comparaison. La somme des `1/cote` vaut ~1.05 à 1.12 : c'est la marge.
   Diviser chaque probabilité par cette somme.

4. **Tout résultat de backtest doit être accompagné** du nombre de paris, de
   l'écart-type et d'un intervalle de confiance. Un ROI sans volume n'a
   aucune signification statistique.

5. **Validation temporelle uniquement.** Jamais de `train_test_split`
   aléatoire ni de `KFold` standard : entraînement sur le passé, test sur le
   futur, avec fenêtre glissante ou expansive.

## Conventions de code

- `from __future__ import annotations`, type hints systématiques
- Pas de boucle sur les lignes d'un DataFrame : vectoriser
- Les agrégations lourdes passent par DuckDB, pas par pandas
- Les fonctions de modélisation reçoivent les données en paramètre et ne
  font aucune I/O — elles doivent être testables sans fichier
- Docstrings au format Google
- Formatage et lint par `ruff`

## Métriques d'évaluation

Dans l'ordre d'importance :

1. **Log loss** du modèle vs log loss des probabilités normalisées Pinnacle.
   Si le modèle ne fait pas mieux, il n'y a pas d'edge — point final.
2. **Brier score** — même comparaison.
3. **Courbe de calibration** — les prédictions à 30 % se réalisent-elles
   30 % du temps ?
4. **ROI simulé** avec mise fixe, sur les cotes de clôture, avec intervalle
   de confiance.

Le ROI vient en dernier : c'est la métrique la plus bruitée et la plus
facile à faire mentir.

## État d'avancement

- [ ] Ingestion des données (`src/ingest.py`)
- [ ] Baseline Dixon-Coles (`src/dixon_coles.py`)
- [ ] Comparaison log loss modèle vs marché
- [ ] Features + XGBoost
- [ ] Backtest avec intervalle de confiance
- [ ] Décision : le projet a-t-il un fond ?
