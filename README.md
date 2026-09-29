# Football Prediction Benchmark

Évaluation de l'efficience des marchés de paris sportifs : un modèle
statistique entraîné sur les seuls résultats historiques peut-il battre les
cotes de clôture d'un bookmaker à faible marge ?

**Réponse courte : non.** Le modèle perd 0,024 de log loss contre Pinnacle sur
29 046 matchs hors échantillon, et l'écart se creuse dans les championnats les
moins liquides. Les résultats détaillés sont plus bas.

---

## La question

Les cotes d'un bookmaker encodent une distribution de probabilité sur l'issue
d'un match. Pinnacle, dont la marge est la plus faible du marché (~2,5 %),
constitue le meilleur estimateur public disponible des probabilités réelles.

Ce projet mesure l'écart entre un modèle de référence de la littérature et
ce benchmark, sur 19 championnats européens et 13 saisons.

## Données

Source : [football-data.co.uk](https://www.football-data.co.uk/) — résultats
et cotes historiques, CSV par championnat et par saison.

- ~110 000 matchs, saisons 2010/11 à 2025/26
- 19 championnats européens (5 grands championnats, divisions inférieures,
  championnats de taille moyenne)
- Cotes d'ouverture et de **clôture** pour plusieurs bookmakers

Seules les cotes de **clôture** sont utilisées pour l'évaluation. Les cotes
d'ouverture sont battables trivialement et ne reflètent pas ce qu'un parieur
peut réellement obtenir.

## Le modèle

Dixon-Coles (1997), un Poisson bivarié :

- Une force d'**attaque** et une force de **défense** par équipe, plus un
  avantage du terrain global
- Une correction `τ` sur les quatre scores faibles (0-0, 1-0, 0-1, 1-1), que
  le Poisson indépendant modélise mal — les deux équipes ne marquent pas
  indépendamment l'une de l'autre
- Une **pondération exponentielle temporelle** : un match ancien pèse moins
  qu'un match récent (ξ = 0,0018/jour, demi-vie ≈ 385 jours)
- Estimation par maximum de vraisemblance pondérée (`scipy.optimize`,
  L-BFGS-B), sous contrainte d'identifiabilité (moyenne des attaques nulle)

Un modèle par championnat : les forces ne sont pas comparables entre ligues,
et les équipes changent rarement de championnat.

## Protocole de validation

Validation temporelle par **fenêtre glissante**, jamais de découpage
aléatoire.

- Réajustement tous les 30 jours
- Chaque ajustement utilise 3 ans d'historique **strictement antérieur** à la
  date de prédiction
- Un garde-fou lève une exception si une date postérieure entre dans
  l'échantillon d'entraînement
- Les équipes absentes de la fenêtre d'entraînement (promues) sont exclues
  plutôt qu'imputées

Environ 130 réajustements par championnat.

## Métriques

Par ordre d'importance :

1. **Log loss** du modèle contre log loss des probabilités Pinnacle
   normalisées (marge retirée par normalisation proportionnelle)
2. **Brier score**
3. **Courbe de calibration** — les prédictions à 30 % se réalisent-elles 30 %
   du temps ?

L'écart de log loss est accompagné de son intervalle de confiance à 95 %,
calculé sur les différences appariées.

Le ROI simulé n'est pas rapporté : c'est la métrique la plus bruitée et la
plus facile à faire mentir.

---

## Résultats

### Global

Sur 29 046 matchs hors échantillon (6 grands championnats, 2013-2026) :

| | modèle | marché | écart |
|---|---|---|---|
| log loss | 1,00827 | 0,97893 | **+0,02934** |
| Brier | 0,60042 | 0,58303 | +0,01739 |

Écart de log loss : **+0,0293 ± 0,0029** (IC 95 %). Le marché gagne de façon
statistiquement nette.

### La calibration n'explique pas l'écart

Hypothèse testée : le modèle est trop confiant aux extrêmes (quand il annonce
6,8 %, l'issue se réalise 14,2 % du temps).

Deux corrections post-hoc, dont le paramètre est appris sur 2013-2020 et
évalué sur 2021-2026 :

| | log loss (test) | vs brut |
|---|---|---|
| Dixon-Coles brut | 1,00692 | — |
| + temperature scaling (t = 1,14) | 1,00652 | −0,00041 |
| + Platt scaling | 1,00589 | −0,00104 |
| Marché (Pinnacle) | 0,98173 | −0,02519 |

**La calibration comble 4 % de l'écart.** Le modèle est déjà bien calibré sur
la plage 0,3-0,6 ; les zones mal calibrées pèsent trop peu dans l'échantillon.

Conclusion : l'écart n'est pas un problème de calibration mais un **déficit
d'information**. Aucune transformation des sorties ne peut créer de
l'information absente en entrée.

### Par championnat

Trié par écart croissant au marché :

| championnat | n | overround | écart | IC 95 % |
|---|---|---|---|---|
| 2. Bundesliga | 3 657 | 1,0295 | 0,0213 | ±0,0074 |
| Premier League | 4 717 | 1,0243 | 0,0228 | ±0,0066 |
| Scottish Premiership | 2 710 | 1,0292 | 0,0252 | ±0,0075 |
| Primeira Liga | 3 694 | 1,0299 | 0,0262 | ±0,0081 |
| Eredivisie | 3 689 | 1,0307 | 0,0272 | ±0,0080 |
| Championship | 6 784 | 1,0267 | 0,0278 | ±0,0057 |
| Bundesliga | 3 760 | 1,0249 | 0,0297 | ±0,0093 |
| Ligue 1 | 4 412 | 1,0248 | 0,0308 | ±0,0079 |
| La Liga | 4 674 | 1,0248 | 0,0312 | ±0,0078 |
| Süper Lig | 4 045 | 1,0308 | 0,0351 | ±0,0081 |
| Serie A | 4 699 | 1,0249 | 0,0363 | ±0,0080 |
| Jupiler League | 3 266 | 1,0327 | 0,0369 | ±0,0113 |
| Super League Grèce | 2 897 | 1,0388 | 0,0371 | ±0,0103 |
| Serie B | 4 874 | 1,0292 | 0,0402 | ±0,0093 |
| Scottish Championship | 1 587 | 1,0363 | 0,0417 | ±0,0194 |
| League Two | 6 515 | 1,0324 | 0,0419 | ±0,0088 |
| Ligue 2 | 4 360 | 1,0319 | 0,0419 | ±0,0094 |
| Segunda División | 5 553 | 1,0302 | 0,0444 | ±0,0091 |
| League One | 6 478 | 1,0317 | 0,0448 | ±0,0089 |

**Le modèle ne bat le marché nulle part.** L'écart le plus faible
(2. Bundesliga) reste statistiquement positif, et son avantage sur les autres
championnats n'est pas distinguable du bruit — c'est le meilleur de 19
tirages, ce qui est attendu par construction.

### L'hypothèse des marchés moins efficients est réfutée

Hypothèse testée : les divisions inférieures, moins travaillées par les
bookmakers, devraient être plus faciles à battre.

**La corrélation entre overround et écart au marché est positive.** Là où
Pinnacle élargit sa marge, le modèle est *plus* mauvais, pas meilleur.

Interprétation : une marge élargie ne signale pas un bookmaker moins
compétent, mais un marché plus **bruité** — effectifs instables, prêts,
turnover élevé. La performance passée y prédit moins bien la performance
future, ce qui pénalise davantage un modèle qui ne dispose que des scores.

League One et League Two, les deux championnats les plus bas du tableau,
affichent un écart presque double de celui de la Premier League.

---

## Ce que le marché sait et que le modèle ignore

L'écart mesuré correspond à l'information absente des données d'entrée :

- Compositions, blessures, suspensions
- Qualité de jeu (xG) plutôt que résultats bruts — une victoire 2-0 en tirant
  4 fois n'est pas une victoire 2-0 en tirant 25 fois
- Changements d'entraîneur, mercato
- Congestion du calendrier, participation aux coupes européennes
- Enjeu du match en fin de saison
- L'argent informé lui-même, qui déplace les cotes

## Conclusion

Un modèle Dixon-Coles correctement validé, entraîné sur 13 saisons et
19 championnats, perd de 0,024 à 0,045 de log loss contre les cotes de
clôture Pinnacle. L'écart est statistiquement net partout et résiste à la
calibration post-hoc.

Pour qu'une stratégie soit exploitable, il ne suffirait pas d'égaler Pinnacle
(2,5 % de marge) : il faudrait battre les bookmakers accessibles au grand
public, dont la marge se situe entre 7 et 10 %.

---

## Structure

```
.
├── src/
│   ├── ingest.py        # téléchargement et normalisation -> DuckDB
│   ├── dixon_coles.py   # modèle et ajustement par MLE
│   ├── evaluate.py      # backtest temporel contre le marché
│   ├── calibrate.py     # temperature et Platt scaling
│   └── by_league.py     # écart au marché par championnat
├── docs/
│   └── plan.md          # contexte, conventions, règles métier
└── data/                # base DuckDB (non versionnée)
```

## Installation

```bash
uv sync
uv run src/ingest.py          # ~10 min, télécharge les données
uv run python -m src.evaluate # ~3-5 h, backtest complet
uv run python -m src.by_league
```

Python 3.12, DuckDB, pandas, scipy, scikit-learn.

## Références

Dixon, M. J. & Coles, S. G. (1997). *Modelling Association Football Scores
and Inefficiencies in the Football Betting Market*. Journal of the Royal
Statistical Society: Series C, 46(2), 265-280.

## Licence

MIT
