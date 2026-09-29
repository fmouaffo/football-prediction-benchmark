"""
Modele Dixon-Coles (1997) : Poisson bivarie avec correction sur les scores
faibles et ponderation temporelle exponentielle.

Chaque equipe recoit une force d'attaque et une force de defense. Le nombre
de buts attendu pour un match est :

    lambda (domicile) = exp(attaque_dom - defense_ext + avantage_terrain)
    mu     (exterieur) = exp(attaque_ext - defense_dom)

Le Poisson simple suppose l'independance des deux scores, ce qui sous-estime
les 0-0 et 1-1 et sur-estime les 1-0 et 0-1. La fonction tau corrige les
quatre scores concernes.

Usage:
    uv run python -m src.dixon_coles
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10  # troncature de la matrice de scores


# --------------------------------------------------------------------------
# Correction tau
# --------------------------------------------------------------------------

def tau(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    lam: np.ndarray,
    mu: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Facteur correctif de Dixon-Coles sur les quatre scores faibles.

    rho < 0 augmente la probabilite des nuls 0-0 et 1-1.
    Vaut 1 partout ailleurs (aucune correction).
    """
    out = np.ones_like(lam, dtype=float)

    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)

    out[m00] = 1.0 - lam[m00] * mu[m00] * rho
    out[m01] = 1.0 + lam[m01] * rho
    out[m10] = 1.0 + mu[m10] * rho
    out[m11] = 1.0 - rho

    # tau doit rester strictement positif : l'optimiseur peut explorer des
    # zones ou la correction rend la vraisemblance negative.
    return np.clip(out, 1e-10, None)


# --------------------------------------------------------------------------
# Modele
# --------------------------------------------------------------------------

@dataclass
class DixonColesModel:
    """Parametres ajustes du modele."""

    teams: list[str]
    attack: dict[str, float]
    defence: dict[str, float]
    home_advantage: float
    rho: float

    def score_matrix(self, home: str, away: str) -> np.ndarray:
        """Matrice (MAX_GOALS+1, MAX_GOALS+1) des probabilites de score exact.

        L'element [i, j] est P(home marque i, away marque j).
        """
        if home not in self.attack or away not in self.attack:
            raise KeyError(f"Equipe inconnue du modele : {home} ou {away}")

        lam = np.exp(self.attack[home] - self.defence[away] + self.home_advantage)
        mu = np.exp(self.attack[away] - self.defence[home])

        goals = np.arange(MAX_GOALS + 1)
        p_home = poisson.pmf(goals, lam)
        p_away = poisson.pmf(goals, mu)
        matrix = np.outer(p_home, p_away)

        # Correction tau sur les 4 cellules concernees
        hg, ag = np.meshgrid(goals, goals, indexing="ij")
        correction = tau(
            hg.ravel(),
            ag.ravel(),
            np.full(hg.size, lam),
            np.full(hg.size, mu),
            self.rho,
        ).reshape(matrix.shape)

        matrix = matrix * correction
        return matrix / matrix.sum()  # renormalisation apres troncature

    def predict_1x2(self, home: str, away: str) -> tuple[float, float, float]:
        """Probabilites (domicile, nul, exterieur)."""
        m = self.score_matrix(home, away)
        p_home = float(np.tril(m, -1).sum())  # home > away
        p_draw = float(np.trace(m))
        p_away = float(np.triu(m, 1).sum())
        return p_home, p_draw, p_away


# --------------------------------------------------------------------------
# Ajustement
# --------------------------------------------------------------------------

def time_weights(
    match_dates: pd.Series, reference_date: pd.Timestamp, xi: float
) -> np.ndarray:
    """Poids exponentiels decroissants avec l'anciennete.

    xi est exprime par jour. xi = 0.0018 donne une demi-vie d'environ
    385 jours : un match d'il y a un an pese moitie moins qu'un match du jour.
    """
    age_days = (reference_date - match_dates).dt.days.to_numpy(dtype=float)
    return np.exp(-xi * np.clip(age_days, 0, None))


def fit(
    matches: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
    xi: float = 0.0018,
) -> DixonColesModel:
    """Ajuste le modele par maximum de vraisemblance ponderee.

    Args:
        matches: doit contenir match_date, home_team, away_team,
            home_goals, away_goals. TOUTES les lignes doivent etre
            anterieures a reference_date (pas de fuite temporelle).
        reference_date: date a laquelle le modele est ajuste. Par defaut,
            le dernier match du jeu fourni.
        xi: taux de decroissance temporelle par jour.

    Returns:
        Le modele ajuste.
    """
    required = {"match_date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    if reference_date is None:
        reference_date = matches["match_date"].max()

    # Garde-fou anti-fuite : aucun match posterieur a la date de reference
    if (matches["match_date"] > reference_date).any():
        raise ValueError(
            "Des matchs sont posterieurs a reference_date : fuite temporelle."
        )

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = matches["home_team"].map(idx).to_numpy()
    away_idx = matches["away_team"].map(idx).to_numpy()
    hg = matches["home_goals"].to_numpy(dtype=int)
    ag = matches["away_goals"].to_numpy(dtype=int)
    w = time_weights(matches["match_date"], reference_date, xi)

    # Vecteur de parametres : [attaque (n), defense (n), avantage terrain, rho]
    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        attack = params[:n]
        defence = params[n : 2 * n]
        return attack, defence, params[2 * n], params[2 * n + 1]

    def neg_log_likelihood(params: np.ndarray) -> float:
        attack, defence, home_adv, rho = unpack(params)

        # Contrainte d'identifiabilite : la moyenne des attaques est nulle.
        # Sans elle, ajouter une constante a toutes les attaques et la
        # retrancher a toutes les defenses donne la meme vraisemblance.
        attack = attack - attack.mean()

        lam = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
        mu = np.exp(attack[away_idx] - defence[home_idx])

        ll = (
            poisson.logpmf(hg, lam)
            + poisson.logpmf(ag, mu)
            + np.log(tau(hg, ag, lam, mu, rho))
        )
        return -float(np.sum(w * ll))

    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
    bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-1, 1), (-0.2, 0.2)]

    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 500},
    )

    if not result.success:
        print(f"[avertissement] convergence incomplete : {result.message}")

    attack, defence, home_adv, rho = unpack(result.x)
    attack = attack - attack.mean()

    return DixonColesModel(
        teams=teams,
        attack=dict(zip(teams, attack)),
        defence=dict(zip(teams, defence)),
        home_advantage=float(home_adv),
        rho=float(rho),
    )


# --------------------------------------------------------------------------
# Demonstration
# --------------------------------------------------------------------------

def main() -> None:
    import duckdb

    con = duckdb.connect("data/betting.duckdb", read_only=True)
    df = con.execute(
        """
        SELECT match_date, home_team, away_team, home_goals, away_goals
        FROM matches
        WHERE league = 'Premier League'
          AND match_date < DATE '2024-08-01'
          AND match_date >= DATE '2021-08-01'
        ORDER BY match_date
        """
    ).fetchdf()
    con.close()

    print(f"Ajustement sur {len(df)} matchs...")
    model = fit(df)

    print(f"\nAvantage du terrain : {model.home_advantage:.3f}")
    print(f"rho (correction scores faibles) : {model.rho:.4f}")

    strengths = (
        pd.DataFrame(
            {
                "attaque": pd.Series(model.attack),
                "defense": pd.Series(model.defence),
            }
        )
        .assign(force=lambda d: d["attaque"] + d["defense"])
        .sort_values("force", ascending=False)
    )
    print("\nTop 5 / Bottom 5 :")
    print(pd.concat([strengths.head(5), strengths.tail(5)]).round(3))


if __name__ == "__main__":
    main()
