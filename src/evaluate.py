"""
Evaluation de Dixon-Coles contre les cotes de cloture Pinnacle.

Protocole : validation temporelle par fenetre glissante. Pour chaque bloc de
matchs, le modele est reajuste UNIQUEMENT sur les matchs anterieurs, puis
predit le bloc suivant. Aucune information future n'entre dans l'ajustement.

La question a laquelle ce script repond :
    le modele fait-il mieux que le marche, en log loss ?
Si non, il n'y a pas d'edge — quelle que soit l'allure du ROI simule.

Usage:
    uv run python -m src.evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dixon_coles import fit

TRAIN_YEARS = 3       # profondeur d'historique pour chaque ajustement
REFIT_EVERY_DAYS = 30  # frequence de reajustement
MIN_TRAIN_MATCHES = 500


def normalise_odds(
    odds_home: np.ndarray, odds_draw: np.ndarray, odds_away: np.ndarray
) -> np.ndarray:
    """Cotes decimales -> probabilites normalisees (marge retiree).

    La somme des inverses des cotes vaut 1 + marge. On divise par cette
    somme pour obtenir des probabilites qui somment a 1. C'est la methode
    proportionnelle : simple, et suffisante pour un bookmaker a faible marge
    comme Pinnacle.
    """
    raw = np.column_stack([1 / odds_home, 1 / odds_draw, 1 / odds_away])
    return raw / raw.sum(axis=1, keepdims=True)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Log loss multiclasse. probs: (n, 3) ordre H/D/A. outcomes: indices 0/1/2."""
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Brier multiclasse (0 = parfait, plus bas est meilleur)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def walk_forward(matches: pd.DataFrame) -> pd.DataFrame:
    """Backtest temporel. Renvoie les predictions hors echantillon.

    Args:
        matches: trie par date, avec les colonnes de cotes de cloture.

    Returns:
        Un DataFrame avec les probabilites du modele et du marche pour
        chaque match predit.
    """
    matches = matches.sort_values("match_date").reset_index(drop=True)
    start = matches["match_date"].min() + pd.DateOffset(years=TRAIN_YEARS)
    end = matches["match_date"].max()

    rows: list[dict] = []
    cutoff = start
    n_refits = 0

    while cutoff <= end:
        window_start = cutoff - pd.DateOffset(years=TRAIN_YEARS)
        train = matches[
            (matches["match_date"] >= window_start)
            & (matches["match_date"] < cutoff)   # strictement anterieur
        ]
        test = matches[
            (matches["match_date"] >= cutoff)
            & (matches["match_date"] < cutoff + pd.Timedelta(days=REFIT_EVERY_DAYS))
        ]

        if len(train) < MIN_TRAIN_MATCHES or test.empty:
            cutoff += pd.Timedelta(days=REFIT_EVERY_DAYS)
            continue

        model = fit(train, reference_date=cutoff - pd.Timedelta(days=1))
        n_refits += 1

        for row in test.itertuples():
            try:
                ph, pd_, pa = model.predict_1x2(row.home_team, row.away_team)
            except KeyError:
                continue  # equipe promue, absente de la fenetre d'entrainement

            rows.append(
                {
                    "match_date": row.match_date,
                    "home_team": row.home_team,
                    "away_team": row.away_team,
                    "result": row.result,
                    "model_h": ph,
                    "model_d": pd_,
                    "model_a": pa,
                    "psch": row.psch,
                    "pscd": row.pscd,
                    "psca": row.psca,
                }
            )

        cutoff += pd.Timedelta(days=REFIT_EVERY_DAYS)

    print(f"{n_refits} reajustements, {len(rows)} predictions hors echantillon")
    return pd.DataFrame(rows)


def evaluate(preds: pd.DataFrame) -> None:
    """Compare le modele au marche et affiche le verdict."""
    preds = preds.dropna(subset=["psch", "pscd", "psca"])
    if preds.empty:
        raise SystemExit("Aucune prediction avec cotes de cloture.")

    outcomes = preds["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()

    model_probs = preds[["model_h", "model_d", "model_a"]].to_numpy()
    market_probs = normalise_odds(
        preds["psch"].to_numpy(), preds["pscd"].to_numpy(), preds["psca"].to_numpy()
    )

    n = len(preds)
    ll_model = log_loss(model_probs, outcomes)
    ll_market = log_loss(market_probs, outcomes)
    bs_model = brier_score(model_probs, outcomes)
    bs_market = brier_score(market_probs, outcomes)

    # Erreur standard de la difference de log loss, par paires
    p_m = np.clip(model_probs[np.arange(n), outcomes], 1e-15, 1.0)
    p_k = np.clip(market_probs[np.arange(n), outcomes], 1e-15, 1.0)
    diffs = -np.log(p_m) + np.log(p_k)
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    delta = ll_model - ll_market

    print(f"\n{'=' * 58}")
    print(f"{n} matchs evalues, du {preds['match_date'].min():%Y-%m-%d} "
          f"au {preds['match_date'].max():%Y-%m-%d}")
    print(f"{'=' * 58}")
    print(f"{'':<22}{'modele':>12}{'marche':>12}{'ecart':>12}")
    print(f"{'log loss':<22}{ll_model:>12.5f}{ll_market:>12.5f}{delta:>+12.5f}")
    print(f"{'Brier':<22}{bs_model:>12.5f}{bs_market:>12.5f}"
          f"{bs_model - bs_market:>+12.5f}")
    print(f"\nEcart de log loss : {delta:+.5f} +/- {1.96 * se:.5f} (IC 95 %)")

    if delta - 1.96 * se > 0:
        print("\n=> Le marche bat le modele de facon statistiquement nette.")
        print("   Aucune edge. C'est le resultat attendu pour une baseline.")
    elif delta + 1.96 * se < 0:
        print("\n=> Le modele bat le marche de facon statistiquement nette.")
        print("   Resultat inhabituel : verifier la fuite temporelle avant")
        print("   toute conclusion (cotes de cloture ? donnees futures ?).")
    else:
        print("\n=> Ecart non distinguable du bruit a ce volume.")

    calibration_report(model_probs, market_probs, outcomes)


def calibration_report(
    model_probs: np.ndarray, market_probs: np.ndarray, outcomes: np.ndarray
) -> None:
    """Les predictions a 30 % se realisent-elles 30 % du temps ?"""
    print(f"\n{'-' * 58}")
    print("Calibration sur la victoire a domicile")
    print(f"{'-' * 58}")
    print(f"{'tranche':<14}{'n':>7}{'p_modele':>11}{'p_marche':>11}{'observe':>11}")

    bins = np.arange(0.0, 1.01, 0.1)
    realised = (outcomes == 0).astype(float)
    bucket = np.digitize(model_probs[:, 0], bins) - 1

    for b in range(len(bins) - 1):
        mask = bucket == b
        if mask.sum() < 30:
            continue
        print(
            f"{bins[b]:.1f}-{bins[b + 1]:.1f}{'':<6}"
            f"{mask.sum():>7}"
            f"{model_probs[mask, 0].mean():>11.3f}"
            f"{market_probs[mask, 0].mean():>11.3f}"
            f"{realised[mask].mean():>11.3f}"
        )


def main() -> None:
    import duckdb

    con = duckdb.connect("data/betting.duckdb", read_only=True)
    df = con.execute(
        """
        SELECT match_date, home_team, away_team, home_goals, away_goals,
               result, league, psch, pscd, psca
        FROM matches
        ORDER BY match_date
        """
    ).fetchdf()
    con.close()

    all_preds = []
    for league, sub in df.groupby("league"):
        print(f"\n--- {league} ({len(sub)} matchs) ---")
        p = walk_forward(sub.reset_index(drop=True))
        p["league"] = league
        all_preds.append(p)

    preds = pd.concat(all_preds, ignore_index=True)
    evaluate(preds)
    preds.to_parquet("data/predictions_dixon_coles.parquet", index=False)


if __name__ == "__main__":
    main()
