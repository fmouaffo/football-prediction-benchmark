"""
Calibration post-hoc des probabilites Dixon-Coles.

Le modele est bien calibre au centre (0.3-0.6) mais trop confiant aux
extremes : quand il annonce 6.8 %, la realite est 14.2 %. Ce script teste
deux corrections qui n'exigent aucun reajustement du modele.

  1. Temperature scaling : un seul parametre t. t > 1 aplatit la
     distribution proportionnellement sur l'echelle log. Corrige fortement
     les extremes, laisse le centre quasi intact.

  2. Platt scaling multiclasse : une regression logistique sur les
     log-probabilites. Plus souple, apprend aussi a corriger un biais
     systematique entre H, D et A.

PROTOCOLE : le parametre est choisi sur la periode ANCIENNE et evalue sur
la periode RECENTE, jamais la meme. Optimiser et mesurer sur le meme
echantillon produit un gain fictif — c'est la forme de fuite la plus
courante et la plus difficile a reperer apres coup.

Usage:
    uv run python -m src.calibrate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

SPLIT_DATE = pd.Timestamp("2021-01-01")
PARQUET = "data/predictions_dixon_coles.parquet"

PROB_COLS = ["model_h", "model_d", "model_a"]
ODDS_COLS = ["psch", "pscd", "psca"]


# --------------------------------------------------------------------------
# Metriques (dupliquees ici pour rendre le module autonome)
# --------------------------------------------------------------------------

def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(outcomes)), outcomes], 1e-15, 1.0)
    return float(-np.mean(np.log(p)))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(outcomes)), outcomes] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def normalise_odds(odds: np.ndarray) -> np.ndarray:
    """Cotes decimales (n, 3) -> probabilites normalisees."""
    raw = 1.0 / odds
    return raw / raw.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------
# Calibrateurs
# --------------------------------------------------------------------------

def temperature_scale(probs: np.ndarray, t: float) -> np.ndarray:
    """Applique une temperature. t > 1 reduit la confiance."""
    logits = np.log(np.clip(probs, 1e-15, 1.0))
    scaled = np.exp(logits / t)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(
    probs: np.ndarray, outcomes: np.ndarray, grid: np.ndarray | None = None
) -> tuple[float, float]:
    """Cherche la temperature qui minimise le log loss.

    Returns:
        (t optimal, log loss obtenu sur ces memes donnees)
    """
    if grid is None:
        grid = np.arange(0.80, 2.01, 0.01)

    losses = np.array([log_loss(temperature_scale(probs, t), outcomes) for t in grid])
    best = int(np.argmin(losses))

    if best in (0, len(grid) - 1):
        print(f"[avertissement] t optimal en bord de grille ({grid[best]:.2f}) "
              f"— elargir la grille.")

    return float(grid[best]), float(losses[best])


def fit_platt(probs: np.ndarray, outcomes: np.ndarray) -> LogisticRegression:
    """Regression logistique multinomiale sur les log-probabilites.

    Plus souple que la temperature : apprend une matrice 3x3 de correction
    plutot qu'un scalaire. Peut donc corriger un biais systematique sur les
    nuls, que la temperature ne touche pas.
    """
    features = np.log(np.clip(probs, 1e-15, 1.0))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(features, outcomes)
    return clf


def apply_platt(clf: LogisticRegression, probs: np.ndarray) -> np.ndarray:
    return clf.predict_proba(np.log(np.clip(probs, 1e-15, 1.0)))


# --------------------------------------------------------------------------
# Diagnostic
# --------------------------------------------------------------------------

def calibration_table(
    probs: np.ndarray, market: np.ndarray, outcomes: np.ndarray, label: str
) -> None:
    print(f"\n{'-' * 62}")
    print(f"Calibration victoire domicile — {label}")
    print(f"{'-' * 62}")
    print(f"{'tranche':<14}{'n':>7}{'predit':>11}{'marche':>11}{'observe':>11}")

    bins = np.arange(0.0, 1.01, 0.1)
    realised = (outcomes == 0).astype(float)
    bucket = np.clip(np.digitize(probs[:, 0], bins) - 1, 0, len(bins) - 2)

    for b in range(len(bins) - 1):
        mask = bucket == b
        if mask.sum() < 30:
            continue
        print(
            f"{bins[b]:.1f}-{bins[b + 1]:.1f}{'':<6}"
            f"{mask.sum():>7}"
            f"{probs[mask, 0].mean():>11.3f}"
            f"{market[mask, 0].mean():>11.3f}"
            f"{realised[mask].mean():>11.3f}"
        )


def compare(
    name: str,
    probs: np.ndarray,
    market: np.ndarray,
    outcomes: np.ndarray,
    baseline_ll: float,
) -> float:
    """Affiche une ligne de resultat. Renvoie le log loss."""
    ll = log_loss(probs, outcomes)
    bs = brier_score(probs, outcomes)
    ll_market = log_loss(market, outcomes)
    print(
        f"{name:<26}{ll:>10.5f}{bs:>10.5f}"
        f"{ll - ll_market:>+11.5f}{ll - baseline_ll:>+11.5f}"
    )
    return ll


# --------------------------------------------------------------------------

def main() -> None:
    preds = pd.read_parquet(PARQUET).dropna(subset=ODDS_COLS)
    preds["match_date"] = pd.to_datetime(preds["match_date"])

    train = preds[preds["match_date"] < SPLIT_DATE]
    test = preds[preds["match_date"] >= SPLIT_DATE]

    if len(train) < 2000 or len(test) < 2000:
        raise SystemExit(
            f"Echantillons trop petits (train={len(train)}, test={len(test)}). "
            "Ajuster SPLIT_DATE."
        )

    def unpack(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = df[PROB_COLS].to_numpy()
        market = normalise_odds(df[ODDS_COLS].to_numpy())
        outcomes = df["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        return probs, market, outcomes

    p_tr, m_tr, y_tr = unpack(train)
    p_te, m_te, y_te = unpack(test)

    print(f"Train : {len(train)} matchs, jusqu'au {train['match_date'].max():%Y-%m-%d}")
    print(f"Test  : {len(test)} matchs, a partir du {test['match_date'].min():%Y-%m-%d}")

    # --- Apprentissage sur TRAIN uniquement ---
    t_opt, ll_train = fit_temperature(p_tr, y_tr)
    platt = fit_platt(p_tr, y_tr)

    print(f"\nTemperature retenue : t = {t_opt:.2f}")
    print(f"  (log loss sur train : {log_loss(p_tr, y_tr):.5f} -> {ll_train:.5f})")

    # --- Evaluation sur TEST, parametres figes ---
    baseline = log_loss(p_te, y_te)

    print(f"\n{'=' * 62}")
    print(f"RESULTATS SUR TEST ({len(test)} matchs) — parametres figes")
    print(f"{'=' * 62}")
    print(f"{'':<26}{'log loss':>10}{'Brier':>10}{'vs marche':>11}{'vs brut':>11}")

    compare("Dixon-Coles brut", p_te, m_te, y_te, baseline)
    p_temp = temperature_scale(p_te, t_opt)
    ll_temp = compare(f"+ temperature ({t_opt:.2f})", p_temp, m_te, y_te, baseline)
    p_platt = apply_platt(platt, p_te)
    ll_platt = compare("+ Platt", p_platt, m_te, y_te, baseline)
    compare("Marche (Pinnacle)", m_te, m_te, y_te, baseline)

    # --- Verdict ---
    ll_market = log_loss(m_te, y_te)
    best_name, best_ll = min(
        [("temperature", ll_temp), ("Platt", ll_platt), ("brut", baseline)],
        key=lambda x: x[1],
    )
    gain = baseline - best_ll
    remaining = best_ll - ll_market

    print(f"\nMeilleure calibration : {best_name}")
    print(f"Gain vs brut       : {gain:+.5f}")
    print(f"Ecart restant      : {remaining:+.5f}")

    if gain > 0:
        pct = 100 * gain / (baseline - ll_market)
        print(f"La calibration comble {pct:.0f} % de l'ecart initial au marche.")
    else:
        print("Aucun gain : la calibration apprise sur train ne generalise pas.")

    # --- Diagnostic visuel ---
    calibration_table(p_te, m_te, y_te, "brut")
    calibration_table(p_temp, m_te, y_te, f"temperature t={t_opt:.2f}")

    # --- Sauvegarde ---
    out = test.copy()
    out[["cal_h", "cal_d", "cal_a"]] = (
        p_temp if best_name == "temperature" else p_platt
    )
    out.to_parquet("data/predictions_calibrees.parquet", index=False)
    print("\nPredictions calibrees -> data/predictions_calibrees.parquet")


if __name__ == "__main__":
    main()
