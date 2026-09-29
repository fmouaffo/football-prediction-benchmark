# src/by_league.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.calibrate import log_loss, normalise_odds

ODDS = ["psch", "pscd", "psca"]
PROBS = ["model_h", "model_d", "model_a"]


def main() -> None:
    preds = pd.read_parquet("data/predictions_dixon_coles.parquet").dropna(subset=ODDS)

    rows = []
    for league, sub in preds.groupby("league"):
        if len(sub) < 1000:
            continue

        y = sub["result"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
        p_model = sub[PROBS].to_numpy()
        odds = sub[ODDS].to_numpy()
        p_market = normalise_odds(odds)

        ll_m = log_loss(p_model, y)
        ll_k = log_loss(p_market, y)

        # Ecart-type de la difference, par paires
        d = -np.log(np.clip(p_model[np.arange(len(y)), y], 1e-15, 1)) \
            + np.log(np.clip(p_market[np.arange(len(y)), y], 1e-15, 1))
        se = float(np.std(d, ddof=1) / np.sqrt(len(y)))

        rows.append({
            "league": league,
            "n": len(sub),
            "overround": float(np.mean(1 / odds[:, 0] + 1 / odds[:, 1] + 1 / odds[:, 2])),
            "ll_modele": ll_m,
            "ll_marche": ll_k,
            "ecart": ll_m - ll_k,
            "ic95": 1.96 * se,
        })

    df = pd.DataFrame(rows).sort_values("ecart")
    print(df.round(4).to_string(index=False))

    print("\nCorrelation overround / ecart au marche :",
          round(df["overround"].corr(df["ecart"]), 3))


if __name__ == "__main__":
    main()