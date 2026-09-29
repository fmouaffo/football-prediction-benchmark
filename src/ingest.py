"""
Ingestion des historiques de matchs + cotes depuis football-data.co.uk vers DuckDB.

Usage:
    uv run src/ingest.py
"""

from __future__ import annotations

import io
from pathlib import Path

import duckdb
import pandas as pd
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Codes de division. Ajoute/retire selon les championnats qui t'intéressent.
DIVISIONS = {
    "E0": "Premier League",
    "E1": "Championship",
    "F1": "Ligue 1",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "SP1": "La Liga",
}

# Saisons a recuperer : 2010/11 -> 2025/26
START_YEAR = 2010
END_YEAR = 2025

DB_PATH = Path("data/betting.duckdb")

# Colonnes conservees.
# FTHG/FTAG = buts finaux domicile/exterieur, FTR = resultat (H/D/A)
CORE_COLS = ["Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Cotes 1X2. Les colonnes en *C* sont les cotes de CLOTURE (closing) :
# ce sont les seules valables pour un backtest honnete.
ODDS_COLS = [
    "PSH", "PSD", "PSA",        # Pinnacle ouverture
    "PSCH", "PSCD", "PSCA",     # Pinnacle cloture  <- le benchmark
    "B365H", "B365D", "B365A",  # Bet365 ouverture
    "B365CH", "B365CD", "B365CA",  # Bet365 cloture
    "MaxCH", "MaxCD", "MaxCA",  # meilleure cote du marche a la cloture
    "AvgCH", "AvgCD", "AvgCA",  # cote moyenne du marche a la cloture
]

NUMERIC_COLS = ["FTHG", "FTAG"] + ODDS_COLS


def season_code(year: int) -> str:
    """2024 -> '2425' (saison 2024/2025)."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def fetch_csv(division: str, year: int) -> pd.DataFrame | None:
    """Telecharge un CSV saison/division. Renvoie None si indisponible."""
    url = f"{BASE_URL}/{season_code(year)}/{division}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [skip] {division} {year}: {exc}")
        return None

    # Encodage latin-1 : les fichiers contiennent des accents mal encodes.
    # on_bad_lines='skip' : certains fichiers ont des lignes vides en fin.
    df = pd.read_csv(
        io.StringIO(resp.content.decode("latin-1")),
        on_bad_lines="skip",
    )

    if df.empty or "HomeTeam" not in df.columns:
        print(f"  [skip] {division} {year}: fichier vide ou malforme")
        return None

    df["season"] = f"{year}/{year + 1}"
    df["division"] = division
    df["league"] = DIVISIONS.get(division, division)
    return df


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Aligne les colonnes, type les dates et les numeriques."""
    # Les colonnes de cotes varient selon les saisons -> on cree les manquantes
    wanted = CORE_COLS + ODDS_COLS + ["season", "division", "league"]
    for col in wanted:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[wanted].copy()

    # Format des dates : dd/mm/yy avant ~2018, dd/mm/yyyy ensuite.
    # dayfirst=True gere les deux, format='mixed' tolere l'heterogeneite.
    df["Date"] = pd.to_datetime(
        df["Date"], dayfirst=True, format="mixed", errors="coerce"
    )

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Un match sans date ou sans score n'est pas exploitable
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])

    df = df.rename(
        columns={
            "Date": "match_date",
            "Time": "match_time",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "result",
        }
    )
    df.columns = [c.lower() for c in df.columns]

    df["home_goals"] = df["home_goals"].astype("int16")
    df["away_goals"] = df["away_goals"].astype("int16")
    return df


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for division in DIVISIONS:
        print(f"{division} ({DIVISIONS[division]})")
        for year in range(START_YEAR, END_YEAR + 1):
            raw = fetch_csv(division, year)
            if raw is not None:
                frames.append(normalise(raw))

    if not frames:
        raise SystemExit("Aucune donnee recuperee.")

    matches = pd.concat(frames, ignore_index=True)
    matches = matches.sort_values("match_date").reset_index(drop=True)

    con = duckdb.connect(str(DB_PATH))
    con.register("matches_df", matches)
    con.execute("DROP TABLE IF EXISTS matches")
    con.execute("CREATE TABLE matches AS SELECT * FROM matches_df")
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)"
    )

    total = con.execute("SELECT count(*) FROM matches").fetchone()[0]
    with_closing = con.execute(
        "SELECT count(*) FROM matches WHERE psch IS NOT NULL"
    ).fetchone()[0]

    print(f"\n{total} matchs charges dans {DB_PATH}")
    print(f"{with_closing} avec cotes de cloture Pinnacle ({with_closing / total:.1%})")

    print("\nCouverture par championnat :")
    print(
        con.execute(
            """
            SELECT league,
                   count(*) AS matchs,
                   min(match_date)::DATE AS depuis,
                   max(match_date)::DATE AS jusqua,
                   round(100.0 * count(psch) / count(*), 1) AS pct_cotes_cloture
            FROM matches
            GROUP BY league
            ORDER BY matchs DESC
            """
        ).fetchdf()
    )

    con.close()


if __name__ == "__main__":
    main()
