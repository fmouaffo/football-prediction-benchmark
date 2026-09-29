"""Arbitrage (surebet) detection and stake sizing. Pure functions, no I/O."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Quote:
    bookmaker: str
    outcome: str
    odds: float
    updated: str | None = None


@dataclass(frozen=True)
class Arb:
    legs: tuple[Quote, ...]

    @property
    def implied_sum(self) -> float:
        return sum(1 / leg.odds for leg in self.legs)

    @property
    def margin(self) -> float:
        """Guaranteed return on the total stake, before rounding (0.01 = 1 %)."""
        return 1 / self.implied_sum - 1


@dataclass(frozen=True)
class Basket:
    arb: Arb
    stakes: tuple[float, ...]

    @property
    def total(self) -> float:
        return sum(self.stakes)

    def profit_if(self, outcome: str) -> float:
        for leg, stake in zip(self.arb.legs, self.stakes):
            if leg.outcome == outcome:
                return stake * leg.odds - self.total
        raise KeyError(outcome)

    @property
    def min_profit(self) -> float:
        return min(self.profit_if(leg.outcome) for leg in self.arb.legs)

    @property
    def max_profit(self) -> float:
        return max(self.profit_if(leg.outcome) for leg in self.arb.legs)


def _valid(odds: float) -> bool:
    return odds is not None and not math.isnan(odds) and odds > 1.0


def best_quotes(quotes: Iterable[Quote]) -> dict[str, Quote]:
    """Highest odds for each outcome."""
    best: dict[str, Quote] = {}
    for q in quotes:
        if _valid(q.odds) and (q.outcome not in best or q.odds > best[q.outcome].odds):
            best[q.outcome] = q
    return best


def find_arb(
    quotes: Iterable[Quote], outcomes: Sequence[str], min_margin: float = 0.0
) -> Arb | None:
    """Return the arbitrage built from the best odds, if every outcome is covered
    and the guaranteed margin is strictly above `min_margin`."""
    best = best_quotes(quotes)
    if any(o not in best for o in outcomes):
        return None
    arb = Arb(tuple(best[o] for o in outcomes))
    return arb if arb.margin > min_margin else None


def stake_basket(arb: Arb, bankroll: float, round_to: float = 0.01) -> Basket | None:
    """Split `bankroll` so every outcome pays the same, then round each stake.
    Returns None if rounding makes any outcome a loss."""
    s = arb.implied_sum
    stakes = tuple(
        round(round((bankroll / leg.odds / s) / round_to) * round_to, 2) for leg in arb.legs
    )
    if any(st <= 0 for st in stakes):
        return None
    basket = Basket(arb, stakes)
    return basket if basket.min_profit > 0 else None
