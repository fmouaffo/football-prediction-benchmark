import pytest

from betting.arbitrage import Quote, find_arb, stake_basket

OUTCOMES_1X2 = ("H", "D", "A")


def surebet_quotes():
    return [
        Quote("A", "H", 2.10),
        Quote("A", "D", 3.30),
        Quote("A", "A", 3.80),
        Quote("B", "H", 1.95),
        Quote("B", "D", 3.60),
        Quote("B", "A", 3.90),
        Quote("C", "H", 2.00),
        Quote("C", "D", 3.40),
        Quote("C", "A", 4.20),
    ]


def test_finds_known_surebet():
    arb = find_arb(surebet_quotes(), OUTCOMES_1X2)
    assert arb is not None
    assert [leg.bookmaker for leg in arb.legs] == ["A", "B", "C"]
    assert arb.implied_sum == pytest.approx(0.99206, abs=1e-5)
    assert arb.margin == pytest.approx(0.0080, abs=1e-4)


def test_no_arb_when_single_bookmaker():
    quotes = [Quote("A", "H", 2.0), Quote("A", "D", 3.3), Quote("A", "A", 3.6)]
    assert find_arb(quotes, OUTCOMES_1X2) is None


def test_no_arb_when_outcome_missing():
    quotes = [Quote("A", "H", 5.0), Quote("B", "D", 5.0)]
    assert find_arb(quotes, OUTCOMES_1X2) is None


def test_min_margin_filter():
    assert find_arb(surebet_quotes(), OUTCOMES_1X2, min_margin=0.01) is None


def test_invalid_odds_ignored():
    quotes = surebet_quotes() + [Quote("X", "H", float("nan")), Quote("Y", "D", 0.5)]
    assert find_arb(quotes, OUTCOMES_1X2).legs[1].bookmaker == "B"


def test_profit_identical_whatever_the_outcome():
    arb = find_arb(surebet_quotes(), OUTCOMES_1X2)
    basket = stake_basket(arb, 1000, round_to=0.01)
    assert basket.total == pytest.approx(1000, abs=0.02)
    profits = [basket.profit_if(o) for o in OUTCOMES_1X2]
    assert max(profits) - min(profits) < 0.05
    assert basket.min_profit > 0


def test_rounding_that_kills_profit_rejects_basket():
    arb = find_arb(surebet_quotes(), OUTCOMES_1X2)
    assert stake_basket(arb, 100, round_to=5) is None
