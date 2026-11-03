from datetime import date
from decimal import Decimal

import pytest

from app.finance import (
    compare_extra_payment,
    debt_payoff,
    emergency_fund_status,
    months_to_goal,
    payoff_date,
)


def test_debt_payoff_standard_payment() -> None:
    result = debt_payoff("2450", "0.249", "350")

    assert result.pays_off
    assert result.months == 8
    assert abs(result.total_interest - Decimal("225")) < Decimal("1")
    assert result.total_paid == Decimal("2675.48")
    assert result.final_payment < Decimal("350.00")


def test_debt_payoff_higher_payment() -> None:
    result = debt_payoff("2450", "0.249", "470")

    assert result.months == 6
    assert abs(result.total_interest - Decimal("171")) < Decimal("1")
    assert result.final_payment < Decimal("470.00")


def test_compare_extra_payment() -> None:
    result = compare_extra_payment("2450", "0.249", "350", "120")

    assert result.months_saved == 2
    assert abs(result.interest_saved - Decimal("55")) < Decimal("1")


def test_emergency_fund_status() -> None:
    result = emergency_fund_status("3200", "2000")

    assert result.months_covered == Decimal("1.60")
    assert result.target_amount == Decimal("6000.00")
    assert result.amount_left == Decimal("2800.00")
    assert result.percent_complete == Decimal("53.33")


def test_zero_apr() -> None:
    result = debt_payoff("100", "0", "30")

    assert result.months == 4
    assert result.total_interest == Decimal("0.00")
    assert result.total_paid == Decimal("100.00")
    assert result.final_payment == Decimal("10.00")


def test_payment_not_covering_first_month_interest_never_pays_off() -> None:
    result = debt_payoff("1000", "0.12", "10")

    assert not result.pays_off
    assert result.months is None
    assert result.total_interest is None


def test_zero_balance_is_already_paid() -> None:
    result = debt_payoff("0", "0.249", "350")

    assert result.pays_off
    assert result.months == 0
    assert result.total_paid == Decimal("0.00")


def test_months_to_goal_with_zero_saving() -> None:
    result = months_to_goal("100", "250", "0")

    assert result.months is None
    assert result.amount_remaining == Decimal("150.00")


def test_months_to_goal_is_zero_if_already_reached() -> None:
    result = months_to_goal("250", "200", "0")

    assert result.months == 0
    assert result.amount_remaining == Decimal("0.00")


def test_payoff_date_adds_calendar_months() -> None:
    assert payoff_date(date(2026, 10, 20), 8) == date(2027, 6, 1)


@pytest.mark.parametrize(
    ("balance", "apr", "payment"),
    [
        ("-1", "0.1", "10"),
        ("10", "-0.1", "10"),
        ("10", "1.01", "10"),
        ("10", "0.1", "-1"),
        ("NaN", "0.1", "10"),
        ("10", "not-a-rate", "10"),
    ],
)
def test_debt_payoff_rejects_invalid_inputs(balance: str, apr: str, payment: str) -> None:
    with pytest.raises(ValueError):
        debt_payoff(balance, apr, payment)


def test_emergency_fund_rejects_zero_essentials() -> None:
    with pytest.raises(ValueError, match="monthly_essentials"):
        emergency_fund_status("100", "0")


def test_months_to_goal_rejects_negative_saving() -> None:
    with pytest.raises(ValueError, match="monthly_saving"):
        months_to_goal("0", "100", "-1")


# --- Edge cases and limits added with the calculation hardening ------------

from app.finance import MAX_AMOUNT, MAX_MONTHS  # noqa: E402


def _reference_payoff(balance: Decimal, apr: Decimal, payment: Decimal) -> tuple[int, Decimal]:
    """Independent re-implementation used to cross-check the engine."""
    months, interest_total = 0, Decimal("0")
    while balance > 0:
        interest = (balance * apr / 12).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        interest_total += interest
        balance = balance + interest - payment
        months += 1
    return months, interest_total


@pytest.mark.parametrize(
    ("balance", "apr", "payment"),
    [("2450", "0.249", "350"), ("999.99", "0.399", "45.50"), ("15000", "0.069", "300")],
)
def test_debt_payoff_matches_independent_reference(balance: str, apr: str, payment: str) -> None:
    result = debt_payoff(balance, apr, payment)
    months, interest = _reference_payoff(Decimal(balance), Decimal(apr), Decimal(payment))

    assert result.months == months
    assert result.total_interest == interest
    assert result.total_paid == Decimal(balance) + interest


def test_total_paid_equals_balance_plus_interest_to_the_penny() -> None:
    result = debt_payoff("1234.56", "0.199", "100")

    assert result.total_paid == Decimal("1234.56") + result.total_interest
    assert result.final_payment <= Decimal("100")


def test_payment_exactly_equal_to_interest_never_pays_off() -> None:
    # £1,200 at 10% APR accrues exactly £10.00 of interest in the first month.
    result = debt_payoff("1200", "0.10", "10")

    assert not result.pays_off
    assert result.reason == "payment_does_not_cover_interest"


def test_payment_a_penny_above_interest_is_bounded_by_max_term() -> None:
    result = debt_payoff("1200", "0.10", "10.01")

    assert not result.pays_off
    assert result.reason == "exceeds_max_term"
    assert result.months is None


def test_very_large_balance_completes_quickly_and_correctly() -> None:
    result = debt_payoff(str(MAX_AMOUNT), "0.05", "500000")

    assert result.pays_off
    assert result.months is not None
    assert result.months < MAX_MONTHS
    assert result.total_paid == MAX_AMOUNT + result.total_interest


def test_tiny_payment_on_zero_interest_debt_hits_term_limit() -> None:
    result = debt_payoff("1000", "0", "0.01")

    assert not result.pays_off
    assert result.reason == "exceeds_max_term"


def test_inputs_are_rounded_to_the_penny_half_up() -> None:
    assert debt_payoff("100.005", "0", "100.005").total_paid == Decimal("100.01")


@pytest.mark.parametrize(
    "value", ["Infinity", "-Infinity", "NaN", "sNaN", float("inf"), True, None, "1e999999"]
)
def test_non_finite_and_non_numeric_inputs_are_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        debt_payoff(value, "0.1", "10")  # type: ignore[arg-type]


def test_amounts_above_the_supported_maximum_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        debt_payoff(str(MAX_AMOUNT + 1), "0.1", "10")


def test_compare_extra_when_only_the_increased_payment_pays_off() -> None:
    result = compare_extra_payment("1000", "0.12", "10", "40")

    assert not result.baseline.pays_off
    assert result.with_extra.pays_off
    assert result.months_saved is None
    assert result.interest_saved is None


def test_compare_extra_of_zero_saves_nothing() -> None:
    result = compare_extra_payment("2450", "0.249", "350", "0")

    assert result.months_saved == 0
    assert result.interest_saved == Decimal("0.00")


def test_extra_payment_never_increases_interest() -> None:
    for extra in ("1", "10", "50", "500", "5000"):
        result = compare_extra_payment("2450", "0.249", "350", extra)
        assert result.interest_saved is not None
        assert result.interest_saved >= 0
        assert result.months_saved is not None
        assert result.months_saved >= 0


def test_payoff_date_handles_year_rollover_and_zero_months() -> None:
    assert payoff_date(date(2026, 12, 31), 1) == date(2027, 1, 1)
    assert payoff_date(date(2026, 5, 15), 0) == date(2026, 5, 1)
    with pytest.raises(ValueError):
        payoff_date(date(2026, 1, 1), -1)


@pytest.mark.parametrize("months", [0, 25, True, 2.5])
def test_emergency_fund_rejects_invalid_target_months(months: object) -> None:
    with pytest.raises(ValueError, match="target_months"):
        emergency_fund_status("100", "100", months)  # type: ignore[arg-type]


def test_emergency_fund_caps_percentage_at_100() -> None:
    result = emergency_fund_status("9000", "2000")

    assert result.percent_complete == Decimal("100.00")
    assert result.amount_left == Decimal("0.00")
    assert result.months_covered == Decimal("4.50")


def test_months_to_goal_rounds_partial_months_up() -> None:
    result = months_to_goal("0", "1000", "300")

    assert result.months == 4
    assert result.reason is None


def test_months_to_goal_beyond_term_limit_is_reported() -> None:
    result = months_to_goal("0", "1000000", "1")

    assert result.months is None
    assert result.reason == "exceeds_max_term"


def test_months_to_goal_without_contribution_explains_why() -> None:
    assert months_to_goal("0", "100", "0").reason == "no_contribution"
