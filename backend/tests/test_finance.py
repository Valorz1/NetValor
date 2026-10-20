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
def test_debt_payoff_rejects_invalid_inputs(
    balance: str, apr: str, payment: str
) -> None:
    with pytest.raises(ValueError):
        debt_payoff(balance, apr, payment)


def test_emergency_fund_rejects_zero_essentials() -> None:
    with pytest.raises(ValueError, match="monthly_essentials"):
        emergency_fund_status("100", "0")


def test_months_to_goal_rejects_negative_saving() -> None:
    with pytest.raises(ValueError, match="monthly_saving"):
        months_to_goal("0", "100", "-1")
