"""Deterministic financial calculations used by the NetValor API and co-pilot.

Every monetary value is a `Decimal` rounded to the penny with ROUND_HALF_UP.
The language model never performs arithmetic; it only explains these results.

Assumptions and limitations (also published in API responses):

* **Interest model.** Each month, interest = remaining balance x APR / 12,
  rounded to the penny, is added before the payment is taken. Real UK card
  providers usually charge daily interest and a quoted APR is an effective
  annual rate that can include fees, so actual figures will differ. Dividing
  the APR by 12 slightly *overstates* interest for an effective APR, which
  errs on the side of not understating costs.
* **Fixed payment.** The same payment is made every month; no new spending,
  fees, promotional rates, rate changes or missed payments are modelled.
  Minimum payments that fall as the balance falls are not modelled.
* **APR is a fraction** between 0 and 1 (0.249 means 24.9%). Rates above 100%
  are rejected rather than risk confusing "24.9" with 2,490%.
* **Term limit.** Simulations stop after `MAX_MONTHS` (50 years). A debt that
  is not repaid by then is reported as not paying off, with a reason.
* **Goals and emergency funds** ignore interest earned and inflation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Literal

PENNY = Decimal("0.01")
ZERO = Decimal("0.00")
MAX_AMOUNT = Decimal("10000000")  # £10m: generous for education, bounds work.
MAX_MONTHS = 600
MAX_TARGET_MONTHS = 24

ASSUMPTIONS: tuple[str, ...] = (
    "Interest is estimated monthly as balance x APR / 12; real providers usually "
    "charge daily interest, so actual figures will differ.",
    "The same payment is made every month with no new spending, fees or rate changes.",
    "Estimates are for learning purposes and are not a quote or financial advice.",
)

NonPayoffReason = Literal["payment_does_not_cover_interest", "exceeds_max_term"]
Number = Decimal | int | float | str


def _decimal(value: Number, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a finite number.") from None
    if not amount.is_finite():
        raise ValueError(f"{name} must be a finite number.")
    return amount


def _money(value: Number, name: str) -> Decimal:
    amount = _decimal(value, name)
    if amount < 0:
        raise ValueError(f"{name} cannot be negative.")
    if amount > MAX_AMOUNT:
        raise ValueError(f"{name} cannot exceed £{MAX_AMOUNT:,}.")
    return amount.quantize(PENNY, rounding=ROUND_HALF_UP)


def _apr(value: Number) -> Decimal:
    rate = _decimal(value, "apr")
    if rate < 0 or rate > 1:
        raise ValueError("apr must be a fraction between 0 and 1 (for example 0.249 for 24.9%).")
    return rate


def _monthly_interest(balance: Decimal, rate: Decimal) -> Decimal:
    return (balance * rate / 12).quantize(PENNY, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class DebtPayoffResult:
    """The outcome of a fixed-payment debt payoff calculation."""

    months: int | None
    total_interest: Decimal | None
    total_paid: Decimal | None
    final_payment: Decimal | None
    pays_off: bool
    reason: NonPayoffReason | None = None


@dataclass(frozen=True)
class ExtraPaymentComparison:
    """Baseline and increased-payment outcomes, plus their savings."""

    baseline: DebtPayoffResult
    with_extra: DebtPayoffResult
    months_saved: int | None
    interest_saved: Decimal | None


@dataclass(frozen=True)
class EmergencyFundStatus:
    """Progress toward an emergency-fund target."""

    months_covered: Decimal
    target_amount: Decimal
    amount_left: Decimal
    percent_complete: Decimal


@dataclass(frozen=True)
class GoalTimeline:
    """The number of monthly contributions needed to reach a savings goal."""

    months: int | None
    amount_remaining: Decimal
    reason: Literal["no_contribution", "exceeds_max_term"] | None = None


def debt_payoff(balance: Number, apr: Number, monthly_payment: Number) -> DebtPayoffResult:
    """Simulate monthly interest and payments until the balance is cleared.

    The loop is bounded by `MAX_MONTHS`, so worst-case work is constant.
    """
    remaining = _money(balance, "balance")
    payment = _money(monthly_payment, "monthly_payment")
    rate = _apr(apr)
    if remaining == 0:
        return DebtPayoffResult(0, ZERO, ZERO, ZERO, True)
    if payment <= _monthly_interest(remaining, rate):
        return DebtPayoffResult(None, None, None, None, False, "payment_does_not_cover_interest")

    total_interest = ZERO
    total_paid = ZERO
    final_payment = ZERO
    for month in range(1, MAX_MONTHS + 1):
        interest = _monthly_interest(remaining, rate)
        amount_due = remaining + interest
        total_interest += interest
        if payment >= amount_due:
            final_payment = amount_due
            total_paid += amount_due
            return DebtPayoffResult(month, total_interest, total_paid, final_payment, True)
        total_paid += payment
        remaining = amount_due - payment
    return DebtPayoffResult(None, None, None, None, False, "exceeds_max_term")


def compare_extra_payment(
    balance: Number, apr: Number, monthly_payment: Number, extra: Number
) -> ExtraPaymentComparison:
    """Compare payoff time and interest with an additional monthly payment."""
    base_payment = _money(monthly_payment, "monthly_payment")
    extra_payment = _money(extra, "extra")
    baseline = debt_payoff(balance, apr, base_payment)
    with_extra = debt_payoff(balance, apr, min(base_payment + extra_payment, MAX_AMOUNT))
    if baseline.pays_off and with_extra.pays_off:
        assert baseline.months is not None
        assert with_extra.months is not None
        assert baseline.total_interest is not None
        assert with_extra.total_interest is not None
        months_saved: int | None = baseline.months - with_extra.months
        interest_saved: Decimal | None = baseline.total_interest - with_extra.total_interest
    else:
        months_saved = None
        interest_saved = None
    return ExtraPaymentComparison(baseline, with_extra, months_saved, interest_saved)


def payoff_date(start_date: date, months: int) -> date:
    """Return the first day of the month in which the final payment falls.

    Assumes the first payment is made in the month after `start_date`.
    """
    if not isinstance(start_date, date):
        raise ValueError("start_date must be a date.")
    if isinstance(months, bool) or not isinstance(months, int) or months < 0:
        raise ValueError("months must be a non-negative integer.")
    month_index = start_date.year * 12 + start_date.month - 1 + months
    year, month = divmod(month_index, 12)
    return date(year, month + 1, 1)


def emergency_fund_status(
    saved: Number, monthly_essentials: Number, target_months: int = 3
) -> EmergencyFundStatus:
    """Measure emergency-fund coverage and progress toward the target."""
    amount_saved = _money(saved, "saved")
    essentials = _money(monthly_essentials, "monthly_essentials")
    if essentials == 0:
        raise ValueError("monthly_essentials must be greater than zero.")
    if (
        isinstance(target_months, bool)
        or not isinstance(target_months, int)
        or not 1 <= target_months <= MAX_TARGET_MONTHS
    ):
        raise ValueError(f"target_months must be a whole number from 1 to {MAX_TARGET_MONTHS}.")

    target = (essentials * target_months).quantize(PENNY)
    months_covered = (amount_saved / essentials).quantize(PENNY, rounding=ROUND_HALF_UP)
    amount_left = max(target - amount_saved, ZERO).quantize(PENNY)
    percent_complete = min(amount_saved / target * 100, Decimal("100")).quantize(
        PENNY, rounding=ROUND_HALF_UP
    )
    return EmergencyFundStatus(months_covered, target, amount_left, percent_complete)


def months_to_goal(current: Number, target: Number, monthly_saving: Number) -> GoalTimeline:
    """Estimate whole monthly contributions to reach a savings goal, ignoring interest."""
    saved = _money(current, "current")
    goal = _money(target, "target")
    contribution = _money(monthly_saving, "monthly_saving")
    remaining = max(goal - saved, ZERO).quantize(PENNY)
    if remaining == 0:
        return GoalTimeline(0, remaining)
    if contribution == 0:
        return GoalTimeline(None, remaining, "no_contribution")
    months = int((remaining / contribution).to_integral_value(rounding=ROUND_CEILING))
    if months > MAX_MONTHS:
        return GoalTimeline(None, remaining, "exceeds_max_term")
    return GoalTimeline(months, remaining)
