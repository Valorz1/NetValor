"""Deterministic financial calculations used by the NetValor co-pilot."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_UP

_PENNY = Decimal("0.01")
_ZERO = Decimal("0.00")


def _money(value: Decimal | int | float | str, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite monetary amount.")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"{name} must be a finite monetary amount.") from None
    if not amount.is_finite():
        raise ValueError(f"{name} must be a finite monetary amount.")
    return amount.quantize(_PENNY, rounding=ROUND_HALF_UP)


def _non_negative_money(value: Decimal | int | float | str, name: str) -> Decimal:
    amount = _money(value, name)
    if amount < 0:
        raise ValueError(f"{name} cannot be negative.")
    return amount


def _apr(value: Decimal | int | float | str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("apr must be a finite rate between 0 and 1.")
    try:
        rate = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("apr must be a finite rate between 0 and 1.") from None
    if not rate.is_finite() or rate < 0 or rate > 1:
        raise ValueError("apr must be a finite rate between 0 and 1.")
    return rate


@dataclass(frozen=True)
class DebtPayoffResult:
    """The outcome of a fixed-payment debt payoff calculation."""

    months: int | None
    total_interest: Decimal | None
    total_paid: Decimal | None
    final_payment: Decimal | None
    pays_off: bool


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


def debt_payoff(
    balance: Decimal | int | float | str,
    apr: Decimal | int | float | str,
    monthly_payment: Decimal | int | float | str,
) -> DebtPayoffResult:
    """Simulate monthly interest and payments, rounding money to the nearest penny."""
    remaining = _non_negative_money(balance, "balance")
    payment = _non_negative_money(monthly_payment, "monthly_payment")
    rate = _apr(apr)
    if remaining == 0:
        return DebtPayoffResult(0, _ZERO, _ZERO, _ZERO, True)

    first_interest = (remaining * rate / 12).quantize(_PENNY, rounding=ROUND_HALF_UP)
    if payment <= first_interest:
        return DebtPayoffResult(None, None, None, None, False)

    total_interest = _ZERO
    total_paid = _ZERO
    months = 0
    final_payment = _ZERO
    while remaining > 0:
        interest = (remaining * rate / 12).quantize(_PENNY, rounding=ROUND_HALF_UP)
        amount_due = remaining + interest
        months += 1
        total_interest += interest
        if payment >= amount_due:
            final_payment = amount_due
            total_paid += final_payment
            remaining = _ZERO
        else:
            final_payment = payment
            total_paid += payment
            remaining = amount_due - payment

    return DebtPayoffResult(
        months,
        total_interest.quantize(_PENNY),
        total_paid.quantize(_PENNY),
        final_payment.quantize(_PENNY),
        True,
    )


def compare_extra_payment(
    balance: Decimal | int | float | str,
    apr: Decimal | int | float | str,
    monthly_payment: Decimal | int | float | str,
    extra: Decimal | int | float | str,
) -> ExtraPaymentComparison:
    """Compare the payoff time and interest with an additional monthly payment."""
    base_payment = _non_negative_money(monthly_payment, "monthly_payment")
    extra_payment = _non_negative_money(extra, "extra")
    baseline = debt_payoff(balance, apr, base_payment)
    with_extra = debt_payoff(balance, apr, base_payment + extra_payment)
    if baseline.pays_off and with_extra.pays_off:
        assert baseline.months is not None and with_extra.months is not None
        assert baseline.total_interest is not None and with_extra.total_interest is not None
        months_saved = baseline.months - with_extra.months
        interest_saved = (baseline.total_interest - with_extra.total_interest).quantize(_PENNY)
    else:
        months_saved = None
        interest_saved = None
    return ExtraPaymentComparison(baseline, with_extra, months_saved, interest_saved)


def payoff_date(start_date: date, months: int) -> date:
    """Return the first day of the payoff month after the given number of months."""
    if not isinstance(start_date, date):
        raise ValueError("start_date must be a date.")
    if isinstance(months, bool) or not isinstance(months, int) or months < 0:
        raise ValueError("months must be a non-negative integer.")
    month_index = start_date.year * 12 + start_date.month - 1 + months
    year, month = divmod(month_index, 12)
    return date(year, month + 1, 1)


def emergency_fund_status(
    saved: Decimal | int | float | str,
    monthly_essentials: Decimal | int | float | str,
    target_months: int = 3,
) -> EmergencyFundStatus:
    """Measure emergency-fund coverage and progress toward the target."""
    amount_saved = _non_negative_money(saved, "saved")
    essentials = _non_negative_money(monthly_essentials, "monthly_essentials")
    if essentials == 0:
        raise ValueError("monthly_essentials must be greater than zero.")
    if isinstance(target_months, bool) or not isinstance(target_months, int) or target_months <= 0:
        raise ValueError("target_months must be a positive integer.")

    target = (essentials * target_months).quantize(_PENNY)
    months_covered = (amount_saved / essentials).quantize(_PENNY, rounding=ROUND_HALF_UP)
    amount_left = max(target - amount_saved, _ZERO).quantize(_PENNY)
    percent_complete = min(amount_saved / target * 100, Decimal("100")).quantize(
        _PENNY, rounding=ROUND_HALF_UP
    )
    return EmergencyFundStatus(months_covered, target, amount_left, percent_complete)


def months_to_goal(
    current: Decimal | int | float | str,
    target: Decimal | int | float | str,
    monthly_saving: Decimal | int | float | str,
) -> GoalTimeline:
    """Estimate contribution months to a savings goal, ignoring interest."""
    saved = _non_negative_money(current, "current")
    goal = _non_negative_money(target, "target")
    contribution = _non_negative_money(monthly_saving, "monthly_saving")
    remaining = max(goal - saved, _ZERO).quantize(_PENNY)
    if remaining == 0:
        return GoalTimeline(0, remaining)
    if contribution == 0:
        return GoalTimeline(None, remaining)
    months = int((remaining / contribution).to_integral_value(rounding=ROUND_CEILING))
    return GoalTimeline(months, remaining)