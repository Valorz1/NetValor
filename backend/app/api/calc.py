"""Deterministic calculation endpoints. See `app.finance` for assumptions."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.api.errors import ErrorResponse
from app.finance import (
    ASSUMPTIONS,
    MAX_AMOUNT,
    MAX_TARGET_MONTHS,
    compare_extra_payment,
    debt_payoff,
    emergency_fund_status,
    months_to_goal,
    payoff_date,
)

router = APIRouter(
    prefix="/api/calc", tags=["calculations"], responses={422: {"model": ErrorResponse}}
)
Money = Annotated[Decimal, Field(ge=0, le=MAX_AMOUNT)]


class DebtPayoffRequest(BaseModel):
    """Inputs for estimating a debt payoff."""

    model_config = ConfigDict(extra="forbid")

    balance: Money
    apr: Decimal = Field(ge=0, le=1, description="Annual rate as a fraction: 0.249 = 24.9%.")
    monthly_payment: Money


class DebtPayoffResponse(BaseModel):
    """Debt payoff estimate returned by the calculation engine."""

    months: int | None
    total_interest: Decimal | None
    total_paid: Decimal | None
    final_payment: Decimal | None
    pays_off: bool
    reason: str | None = Field(
        default=None,
        description=(
            "Why the debt doesn't pay off: payment_does_not_cover_interest or exceeds_max_term."
        ),
    )


class CompareExtraPaymentRequest(DebtPayoffRequest):
    """Inputs for comparing a regular and increased monthly payment."""

    extra: Money


class CompareExtraPaymentResponse(BaseModel):
    """Payment comparison and estimated payoff months."""

    baseline: DebtPayoffResponse
    with_extra: DebtPayoffResponse
    months_saved: int | None
    interest_saved: Decimal | None
    baseline_payoff_date: date | None
    with_extra_payoff_date: date | None
    assumptions: list[str]


class EmergencyFundRequest(BaseModel):
    """Inputs for checking emergency-fund coverage."""

    model_config = ConfigDict(extra="forbid")

    saved: Money
    monthly_essentials: Annotated[Decimal, Field(gt=0, le=MAX_AMOUNT)]
    target_months: int = Field(default=3, ge=1, le=MAX_TARGET_MONTHS)


class EmergencyFundResponse(BaseModel):
    """Emergency-fund target and progress."""

    months_covered: Decimal
    target_amount: Decimal
    amount_left: Decimal
    percent_complete: Decimal


class GoalTimelineRequest(BaseModel):
    """Inputs for estimating how long a savings goal will take."""

    model_config = ConfigDict(extra="forbid")

    current: Money
    target: Money
    monthly_saving: Money


class GoalTimelineResponse(BaseModel):
    months: int | None
    amount_remaining: Decimal
    reason: str | None
    target_date: date | None


def _payoff(result: object) -> DebtPayoffResponse:
    return DebtPayoffResponse.model_validate(result, from_attributes=True)


def _month_date(months: int | None) -> date | None:
    return payoff_date(date.today(), months) if months is not None else None


@router.post("/debt-payoff", response_model=DebtPayoffResponse)
def calculate_debt_payoff(request: DebtPayoffRequest) -> DebtPayoffResponse:
    """Estimate months and interest to clear a debt with a fixed monthly payment."""
    try:
        result = debt_payoff(request.balance, request.apr, request.monthly_payment)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _payoff(result)


@router.post("/compare-extra", response_model=CompareExtraPaymentResponse)
def calculate_extra_payment(request: CompareExtraPaymentRequest) -> CompareExtraPaymentResponse:
    """Compare a debt payoff with and without an extra monthly payment."""
    try:
        result = compare_extra_payment(
            request.balance, request.apr, request.monthly_payment, request.extra
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return CompareExtraPaymentResponse(
        baseline=_payoff(result.baseline),
        with_extra=_payoff(result.with_extra),
        months_saved=result.months_saved,
        interest_saved=result.interest_saved,
        baseline_payoff_date=_month_date(result.baseline.months),
        with_extra_payoff_date=_month_date(result.with_extra.months),
        assumptions=list(ASSUMPTIONS),
    )


@router.post("/emergency-fund", response_model=EmergencyFundResponse)
def calculate_emergency_fund(request: EmergencyFundRequest) -> EmergencyFundResponse:
    """Calculate emergency-fund coverage and progress toward a target."""
    try:
        result = emergency_fund_status(
            request.saved, request.monthly_essentials, request.target_months
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return EmergencyFundResponse.model_validate(result, from_attributes=True)


@router.post("/goal-timeline", response_model=GoalTimelineResponse)
def calculate_goal_timeline(request: GoalTimelineRequest) -> GoalTimelineResponse:
    """Estimate monthly contributions needed to reach a savings goal (no interest)."""
    try:
        result = months_to_goal(request.current, request.target, request.monthly_saving)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return GoalTimelineResponse(
        months=result.months,
        amount_remaining=result.amount_remaining,
        reason=result.reason,
        target_date=_month_date(result.months),
    )
