"""Regression tests for interpreting calculation requests in chat messages."""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.calc_intent import parse_message, plan_calculations
from app.memory import Debt, UserProfile


def _profile(*debts: Debt) -> UserProfile:
    return UserProfile(
        name="Sam",
        monthly_take_home_income=Decimal("2800"),
        monthly_essentials=Decimal("1700"),
        savings=Decimal("3400"),
        debts=list(debts),
    )


CARD = Debt(
    name="Credit card",
    balance=Decimal("2450"),
    apr=Decimal("0.249"),
    monthly_payment=Decimal("350"),
)
LOAN = Debt(
    name="Car loan", balance=Decimal("6000"), apr=Decimal("0.079"), monthly_payment=Decimal("220")
)


def _clarification(message: str, profile: UserProfile | None, previous: str | None = None) -> str:
    plan = plan_calculations(message, profile, previous)
    assert plan.status == "clarify"
    assert plan.clarification is not None
    return plan.clarification


def _roles(message: str) -> list[tuple[Decimal, str]]:
    return [(amount.value, amount.role) for amount in parse_message(message).amounts]


# --- The known bug: balance confused with the extra payment ----------------


def test_known_bug_balance_is_not_used_as_extra_payment() -> None:
    message = "I owe £2,000 on my credit card. What happens if I pay £50 extra?"

    assert _roles(message) == [(Decimal("2000"), "balance"), (Decimal("50"), "extra")]
    plan = plan_calculations(message, _profile(CARD))

    assert plan.status == "ready"
    calc = plan.calculations[0]
    assert calc["type"] == "extra_payment_comparison"
    assert calc["inputs"] == {
        "balance": "2000",
        "apr": "0.249",
        "monthly_payment": "350",
        "extra_monthly_payment": "50",
    }
    assert calc["input_sources"]["balance"] == "message"
    assert calc["input_sources"]["apr"] == "profile"
    assert calc["baseline"]["months"] == 7
    assert calc["with_extra"]["months"] == 6


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "What happens if I pay £50 extra on my £2,000 balance?",
            [("50", "extra"), ("2000", "balance")],
        ),
        (
            "I owe more than £2,000 on my card, what if I pay an extra £50 a month?",
            [("2000", "balance"), ("50", "extra")],
        ),
        ("My balance is £2k. Could I overpay £75?", [("2000", "balance"), ("75", "extra")]),
        (
            "I have a £1,500.50 debt and want to pay £25 more each month",
            [("1500.50", "balance"), ("25", "more")],
        ),
        (
            "I currently pay £350 a month on a £2,450 balance",
            [("350", "payment"), ("2450", "balance")],
        ),
        ("My minimum payment is £60", [("60", "payment")]),
        ("What if I increase my payment to £400?", [("400", "new_payment")]),
        ("What if I increase my payments by £40?", [("40", "extra")]),
        ("I owe 2000 pounds and pay GBP 100 per month", [("2000", "balance"), ("100", "payment")]),
    ],
)
def test_amount_roles_from_nearby_cues(message: str, expected: list[tuple[str, str]]) -> None:
    roles = _roles(message)
    normalised = [(value, "extra" if role == "extra" else role) for value, role in roles]
    assert normalised == [
        (Decimal(value), "extra" if role == "more" else role) for value, role in expected
    ]


@pytest.mark.parametrize(
    "message",
    [
        "I'm 25 and owe £3,000 on a loan",  # age is not money
        "I want to be debt free by 2027 and owe £3,000",  # year is not money
        "It will take 12 months to clear £3,000 I owe",  # months are not money
    ],
)
def test_non_money_numbers_are_ignored(message: str) -> None:
    assert [value for value, _ in _roles(message)] == [Decimal("3000")]


def test_percentage_is_read_as_apr_and_not_as_money() -> None:
    parsed = parse_message("I owe £2,000 on my card at 24.9% APR")

    assert parsed.aprs == [Decimal("0.249")]
    assert [amount.value for amount in parsed.amounts] == [Decimal("2000")]


# --- Ambiguity leads to clarification, never a silent guess ----------------


@pytest.mark.parametrize(
    ("message", "fragment"),
    [
        ("What if I pay £50 on my card?", "wasn't sure what £50 refers to"),
        ("I have £2,000 and £3,000 on my card", "wasn't sure what £2,000"),
        ("What if I paid a lump sum of £500 off my card?", "one-off payment"),
        ("I owe £3k on a loan, what if I pay 10% more?", "not as 10%"),
        ("I owe £2,000 and £3,000 on my cards", "£3,000"),
        ("I owe £2,000 on my card and my balance is £3,000", "more than one balance"),
        ("My card is at 20% or maybe 25% interest, I owe £1,000", "more than one interest rate"),
    ],
)
def test_ambiguous_requests_ask_for_clarification(message: str, fragment: str) -> None:
    plan = plan_calculations(message, _profile(CARD))

    assert plan.calculations == []
    assert fragment in _clarification(message, _profile(CARD))


def test_missing_rate_and_payment_without_profile_asks_for_them() -> None:
    text = _clarification("I owe £2,000 on my card. What if I pay £50 extra?", None)

    assert "APR" in text
    assert "monthly payment" in text


def test_extra_amount_with_several_saved_debts_asks_which_one() -> None:
    text = _clarification("What if I pay £50 extra a month?", _profile(CARD, LOAN))

    assert "Credit card" in text
    assert "Car loan" in text


def test_named_debt_is_selected_from_several() -> None:
    plan = plan_calculations("What if I pay £50 extra on my car loan?", _profile(CARD, LOAN))

    assert plan.status == "ready"
    assert plan.calculations[0]["debt_name"] == "Car loan"
    assert plan.calculations[0]["inputs"]["balance"] == "6000"


def test_mentioned_debt_type_not_in_profile_is_not_filled_from_another_debt() -> None:
    text = _clarification("I owe £900 on my overdraft, what if I pay £20 extra?", _profile(CARD))

    assert "APR" in text


def test_new_total_payment_becomes_extra_over_current_payment() -> None:
    plan = plan_calculations("What if I increase my card payment to £400?", _profile(CARD))

    calc = plan.calculations[0]
    assert calc["inputs"]["extra_monthly_payment"] == "50"
    assert calc["input_sources"]["extra_monthly_payment"].startswith("message")


def test_new_total_payment_below_current_asks_again() -> None:
    text = _clarification("What if I change my card payment to £300?", _profile(CARD))

    assert "isn't more than" in text


def test_fully_stated_debt_needs_no_profile() -> None:
    plan = plan_calculations(
        "I owe £1,000 on a card at 20% APR and pay £100 a month. How long will it take?", None
    )

    calc = plan.calculations[0]
    assert calc["type"] == "debt_payoff"
    assert calc["debt_name"] == "The debt you described"
    assert set(calc["input_sources"].values()) == {"message"}
    assert calc["result"]["months"] == 12  # closed form gives 11.03 payments


def test_bare_numbers_with_money_cues_record_the_currency_assumption() -> None:
    plan = plan_calculations("I owe 2000 on my card and pay 100 a month at 24.9% APR", None)

    assumptions = plan.calculations[0]["assumptions"]
    assert any("2000" in note and "pounds" in note for note in assumptions)


# --- No figures: saved debts only for personal questions -------------------


def test_personal_question_without_figures_uses_saved_debts() -> None:
    plan = plan_calculations("Should I pay extra on my card?", _profile(CARD))

    assert plan.status == "ready"
    assert plan.calculations[0]["type"] == "debt_payoff"
    assert plan.calculations[0]["extra_amount_was_specified"] is False


def test_general_question_does_not_expose_saved_debts() -> None:
    assert plan_calculations("How does credit card interest work?", _profile(CARD)).status == "none"


def test_unrelated_question_needs_no_calculation() -> None:
    assert plan_calculations("What is a budget?", _profile(CARD)).status == "none"


# --- Multi-turn clarification ---------------------------------------------


def test_follow_up_resolves_the_ambiguous_amount() -> None:
    first = "What if I pay £50 on my card?"
    assert plan_calculations(first, _profile(CARD)).status == "clarify"

    plan = plan_calculations("It's £50 extra on top of my usual payment", _profile(CARD), first)

    assert plan.status == "ready"
    assert plan.calculations[0]["inputs"]["extra_monthly_payment"] == "50"
    assert plan.calculations[0]["inputs"]["balance"] == "2450"


def test_one_word_follow_up_assigns_the_single_unknown_amount() -> None:
    first = "What if I pay £50 on my card?"

    plan = plan_calculations("the extra amount", _profile(CARD), first)

    assert plan.status == "ready"
    assert plan.calculations[0]["inputs"]["extra_monthly_payment"] == "50"


def test_follow_up_supplies_missing_details() -> None:
    first = "I owe £2,000 on my card. What if I pay £50 extra?"
    assert plan_calculations(first, None).status == "clarify"

    plan = plan_calculations("It's 19.9% APR and I pay £150 a month", None, first)

    assert plan.status == "ready"
    assert plan.calculations[0]["inputs"] == {
        "balance": "2000",
        "apr": "0.199",
        "monthly_payment": "150",
        "extra_monthly_payment": "50",
    }


# --- Savings ---------------------------------------------------------------


def test_emergency_fund_question_uses_saved_figures() -> None:
    plan = plan_calculations("How is my emergency fund looking?", _profile(CARD))

    calc = plan.calculations[0]
    assert calc["type"] == "emergency_fund"
    assert calc["result"]["months_covered"] == "2.00"


def test_emergency_fund_with_stated_figures() -> None:
    plan = plan_calculations(
        "I have £1,000 saved and my essentials are £800. Is my safety net enough?", None
    )

    assert plan.calculations[0]["result"]["months_covered"] == "1.25"
