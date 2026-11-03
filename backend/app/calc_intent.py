"""Deterministic interpretation of calculation requests in chat messages.

The co-pilot must never guess which number is which. For example, in
"I owe £2,000 on my credit card. What happens if I pay £50 extra?" the
£2,000 is a *balance* and the £50 is an *extra monthly payment*.

Each monetary amount is given a role from the cue words closest to it
(within its clause, never reaching past another amount). Strong cues such as
"owe", "balance" or "extra" decide a role; weak cues such as the bare verb
"pay" are not enough on their own. Anything uncertain produces a
clarification question instead of a calculation. Missing values may be filled
from the user's saved profile, and every input records where it came from so
the answer can be transparent about it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from app.finance import (
    ASSUMPTIONS,
    MAX_AMOUNT,
    DebtPayoffResult,
    compare_extra_payment,
    debt_payoff,
    emergency_fund_status,
)
from app.memory import Debt, UserProfile

Role = Literal[
    "balance", "payment", "extra", "new_payment", "lump_sum", "saved", "essentials", "unknown"
]

_AMOUNT = re.compile(
    r"(?P<prefix>£|\bGBP\s?)?"
    r"(?P<number>\d{1,3}(?:,\d{3})+|\d+)(?P<decimals>\.\d{1,2})?"
    r"(?P<k>k\b)?"
    r"(?:\s?(?P<suffix>pounds?\b|quid\b|GBP\b))?",
    re.IGNORECASE,
)
_PERCENT = re.compile(r"(?P<number>\d{1,3}(?:\.\d+)?)\s?(?:%|per\s?cent\b)", re.IGNORECASE)
_TIME_OR_PERCENT_AFTER = re.compile(
    r"\s*(?:%|per\s?cent|months?|years?|yrs?|weeks?|days?|mins?|minutes?|hours?)\b|\s*%",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(r"[.?!;:](?=\s|$)|,(?=\s)|\n")
_WORD = re.compile(r"[a-z']+")

# Phrases are matched against the words *before* an amount (ending at the
# amount) or *after* it (starting at the amount). Distance is measured in
# words between the cue and the amount; the nearest strong cue wins.
_STRONG_BEFORE: dict[Role, tuple[str, ...]] = {
    "extra": ("extra", "additional", "another", "more", "overpay", "overpaying", "on top of"),
    "balance": ("owe", "owing", "balance", "balance of", "debt of", "borrowed", "outstanding"),
    "payment": (
        "minimum payment",
        "monthly payment",
        "repayment",
        "repayments",
        "currently pay",
        "usually pay",
        "normally pay",
        "already pay",
        "paying",
    ),
    "lump_sum": ("lump sum", "lump sum of", "one off", "one-off", "once off", "windfall"),
    "saved": ("saved", "savings", "savings of", "put aside", "emergency fund of"),
    "essentials": ("essentials", "essential costs", "living costs", "outgoings", "bills"),
}
_STRONG_AFTER: dict[Role, tuple[str, ...]] = {
    "extra": ("extra", "more", "additional", "on top", "over"),
    "balance": (
        "balance",
        "debt",
        "owed",
        "on my card",
        "on my credit card",
        "on a credit card",
        "credit card",
        "on my loan",
        "on a loan",
        "loan",
        "overdraft",
        "on my overdraft",
    ),
    "new_payment": ("instead", "in total", "altogether"),
    "lump_sum": ("lump sum", "one off", "one-off", "in one go", "all at once"),
    "saved": ("saved", "in savings", "savings", "put aside"),
    "essentials": ("on essentials", "in essentials", "of essentials", "essentials"),
}
_FREQUENCY_AFTER = ("a month", "per month", "monthly", "each month", "every month", "pcm")
_PAYMENT_VERBS = {"pay", "paid", "paying", "put", "putting", "repay"}
_PERSONAL = re.compile(r"\b(?:my|mine|i've|i'm|i owe|i have)\b", re.IGNORECASE)
_APPROXIMATORS = (
    ("more", "than"),
    ("less", "than"),
    ("about",),
    ("around",),
    ("roughly",),
    ("approximately",),
    ("nearly",),
    ("almost",),
    ("just", "over"),
    ("just", "under"),
    ("under",),
    ("over",),
)
_NEW_PAYMENT_BEFORE = re.compile(r"\b(?:increase|raise|up|change|bump)\w*\s+(?:\w+\s+){0,3}to\s*$")
_EXTRA_BY_BEFORE = re.compile(r"\b(?:increase|raise|up|bump)\w*\s+(?:\w+\s+){0,3}by\s*$")

_DEBT_CONTEXT = re.compile(
    r"\b(?:debts?|credit cards?|store cards?|cards?|loans?|overdrafts?|repay\w*|"
    r"overpay\w*|pay(?:ing)? (?:off|extra|more)|payoff|owe|owing|"
    r"car finance|buy now,? pay later|bnpl|apr|interest)\b",
    re.IGNORECASE,
)
_BALANCE_WORD = re.compile(r"\bbalance\b", re.IGNORECASE)
_CALC_INTENT = re.compile(
    r"\b(?:how long|how much (?:interest|will|would|could|do|does)|what (?:happens|if)|"
    r"if i (?:pay|paid|put|overpay|increase)|when will|when would|how many months|"
    r"pay (?:off|extra|more)|overpay\w*|save me|clear (?:it|my|the))\b",
    re.IGNORECASE,
)
_SAVINGS_CONTEXT = re.compile(
    r"\b(?:emergency fund|safety net|rainy day|cushion|buffer|"
    r"months? of essentials|how long (?:will|would) my savings last)\b",
    re.IGNORECASE,
)
_GENERIC_DEBT_WORDS = {"my", "the", "a", "debt", "account", "balance", "and", "of", "for"}
_DEBT_TYPE_WORDS = {
    "card",
    "cards",
    "credit",
    "store",
    "loan",
    "loans",
    "overdraft",
    "finance",
    "mortgage",
    "bnpl",
    "catalogue",
}

_ROLE_LABELS: dict[str, str] = {
    "balance": "balance",
    "payment": "regular monthly payment",
    "extra": "extra monthly payment",
    "new_payment": "new total monthly payment",
    "saved": "savings",
    "essentials": "monthly essentials",
}


@dataclass(frozen=True)
class AmountMention:
    """One monetary amount found in a message and its interpreted role."""

    value: Decimal
    text: str
    role: Role
    currency_assumed: bool = False


@dataclass
class ParsedMessage:
    """Everything the parser could confidently extract from one message."""

    text: str = ""
    amounts: list[AmountMention] = field(default_factory=list)
    aprs: list[Decimal] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    words: set[str] = field(default_factory=set)
    debt_context: bool = False
    savings_context: bool = False
    calc_intent: bool = False
    extra_requested: bool = False
    personal: bool = False

    def by_role(self, role: Role) -> list[AmountMention]:
        return [amount for amount in self.amounts if amount.role == role]


@dataclass
class CalculationPlan:
    """Verified calculations to share, or a clarification question to ask."""

    status: Literal["none", "ready", "clarify"]
    calculations: list[dict[str, Any]] = field(default_factory=list)
    clarification: str | None = None


def is_personal_question(text: str) -> bool:
    """True when a message is about the user's own situation, not a general topic."""
    return bool(_PERSONAL.search(text) or _CALC_INTENT.search(text))


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("-", " "))


def _cue_distance(window: list[str], phrase: str, *, before: bool) -> int | None:
    """Return how many words separate `phrase` from the amount, if present."""
    tokens = phrase.replace("-", " ").split()
    size = len(tokens)
    best: int | None = None
    for start in range(len(window) - size + 1):
        if window[start : start + size] == tokens:
            distance = len(window) - (start + size) if before else start
            best = distance if best is None else min(best, distance)
    return best


def _clause_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    for match in _CLAUSE_BREAK.finditer(text, 0, start):
        left = match.end()
    right_match = _CLAUSE_BREAK.search(text, end)
    right = right_match.start() if right_match else len(text)
    return left, right


def _parse_value(match: re.Match[str]) -> Decimal | None:
    try:
        value = Decimal(match.group("number").replace(",", "") + (match.group("decimals") or ""))
    except InvalidOperation:
        return None
    if match.group("k"):
        value *= 1000
    return value


def _classify(before: list[str], after: list[str], before_text: str) -> Role | None:
    """Pick a role from nearby cues, or None when no cue applies."""
    if _NEW_PAYMENT_BEFORE.search(before_text):
        return "new_payment"
    if _EXTRA_BY_BEFORE.search(before_text):
        return "extra"
    candidates: dict[Role, int] = {}
    before = list(before)
    stripped = True
    while stripped:  # "owe more than £2,000" is still a balance.
        stripped = False
        for approximator in _APPROXIMATORS:
            size = len(approximator)
            if len(before) >= size and tuple(before[-size:]) == approximator:
                del before[-size:]
                stripped = True
    window_before = before[-5:]
    window_after = after[:5]
    for role, phrases in _STRONG_BEFORE.items():
        for phrase in phrases:
            distance = _cue_distance(window_before, phrase, before=True)
            if distance is not None:
                candidates[role] = min(candidates.get(role, distance), distance)
    for role, phrases in _STRONG_AFTER.items():
        for phrase in phrases:
            distance = _cue_distance(window_after, phrase, before=False)
            if distance is not None:
                candidates[role] = min(candidates.get(role, distance), distance)
    if candidates:
        nearest = min(candidates.values())
        roles = [role for role, distance in candidates.items() if distance == nearest]
        if len(roles) != 1:
            return "unknown"
        if roles[0] == "balance" and before and before[-1] in _PAYMENT_VERBS:
            return "unknown"  # "pay £50 on my card": a payment, but which kind?
        return roles[0]
    # Weak evidence: a stated frequency means a regular monthly amount.
    if any(_cue_distance(window_after, phrase, before=False) == 0 for phrase in _FREQUENCY_AFTER):
        return "payment"
    return None


def parse_message(text: str) -> ParsedMessage:
    """Extract amounts, APRs and calculation context from a message."""
    lowered = text.lower()
    savings_context = bool(_SAVINGS_CONTEXT.search(text))
    parsed = ParsedMessage(
        text=text,
        words=set(_words(text)),
        debt_context=bool(
            _DEBT_CONTEXT.search(text) or (_BALANCE_WORD.search(text) and not savings_context)
        ),
        savings_context=savings_context,
        calc_intent=bool(_CALC_INTENT.search(text)),
        personal=bool(_PERSONAL.search(text)),
        extra_requested=bool(re.search(r"\b(?:extra|overpay\w*|more|additional)\b", lowered)),
    )
    matches = [
        match
        for match in _AMOUNT.finditer(text)
        if not _TIME_OR_PERCENT_AFTER.match(text, match.end())
        and not (match.start() > 0 and text[match.start() - 1] in ".,%")
    ]
    spans = [(match.start(), match.end()) for match in matches]
    for index, match in enumerate(matches):
        value = _parse_value(match)
        if value is None or value > MAX_AMOUNT:
            continue
        explicit_currency = bool(match.group("prefix") or match.group("suffix"))
        clause_start, clause_end = _clause_bounds(text, match.start(), match.end())
        previous_end = spans[index - 1][1] if index > 0 else 0
        next_start = spans[index + 1][0] if index + 1 < len(spans) else len(text)
        before_text = text[max(clause_start, previous_end) : match.start()].lower()
        after_text = text[match.end() : min(clause_end, next_start)].lower()
        role = _classify(_words(before_text), _words(after_text), before_text)
        if role is None:
            if not explicit_currency:
                continue  # A bare number with no money cue (an age, a year...).
            role = "unknown"
        parsed.amounts.append(
            AmountMention(
                value=value,
                text=match.group(0).strip(),
                role=role,
                currency_assumed=not explicit_currency,
            )
        )
    for match in _PERCENT.finditer(text):
        after = _words(text[match.end() : match.end() + 30])[:2]
        if after and after[0] in {"more", "extra", "of"}:
            parsed.unsupported.append(match.group(0))
            continue
        parsed.aprs.append(Decimal(match.group("number")) / 100)
    return parsed


def _merge(previous: ParsedMessage, current: ParsedMessage) -> ParsedMessage:
    """Combine a clarified follow-up with the message that needed clarifying."""
    resolved_values = {amount.value for amount in current.amounts if amount.role not in {"unknown"}}
    current_roles = {amount.role for amount in current.amounts if amount.role != "unknown"}
    carried = []
    unknown_previous = [a for a in previous.amounts if a.role == "unknown"]
    for amount in previous.amounts:
        if amount.role == "unknown" and amount.value in resolved_values:
            continue  # The follow-up restated this amount with a clear role.
        if amount.role != "unknown" and amount.role in current_roles:
            continue  # The follow-up replaces this value.
        carried.append(amount)
    if not current.amounts and len(unknown_previous) == 1:
        # e.g. "It's the extra amount" answering "what is £50?"
        reply_words = _words(current.text)
        role = _classify(reply_words, [], " ".join(reply_words))
        if role and role != "unknown":
            carried = [a for a in carried if a is not unknown_previous[0]]
            carried.append(
                AmountMention(
                    unknown_previous[0].value,
                    unknown_previous[0].text,
                    role,
                    unknown_previous[0].currency_assumed,
                )
            )
    current_unknown = [a for a in current.amounts if a.role == "unknown"]
    current_known = [a for a in current.amounts if a.role != "unknown"]
    return ParsedMessage(
        text=current.text,
        amounts=carried + current_known + current_unknown,
        aprs=current.aprs or previous.aprs,
        unsupported=current.unsupported,
        words=previous.words | current.words,
        debt_context=previous.debt_context or current.debt_context,
        savings_context=previous.savings_context or current.savings_context,
        calc_intent=previous.calc_intent or current.calc_intent,
        personal=previous.personal or current.personal,
        extra_requested=previous.extra_requested or current.extra_requested,
    )


def _money_text(value: Decimal) -> str:
    return f"£{value:,.2f}".replace(".00", "")


def _result_dict(result: DebtPayoffResult) -> dict[str, Any]:
    return {
        "months": result.months,
        "total_interest": None if result.total_interest is None else str(result.total_interest),
        "total_paid": None if result.total_paid is None else str(result.total_paid),
        "final_payment": None if result.final_payment is None else str(result.final_payment),
        "pays_off": result.pays_off,
        "reason": result.reason,
    }


def _matching_debts(profile: UserProfile | None, words: set[str]) -> tuple[list[Debt], bool]:
    """Return profile debts named in the message, and whether a type was named."""
    if profile is None:
        return [], False
    mentioned_types = words & _DEBT_TYPE_WORDS
    matches = []
    for debt in profile.debts:
        name_words = set(_words(debt.name)) - _GENERIC_DEBT_WORDS
        if name_words & words:
            matches.append(debt)
    return matches, bool(mentioned_types)


def _single_value(amounts: list[AmountMention], role: Role) -> tuple[Decimal | None, str | None]:
    values = {amount.value for amount in amounts if amount.role == role}
    if len(values) > 1:
        listed = " and ".join(_money_text(value) for value in sorted(values))
        return None, f"You mentioned more than one {_ROLE_LABELS[role]} ({listed})."
    return (values.pop() if values else None), None


def _debt_calculation(
    *,
    name: str,
    balance: Decimal,
    apr: Decimal,
    payment: Decimal,
    extra: Decimal | None,
    sources: dict[str, str],
    assumed_currency: list[str],
) -> dict[str, Any]:
    inputs = {"balance": str(balance), "apr": str(apr), "monthly_payment": str(payment)}
    notes = list(ASSUMPTIONS)
    notes.extend(f"I read {text} as an amount in pounds." for text in assumed_currency)
    if extra is None:
        result = debt_payoff(balance, apr, payment)
        return {
            "type": "debt_payoff",
            "debt_name": name,
            "inputs": inputs,
            "input_sources": sources,
            "extra_amount_was_specified": False,
            "result": _result_dict(result),
            "assumptions": notes,
        }
    comparison = compare_extra_payment(balance, apr, payment, extra)
    return {
        "type": "extra_payment_comparison",
        "debt_name": name,
        "inputs": {**inputs, "extra_monthly_payment": str(extra)},
        "input_sources": sources,
        "extra_amount_was_specified": True,
        "baseline": _result_dict(comparison.baseline),
        "with_extra": _result_dict(comparison.with_extra),
        "months_saved": comparison.months_saved,
        "interest_saved": (
            None if comparison.interest_saved is None else str(comparison.interest_saved)
        ),
        "assumptions": notes,
    }


def _plan_debt(parsed: ParsedMessage, profile: UserProfile | None) -> CalculationPlan:
    amounts = parsed.amounts
    problems: list[str] = []
    for amount in amounts:
        if amount.role == "unknown":
            problems.append(
                f"I wasn't sure what {amount.text} refers to: is it the balance you owe, "
                "your usual monthly payment, or an extra amount on top each month?"
            )
        elif amount.role == "lump_sum":
            problems.append(
                f"I can compare regular extra monthly payments, but I can't yet model a "
                f"one-off payment like {amount.text}. Could you tell me a monthly amount instead?"
            )
        elif amount.role in {"saved", "essentials"}:
            problems.append(
                f"I wasn't sure how {amount.text} fits into a debt calculation. "
                "Could you say whether it's the balance, a monthly payment or an extra payment?"
            )
    for text in parsed.unsupported:
        problems.append(
            f"I can only compare extra payments given in pounds, not as {text}. "
            "How much extra in pounds would you like to try each month?"
        )
    balance, problem = _single_value(amounts, "balance")
    problems += [problem] if problem else []
    payment, problem = _single_value(amounts, "payment")
    problems += [problem] if problem else []
    extra, problem = _single_value(amounts, "extra")
    problems += [problem] if problem else []
    new_payment, problem = _single_value(amounts, "new_payment")
    problems += [problem] if problem else []
    if len(set(parsed.aprs)) > 1:
        problems.append("You mentioned more than one interest rate. Which APR should I use?")
    if problems:
        return CalculationPlan("clarify", clarification=" ".join(problems))

    stated_any = any(value is not None for value in (balance, payment, extra, new_payment))
    stated_apr = parsed.aprs[0] if parsed.aprs else None
    matches, type_named = _matching_debts(profile, parsed.words)
    debts = profile.debts if profile else []

    if not stated_any and stated_apr is None:
        # No figures given: share baseline estimates for the user's own saved
        # debts, but only for personal questions. A general question such as
        # "How does card interest work?" doesn't need their financial details.
        if not (parsed.personal or parsed.calc_intent):
            return CalculationPlan("none")
        selected = matches or ([] if type_named else debts)
        if not selected:
            if parsed.calc_intent and not debts:
                return CalculationPlan(
                    "clarify",
                    clarification=(
                        "I can work that out with NetValor's calculator if you tell me the "
                        "balance, the APR (interest rate) and your usual monthly payment."
                    ),
                )
            return CalculationPlan("none")
        calculations = [
            _debt_calculation(
                name=debt.name,
                balance=debt.balance,
                apr=debt.apr,
                payment=debt.monthly_payment,
                extra=None,
                sources={"balance": "profile", "apr": "profile", "monthly_payment": "profile"},
                assumed_currency=[],
            )
            for debt in selected
        ]
        return CalculationPlan("ready", calculations)

    # Figures were given. Pick at most one saved debt to fill any gaps.
    base: Debt | None = None
    if len(matches) == 1:
        base = matches[0]
    elif len(matches) > 1:
        names = ", ".join(debt.name for debt in matches)
        return CalculationPlan(
            "clarify",
            clarification=f"Which debt do you mean: {names}?",
        )
    elif not type_named and len(debts) == 1:
        base = debts[0]
    elif not type_named and len(debts) > 1 and balance is None:
        names = ", ".join(debt.name for debt in debts)
        return CalculationPlan(
            "clarify", clarification=f"Which of your saved debts should I use: {names}?"
        )

    sources: dict[str, str] = {}

    def pick(stated: Decimal | None, saved: Decimal | None, key: str) -> Decimal | None:
        if stated is not None:
            sources[key] = "message"
            return stated
        if saved is not None:
            sources[key] = "profile"
            return saved
        return None

    final_balance = pick(balance, base.balance if base else None, "balance")
    final_apr = pick(stated_apr, base.apr if base else None, "apr")
    final_payment = pick(payment, base.monthly_payment if base else None, "monthly_payment")
    missing = [
        label
        for label, value in (
            ("the balance you owe", final_balance),
            ("the APR (interest rate)", final_apr),
            ("your usual monthly payment", final_payment),
        )
        if value is None
    ]
    if missing:
        return CalculationPlan(
            "clarify",
            clarification="To work this out I also need " + ", ".join(missing) + ".",
        )
    assert final_balance is not None
    assert final_apr is not None
    assert final_payment is not None
    if final_apr > 1:
        return CalculationPlan(
            "clarify", clarification="That interest rate looks above 100%. Could you check it?"
        )
    if new_payment is not None:
        if extra is not None:
            return CalculationPlan(
                "clarify",
                clarification=(
                    "Should I use a new total monthly payment of "
                    f"{_money_text(new_payment)} or {_money_text(extra)} extra on top?"
                ),
            )
        if new_payment <= final_payment:
            return CalculationPlan(
                "clarify",
                clarification=(
                    f"{_money_text(new_payment)} isn't more than the current monthly payment of "
                    f"{_money_text(final_payment)}. What new payment would you like to compare?"
                ),
            )
        extra = new_payment - final_payment
        sources["extra_monthly_payment"] = "message (new total minus current payment)"
    elif extra is not None:
        sources["extra_monthly_payment"] = "message"

    assumed = [amount.text for amount in amounts if amount.currency_assumed]
    name = base.name if base else "The debt you described"
    return CalculationPlan(
        "ready",
        [
            _debt_calculation(
                name=name,
                balance=final_balance,
                apr=final_apr,
                payment=final_payment,
                extra=extra,
                sources=sources,
                assumed_currency=assumed,
            )
        ],
    )


def _plan_savings(parsed: ParsedMessage, profile: UserProfile | None) -> CalculationPlan:
    if any(amount.role not in {"saved", "essentials"} for amount in parsed.amounts):
        return CalculationPlan(
            "clarify",
            clarification=(
                "For an emergency-fund check I need how much you have saved and your "
                "monthly essential costs. Which of the amounts you mentioned is which?"
            ),
        )
    saved, problem_saved = _single_value(parsed.amounts, "saved")
    essentials, problem_ess = _single_value(parsed.amounts, "essentials")
    problems = [p for p in (problem_saved, problem_ess) if p]
    if problems:
        return CalculationPlan("clarify", clarification=" ".join(problems))
    sources = {
        "saved": "message" if saved is not None else "profile",
        "monthly_essentials": "message" if essentials is not None else "profile",
    }
    if saved is None and profile is not None:
        saved = profile.savings
    if essentials is None and profile is not None and profile.monthly_essentials > 0:
        essentials = profile.monthly_essentials
    if saved is None or essentials is None or essentials == 0:
        if parsed.calc_intent or parsed.amounts:
            return CalculationPlan(
                "clarify",
                clarification=(
                    "To check an emergency fund I need how much you have saved and "
                    "your monthly essential costs."
                ),
            )
        return CalculationPlan("none")
    status = emergency_fund_status(saved, essentials, 3)
    return CalculationPlan(
        "ready",
        [
            {
                "type": "emergency_fund",
                "inputs": {
                    "saved": str(saved),
                    "monthly_essentials": str(essentials),
                    "target_months": 3,
                },
                "input_sources": sources,
                "result": {
                    "months_covered": str(status.months_covered),
                    "target_amount": str(status.target_amount),
                    "amount_left": str(status.amount_left),
                    "percent_complete": str(status.percent_complete),
                },
                "assumptions": [
                    "A three-month target is a common starting point, not a rule.",
                    "Interest earned on savings and inflation are ignored.",
                ],
            }
        ],
    )


def plan_calculations(
    message: str,
    profile: UserProfile | None,
    clarified_message: str | None = None,
) -> CalculationPlan:
    """Decide which verified calculations, if any, answer a chat message.

    `clarified_message` is the earlier user message that the co-pilot asked
    to clarify; its unambiguous amounts carry over into this turn.
    """
    parsed = parse_message(message)
    if clarified_message:
        parsed = _merge(parse_message(clarified_message), parsed)
    if parsed.savings_context and not parsed.debt_context:
        return _plan_savings(parsed, profile)
    if parsed.by_role("extra") or parsed.by_role("new_payment") or parsed.by_role("balance"):
        parsed.debt_context = True  # Extra repayments and balances only apply to debts.
    if parsed.debt_context or (parsed.amounts and parsed.calc_intent):
        if not parsed.debt_context:
            return CalculationPlan(
                "clarify",
                clarification=(
                    "Is this about paying off a debt or about savings? Let me know so I "
                    "can run the right calculation."
                ),
            )
        return _plan_debt(parsed, profile)
    return CalculationPlan("none")
