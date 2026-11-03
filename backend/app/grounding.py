"""Evidence handling, citation checks and safety signals for co-pilot answers.

These checks are deliberately simple and deterministic. They catch common
failure modes (invented citation IDs, figures that appear nowhere in the
evidence, citations whose sentence shares little with the cited passage,
instructions hidden in retrieved text) but they do **not** prove an answer
is factually correct. Their results are reported alongside each answer.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Untrusted text: prompt-injection screening and escaping
# --------------------------------------------------------------------------

_INJECTION = re.compile(
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier|system) "
    r"(?:instructions?|prompts?|rules?)"
    r"|disregard (?:all |any |the )?(?:previous |prior |above |system )?(?:instructions?|rules?)"
    r"|forget (?:all |your )?(?:previous )?instructions"
    r"|you are now\b|new instructions?:|system prompt|developer mode"
    r"|<\|?(?:im_start|im_end|system|endoftext)\|?>|\[/?inst\]|</?(?:source|evidence|system)\b",
    re.IGNORECASE,
)


def looks_like_injection(text: str) -> bool:
    """True if untrusted text contains instruction-like or delimiter-breaking content."""
    return bool(_INJECTION.search(text))


def escape_untrusted(text: str) -> str:
    """Neutralise angle brackets so untrusted text can't close our delimiters."""
    return text.replace("<", "‹").replace(">", "›")


# --------------------------------------------------------------------------
# Evidence selection and formatting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """A retrieved passage the model may cite as [S<n>]."""

    source_id: str
    lesson_id: str
    lesson_title: str
    heading: str
    urls: list[str]
    review_status: str
    chunk: str
    score: float


def select_evidence(
    results: Iterable[dict[str, Any]],
    *,
    allowed_statuses: frozenset[str],
    min_score: float,
    k: int,
) -> tuple[list[Source], dict[str, int]]:
    """Filter retrieved chunks to reviewed, relevant, safe and distinct passages."""
    dropped = {"status": 0, "score": 0, "injection": 0, "duplicate": 0}
    seen: set[str] = set()
    sources: list[Source] = []
    for result in sorted(results, key=lambda item: item.get("score", 0.0), reverse=True):
        metadata = result.get("metadata") or {}
        status = str(metadata.get("review_status", ""))
        chunk = str(result.get("chunk", ""))
        if status not in allowed_statuses:
            dropped["status"] += 1
            continue
        if float(result.get("score", 0.0)) < min_score:
            dropped["score"] += 1
            continue
        if looks_like_injection(chunk):
            dropped["injection"] += 1
            logger.warning(
                "Dropped lesson chunk %s: instruction-like content.", result.get("chunk_id")
            )
            continue
        key = str(result.get("chunk_id") or chunk)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        urls = metadata.get("source_urls", [])
        if isinstance(urls, str):
            try:
                urls = json.loads(urls)
            except json.JSONDecodeError:
                urls = []
        sources.append(
            Source(
                source_id=f"S{len(sources) + 1}",
                lesson_id=str(metadata.get("lesson_id", "")),
                lesson_title=str(metadata.get("title", "Lesson")),
                heading=str(metadata.get("heading", "")),
                urls=[str(url) for url in urls],
                review_status=status,
                chunk=chunk,
                score=float(result.get("score", 0.0)),
            )
        )
        if len(sources) == k:
            break
    return sources, dropped


def format_sources(sources: list[Source]) -> str:
    """Render sources inside delimiters the model is told to treat as data."""
    if not sources:
        return "<sources>\nNo lesson passages matched this question.\n</sources>"
    blocks = []
    for source in sources:
        title = escape_untrusted(source.lesson_title)
        section = escape_untrusted(source.heading)
        blocks.append(
            f'<source id="{source.source_id}" title="{title}" section="{section}" '
            f'review_status="{source.review_status}">\n'
            f"{escape_untrusted(source.chunk)}\n</source>"
        )
    return "<sources>\n" + "\n".join(blocks) + "\n</sources>"


# --------------------------------------------------------------------------
# Streaming citation filter
# --------------------------------------------------------------------------

_CITATION_GROUP = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\]", re.IGNORECASE)
_CITATION_ID = re.compile(r"S?(\d+)", re.IGNORECASE)
_PARTIAL_CITATION = re.compile(r"\[\s*(?:S\d*(?:\s*[,;]\s*S?\d*)*\s*)?$", re.IGNORECASE)


class CitationFilter:
    """Remove citations to unknown sources from a token stream.

    Tokens may split a citation (for example "[S" then "1]"), so a possible
    partial citation at the end of the buffer is held back until complete.
    """

    def __init__(self, valid_ids: set[str]) -> None:
        self.valid_ids = valid_ids
        self.cited: set[str] = set()
        self.invalid: set[str] = set()
        self._pending = ""

    def _rewrite(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            ids = [f"S{number}" for number in _CITATION_ID.findall(match.group(0)[1:-1])]
            kept = []
            for source_id in ids:
                if source_id in self.valid_ids:
                    if source_id not in kept:
                        kept.append(source_id)
                    self.cited.add(source_id)
                else:
                    self.invalid.add(source_id)
                    logger.warning("Removed invalid lesson citation [%s].", source_id)
            return "".join(f"[{source_id}]" for source_id in kept)

        return _CITATION_GROUP.sub(replace, text)

    def feed(self, token: str) -> str:
        self._pending += token
        opening = self._pending.rfind("[")
        if opening >= 0 and _PARTIAL_CITATION.fullmatch(self._pending[opening:]):
            ready, self._pending = self._pending[:opening], self._pending[opening:]
        else:
            ready, self._pending = self._pending, ""
        return self._rewrite(ready)

    def flush(self) -> str:
        ready, self._pending = self._pending, ""
        return self._rewrite(ready)


# --------------------------------------------------------------------------
# Post-generation grounding report
# --------------------------------------------------------------------------

_FIGURE = re.compile(
    r"£\s?(?P<money>\d[\d,]*(?:\.\d+)?)(?P<k>k\b)?"
    r"|(?P<percent>\d+(?:\.\d+)?)\s?%"
    r"|(?P<period>\d+(?:\.\d+)?)\s(?:months?|years?)\b",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMBER_WORDS: dict[str, Decimal | int] = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "half": Decimal("0.5"),
}
_CONTENT_WORD = re.compile(r"[a-z][a-z'-]{3,}")
_STOPWORDS = frozenset(
    [
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "your",
        "they",
        "them",
        "their",
        "there",
        "these",
        "those",
        "what",
        "when",
        "where",
        "which",
        "while",
        "about",
        "could",
        "would",
        "should",
        "into",
        "than",
        "then",
        "also",
        "just",
        "more",
        "most",
        "some",
        "such",
        "only",
        "very",
        "much",
        "many",
        "each",
        "other",
        "because",
        "being",
        "been",
        "were",
        "does",
        "doing",
        "over",
        "under",
        "after",
        "before",
        "again",
        "further",
        "here",
        "once",
        "both",
        "same",
        "make",
        "made",
        "like",
        "your",
        "yours",
        "mean",
        "means",
        "can't",
        "don't",
        "it's",
        "you're",
        "isn't",
        "aren't",
        "might",
    ]
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def allowed_figures(texts: Iterable[str], calculations: list[dict[str, Any]]) -> set[Decimal]:
    """Every number the answer may legitimately mention."""
    values: set[Decimal] = set()
    for text in texts:
        for match in _NUMBER.finditer(text):
            value = _decimal(match.group(0))
            if value is not None:
                values.add(value)
        for word in re.findall(r"[a-z]+", text.lower()):
            if word in _NUMBER_WORDS:
                values.add(Decimal(_NUMBER_WORDS[word]))

    def walk(item: Any, key: str = "") -> None:
        if isinstance(item, dict):
            for child_key, child in item.items():
                walk(child, child_key)
        elif isinstance(item, list):
            for child in item:
                walk(child, key)
        elif isinstance(item, (int, str, Decimal)) and not isinstance(item, bool):
            value = _decimal(str(item))
            if value is None:
                return
            values.add(value)
            if key == "apr":
                values.add(value * 100)
            if key in {"months", "months_saved", "target_months"}:
                values.add((value / 12).quantize(Decimal("0.1")))

    walk(calculations)
    return values


def _is_supported(value: Decimal, allowed: set[Decimal]) -> bool:
    for candidate in allowed:
        tolerance = max(Decimal("0.5"), abs(candidate) * Decimal("0.005"))
        if abs(value - candidate) <= tolerance:
            return True
    return False


@dataclass
class GroundingReport:
    """Heuristic signals about how well an answer is supported."""

    unverified_figures: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    weak_citations: list[str] = field(default_factory=list)
    uncited_sentences: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "unverified_figures": self.unverified_figures,
            "invalid_citations": self.invalid_citations,
            "weak_citations": self.weak_citations,
            "uncited_sentences": self.uncited_sentences,
            "note": "Automated heuristics; they do not guarantee factual accuracy.",
        }


_SUFFIXES = ("ations", "ation", "ings", "ing", "ison", "ed", "es", "s", "ly")


def _stem(word: str) -> str:
    """Crude suffix stripping so "cards"/"card" and "comparing"/"comparison" match."""
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _content_words(text: str) -> set[str]:
    return {_stem(word) for word in _CONTENT_WORD.findall(text.lower()) if word not in _STOPWORDS}


def _figures(text: str) -> list[tuple[str, Decimal]]:
    found = []
    for match in _FIGURE.finditer(text):
        raw = match.group("money") or match.group("percent") or match.group("period")
        value = _decimal(raw)
        if value is None:
            continue
        if match.group("k"):
            value *= 1000
        found.append((match.group(0).strip(), value))
    return found


def check_grounding(
    answer: str,
    sources: list[Source],
    allowed: set[Decimal],
    invalid_citations: set[str],
) -> GroundingReport:
    """Compare an answer with its evidence and verified calculations."""
    report = GroundingReport(invalid_citations=sorted(invalid_citations))
    for figure, value in _figures(answer):
        if not _is_supported(value, allowed) and figure not in report.unverified_figures:
            report.unverified_figures.append(figure)

    by_id = {
        source.source_id: _content_words(source.chunk + " " + source.heading) for source in sources
    }
    weak: set[str] = set()
    for sentence in _SENTENCE.split(answer):
        cited = {f"S{n}" for n in re.findall(r"\[S(\d+)\]", sentence)}
        words = _content_words(re.sub(r"\[S\d+\]", "", sentence))
        if not cited:
            if len(words) >= 6:
                report.uncited_sentences += 1
            continue
        figures = _figures(sentence)
        if len(words) < 3 or (figures and all(_is_supported(v, allowed) for _, v in figures)):
            continue  # Too short to judge, or grounded in verified figures.
        for source_id in cited:
            source_words = by_id.get(source_id, set())
            overlap = len(words & source_words) / len(words)
            if overlap < 0.2:
                weak.add(source_id)
    report.weak_citations = sorted(weak)
    return report


# --------------------------------------------------------------------------
# Sensitive situations and advice requests
# --------------------------------------------------------------------------

_CRISIS = re.compile(
    r"\bbailiffs?\b|\beviction\b|\bevicted\b|\bcourt\b|\bccj\b|\brent arrears\b|"
    r"\bmortgage arrears\b|\bcouncil tax arrears\b|\bdisconnect(?:ed|ion)?\b|"
    r"\bcan'?t afford (?:food|rent|to eat|my rent|bills)\b|\bcannot afford (?:food|rent)\b|"
    r"\bno food\b|\bfood ?bank\b|\brepossess(?:ion|ed)?\b|\bdebt collectors?\b|"
    r"\bdefault notice\b|\bhomeless\b|\bloan sharks?\b|\billegal lender\b",
    re.IGNORECASE,
)
_WELLBEING = re.compile(
    r"\bsuicid\w*|\bkill myself\b|\bend (?:it all|my life)\b|\bself[- ]harm\w*|"
    r"\bdon'?t want to (?:live|be here)\b|\bcan'?t cope\b|\bhopeless\b",
    re.IGNORECASE,
)
_ADVICE = re.compile(
    r"\bshould i (?:invest|buy|sell|switch|move|take out|transfer|open|cash in)\b|"
    r"\bwhich (?:isa|fund|shares?|stocks?|pension|mortgage|credit card|loan|provider|bank|"
    r"crypto\w*)\b.{0,20}\b(?:should|best|recommend)|"
    r"\bbest (?:isa|fund|investment|mortgage|credit card|loan|pension|savings account)\b|"
    r"\brecommend (?:a|an|me)\b|"
    r"\b(?:which|what) (?:shares?|stocks?|funds?|crypto\w*|investments?|isas?)\b"
    r".{0,25}\b(?:to buy|to invest|should i|do i buy)\b|"
    r"\btell me (?:what|which) to (?:buy|invest in)\b",
    re.IGNORECASE,
)

DEBT_SIGNPOSTING = (
    "If you're facing urgent debt, housing or essential-bill problems, free and "
    "confidential help is available from [MoneyHelper](https://www.moneyhelper.org.uk/), "
    "[StepChange](https://www.stepchange.org/), "
    "[National Debtline](https://nationaldebtline.org/) and "
    "[Citizens Advice](https://www.citizensadvice.org.uk/)."
)
WELLBEING_SIGNPOSTING = (
    "If you're struggling to cope, you don't have to deal with it alone. You can call "
    "Samaritans free, day or night, on 116 123. If you're in immediate danger, call 999."
)
ADVICE_NOTICE = (
    "NetValor explains options for learning purposes and can't recommend specific "
    "products or tell you what to do. For regulated advice, "
    "[MoneyHelper](https://www.moneyhelper.org.uk/) can help you find an "
    "FCA-authorised adviser."
)


def is_crisis_message(message: str) -> bool:
    """Check for money-crisis phrases that require free-advice signposting."""
    return bool(_CRISIS.search(message))


def is_wellbeing_risk(message: str) -> bool:
    """Check for distress phrases that require wellbeing signposting."""
    return bool(_WELLBEING.search(message))


def is_advice_request(message: str) -> bool:
    """Check for requests for a personal product recommendation."""
    return bool(_ADVICE.search(message))
