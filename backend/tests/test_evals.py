"""Run the offline co-pilot evaluation suite as part of the standard tests."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evals.harness import LexicalRetriever, load_cases, run_case, summarise

CASES = load_cases()
CATEGORIES = {
    "retrieval_relevance",
    "citation_grounding",
    "unsupported_claims",
    "ambiguous_question",
    "financial_calculation",
    "missing_evidence",
    "sensitive_circumstances",
    "prompt_injection",
    "uk_terminology",
    "privacy",
}


@pytest.fixture(scope="module")
def retriever() -> LexicalRetriever:
    return LexicalRetriever()


def test_every_required_category_is_covered() -> None:
    assert {case["category"] for case in CASES} == CATEGORIES


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_offline_evaluation_case(case: dict[str, Any], retriever: LexicalRetriever) -> None:
    result = asyncio.run(run_case(case, retriever=retriever))

    assert result.checks, "each case must check something"
    failures = [f"{check.name}: {check.detail}" for check in result.checks if not check.passed]
    assert not failures, failures


def test_summary_counts(retriever: LexicalRetriever) -> None:
    results = [asyncio.run(run_case(case, retriever=retriever)) for case in CASES[:3]]

    summary = summarise(results)

    assert summary["cases"] == 3
    assert sum(bucket["cases"] for bucket in summary["categories"].values()) == 3
