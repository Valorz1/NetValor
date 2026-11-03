"""Evaluation harness for the NetValor co-pilot pipeline.

Offline mode is deterministic and needs no services: retrieval uses TF-IDF
over the real lesson chunks and the model's answer is scripted per case. It
evaluates NetValor's own logic (what is retrieved, which numbers are
calculated, when to clarify, which notices and grounding flags appear, and
what reaches the model).

Live mode uses the configured Ollama models and Chroma index, so it also
measures real model behaviour. Model output varies between runs, so live
results are a signal to review, not a pass/fail guarantee.
"""

from __future__ import annotations

import math
import re
import tempfile
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

import yaml

from app.chat import ChatOptions, ChatService
from app.conversations import ConversationStore
from app.db import Database
from app.knowledge.lessons import chunk_lesson, load_lessons
from app.memory import ProfileStore, UserProfile
from app.users import UserStore

CASES_PATH = Path(__file__).with_name("cases.yaml")
LESSONS_DIR = Path(__file__).resolve().parents[1] / "app" / "knowledge" / "lessons"
TOP_K = 4
EVAL_PROFILE: dict[str, Any] = {
    "name": "Sam",
    "monthly_take_home_income": "2800",
    "monthly_essentials": "1700",
    "savings": "500",
    "debts": [{"name": "Credit card", "balance": "2450", "apr": "0.249", "monthly_payment": "350"}],
    "goals": [
        {
            "title": "A fresh debt-free start",
            "type": "debt_free",
            "target_amount": "2450",
            "target_date": "2027-06-01",
        }
    ],
}
_WORD = re.compile(r"[a-z][a-z'-]+")
_STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "is",
        "are",
        "be",
        "it",
        "you",
        "your",
        "my",
        "i",
        "me",
        "what",
        "how",
        "why",
        "when",
        "do",
        "does",
        "can",
        "should",
        "will",
        "would",
        "with",
        "if",
        "this",
        "that",
        "at",
        "as",
        "by",
        "from",
        "so",
        "like",
        "each",
        "there",
        "their",
        "they",
    ]
)


def _tokens(text: str) -> list[str]:
    return [word.rstrip("s") for word in _WORD.findall(text.lower()) if word not in _STOP]


class LexicalRetriever:
    """Deterministic TF-IDF cosine retrieval over the real lesson chunks."""

    def __init__(self, lessons_dir: Path = LESSONS_DIR) -> None:
        self.documents: list[dict[str, Any]] = []
        for lesson in load_lessons(lessons_dir):
            meta = lesson.metadata
            for index, chunk in enumerate(chunk_lesson(lesson)):
                self.documents.append(
                    {
                        "chunk_id": f"{meta['id']}:{index}",
                        "chunk": chunk["text"],
                        "metadata": {
                            "lesson_id": meta["id"],
                            "title": meta["title"],
                            "module": meta["module"],
                            "heading": chunk["heading"],
                            "source_urls": [source["url"] for source in meta["sources"]],
                            "review_status": meta["review_status"],
                            "embedding_model": "tf-idf",
                        },
                        "terms": Counter(
                            _tokens(f"{meta['title']} {meta['title']} {chunk['text']}")
                        ),
                    }
                )
        frequency: Counter[str] = Counter()
        for document in self.documents:
            frequency.update(document["terms"].keys())
        total = len(self.documents)
        self.idf = {
            term: math.log((1 + total) / (1 + count)) + 1 for term, count in frequency.items()
        }
        for document in self.documents:
            document["vector"] = self._vector(document["terms"])

    def _vector(self, terms: Counter[str]) -> dict[str, float]:
        vector = {term: count * self.idf.get(term, 0.0) for term, count in terms.items()}
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {term: value / norm for term, value in vector.items()}

    async def __call__(
        self,
        question: str,
        *,
        k: int = 4,
        module: str | None = None,
        review_status: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query = self._vector(Counter(_tokens(question)))
        scored = []
        for document in self.documents:
            score = sum(
                weight * document["vector"].get(term, 0.0) for term, weight in query.items()
            )
            scored.append((score, document))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "chunk_id": document["chunk_id"],
                "chunk": document["chunk"],
                "score": round(score, 4),
                "metadata": dict(document["metadata"]),
            }
            for score, document in scored[:k]
        ]


class RecordingRetriever:
    """Wrap a retriever to record results and simulate injection or no evidence."""

    def __init__(self, inner: Any, case: dict[str, Any]) -> None:
        self.inner = inner
        self.case = case
        self.results: list[dict[str, Any]] = []
        self.called = False

    async def __call__(self, question: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.called = True
        if self.case.get("retrieval") == "none":
            return []
        results = list(await self.inner(question, **kwargs))
        self.results = sorted(results, key=lambda item: item["score"], reverse=True)
        if self.case.get("inject_chunk"):
            injected = dict(self.results[0]) if self.results else {"metadata": {}}
            injected.update(
                {"chunk_id": "injected:0", "chunk": self.case["inject_chunk"], "score": 0.99}
            )
            results.insert(0, injected)
        return results


class RecordingLLM:
    """Scripted (offline) or real (live) model that records its input."""

    def __init__(self, answer: str | None = None, inner: Any = None) -> None:
        self.answer = answer
        self.inner = inner
        self.messages: list[dict[str, str]] | None = None

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.messages = messages
        if self.inner is not None:
            async for token in self.inner.stream_chat(messages):
                yield token
            return
        words = (self.answer or "I can only explain this in general terms.").split(" ")
        for index, word in enumerate(words):
            yield word if index == 0 else " " + word


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class CaseResult:
    case_id: str
    category: str
    checks: list[Check] = field(default_factory=list)
    answer: str = ""
    done: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case IDs must be unique.")
    return cases


async def run_case(
    case: dict[str, Any],
    *,
    retriever: Any,
    live_llm: Any = None,
    min_score: float = 0.05,
) -> CaseResult:
    """Run one case through the real ChatService and check expectations."""
    live = live_llm is not None
    with tempfile.TemporaryDirectory() as directory:
        database = Database(Path(directory) / "eval.db")
        database.initialise()
        UserStore(database, timedelta(hours=1)).create_user("sam", None)
        profiles = ProfileStore(database)
        if case.get("profile") != "none":
            profiles.put_profile("sam", UserProfile.model_validate(EVAL_PROFILE))
        profiles.add_note("sam", "I want to pay down my credit card steadily.")
        if case.get("note"):
            profiles.add_note("sam", case["note"])
        recording_retriever = RecordingRetriever(retriever, case)
        llm = RecordingLLM(case.get("scripted_answer"), inner=live_llm)
        service = ChatService(
            profiles,
            ConversationStore(database),
            recording_retriever,
            llm,
            ChatOptions(
                allowed_statuses=frozenset({"approved", "draft"}), k=TOP_K, min_score=min_score
            ),
        )
        events = [event async for event in service.stream("sam", case["message"])]

    result = CaseResult(case["id"], case["category"])
    result.answer = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    done = events[-1]["data"] if events and events[-1]["event"] == "done" else None
    result.done = done
    if done is None:
        result.checks.append(Check("completed", False, f"stream ended with {events[-1:]}"))
        return result
    expect: dict[str, Any] = case.get("expect", {})
    skip_scripted = live and case.get("offline_only")
    system_prompt = llm.messages[0]["content"] if llm.messages else ""
    full_prompt = "\n".join(message["content"] for message in llm.messages or [])

    def add(name: str, passed: bool, detail: str = "") -> None:
        result.checks.append(Check(name, passed, detail))

    if "lessons_in_top_k" in expect:
        found = [item["metadata"]["lesson_id"] for item in recording_retriever.results[:TOP_K]]
        missing = [lesson for lesson in expect["lessons_in_top_k"] if lesson not in found]
        add("retrieval_relevance", not missing, f"top {TOP_K}: {found}")
    if "llm_called" in expect:
        add("llm_called", (llm.messages is not None) == expect["llm_called"])
    if "clarification" in expect:
        add("clarification", done["clarification_needed"] == expect["clarification"])
    if "calculation" in expect:
        calculations = done["calculations"]
        wanted = expect["calculation"]
        ok = bool(calculations) and calculations[0]["type"] == wanted["type"]
        if ok:
            inputs = calculations[0]["inputs"]
            ok = all(inputs.get(key) == value for key, value in wanted.get("inputs", {}).items())
        add("calculation", ok, str(calculations[0]["inputs"] if calculations else None))
    for code in expect.get("notices", []):
        add(f"notice:{code}", code in done["notices"], str(done["notices"]))
    for code in expect.get("no_notices", []):
        add(f"no_notice:{code}", code not in done["notices"])
    for text in expect.get("text_contains", []):
        add(f"text_contains:{text[:30]}", text in result.answer)
    for text in expect.get("prompt_excludes", []):
        add(f"prompt_excludes:{text[:30]}", text not in full_prompt)
    if "prompt_in_user_turn_only" in expect:
        text = expect["prompt_in_user_turn_only"]
        user_turn = llm.messages[-1]["content"] if llm.messages else ""
        add("injection_confined_to_user_turn", text in user_turn and text not in system_prompt)
    grounding = done.get("grounding") or {}
    if not skip_scripted:
        if "unverified_figures" in expect and not live:
            add(
                "unverified_figures",
                grounding.get("unverified_figures", []) == expect["unverified_figures"],
                str(grounding.get("unverified_figures")),
            )
        if "invalid_citations" in expect and not live:
            add(
                "invalid_citations",
                grounding.get("invalid_citations") == expect["invalid_citations"],
            )
        if "weak_citations" in expect and not live:
            add("weak_citations", grounding.get("weak_citations") == expect["weak_citations"])
    return result


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    """Aggregate pass rates per category and pipeline-wide grounding signals."""
    categories: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = categories.setdefault(result.category, {"cases": 0, "passed": 0})
        bucket["cases"] += 1
        bucket["passed"] += int(result.passed)
    answered = [r for r in results if r.done and r.done.get("grounding")]
    return {
        "cases": len(results),
        "passed": sum(result.passed for result in results),
        "categories": categories,
        "answers_with_invalid_citations": sum(
            bool(r.done and r.done["grounding"]["invalid_citations"]) for r in answered
        ),
        "answers_with_unverified_figures": sum(
            bool(r.done and r.done["grounding"]["unverified_figures"]) for r in answered
        ),
        "answers_with_weak_citations": sum(
            bool(r.done and r.done["grounding"]["weak_citations"]) for r in answered
        ),
        "answers_checked": len(answered),
    }
