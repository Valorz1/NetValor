"""Source-grounded co-pilot response pipeline.

For each user message the service:

1. loads the user's profile, chosen memory notes and conversation history;
2. interprets any calculation request deterministically (`calc_intent`),
   asking for clarification instead of guessing when it is ambiguous;
3. retrieves reviewed lesson passages, dropping low-relevance, draft (unless
   allowed), duplicate and instruction-like chunks;
4. answers without the language model when there is no evidence to ground
   an answer;
5. streams the model's answer, removing citations to unknown sources;
6. checks the finished answer for unverified figures and weak citations and
   appends code-owned notices (draft content, signposting, advice limits);
7. stores the turn atomically.

Events are dicts: ``{"event": "token"|"done"|"error", "data": {...}}``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import anyio

from app.calc_intent import is_personal_question, plan_calculations
from app.conversations import (
    ConversationFullError,
    ConversationNotFoundError,
    ConversationStore,
    MessageStatus,
)
from app.grounding import (
    ADVICE_NOTICE,
    DEBT_SIGNPOSTING,
    WELLBEING_SIGNPOSTING,
    CitationFilter,
    Source,
    allowed_figures,
    check_grounding,
    escape_untrusted,
    format_sources,
    is_advice_request,
    is_crisis_message,
    is_wellbeing_risk,
    select_evidence,
)
from app.knowledge.embeddings import EmbeddingError
from app.llm import LLMError
from app.memory import MemoryNote, ProfileStore, UserProfile
from app.retrieval import RetrievalError, Retriever
from app.security import DecryptionError

logger = logging.getLogger(__name__)
_SYSTEM_PROMPT = (Path(__file__).resolve().parent / "prompts" / "system.md").read_text(
    encoding="utf-8"
)
_TOKEN_PATTERN = re.compile(r"\b[\w'-]{3,}\b")
_TOPIC_GROUPS = {
    "debt": {"debt", "debts", "card", "credit", "loan", "repayment", "pay", "extra", "owe"},
    "savings": {"saving", "savings", "save", "safety", "buffer", "emergency", "cushion"},
}
_GOAL_TOPICS = {"debt_free": "debt", "safety_net": "savings"}
_MAX_HISTORY_MESSAGE_CHARS = 1500

NO_EVIDENCE_ANSWER = (
    "I couldn't find anything in NetValor's lessons that covers this, so I'd rather not "
    "guess. You could try asking in a different way, or look at "
    "[MoneyHelper](https://www.moneyhelper.org.uk/), which offers free, impartial "
    "guidance backed by the government."
)
EMPTY_ANSWER = "Sorry, I wasn't able to put together an answer to that. Could you try rephrasing?"
DRAFT_NOTICE = (
    "Note: some of the lesson material used here is still a draft and hasn't been fact-checked yet."
)
TRUNCATED_NOTICE = "(This answer was cut short because it was getting very long.)"

ERROR_MESSAGES = {
    "llm_unavailable": (
        "NetValor couldn't connect to its local AI or lesson search. "
        "Check that Ollama is running and the lesson index is available."
    ),
    "llm_interrupted": (
        "NetValor's local AI stopped responding part-way through this answer, so it may be "
        "incomplete. Please try again."
    ),
    "retrieval_unavailable": (
        "NetValor couldn't search its lessons right now. Check that Ollama is running and "
        "the lesson index has been built."
    ),
    "conversation_not_found": "That conversation wasn't found.",
    "conversation_full": "This conversation is full. Please start a new conversation.",
    "storage_unavailable": "NetValor couldn't load your saved data. Please try again shortly.",
}


class ChatLLM(Protocol):
    """The async streaming interface required by the response pipeline."""

    def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]: ...


@dataclass(frozen=True)
class ChatOptions:
    """Tunable behaviour for the response pipeline."""

    allowed_statuses: frozenset[str] = frozenset({"approved"})
    k: int = 4
    min_score: float = 0.35
    max_history_messages: int = 6
    max_history_chars: int = 6000
    max_answer_chars: int = 6000


def _event(name: str, **data: Any) -> dict[str, Any]:
    return {"event": name, "data": data}


def _error(code: str, conversation_id: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {"code": code, "message": ERROR_MESSAGES[code]}
    if conversation_id:
        data["conversation_id"] = conversation_id
    return {"event": "error", "data": data}


def _topic_tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _shares_topic(text_tokens: set[str], question_tokens: set[str]) -> bool:
    if text_tokens & question_tokens:
        return True
    return any(question_tokens & group and text_tokens & group for group in _TOPIC_GROUPS.values())


def relevant_context(
    profile: UserProfile | None, notes: list[MemoryNote], question: str
) -> dict[str, Any]:
    """Select only the saved goals and notes that relate to a personal question.

    General questions ("How does APR work?") get no personal context at all.
    """
    if not is_personal_question(question):
        return {"goals": [], "notes": []}
    question_tokens = _topic_tokens(question)
    goals = []
    for goal in profile.goals if profile else []:
        topic = _TOPIC_GROUPS[_GOAL_TOPICS[goal.type]]
        if _shares_topic(_topic_tokens(goal.title), question_tokens) or question_tokens & topic:
            goals.append(goal.model_dump(mode="json"))
    relevant_notes = [
        note.content
        for note in notes
        if _shares_topic(_topic_tokens(note.content), question_tokens)
    ]
    return {"goals": goals, "notes": relevant_notes}


def retrieval_query(message: str, history: list[dict[str, Any]]) -> str:
    """Expand short follow-ups ("what about £100?") with the previous question."""
    if len(_topic_tokens(message)) >= 6:
        return message
    previous = next((item["content"] for item in reversed(history) if item["role"] == "user"), None)
    return f"{previous}\n{message}" if previous else message


def trim_history(
    history: list[dict[str, Any]], max_messages: int, max_chars: int
) -> list[dict[str, str]]:
    """Keep recent complete turns within message and character budgets."""
    # Drop interrupted answers together with the question they answered, so
    # the remaining history still alternates user/assistant.
    usable: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        following = history[index + 1] if index + 1 < len(history) else None
        if item.get("status") == "incomplete":
            continue
        if item["role"] == "user" and following and following.get("status") == "incomplete":
            continue
        usable.append(item)
    selected: list[dict[str, str]] = []
    total = 0
    for item in reversed(usable[-max_messages:] if max_messages else []):
        content = item["content"][:_MAX_HISTORY_MESSAGE_CHARS]
        if total + len(content) > max_chars:
            break
        total += len(content)
        selected.append({"role": item["role"], "content": content})
    selected.reverse()
    # Chat templates expect the history to start with a user turn.
    while selected and selected[0]["role"] != "user":
        selected.pop(0)
    return selected


def build_messages(
    question: str,
    context: dict[str, Any],
    sources: list[Source],
    calculations: list[dict[str, Any]],
    history: list[dict[str, str]],
    crisis: bool,
) -> list[dict[str, str]]:
    """Assemble the model input: instructions, delimited data, history, question."""
    saved = escape_untrusted(json.dumps(context, ensure_ascii=False))
    calc_json = json.dumps(calculations, ensure_ascii=False, default=str)
    system = (
        f"{_SYSTEM_PROMPT}\n"
        f"<saved_context>\n{saved}\n</saved_context>\n\n"
        f"{format_sources(sources)}\n\n"
        f"<verified_calculations>\n{calc_json if calculations else 'None.'}\n"
        "</verified_calculations>"
    )
    if crisis:
        system += (
            "\n\nThe user's message suggests an urgent financial or personal difficulty. "
            "Respond with care. NetValor will add contact details for free support services "
            "after your answer, so don't list phone numbers yourself."
        )
    return [{"role": "system", "content": system}, *history, {"role": "user", "content": question}]


class ChatService:
    """Assemble relevant evidence and stream a safe, cited response."""

    def __init__(
        self,
        profile_store: ProfileStore,
        conversation_store: ConversationStore,
        retriever: Retriever,
        llm: ChatLLM,
        options: ChatOptions | None = None,
    ) -> None:
        self.profiles = profile_store
        self.conversations = conversation_store
        self.retriever = retriever
        self.llm = llm
        self.options = options or ChatOptions()

    async def _persist(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        answer: str,
        status: MessageStatus,
    ) -> None:
        await anyio.to_thread.run_sync(
            self.conversations.append_turn, user_id, conversation_id, message, answer, status
        )

    async def stream(
        self,
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        use_saved_context: bool = True,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Yield token, done or error events for one user message.

        With `use_saved_context=False` the user's profile, goals and memory
        notes are neither read nor shared with the language model.
        """
        conv_id = conversation_id or self.conversations.create_id()
        try:
            profile: UserProfile | None = None
            notes: list[MemoryNote] = []
            if use_saved_context:
                profile = await anyio.to_thread.run_sync(self.profiles.find_profile, user_id)
                notes = await anyio.to_thread.run_sync(self.profiles.get_notes, user_id)
            history: list[dict[str, Any]] = []
            if conversation_id is not None:
                conversation = await anyio.to_thread.run_sync(
                    self.conversations.get, user_id, conversation_id
                )
                history = conversation["messages"]
        except ConversationNotFoundError:
            yield _error("conversation_not_found")
            return
        except (sqlite3.Error, DecryptionError, ValueError):
            logger.exception("Could not load chat state.")
            yield _error("storage_unavailable")
            return

        crisis = is_crisis_message(message)
        wellbeing = is_wellbeing_risk(message)
        closing_notes: list[tuple[str, str]] = []
        if is_advice_request(message):
            closing_notes.append(("advice_limits", ADVICE_NOTICE))
        if crisis:
            closing_notes.append(("debt_signposting", DEBT_SIGNPOSTING))
        if wellbeing:
            closing_notes.append(("wellbeing_signposting", WELLBEING_SIGNPOSTING))

        # 1. Calculations: deterministic, and never guessed.
        clarified = None
        if (
            history
            and history[-1]["role"] == "assistant"
            and history[-1]["status"] == "clarification"
        ):
            clarified = next(
                (item["content"] for item in reversed(history) if item["role"] == "user"), None
            )
        try:
            plan = plan_calculations(message, profile, clarified)
        except ValueError:
            logger.exception("Calculation planning failed.")
            plan = None
        if plan is None or plan.status == "clarify":
            text = (
                plan.clarification
                if plan and plan.clarification
                else "I couldn't run that calculation. Could you check the figures?"
            )
            async for event in self._finish_without_llm(
                user_id,
                conv_id,
                message,
                text,
                closing_notes,
                "clarification",
                calculations=[],
                clarification_needed=True,
            ):
                yield event
            return

        # 2. Retrieval.
        options = self.options
        try:
            raw_results = await self.retriever(
                retrieval_query(message, history),
                k=options.k + 2,
                review_status=sorted(options.allowed_statuses),
            )
        except (RetrievalError, EmbeddingError, ValueError) as exc:
            logger.warning("Lesson retrieval failed: %s", exc)
            yield _error("retrieval_unavailable")
            return
        sources, dropped = select_evidence(
            raw_results,
            allowed_statuses=options.allowed_statuses,
            min_score=options.min_score,
            k=options.k,
        )
        if any(dropped.values()):
            logger.info("Evidence filtered: %s", dropped)

        # 3. No evidence and nothing calculated: don't let the model improvise.
        if not sources and not plan.calculations:
            async for event in self._finish_without_llm(
                user_id,
                conv_id,
                message,
                NO_EVIDENCE_ANSWER,
                closing_notes,
                "complete",
                calculations=[],
                clarification_needed=False,
                evidence="none",
            ):
                yield event
            return

        # 4. Stream the model's answer.
        context = relevant_context(profile, notes, message)
        llm_messages = build_messages(
            message,
            context,
            sources,
            plan.calculations,
            trim_history(history, options.max_history_messages, options.max_history_chars),
            crisis or wellbeing,
        )
        citations = CitationFilter({source.source_id for source in sources})
        parts: list[str] = []
        length = 0
        truncated = False
        persisted = False
        try:
            async for token in self.llm.stream_chat(llm_messages):
                text = citations.feed(token)
                if text:
                    parts.append(text)
                    length += len(text)
                    yield _event("token", text=text)
                if length >= options.max_answer_chars:
                    truncated = True
                    break
            tail = citations.flush()
            if tail:
                parts.append(tail)
                yield _event("token", text=tail)
        except LLMError as exc:
            logger.warning("Language model failed: %s", exc)
            partial = "".join(parts).strip()
            if partial:
                try:
                    await self._persist(user_id, conv_id, message, partial, "incomplete")
                    persisted = True
                except (sqlite3.Error, ConversationFullError):
                    logger.exception("Could not save a partial answer.")
                yield _error("llm_interrupted", conv_id if persisted else None)
            else:
                yield _error("llm_unavailable")
            return
        except (asyncio.CancelledError, GeneratorExit):
            # The client disconnected. Keep what they saw, marked incomplete.
            partial = "".join(parts).strip()
            if partial:
                try:
                    self.conversations.append_turn(user_id, conv_id, message, partial, "incomplete")
                except Exception:
                    logger.exception("Could not save an interrupted answer.")
            raise

        answer = "".join(parts).strip()
        if not answer:
            answer = EMPTY_ANSWER
            yield _event("token", text=answer)

        # 5. Post-generation checks and code-owned notices.
        evidence_texts = [message, json.dumps(context, ensure_ascii=False)]
        evidence_texts += [source.chunk for source in sources]
        evidence_texts += [item["content"] for item in history if item["role"] == "user"]
        report = check_grounding(
            answer, sources, allowed_figures(evidence_texts, plan.calculations), citations.invalid
        )
        notices: list[tuple[str, str]] = []
        if truncated:
            notices.append(("truncated", TRUNCATED_NOTICE))
        if report.unverified_figures:
            notices.append(
                (
                    "unverified_figures",
                    "Note: this answer mentions figures ("
                    + ", ".join(report.unverified_figures)
                    + ") that NetValor couldn't match to its calculations or lessons, "
                    "so please treat them with caution.",
                )
            )
        used = [source for source in sources if source.source_id in citations.cited]
        if any(source.review_status != "approved" for source in used or sources):
            notices.append(("draft_content", DRAFT_NOTICE))
        notices += closing_notes
        for _, text in notices:
            addition = "\n\n" + text
            answer += addition
            yield _event("token", text=addition)

        try:
            await self._persist(user_id, conv_id, message, answer, "complete")
        except ConversationFullError:
            yield _error("conversation_full")
            return
        except sqlite3.Error:
            logger.exception("Could not save the conversation.")
            yield _error("storage_unavailable")
            return
        yield _event(
            "done",
            conversation_id=conv_id,
            citations=[
                {
                    "source_id": source.source_id,
                    "lesson_id": source.lesson_id,
                    "lesson_title": source.lesson_title,
                    "heading": source.heading,
                    "review_status": source.review_status,
                    "url": url,
                }
                for source in used
                for url in source.urls
            ],
            calculations=plan.calculations,
            clarification_needed=False,
            evidence="sources" if sources else "calculations_only",
            grounding=report.as_dict(),
            notices=[code for code, _ in notices],
        )

    async def _finish_without_llm(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        text: str,
        closing_notes: list[tuple[str, str]],
        status: MessageStatus,
        *,
        calculations: list[dict[str, Any]],
        clarification_needed: bool,
        evidence: str = "none",
    ) -> AsyncGenerator[dict[str, Any]]:
        """Emit a code-owned answer (clarification or no-evidence) and save it."""
        answer = text + "".join("\n\n" + note for _, note in closing_notes)
        yield _event("token", text=answer)
        try:
            await self._persist(user_id, conversation_id, message, answer, status)
        except ConversationFullError:
            yield _error("conversation_full")
            return
        except sqlite3.Error:
            logger.exception("Could not save the conversation.")
            yield _error("storage_unavailable")
            return
        yield _event(
            "done",
            conversation_id=conversation_id,
            citations=[],
            calculations=calculations,
            clarification_needed=clarification_needed,
            evidence=evidence,
            grounding=None,
            notices=[code for code, _ in closing_notes],
        )
