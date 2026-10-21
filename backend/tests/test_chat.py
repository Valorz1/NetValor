import asyncio
import json
import os
import socket
from datetime import date
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.chat import ChatService, ConversationStore
from app.llm import LLMError, OllamaChatClient
from app.memory import Debt, Goal, JsonProfileStore, UserProfile


def _profile_store(path: Path) -> JsonProfileStore:
    store = JsonProfileStore(path)
    store.update_profile(
        "alex",
        UserProfile(
            name="Alex",
            monthly_take_home_income=Decimal("3600"),
            monthly_essentials=Decimal("2000"),
            savings=Decimal("3200"),
            debts=[
                Debt(
                    name="Credit card",
                    balance=Decimal("2450"),
                    apr=Decimal("0.249"),
                    monthly_payment=Decimal("350"),
                )
            ],
            goals=[
                Goal(
                    title="A fresh debt-free start",
                    type="debt_free",
                    target_amount=Decimal("2450"),
                    target_date=date(2027, 6, 1),
                ),
                Goal(
                    title="Build a safety net",
                    type="safety_net",
                    target_amount=Decimal("6000"),
                    target_date=date(2027, 6, 1),
                ),
            ],
        ),
    )
    store.add_note("alex", "I want to pay down my credit card steadily.")
    store.add_note("alex", "Planning a summer holiday.")
    return store


def _retrieval_result(
    status: str = "approved",
    lesson_id: str = "card-interest",
) -> dict:
    return {
        "chunk": "## Interest\n\nInterest can mean a card balance falls slowly.",
        "score": 0.9,
        "metadata": {
            "lesson_id": lesson_id,
            "title": "How credit card interest works",
            "module": "Build your debt strategy",
            "heading": "Interest",
            "source_urls": ["https://example.org/card"],
            "review_status": status,
            "embedding_model": "fake-model",
        },
    }


class FakeLLM:
    def __init__(self, tokens: list[str] | None = None, error: Exception | None = None):
        self.tokens = tokens if tokens is not None else ["This is supported [S1]."]
        self.error = error
        self.messages = None

    async def stream_chat(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        for token in self.tokens:
            yield token


def _service(
    tmp_path: Path,
    llm: FakeLLM,
    results: list[dict] | None = None,
    allow_draft: bool = False,
):
    calls = []

    def retrieve(question, *, k, review_status):
        calls.append((question, k, review_status))
        return results if results is not None else [_retrieval_result()]

    service = ChatService(
        profile_store=_profile_store(tmp_path / "profiles"),
        conversation_store=ConversationStore(tmp_path / "conversations"),
        retriever=retrieve,
        llm=llm,
        allow_draft=allow_draft,
    )
    return service, calls


async def _collect(service: ChatService, message: str, conversation_id=None):
    return [
        event
        async for event in service.stream("alex", message, conversation_id)
    ]


def test_prompt_contains_relevant_sources_goal_and_note(tmp_path) -> None:
    llm = FakeLLM()
    service, calls = _service(tmp_path, llm)

    events = asyncio.run(_collect(service, "Should I pay extra on my card?"))
    system_prompt = llm.messages[0]["content"]

    assert calls == [("Should I pay extra on my card?", 4, "approved")]
    assert "[S1] How credit card interest works — Interest" in system_prompt
    assert "https://example.org/card" in system_prompt
    assert "fresh debt-free start" in system_prompt
    assert "pay down my credit card steadily" in system_prompt
    assert "summer holiday" not in system_prompt
    assert events[-1]["event"] == "done"


def test_draft_sources_are_only_included_when_enabled(tmp_path) -> None:
    draft = [_retrieval_result(status="draft")]
    denied_llm = FakeLLM()
    denied, _ = _service(tmp_path / "denied", denied_llm, draft, allow_draft=False)
    approved_events = asyncio.run(_collect(denied, "Explain credit card interest"))
    assert "How credit card interest works" not in denied_llm.messages[0]["content"]
    assert approved_events[-1]["data"]["citations"] == []

    allowed_llm = FakeLLM()
    allowed, calls = _service(
        tmp_path / "allowed", allowed_llm, draft, allow_draft=True
    )
    asyncio.run(_collect(allowed, "Explain credit card interest"))
    assert calls[0][2] == ["approved", "draft"]
    assert "[S1] How credit card interest works" in allowed_llm.messages[0]["content"]


def test_debt_question_injects_verified_calculations(tmp_path) -> None:
    llm = FakeLLM()
    service, _ = _service(tmp_path, llm)

    events = asyncio.run(_collect(service, "Should I pay extra on my card?"))
    calculation_block = llm.messages[0]["content"].split(
        "Verified calculations", 1
    )[1]

    assert "2450" in calculation_block
    assert "0.249" in calculation_block
    assert "extra_amount_was_specified" in calculation_block
    assert events[-1]["data"]["calculations"][0]["debt_name"] == "Credit card"


def test_explicit_extra_amount_is_used_in_verified_comparison(tmp_path) -> None:
    llm = FakeLLM()
    service, _ = _service(tmp_path, llm)

    events = asyncio.run(_collect(service, "What if I pay GBP 50 extra on my card?"))

    assert events[-1]["data"]["calculations"][0]["extra_payment_scenario"] == "50"
    assert events[-1]["data"]["calculations"][0]["extra_amount_was_specified"] is True


def test_crisis_message_adds_code_owned_signposting(tmp_path) -> None:
    llm = FakeLLM(tokens=["I'm sorry this is difficult."])
    service, _ = _service(tmp_path, llm)

    events = asyncio.run(_collect(service, "I'm facing eviction and rent arrears."))
    tokens = "".join(event["data"]["text"] for event in events if event["event"] == "token")

    assert "MoneyHelper" in tokens
    assert "StepChange" in tokens
    assert "Citizens Advice" in tokens
    assert events[-1]["event"] == "done"


def test_invalid_citations_are_removed_and_logged(tmp_path, caplog) -> None:
    llm = FakeLLM(tokens=["Use [S1]", " and avoid [S9]."])
    service, _ = _service(tmp_path, llm)

    events = asyncio.run(_collect(service, "Explain interest"))
    answer = "".join(event["data"]["text"] for event in events if event["event"] == "token")

    assert answer == "Use [S1] and avoid ."
    assert "Removed invalid lesson citation [S9]" in caplog.text
    assert events[-1]["data"]["citations"] == [
        {
            "source_id": "S1",
            "lesson_title": "How credit card interest works",
            "url": "https://example.org/card",
        }
    ]


def test_llm_unavailable_yields_friendly_error_event(tmp_path) -> None:
    service, _ = _service(
        tmp_path, FakeLLM(error=LLMError("connection refused"))
    )

    events = asyncio.run(_collect(service, "Explain interest"))

    assert events == [
        {
            "event": "error",
            "data": {
                "message": (
                    "NetValor couldn't connect to its local AI or lesson search. "
                    "Check that Ollama is running and the lesson index is available."
                )
            },
        }
    ]


def test_conversation_saved_and_can_be_loaded_from_endpoints(tmp_path, monkeypatch) -> None:
    original_profiles = main.profile_store
    original_conversations = main.conversation_store
    original_llm = main.llm_client
    original_retriever = main.search_lessons
    profiles = _profile_store(tmp_path / "profiles")
    store = ConversationStore(tmp_path / "conversations")
    main.profile_store = profiles
    main.conversation_store = store
    main.llm_client = FakeLLM(tokens=["A short answer [S1]."])
    monkeypatch.setattr(main, "search_lessons", lambda *_args, **_kwargs: [_retrieval_result()])
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/chat/alex",
                json={"message": "Explain my card interest."},
            )
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            lines = [line for line in response.text.splitlines() if line.startswith("event:")]
            assert lines == ["event: token", "event: done"]
            done_event = json.loads(response.text.split("event: done\ndata: ", 1)[1].split("\n", 1)[0])
            conversation_id = done_event["conversation_id"]

            listed = client.get("/api/conversations/alex")
            detail = client.get(f"/api/conversations/alex/{conversation_id}")
        assert listed.status_code == 200
        assert listed.json()["conversations"][0]["conversation_id"] == conversation_id
        assert detail.status_code == 200
        assert [message["role"] for message in detail.json()["messages"]] == [
            "user",
            "assistant",
        ]
    finally:
        main.profile_store = original_profiles
        main.conversation_store = original_conversations
        main.llm_client = original_llm
        main.search_lessons = original_retriever


def test_ollama_chat_request_includes_streaming_settings(monkeypatch) -> None:
    requests = []

    class FakeResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield '{"message":{"content":"ready"},"done":false}'
            yield '{"message":{"content":""},"done":true}'

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, method, url, *, json):
            requests.append((method, url, json))
            return FakeResponse()

    monkeypatch.setattr("app.llm.httpx.AsyncClient", FakeClient)
    client = OllamaChatClient(
        base_url="http://ollama.test", model="fake-model", temperature=0.1, num_ctx=8192
    )
    tokens = asyncio.run(_consume_llm(client))

    assert tokens == ["ready"]
    request = requests[0][2]
    assert requests[0][:2] == ("POST", "http://ollama.test/api/chat")
    assert request["think"] is False
    assert request["stream"] is True
    assert request["options"] == {"temperature": 0.1, "num_ctx": 8192}


async def _consume_llm(client: OllamaChatClient):
    return [token async for token in client.stream_chat([{"role": "user", "content": "hi"}])]


def test_live_ollama_chat_optional() -> None:
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    parsed_url = urlparse(ollama_url)
    try:
        with socket.create_connection(
            (parsed_url.hostname, parsed_url.port or 80), timeout=0.5
        ):
            response = httpx.get(f"{ollama_url}/api/tags", timeout=1.0)
    except (OSError, httpx.HTTPError):
        pytest.skip("Ollama is not running.")
    if response.status_code != 200:
        pytest.skip("Ollama is not responding to its API.")
    model = OllamaChatClient().model
    models = {entry.get("name") for entry in response.json().get("models", [])}
    if model not in models:
        pytest.skip(f"Ollama model {model} is not pulled.")
    client = OllamaChatClient(timeout=60)
    tokens = asyncio.run(
        _consume_llm(
            client,
        )
    )
    assert tokens
