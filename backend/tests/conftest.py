"""Shared fixtures. Nothing here needs Ollama, Chroma or network access."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import security
from app.config import Settings
from app.main import create_app

PASSWORD = "correct horse battery staple"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore NetValor settings exported in the developer's shell."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture(autouse=True)
def fast_password_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a cheaper scrypt cost in tests. Hashes record their own parameters."""
    monkeypatch.setattr(security, "_SCRYPT_N", 2**10)


class FakeLLM:
    """Scripted streaming model that records the messages it receives."""

    def __init__(
        self,
        tokens: list[str] | None = None,
        error: Exception | None = None,
        error_after: int | None = None,
    ) -> None:
        self.tokens = tokens if tokens is not None else ["Interest adds to what you owe [S1]."]
        self.error = error
        self.error_after = error_after
        self.messages: list[dict[str, str]] | None = None
        self.calls = 0

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.calls += 1
        self.messages = messages
        if self.error is not None and self.error_after is None:
            raise self.error
        for index, token in enumerate(self.tokens):
            if self.error is not None and index == self.error_after:
                raise self.error
            yield token


DEFAULT_CHUNK = "## Interest\n\nInterest adds to what you owe when a card balance is carried over."


def lesson_result(
    chunk: str = DEFAULT_CHUNK,
    *,
    status: str = "approved",
    score: float = 0.8,
    chunk_id: str = "how-credit-card-interest-works:0",
    title: str = "How credit card interest works",
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "chunk": chunk,
        "score": score,
        "metadata": {
            "lesson_id": chunk_id.split(":")[0],
            "title": title,
            "module": "Build your debt strategy",
            "heading": "Interest",
            "source_urls": ["https://www.moneyhelper.org.uk/card"],
            "review_status": status,
            "embedding_model": "fake-model",
        },
    }


class FakeRetriever:
    """Async retriever returning fixed results and recording its calls."""

    def __init__(self, results: list[dict[str, Any]] | None = None, error: Exception | None = None):
        self.results = [lesson_result()] if results is None else results
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        question: str,
        *,
        k: int = 4,
        module: str | None = None,
        review_status: str | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {"question": question, "k": k, "module": module, "review_status": review_status}
        )
        if self.error is not None:
            raise self.error
        return self.results


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "database_path": tmp_path / "netvalor-test.db",
        "auth_rate_limit_per_minute": 1000,
        "chat_rate_limit_per_minute": 1000,
        "search_rate_limit_per_minute": 1000,
        "log_level": "WARNING",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    return FakeRetriever()


ClientFactory = Callable[..., Any]


@pytest.fixture
def make_client(tmp_path: Path) -> Iterator[ClientFactory]:
    """Build a TestClient; extra kwargs override settings."""
    stack: list[TestClient] = []

    def factory(llm: Any = None, retriever: Any = None, **setting_overrides: Any) -> TestClient:
        app = create_app(
            make_settings(tmp_path, **setting_overrides),
            llm=llm or FakeLLM(),
            retriever=retriever or FakeRetriever(),
        )
        client = TestClient(app)
        client.__enter__()
        stack.append(client)
        return client

    yield factory
    for client in reversed(stack):
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client: ClientFactory) -> TestClient:
    client: TestClient = make_client()
    return client


def register_and_login(
    client: TestClient, user_id: str = "sam", password: str = PASSWORD
) -> dict[str, str]:
    """Create an account and return Authorization headers for it."""
    created = client.post("/api/auth/register", json={"user_id": user_id, "password": password})
    assert created.status_code == 201, created.text
    response = client.post("/api/auth/login", json={"user_id": user_id, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


SAMPLE_PROFILE: dict[str, Any] = {
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
        },
        {
            "title": "Build a safety net",
            "type": "safety_net",
            "target_amount": "6000",
            "target_date": "2027-06-01",
        },
    ],
}
