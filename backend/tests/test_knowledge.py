import re
import socket
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.knowledge import embeddings as embedding_module
from app.knowledge.embeddings import OllamaEmbeddings
from app.knowledge.lessons import chunk_lesson, load_lessons, parse_lesson
from app.knowledge.scripts.ingest import build_index
from app.retrieval import COLLECTION_NAME, LessonRetriever, RetrievalError


SAMPLE_LESSON = """---
id: sample
title: Sample lesson
module: Sample module
order: 1
difficulty: beginner
jurisdiction: UK
sources:
  - publisher: Example
    url: https://example.org/
licence: original-summary
review_status: draft
last_reviewed: 2026-10-20
---

## A useful heading

{body}
"""


class FakeCollection:
    metadata = {"embedding_model": "fake-model"}

    def __init__(self) -> None:
        self.query_args = None

    def query(self, **kwargs):
        self.query_args = kwargs
        return {
            "documents": [["## Budgeting\n\nPlan the money coming in."]],
            "metadatas": [[
                {
                    "lesson_id": "sample",
                    "title": "Sample lesson",
                    "module": "Sample module",
                    "heading": "Budgeting",
                    "source_urls": '["https://example.org/"]',
                    "review_status": "draft",
                    "embedding_model": "fake-model",
                }
            ]],
            "distances": [[0.2]],
        }


def test_ollama_embeddings_use_document_and_query_prefixes(monkeypatch) -> None:
    payloads = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"embeddings": [[0.1, 0.2] for _ in self._payload["input"]]}

    def fake_post(_url, *, json, timeout):
        payloads.append(json)
        return FakeResponse(json)

    monkeypatch.setattr(embedding_module.httpx, "post", fake_post)
    embeddings = OllamaEmbeddings(model="fake-model", base_url="http://ollama.test")

    assert embeddings.embed_documents("Cash flow", ["Track income."]) == [[0.1, 0.2]]
    assert embeddings.embed_query("How can I budget?") == [0.1, 0.2]
    assert payloads == [
        {
            "model": "fake-model",
            "input": ["title: Cash flow | text: Track income."],
        },
        {
            "model": "fake-model",
            "input": ["task: question answering | query: How can I budget?"],
        },
    ]


def test_parse_lesson_front_matter() -> None:
    lesson = parse_lesson(SAMPLE_LESSON.format(body="A short lesson body."))

    assert lesson.metadata["id"] == "sample"
    assert lesson.metadata["last_reviewed"] == date(2026, 10, 20)
    assert lesson.metadata["sources"] == [
        {"publisher": "Example", "url": "https://example.org/"}
    ]
    assert lesson.content.startswith("## A useful heading")


def test_parse_lesson_rejects_missing_required_field() -> None:
    with pytest.raises(ValueError, match="missing fields: title"):
        parse_lesson(SAMPLE_LESSON.replace("title: Sample lesson\n", ""))


def test_chunking_keeps_heading_and_overlaps_by_fifty_tokens() -> None:
    words = " ".join(f"word{index}" for index in range(700))
    lesson = parse_lesson(SAMPLE_LESSON.format(body=words))

    chunks = chunk_lesson(lesson)

    assert len(chunks) == 2
    assert all(chunk["text"].startswith("## A useful heading") for chunk in chunks)
    bodies = [chunk["text"].split("\n\n", 1)[1].split() for chunk in chunks]
    assert bodies[0][-50:] == bodies[1][:50]
    assert len(bodies[0]) <= 400
    assert len(bodies[1]) <= 450


def test_all_starter_lessons_have_required_fields_and_headings() -> None:
    lessons_dir = Path(__file__).parents[1] / "app" / "knowledge" / "lessons"
    lessons = load_lessons(lessons_dir)

    assert len(lessons) == 8
    assert [lesson.metadata["order"] for lesson in lessons] == list(range(1, 9))
    assert all(lesson.metadata["review_status"] == "draft" for lesson in lessons)
    assert all(lesson.content.count("\n## ") >= 2 for lesson in lessons)
    assert all(chunk_lesson(lesson) for lesson in lessons)
    for lesson in lessons:
        for chunk in chunk_lesson(lesson):
            assert chunk["text"].startswith("## ")
            token_count = len(
                re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]", chunk["text"])
            )
            assert 250 <= token_count <= 450


def test_retrieval_uses_fake_embeddings_and_decodes_metadata() -> None:
    collection = FakeCollection()
    retriever = LessonRetriever(collection, lambda _question: [0.1, 0.2], "fake-model")

    results = retriever.search("How do I budget?", k=3, module="Sample module")

    assert results[0]["score"] == pytest.approx(0.8)
    assert results[0]["metadata"]["source_urls"] == ["https://example.org/"]
    assert collection.query_args["query_embeddings"] == [[0.1, 0.2]]
    assert collection.query_args["where"] == {"module": "Sample module"}
    assert collection.query_args["n_results"] == 3


def test_retrieval_refuses_a_different_embedding_model() -> None:
    with pytest.raises(RetrievalError, match="Embedding model mismatch"):
        LessonRetriever(FakeCollection(), lambda _question: [0.1], "other-model")


def test_build_index_writes_model_and_lesson_metadata(tmp_path) -> None:
    chromadb = pytest.importorskip("chromadb")
    lessons_dir = Path(__file__).parents[1] / "app" / "knowledge" / "lessons"

    class FakeEmbeddings:
        model = "offline-test-model"

        def embed_documents(self, title: str, texts: list[str]) -> list[list[float]]:
            assert title
            return [[float(len(text)), 1.0] for text in texts]

    lessons, chunks = build_index(lessons_dir, tmp_path / "chroma", FakeEmbeddings())
    collection = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection(
        COLLECTION_NAME
    )

    assert lessons == 8
    assert chunks >= lessons
    assert collection.metadata["embedding_model"] == "offline-test-model"
    record = collection.get(where={"lesson_id": "how-credit-card-interest-works"})
    assert record["metadatas"][0]["review_status"] == "draft"
    assert record["metadatas"][0]["source_urls"]


def test_lessons_endpoints_list_and_return_full_content() -> None:
    with TestClient(main.app) as client:
        listed = client.get("/api/lessons")
        detail = client.get("/api/lessons/how-credit-card-interest-works")
        missing = client.get("/api/lessons/not-a-lesson")

    assert listed.status_code == 200
    assert len(listed.json()) == 8
    assert listed.json()[0] == {
        "id": "get-to-know-your-money",
        "title": "Get to know your money",
        "module": "Understand your cash flow",
        "order": 1,
    }
    assert detail.status_code == 200
    assert detail.json()["content"].startswith("## APR is a comparison measure")
    assert missing.status_code == 404


def test_search_endpoint_returns_chunks_and_metadata(monkeypatch) -> None:
    def fake_search(question: str, k: int = 4, module: str | None = None):
        assert (question, k, module) == ("How does interest work?", 2, "Build your debt strategy")
        return [
            {
                "chunk": "## APR\n\nInterest adds to the amount owed.",
                "score": 0.91,
                "metadata": {
                    "lesson_id": "how-credit-card-interest-works",
                    "title": "How credit card interest really works",
                    "module": "Build your debt strategy",
                    "heading": "APR",
                    "source_urls": ["https://example.org/"],
                    "review_status": "draft",
                    "embedding_model": "fake-model",
                },
            }
        ]

    monkeypatch.setattr(main, "search_lessons", fake_search)
    with TestClient(main.app) as client:
        response = client.get(
            "/api/search",
            params={
                "q": "How does interest work?",
                "k": 2,
                "module": "Build your debt strategy",
            },
        )

    assert response.status_code == 200
    assert response.json()["results"][0]["metadata"]["lesson_id"] == (
        "how-credit-card-interest-works"
    )


def test_search_endpoint_reports_an_unavailable_index(monkeypatch) -> None:
    def fail_search(*_args, **_kwargs):
        raise RetrievalError("Lesson index is unavailable.")

    monkeypatch.setattr(main, "search_lessons", fail_search)
    with TestClient(main.app) as client:
        response = client.get("/api/search", params={"q": "budget"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Lesson index is unavailable."


def test_credit_card_retrieval_with_running_ollama(tmp_path) -> None:
    ollama_url = OllamaEmbeddings().base_url
    parsed_url = urlparse(ollama_url)
    try:
        with socket.create_connection((parsed_url.hostname, parsed_url.port or 80), timeout=0.5):
            response = httpx.get(f"{ollama_url}/api/tags", timeout=1.0)
    except (OSError, httpx.HTTPError):
        pytest.skip("Ollama is not running.")
    if response.status_code != 200:
        pytest.skip("Ollama is not responding to its API.")
    installed_models = {
        model.get("name")
        for model in response.json().get("models", [])
        if isinstance(model, dict)
    }
    if OllamaEmbeddings().model not in installed_models:
        pytest.skip(f"Embedding model {OllamaEmbeddings().model} is not pulled.")

    embeddings = OllamaEmbeddings()
    lessons_dir = Path(__file__).parents[1] / "app" / "knowledge" / "lessons"
    build_index(lessons_dir, tmp_path / "chroma", embeddings)
    chromadb = pytest.importorskip("chromadb")
    collection = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection(
        COLLECTION_NAME
    )
    retriever = LessonRetriever(collection, embeddings.embed_query, embeddings.model)

    results = retriever.search("Why does my credit card balance go down so slowly?", k=4)

    assert any(
        result["metadata"]["lesson_id"] == "how-credit-card-interest-works"
        for result in results
    )
