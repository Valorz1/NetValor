"""Ollama embedding client with model-neutral document and query methods.

`embed_documents` is synchronous because it is only used by the offline
ingestion script. `aembed_query` is asynchronous so that request handlers
never block the event loop while waiting for Ollama.
"""

from __future__ import annotations

from typing import Any

import httpx


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot produce a valid embedding response."""


def _document_inputs(title: str, texts: list[str]) -> list[str]:
    return [f"title: {title} | text: {text}" for text in texts]


def _query_input(question: str) -> str:
    return f"task: question answering | query: {question}"


def _vectors(payload: Any, expected: int) -> list[list[float]]:
    vectors = payload.get("embeddings") if isinstance(payload, dict) else None
    if (
        not isinstance(vectors, list)
        or len(vectors) != expected
        or any(not isinstance(vector, list) or not vector for vector in vectors)
    ):
        raise EmbeddingError("Ollama returned an invalid embeddings response.")
    try:
        return [[float(value) for value in vector] for vector in vectors]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("Ollama returned a non-numeric embedding.") from exc


class OllamaEmbeddings:
    """Create prefixed embeddings through Ollama's HTTP API."""

    def __init__(
        self,
        model: str = "embeddinggemma-2:270m",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = http_client

    def embed_documents(self, title: str, texts: list[str]) -> list[list[float]]:
        """Embed lesson chunks using the model's document instruction prefix."""
        if not texts:
            return []
        inputs = _document_inputs(title, texts)
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs},
                timeout=max(self.timeout, 120.0),
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(f"Could not get embeddings from Ollama: {exc}") from exc
        return _vectors(payload, len(inputs))

    async def aembed_query(self, question: str) -> list[float]:
        """Embed a search question using the model's query instruction prefix."""
        body = {"model": self.model, "input": [_query_input(question)]}
        owned = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            response = await client.post(
                f"{self.base_url}/api/embed", json=body, timeout=self.timeout
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(
                f"Could not get embeddings from Ollama: {type(exc).__name__}"
            ) from exc
        finally:
            if owned:
                await client.aclose()
        return _vectors(payload, 1)[0]
