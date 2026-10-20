"""Ollama embedding client with model-neutral document and query methods."""

import os
from typing import Any

import httpx


class EmbeddingError(RuntimeError):
    """Raised when Ollama cannot produce a valid embedding response."""


class OllamaEmbeddings:
    """Create prefixed embeddings through Ollama's HTTP API."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.getenv("EMBEDDING_MODEL", "embeddinggemma-2:270m")
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def embed_documents(self, title: str, texts: list[str]) -> list[list[float]]:
        """Embed lesson chunks using the model's document instruction prefix."""
        inputs = [f"title: {title} | text: {text}" for text in texts]
        return self._embed(inputs)

    def embed_query(self, question: str) -> list[float]:
        """Embed a search question using the model's query instruction prefix."""
        return self._embed([f"task: question answering | query: {question}"])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        if not inputs:
            return []
        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise EmbeddingError(f"Could not get embeddings from Ollama: {exc}") from exc

        vectors = payload.get("embeddings") if isinstance(payload, dict) else None
        if (
            not isinstance(vectors, list)
            or len(vectors) != len(inputs)
            or any(not isinstance(vector, list) or not vector for vector in vectors)
        ):
            raise EmbeddingError("Ollama returned an invalid embeddings response.")
        try:
            return [[float(value) for value in vector] for vector in vectors]
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("Ollama returned a non-numeric embedding.") from exc
