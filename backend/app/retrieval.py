"""Semantic retrieval over the persistent lesson collection.

Query embedding is awaited over the shared async HTTP client and the
(synchronous) Chroma query runs in a worker thread, so a search never blocks
the event loop that is streaming other users' answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Protocol

import anyio

logger = logging.getLogger(__name__)
COLLECTION_NAME = "netvalor_lessons"
ReviewStatus = str | list[str] | None


class RetrievalError(RuntimeError):
    """Raised when the lesson index is missing or incompatible."""


class Retriever(Protocol):
    """Async lesson search used by the chat pipeline and search endpoint."""

    async def __call__(
        self,
        question: str,
        *,
        k: int = 4,
        module: str | None = None,
        review_status: ReviewStatus = None,
    ) -> list[dict[str, Any]]: ...


class LessonRetriever:
    """Search a Chroma collection using an injected async query embedder."""

    def __init__(
        self,
        collection: Any,
        embed_query: Callable[[str], Awaitable[list[float]]],
        embedding_model: str,
    ) -> None:
        self._collection = collection
        self._embed_query = embed_query
        stored_model = (collection.metadata or {}).get("embedding_model")
        if stored_model != embedding_model:
            raise RetrievalError(
                f"Embedding model mismatch: index uses {stored_model!r}, "
                f"configured model is {embedding_model!r}. Rebuild the lesson index."
            )

    @staticmethod
    def _filters(module: str | None, review_status: ReviewStatus) -> dict[str, Any] | None:
        conditions: list[dict[str, Any]] = []
        if module:
            conditions.append({"module": module})
        if isinstance(review_status, str):
            conditions.append({"review_status": review_status})
        elif review_status:
            conditions.append({"review_status": {"$in": list(review_status)}})
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    async def search(
        self,
        question: str,
        k: int = 4,
        module: str | None = None,
        review_status: ReviewStatus = None,
    ) -> list[dict[str, Any]]:
        """Return the nearest lesson chunks with cosine similarity and metadata."""
        if not question.strip():
            raise ValueError("question must not be empty.")
        if k < 1:
            raise ValueError("k must be at least 1.")
        embedding = await self._embed_query(question)
        filters = self._filters(module, review_status)

        def query() -> Any:
            return self._collection.query(
                query_embeddings=[embedding],
                n_results=k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )

        try:
            matches = await anyio.to_thread.run_sync(query)
        except Exception as exc:  # Chroma raises a variety of internal errors.
            raise RetrievalError(f"Lesson index query failed: {type(exc).__name__}") from exc
        ids = (matches.get("ids") or [[]])[0]
        results = []
        for index, (document, metadata, distance) in enumerate(
            zip(
                matches["documents"][0],
                matches["metadatas"][0],
                matches["distances"][0],
                strict=True,
            )
        ):
            result_metadata = dict(metadata or {})
            source_urls = result_metadata.get("source_urls", "[]")
            if isinstance(source_urls, str):
                try:
                    source_urls = json.loads(source_urls)
                except json.JSONDecodeError:
                    logger.warning("Skipping lesson chunk with malformed source URLs.")
                    continue
            if not isinstance(source_urls, list):
                continue
            result_metadata["source_urls"] = [str(url) for url in source_urls]
            results.append(
                {
                    "chunk_id": ids[index] if index < len(ids) else None,
                    "chunk": document,
                    "score": 1.0 - float(distance),
                    "metadata": result_metadata,
                }
            )
        return results


class ChromaRetriever:
    """Lazily open the persistent Chroma index and search it.

    Opening is retried on the next request after a failure (for example if
    the index is built while the API is running).
    """

    def __init__(
        self,
        chroma_dir: Path,
        embed_query: Callable[[str], Awaitable[list[float]]],
        embedding_model: str,
    ) -> None:
        self._chroma_dir = chroma_dir
        self._embed_query = embed_query
        self._embedding_model = embedding_model
        self._retriever: LessonRetriever | None = None
        self._lock = asyncio.Lock()

    def _open(self) -> LessonRetriever:
        try:
            from chromadb import PersistentClient

            client = PersistentClient(path=str(self._chroma_dir))
            # Older Chroma releases return names, newer ones Collection objects.
            names = {
                str(getattr(collection, "name", collection))
                for collection in client.list_collections()
            }
        except Exception as exc:
            raise RetrievalError(f"Could not open the lesson index: {type(exc).__name__}") from exc
        if COLLECTION_NAME not in names:
            raise RetrievalError(
                "Lesson index is unavailable. Run the knowledge-base ingestion script first."
            )
        collection = client.get_collection(COLLECTION_NAME)
        return LessonRetriever(collection, self._embed_query, self._embedding_model)

    async def __call__(
        self,
        question: str,
        *,
        k: int = 4,
        module: str | None = None,
        review_status: ReviewStatus = None,
    ) -> list[dict[str, Any]]:
        if self._retriever is None:
            async with self._lock:
                if self._retriever is None:
                    self._retriever = await anyio.to_thread.run_sync(self._open)
        return await self._retriever.search(
            question, k=k, module=module, review_status=review_status
        )
