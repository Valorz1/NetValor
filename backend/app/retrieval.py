"""Semantic retrieval over the persistent lesson collection."""

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from app.knowledge.embeddings import OllamaEmbeddings


COLLECTION_NAME = "netvalor_lessons"


class RetrievalError(RuntimeError):
    """Raised when the lesson index is missing or incompatible."""


class LessonRetriever:
    """Search a Chroma collection using an injected query embedder."""

    def __init__(
        self,
        collection: Any,
        embed_query: Callable[[str], list[float]],
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

    def search(
        self, question: str, k: int = 4, module: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the nearest lesson chunks with cosine similarity and metadata."""
        if not question.strip():
            raise ValueError("question must not be empty.")
        if k < 1:
            raise ValueError("k must be at least 1.")
        filters = {"module": module} if module else None
        matches = self._collection.query(
            query_embeddings=[self._embed_query(question)],
            n_results=k,
            where=filters,
            include=["documents", "metadatas", "distances"],
        )
        documents = matches["documents"][0]
        metadatas = matches["metadatas"][0]
        distances = matches["distances"][0]
        results = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            result_metadata = dict(metadata)
            source_urls = result_metadata.get("source_urls", "[]")
            if isinstance(source_urls, str):
                result_metadata["source_urls"] = json.loads(source_urls)
            results.append(
                {
                    "chunk": document,
                    "score": 1.0 - float(distance),
                    "metadata": result_metadata,
                }
            )
        return results


def _chroma_path() -> Path:
    configured = Path(os.getenv("CHROMA_DIR", "data/chroma"))
    backend_dir = Path(__file__).resolve().parents[1]
    return configured if configured.is_absolute() else backend_dir / configured


@lru_cache(maxsize=1)
def _default_retriever() -> LessonRetriever:
    """Open the configured persistent collection and verify its embedding model."""
    from chromadb import PersistentClient

    embeddings = OllamaEmbeddings()
    client = PersistentClient(path=str(_chroma_path()))
    collections = client.list_collections()
    names = {
        collection if isinstance(collection, str) else collection.name
        for collection in collections
    }
    if COLLECTION_NAME not in names:
        raise RetrievalError(
            "Lesson index is unavailable. Run the knowledge-base ingestion script first."
        )
    collection = client.get_collection(COLLECTION_NAME)
    return LessonRetriever(collection, embeddings.embed_query, embeddings.model)


def search(
    question: str, k: int = 4, module: str | None = None
) -> list[dict[str, Any]]:
    """Search the configured NetValor lesson index."""
    return _default_retriever().search(question, k=k, module=module)
