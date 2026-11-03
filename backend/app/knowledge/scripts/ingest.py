"""Rebuild the persistent Chroma index from the reviewed lesson files."""

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from app.config import BACKEND_DIR, get_settings
from app.grounding import looks_like_injection
from app.knowledge.embeddings import OllamaEmbeddings
from app.knowledge.lessons import chunk_lesson, load_lessons
from app.retrieval import COLLECTION_NAME


class DocumentEmbedder(Protocol):
    model: str

    def embed_documents(self, title: str, texts: list[str]) -> list[list[float]]: ...


def _collection_names(collections: Sequence[Any]) -> set[str]:
    return {str(getattr(collection, "name", collection)) for collection in collections}


def build_index(
    lessons_dir: Path,
    chroma_dir: Path,
    embeddings: DocumentEmbedder,
) -> tuple[int, int]:
    """Replace the lesson collection and return lesson and chunk counts.

    All lessons are validated (and checked for instruction-like text that
    could act as a prompt injection) before the existing index is touched.
    """
    from chromadb import PersistentClient

    lessons = load_lessons(lessons_dir)
    if not lessons:
        raise ValueError(f"No Markdown lessons found in {lessons_dir}.")
    for lesson in lessons:
        if looks_like_injection(lesson.content):
            raise ValueError(f"{lesson.path}: contains instruction-like text; refusing to index.")
    client = PersistentClient(path=str(chroma_dir))
    if COLLECTION_NAME in _collection_names(client.list_collections()):
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": embeddings.model, "hnsw:space": "cosine"},
    )

    chunk_count = 0
    for lesson in lessons:
        chunks = chunk_lesson(lesson)
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            vectors = embeddings.embed_documents(
                lesson.metadata["title"], [chunk["text"] for chunk in batch]
            )
            collection.add(
                ids=[f"{lesson.metadata['id']}:{start + offset}" for offset in range(len(batch))],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[
                    {
                        "lesson_id": lesson.metadata["id"],
                        "title": lesson.metadata["title"],
                        "module": lesson.metadata["module"],
                        "heading": chunk["heading"],
                        "source_urls": json.dumps(
                            [source["url"] for source in lesson.metadata["sources"]]
                        ),
                        "review_status": lesson.metadata["review_status"],
                        "embedding_model": embeddings.model,
                    }
                    for chunk in batch
                ],
                embeddings=cast("list[Sequence[float] | Sequence[int]]", vectors),
            )
            chunk_count += len(batch)
    return len(lessons), chunk_count


def main() -> None:
    """Rebuild the lesson index and print a concise ingestion summary."""
    started = time.perf_counter()
    settings = get_settings()
    lessons_dir = BACKEND_DIR / "app" / "knowledge" / "lessons"
    chroma_dir = settings.resolve(settings.chroma_dir)
    embeddings = OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_url)
    lesson_count, chunk_count = build_index(lessons_dir, chroma_dir, embeddings)
    elapsed = time.perf_counter() - started
    print(
        f"Ingested {lesson_count} lessons into {chunk_count} chunks "
        f"in {elapsed:.2f}s using {embeddings.model}."
    )


if __name__ == "__main__":
    main()
