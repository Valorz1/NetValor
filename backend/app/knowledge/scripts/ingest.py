"""Rebuild the persistent Chroma index from the reviewed lesson files."""

import json
import os
import time
from pathlib import Path
from typing import Any

from app.knowledge.embeddings import OllamaEmbeddings
from app.knowledge.lessons import chunk_lesson, load_lessons
from app.retrieval import COLLECTION_NAME


def _backend_path() -> Path:
    return Path(__file__).resolve().parents[3]


def _configured_path(variable: str, default: str) -> Path:
    configured = Path(os.getenv(variable, default))
    return configured if configured.is_absolute() else _backend_path() / configured


def _collection_names(collections: list[Any]) -> set[str]:
    return {
        collection if isinstance(collection, str) else collection.name
        for collection in collections
    }


def build_index(
    lessons_dir: Path,
    chroma_dir: Path,
    embeddings: OllamaEmbeddings,
) -> tuple[int, int]:
    """Replace the lesson collection and return lesson and chunk counts."""
    from chromadb import PersistentClient

    lessons = load_lessons(lessons_dir)
    if not lessons:
        raise ValueError(f"No Markdown lessons found in {lessons_dir}.")
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
            batch = chunks[start:start + 32]
            vectors = embeddings.embed_documents(
                lesson.metadata["title"], [chunk["text"] for chunk in batch]
            )
            collection.add(
                ids=[
                    f"{lesson.metadata['id']}:{start + offset}"
                    for offset in range(len(batch))
                ],
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
                embeddings=vectors,
            )
            chunk_count += len(batch)
    return len(lessons), chunk_count


def main() -> None:
    """Rebuild the lesson index and print a concise ingestion summary."""
    started = time.perf_counter()
    lessons_dir = _backend_path() / "app" / "knowledge" / "lessons"
    chroma_dir = _configured_path("CHROMA_DIR", "data/chroma")
    embeddings = OllamaEmbeddings()
    lesson_count, chunk_count = build_index(lessons_dir, chroma_dir, embeddings)
    elapsed = time.perf_counter() - started
    print(
        f"Ingested {lesson_count} lessons into {chunk_count} chunks "
        f"in {elapsed:.2f}s using {embeddings.model}."
    )


if __name__ == "__main__":
    main()
