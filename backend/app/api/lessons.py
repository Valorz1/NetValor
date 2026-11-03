"""Public learning-path lessons and lesson search."""

from __future__ import annotations

import logging
from datetime import date
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel

from app.api.deps import ServicesDep, client_ip, enforce_rate_limit
from app.api.errors import ErrorResponse
from app.knowledge.embeddings import EmbeddingError
from app.knowledge.lessons import Lesson
from app.retrieval import RetrievalError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["lessons"])


class LessonSummary(BaseModel):
    """Public index fields for one learning-path lesson."""

    id: str
    title: str
    module: str
    order: int
    review_status: str
    last_reviewed: date


class LessonDetail(LessonSummary):
    """A lesson and its Markdown content."""

    content: str
    source_urls: list[str]


class SearchMetadata(BaseModel):
    """Lesson metadata attached to a retrieved chunk."""

    lesson_id: str
    title: str
    module: str
    heading: str
    source_urls: list[str]
    review_status: str
    embedding_model: str


class SearchResult(BaseModel):
    """One retrieved lesson chunk and its cosine similarity."""

    chunk: str
    score: float
    metadata: SearchMetadata


class SearchResponse(BaseModel):
    """Search results ordered by relevance."""

    results: list[SearchResult]


def _visible(services: ServicesDep) -> list[Lesson]:
    allowed = services.settings.allowed_review_statuses
    return [lesson for lesson in services.lessons if lesson.metadata["review_status"] in allowed]


def _summary(lesson: Lesson) -> dict[str, object]:
    meta = lesson.metadata
    return {
        "id": meta["id"],
        "title": meta["title"],
        "module": meta["module"],
        "order": meta["order"],
        "review_status": meta["review_status"],
        "last_reviewed": meta["last_reviewed"],
    }


@router.get("/api/lessons", response_model=list[LessonSummary])
def list_lessons(services: ServicesDep) -> list[LessonSummary]:
    """List lessons in learning-path order.

    Draft lessons are only listed when ALLOW_DRAFT is enabled, and every
    lesson carries its `review_status` so drafts are never presented as reviewed.
    """
    return [LessonSummary.model_validate(_summary(lesson)) for lesson in _visible(services)]


@router.get(
    "/api/lessons/{lesson_id}",
    response_model=LessonDetail,
    responses={404: {"model": ErrorResponse}},
)
def get_lesson(lesson_id: str, services: ServicesDep) -> LessonDetail:
    """Return one full lesson without its authoring front matter."""
    for lesson in _visible(services):
        if lesson.metadata["id"] == lesson_id:
            return LessonDetail.model_validate(
                {
                    **_summary(lesson),
                    "content": lesson.content,
                    "source_urls": [source["url"] for source in lesson.metadata["sources"]],
                }
            )
    raise HTTPException(status.HTTP_404_NOT_FOUND, "Lesson not found.")


@router.get(
    "/api/search",
    response_model=SearchResponse,
    responses={429: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def search(
    request: Request,
    services: ServicesDep,
    q: Annotated[str, Query(min_length=1, max_length=300)],
    k: Annotated[int, Query(ge=1, le=10)] = 4,
    module: Annotated[str | None, Query(max_length=100)] = None,
) -> SearchResponse:
    """Search lesson passages without generating an answer."""
    enforce_rate_limit(services.search_limiter, f"search:{client_ip(request)}")
    if not q.strip():
        raise HTTPException(422, "q must not be blank.")
    try:
        results = await services.retriever(
            q,
            k=k,
            module=module,
            review_status=sorted(services.settings.allowed_review_statuses),
        )
    except (RetrievalError, EmbeddingError) as exc:
        logger.warning("Lesson search unavailable: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Lesson search is unavailable. Check that Ollama is running and the "
            "lesson index has been built.",
        ) from exc
    return SearchResponse.model_validate({"results": results})
