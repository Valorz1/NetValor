"""FastAPI routes for NetValor health, calculations, profiles and memory."""

import os
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Path as PathParameter, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.finance import (
    compare_extra_payment,
    debt_payoff,
    emergency_fund_status,
    payoff_date,
)
from app.memory import (
    JsonProfileStore,
    MemoryNote,
    MemoryResponse,
    NoteCreate,
    NoteDeletedResponse,
    NoteUpdate,
    ProfileNotFoundError,
    UserProfile,
)
from app.knowledge.embeddings import EmbeddingError
from app.knowledge.lessons import load_lessons
from app.retrieval import RetrievalError, search as search_lessons


class HealthResponse(BaseModel):
    """API health status."""

    status: str


class ErrorResponse(BaseModel):
    """Readable API error detail."""

    detail: str


class DebtPayoffRequest(BaseModel):
    """Inputs for estimating a debt payoff."""

    balance: Decimal = Field(ge=0)
    apr: Decimal = Field(ge=0, le=1)
    monthly_payment: Decimal = Field(ge=0)


class DebtPayoffResponse(BaseModel):
    """Debt payoff estimate returned by the calculation engine."""

    months: int | None
    total_interest: Decimal | None
    total_paid: Decimal | None
    final_payment: Decimal | None
    pays_off: bool


class CompareExtraPaymentRequest(DebtPayoffRequest):
    """Inputs for comparing a regular and increased monthly payment."""

    extra: Decimal = Field(ge=0)


class CompareExtraPaymentResponse(BaseModel):
    """Payment comparison and estimated payoff months."""

    baseline: DebtPayoffResponse
    with_extra: DebtPayoffResponse
    months_saved: int | None
    interest_saved: Decimal | None
    baseline_payoff_date: date | None
    with_extra_payoff_date: date | None


class EmergencyFundRequest(BaseModel):
    """Inputs for checking emergency-fund coverage."""

    saved: Decimal = Field(ge=0)
    monthly_essentials: Decimal = Field(gt=0)
    target_months: int = Field(default=3, gt=0)


class EmergencyFundResponse(BaseModel):
    """Emergency-fund target and progress."""

    months_covered: Decimal
    target_amount: Decimal
    amount_left: Decimal
    percent_complete: Decimal


class LessonSummary(BaseModel):
    """Public index fields for one learning-path lesson."""

    id: str
    title: str
    module: str
    order: int


class LessonDetail(LessonSummary):
    """A lesson and its Markdown content."""

    content: str


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


_default_origins = "http://localhost:3000"
_cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

_backend_dir = Path(__file__).resolve().parents[1]
profile_store = JsonProfileStore(_backend_dir / "data" / "profiles")
_lessons_dir = _backend_dir / "app" / "knowledge" / "lessons"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Seed the local demo profile when the API starts."""
    profile_store.seed_demo_user()
    yield


app = FastAPI(title="NetValor API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or [_default_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_profile_store() -> JsonProfileStore:
    """Provide the profile storage implementation to API routes."""
    return profile_store


@app.exception_handler(ValueError)
async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _profile_or_404(store: JsonProfileStore, user_id: str) -> UserProfile:
    try:
        return store.get_profile(user_id)
    except ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return a basic health status."""
    return HealthResponse(status="ok")


@app.get("/api/lessons", response_model=list[LessonSummary])
def list_lessons() -> list[LessonSummary]:
    """List the available learning-path lessons in their intended order."""
    lessons = load_lessons(_lessons_dir)
    return [
        LessonSummary(
            id=lesson.metadata["id"],
            title=lesson.metadata["title"],
            module=lesson.metadata["module"],
            order=lesson.metadata["order"],
        )
        for lesson in lessons
    ]


@app.get("/api/lessons/{lesson_id}", response_model=LessonDetail)
def get_lesson(lesson_id: str) -> LessonDetail:
    """Return one full lesson, excluding its authoring front matter."""
    for lesson in load_lessons(_lessons_dir):
        if lesson.metadata["id"] == lesson_id:
            return LessonDetail(
                id=lesson.metadata["id"],
                title=lesson.metadata["title"],
                module=lesson.metadata["module"],
                order=lesson.metadata["order"],
                content=lesson.content,
            )
    raise HTTPException(status_code=404, detail=f"Lesson '{lesson_id}' was not found.")


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1),
    k: int = Query(default=4, ge=1, le=20),
    module: str | None = None,
) -> SearchResponse:
    """Search indexed lesson passages without generating an answer."""
    try:
        results = search_lessons(q, k=k, module=module)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RetrievalError, EmbeddingError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchResponse(results=results)


@app.post(
    "/api/calc/debt-payoff",
    response_model=DebtPayoffResponse,
    responses={422: {"model": ErrorResponse}},
)
def calculate_debt_payoff(request: DebtPayoffRequest) -> DebtPayoffResponse:
    """Calculate monthly debt payoff totals using the deterministic engine."""
    result = debt_payoff(request.balance, request.apr, request.monthly_payment)
    return DebtPayoffResponse.model_validate(result, from_attributes=True)


@app.post(
    "/api/calc/compare-extra",
    response_model=CompareExtraPaymentResponse,
    responses={422: {"model": ErrorResponse}},
)
def calculate_extra_payment(
    request: CompareExtraPaymentRequest,
) -> CompareExtraPaymentResponse:
    """Compare debt payoff with an extra monthly payment."""
    result = compare_extra_payment(
        request.balance, request.apr, request.monthly_payment, request.extra
    )
    today = date.today()
    baseline_date = (
        payoff_date(today, result.baseline.months)
        if result.baseline.pays_off and result.baseline.months is not None
        else None
    )
    extra_date = (
        payoff_date(today, result.with_extra.months)
        if result.with_extra.pays_off and result.with_extra.months is not None
        else None
    )
    return CompareExtraPaymentResponse(
        baseline=DebtPayoffResponse.model_validate(result.baseline, from_attributes=True),
        with_extra=DebtPayoffResponse.model_validate(result.with_extra, from_attributes=True),
        months_saved=result.months_saved,
        interest_saved=result.interest_saved,
        baseline_payoff_date=baseline_date,
        with_extra_payoff_date=extra_date,
    )


@app.post(
    "/api/calc/emergency-fund",
    response_model=EmergencyFundResponse,
    responses={422: {"model": ErrorResponse}},
)
def calculate_emergency_fund(request: EmergencyFundRequest) -> EmergencyFundResponse:
    """Calculate emergency-fund coverage and progress toward the target."""
    result = emergency_fund_status(
        request.saved, request.monthly_essentials, request.target_months
    )
    return EmergencyFundResponse.model_validate(result, from_attributes=True)


@app.get("/api/profile/{user_id}", response_model=UserProfile)
def get_profile(
    user_id: str = PathParameter(min_length=1),
    store: JsonProfileStore = Depends(get_profile_store),
) -> UserProfile:
    """Read a user's saved financial profile."""
    return _profile_or_404(store, user_id)


@app.put("/api/profile/{user_id}", response_model=UserProfile)
def put_profile(
    profile: UserProfile,
    user_id: str = PathParameter(min_length=1),
    store: JsonProfileStore = Depends(get_profile_store),
) -> UserProfile:
    """Create or replace a user's saved financial profile."""
    try:
        return store.update_profile(user_id, profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/api/memory/{user_id}", response_model=MemoryResponse)
def get_memory(
    user_id: str = PathParameter(min_length=1),
    store: JsonProfileStore = Depends(get_profile_store),
) -> MemoryResponse:
    """List the notes the co-pilot remembers about a user."""
    _profile_or_404(store, user_id)
    try:
        return MemoryResponse(notes=store.get_notes(user_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/api/memory/{user_id}",
    response_model=MemoryNote,
    status_code=201,
    responses={422: {"model": ErrorResponse}},
)
def create_memory_note(
    request: NoteCreate,
    user_id: str = PathParameter(min_length=1),
    store: JsonProfileStore = Depends(get_profile_store),
) -> MemoryNote:
    """Save a new user-visible memory note."""
    _profile_or_404(store, user_id)
    try:
        return store.add_note(user_id, request.content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.patch(
    "/api/memory/{user_id}/{note_id}",
    response_model=MemoryNote,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def edit_memory_note(
    request: NoteUpdate,
    user_id: str = PathParameter(min_length=1),
    note_id: str = PathParameter(min_length=1),
    store: JsonProfileStore = Depends(get_profile_store),
) -> MemoryNote:
    """Edit one of a user's remembered notes."""
    _profile_or_404(store, user_id)
    try:
        return store.update_note(user_id, note_id, request.content)
    except ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.delete(
    "/api/memory/{user_id}/{note_id}",
    response_model=NoteDeletedResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def delete_memory_note(
    user_id: str = PathParameter(min_length=1),
    note_id: str = PathParameter(min_length=1),
    store: JsonProfileStore = Depends(get_profile_store),
) -> NoteDeletedResponse:
    """Delete one of a user's remembered notes."""
    _profile_or_404(store, user_id)
    try:
        store.delete_note(user_id, note_id)
    except ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return NoteDeletedResponse(deleted=True)