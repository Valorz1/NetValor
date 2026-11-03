"""A user's own financial profile and memory notes.

Every route requires a session for the `{user_id}` in the path.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, status
from pydantic import BaseModel

from app.api.deps import OwnerId, ServicesDep
from app.api.errors import ErrorResponse
from app.memory import (
    MemoryNote,
    MemoryResponse,
    NoteCreate,
    NoteDeletedResponse,
    NoteLimitError,
    NoteUpdate,
    ProfileNotFoundError,
    UserProfile,
)

router = APIRouter(
    tags=["profile and memory"],
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
NoteId = Annotated[str, Path(min_length=1, max_length=64)]


class NotesClearedResponse(BaseModel):
    deleted: int


@router.get(
    "/api/profile/{user_id}", response_model=UserProfile, responses={404: {"model": ErrorResponse}}
)
def get_profile(user_id: OwnerId, services: ServicesDep) -> UserProfile:
    """Read your saved financial profile."""
    try:
        return services.profiles.get_profile(user_id)
    except ProfileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.put("/api/profile/{user_id}", response_model=UserProfile)
def put_profile(profile: UserProfile, user_id: OwnerId, services: ServicesDep) -> UserProfile:
    """Create or replace your saved financial profile."""
    return services.profiles.put_profile(user_id, profile)


@router.get("/api/memory/{user_id}", response_model=MemoryResponse)
def get_memory(user_id: OwnerId, services: ServicesDep) -> MemoryResponse:
    """List every note the co-pilot may use. Nothing is remembered without you adding it."""
    return MemoryResponse(notes=services.profiles.get_notes(user_id))


@router.post(
    "/api/memory/{user_id}",
    response_model=MemoryNote,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse}},
)
def create_memory_note(request: NoteCreate, user_id: OwnerId, services: ServicesDep) -> MemoryNote:
    """Save a new memory note."""
    try:
        return services.profiles.add_note(user_id, request.content)
    except NoteLimitError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc


@router.delete("/api/memory/{user_id}", response_model=NotesClearedResponse)
def clear_memory(user_id: OwnerId, services: ServicesDep) -> NotesClearedResponse:
    """Delete all of your memory notes."""
    return NotesClearedResponse(deleted=services.profiles.delete_all_notes(user_id))


@router.patch(
    "/api/memory/{user_id}/{note_id}",
    response_model=MemoryNote,
    responses={404: {"model": ErrorResponse}},
)
def edit_memory_note(
    request: NoteUpdate, user_id: OwnerId, note_id: NoteId, services: ServicesDep
) -> MemoryNote:
    """Edit one of your memory notes."""
    try:
        return services.profiles.update_note(user_id, note_id, request.content)
    except ProfileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.delete(
    "/api/memory/{user_id}/{note_id}",
    response_model=NoteDeletedResponse,
    responses={404: {"model": ErrorResponse}},
)
def delete_memory_note(
    user_id: OwnerId, note_id: NoteId, services: ServicesDep
) -> NoteDeletedResponse:
    """Delete one of your memory notes."""
    try:
        services.profiles.delete_note(user_id, note_id)
    except ProfileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return NoteDeletedResponse(deleted=True)
