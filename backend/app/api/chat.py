"""Streaming co-pilot chat and saved conversations."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

import anyio
from fastapi import APIRouter, HTTPException, Path, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.types import Receive, Scope, Send

from app.api.deps import OwnerId, ServicesDep, enforce_rate_limit
from app.api.errors import ErrorResponse
from app.conversations import (
    MAX_MESSAGES_PER_CONVERSATION,
    ConversationNotFoundError,
)
from app.ratelimit import Lease

logger = logging.getLogger(__name__)
router = APIRouter(
    tags=["co-pilot"],
    responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
MAX_MESSAGE_CHARS = 2000
ConversationId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class ChatRequest(BaseModel):
    """One new user message, optionally continuing a conversation."""

    model_config = ConfigDict(extra="forbid")

    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_MESSAGE_CHARS)
    ]
    conversation_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    use_saved_context: bool = Field(
        default=True,
        description="Set false to answer without reading your profile, goals or memory notes.",
    )


class ConversationSummary(BaseModel):
    conversation_id: str
    created_at: str
    updated_at: str
    message_count: int
    last_message: str


class ConversationList(BaseModel):
    conversations: list[ConversationSummary]


class StoredMessage(BaseModel):
    role: str
    content: str
    status: str
    created_at: str


class ConversationDetail(BaseModel):
    conversation_id: str
    user_id: str
    created_at: str
    updated_at: str
    messages: list[StoredMessage]


def sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


class _LeasedStreamingResponse(StreamingResponse):
    """Release the concurrency slot however the response ends.

    A generator's `finally` doesn't run if the client disconnects before
    streaming starts, so the lease is released around the whole response.
    """

    def __init__(self, content: AsyncIterator[str], lease: Lease, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._lease = lease

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._lease.release()


@router.post(
    "/api/chat/{user_id}",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "token, done or error events"},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
    },
)
async def chat(request: ChatRequest, user_id: OwnerId, services: ServicesDep) -> StreamingResponse:
    """Stream a source-grounded co-pilot answer as Server-Sent Events.

    Events: `token` (`{"text"}`), then either `done` (conversation ID,
    citations, verified calculations, grounding checks, notices) or `error`
    (`{"code", "message"}`).
    """
    enforce_rate_limit(services.chat_limiter, f"chat:{user_id}")
    if request.conversation_id is not None:
        count = await anyio.to_thread.run_sync(
            services.conversations.message_count, user_id, request.conversation_id
        )
        if count == 0 and not await anyio.to_thread.run_sync(
            services.conversations.exists, user_id, request.conversation_id
        ):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "That conversation wasn't found.")
        if count + 2 > MAX_MESSAGES_PER_CONVERSATION:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "This conversation is full. Please start a new one."
            )
    lease = services.chat_gate.try_acquire(user_id)
    if lease is None:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "NetValor is busy answering other questions. Please try again shortly.",
            headers={"Retry-After": "5"},
        )
    service = services.chat_service()

    async def events() -> AsyncIterator[str]:
        try:
            async for event in service.stream(
                user_id,
                request.message,
                request.conversation_id,
                use_saved_context=request.use_saved_context,
            ):
                yield sse_event(event["event"], event["data"])
        finally:
            lease.release()

    return _LeasedStreamingResponse(
        events(),
        lease,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/api/conversations/{user_id}", response_model=ConversationList)
def list_conversations(user_id: OwnerId, services: ServicesDep) -> dict[str, Any]:
    """List your saved conversation summaries, newest first."""
    return {"conversations": services.conversations.list(user_id)}


@router.get(
    "/api/conversations/{user_id}/{conversation_id}",
    response_model=ConversationDetail,
    responses={404: {"model": ErrorResponse}},
)
def get_conversation(
    user_id: OwnerId, conversation_id: ConversationId, services: ServicesDep
) -> dict[str, Any]:
    """Return one of your saved conversations."""
    try:
        return services.conversations.get(user_id, conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That conversation wasn't found.") from exc


@router.delete(
    "/api/conversations/{user_id}/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}},
)
def delete_conversation(
    user_id: OwnerId, conversation_id: ConversationId, services: ServicesDep
) -> None:
    """Permanently delete one of your conversations."""
    try:
        services.conversations.delete(user_id, conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "That conversation wasn't found.") from exc
