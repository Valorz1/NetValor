"""Shared services and FastAPI dependencies (authentication, ownership, limits)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import httpx
from fastapi import Depends, HTTPException, Path, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.chat import ChatLLM, ChatOptions, ChatService
from app.config import Settings
from app.conversations import ConversationStore
from app.db import Database
from app.knowledge.lessons import Lesson
from app.memory import ProfileStore
from app.ratelimit import ConcurrencyGate, RateLimiter
from app.retrieval import Retriever
from app.users import UserStore


@dataclass
class Services:
    """Everything the routes need, built once per application lifespan."""

    settings: Settings
    database: Database
    users: UserStore
    profiles: ProfileStore
    conversations: ConversationStore
    retriever: Retriever
    llm: ChatLLM
    lessons: list[Lesson]
    auth_limiter: RateLimiter
    chat_limiter: RateLimiter
    search_limiter: RateLimiter
    chat_gate: ConcurrencyGate
    http_client: httpx.AsyncClient | None = None

    def chat_service(self) -> ChatService:
        settings = self.settings
        return ChatService(
            profile_store=self.profiles,
            conversation_store=self.conversations,
            retriever=self.retriever,
            llm=self.llm,
            options=ChatOptions(
                allowed_statuses=settings.allowed_review_statuses,
                k=settings.retrieval_k,
                min_score=settings.min_retrieval_score,
                max_history_messages=settings.max_history_messages,
                max_history_chars=settings.max_history_chars,
                max_answer_chars=settings.llm_max_answer_chars,
            ),
        )


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]
_bearer = HTTPBearer(auto_error=False, description="Session token from POST /api/auth/login")


def _unauthorised(detail: str = "Authentication required.") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    services: ServicesDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    """Resolve the bearer token to a user ID, or reject the request with 401."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _unauthorised()
    token = credentials.credentials
    if not token or len(token) > 256:
        raise _unauthorised("Invalid or expired session.")
    user_id = services.users.resolve_session(token)
    if user_id is None:
        raise _unauthorised("Invalid or expired session.")
    return user_id


CurrentUser = Annotated[str, Depends(get_current_user)]


def get_bearer_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str:
    if credentials is None:
        raise _unauthorised()
    return credentials.credentials


def require_owner(
    current_user: CurrentUser,
    user_id: Annotated[str, Path(min_length=1, max_length=64)],
) -> str:
    """Allow access to `/{user_id}/...` routes only for that user's own session."""
    if user_id != current_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own data.",
        )
    return current_user


OwnerId = Annotated[str, Depends(require_owner)]


def client_ip(request: Request) -> str:
    """The direct peer address. Run uvicorn with --proxy-headers behind a proxy."""
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(limiter: RateLimiter, key: str) -> None:
    retry_after = limiter.hit(key)
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait a moment and try again.",
            headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
        )
