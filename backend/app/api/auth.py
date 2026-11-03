"""Registration, login, logout and account deletion."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.api.deps import (
    CurrentUser,
    ServicesDep,
    client_ip,
    enforce_rate_limit,
    get_bearer_token,
)
from app.memory import USER_ID_PATTERN
from app.security import MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH
from app.users import UserExistsError

router = APIRouter(prefix="/api/auth", tags=["auth"])

Password = Annotated[str, Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)]


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(pattern=USER_ID_PATTERN, examples=["alex"])
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)

    @field_validator("password")
    @classmethod
    def _not_trivial(cls, value: str) -> str:
        if len(set(value)) < 4:
            raise ValueError("Password is too simple; use a longer, more varied passphrase.")
        return value


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=64)
    password: Password


class ConfirmPasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: Password


class UserResponse(BaseModel):
    user_id: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 - token scheme name, not a secret
    expires_at: datetime
    user_id: str


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
def register(body: RegisterRequest, request: Request, services: ServicesDep) -> UserResponse:
    """Create an account. Registration can be disabled with ALLOW_REGISTRATION=false."""
    enforce_rate_limit(services.auth_limiter, f"register:{client_ip(request)}")
    if not services.settings.allow_registration:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Registration is disabled.")
    if body.password.casefold() == body.user_id.casefold():
        raise HTTPException(422, "Password must not be the same as the user ID.")
    try:
        services.users.create_user(body.user_id, body.password)
    except UserExistsError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return UserResponse(user_id=body.user_id)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request, services: ServicesDep) -> TokenResponse:
    """Exchange credentials for a bearer token."""
    enforce_rate_limit(services.auth_limiter, f"login-ip:{client_ip(request)}")
    enforce_rate_limit(services.auth_limiter, f"login-user:{body.user_id.lower()}")
    if not services.users.authenticate(body.user_id, body.password):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect user ID or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token, expires_at = services.users.create_session(body.user_id)
    return TokenResponse(access_token=token, expires_at=expires_at, user_id=body.user_id)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    _user: CurrentUser,
    services: ServicesDep,
    token: Annotated[str, Depends(get_bearer_token)],
) -> Response:
    """Revoke the current session token."""
    services.users.revoke_session(token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/me", response_model=UserResponse)
def me(user_id: CurrentUser) -> UserResponse:
    return UserResponse(user_id=user_id)


@router.post("/delete-account", status_code=status.HTTP_204_NO_CONTENT)
async def delete_account(
    body: ConfirmPasswordRequest, user_id: CurrentUser, request: Request, services: ServicesDep
) -> Response:
    """Permanently delete the account and all of its profiles, notes and chats."""
    enforce_rate_limit(services.auth_limiter, f"delete:{client_ip(request)}")
    if not await anyio.to_thread.run_sync(services.users.authenticate, user_id, body.password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect password.")
    await anyio.to_thread.run_sync(services.users.delete_user, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
