"""Consistent error responses that never leak internals."""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("app.api")


class ErrorResponse(BaseModel):
    """Readable API error detail."""

    detail: str


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    # Echoing submitted values back (FastAPI's default) could repeat
    # passwords or financial details into logs and proxies; send only
    # where and why validation failed.
    assert isinstance(exc, RequestValidationError)
    errors = [
        {
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg", ""),
            "type": error.get("type", ""),
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


async def _unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled error (request %s)", request_id)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Something went wrong on our side. Reference: {request_id}."},
    )


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(Exception, _unhandled_error)


class RequestGuardMiddleware:
    """Assign request IDs, add security headers and cap request body size.

    Implemented as plain ASGI so it doesn't buffer streaming responses.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = request_id
        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                too_large = int(declared) > self.max_body_bytes
            except ValueError:
                too_large = True
            if too_large:
                await self._reject(send, request_id)
                return

        received = 0
        exceeded = False
        started = False

        async def limited_receive() -> Message:
            # A chunked body with no Content-Length is counted as it arrives.
            # Once over the limit the app sees a disconnect, and whatever it
            # tries to send is replaced with a 413 below.
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def send_with_headers(message: Message) -> None:
            nonlocal started
            if exceeded:
                if message["type"] == "http.response.start" and not started:
                    started = True
                    await self._reject(send, request_id)
                return
            if message["type"] == "http.response.start":
                started = True
                extra = [
                    (b"x-request-id", request_id.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"DENY"),
                ]
                existing = {name.lower() for name, _ in message.get("headers", [])}
                if b"cache-control" not in existing:
                    extra.append((b"cache-control", b"no-store"))
                message["headers"] = list(message.get("headers", [])) + extra
            await send(message)

        await self.app(scope, limited_receive, send_with_headers)
        if exceeded and not started:
            await self._reject(send, request_id)

    @staticmethod
    async def _reject(send: Send, request_id: str) -> None:
        body = b'{"detail":"Request body is too large."}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"x-request-id", request_id.encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
