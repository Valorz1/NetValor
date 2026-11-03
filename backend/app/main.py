"""NetValor API application factory.

`create_app()` wires configuration, storage, Ollama clients and routers.
Tests pass their own `Settings` and fakes for the retriever and language
model, so the standard suite never needs a running Ollama instance.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api import auth, calc, chat, lessons, profiles
from app.api.deps import Services, ServicesDep
from app.api.errors import RequestGuardMiddleware, install_error_handlers
from app.chat import ChatLLM
from app.config import BACKEND_DIR, Settings, get_settings
from app.conversations import ConversationStore
from app.db import Database
from app.knowledge.embeddings import OllamaEmbeddings
from app.knowledge.lessons import load_lessons
from app.llm import OllamaChatClient
from app.memory import ProfileStore, demo_profile
from app.ratelimit import ConcurrencyGate, RateLimiter
from app.retrieval import ChromaRetriever, Retriever
from app.security import build_cipher
from app.users import UserStore

logger = logging.getLogger("app")
LESSONS_DIR = BACKEND_DIR / "app" / "knowledge" / "lessons"


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    database: bool
    lessons: int


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("app").setLevel(level)


def _seed_demo_user(services: Services) -> None:
    """Create the local 'alex' demo account when explicitly enabled."""
    settings = services.settings
    if not settings.seed_demo_user:
        return
    password = settings.demo_user_password.get_secret_value() if settings.demo_user_password else ""
    if len(password) < 12:
        logger.warning(
            "SEED_DEMO_USER is enabled but DEMO_USER_PASSWORD is not set to at least "
            "12 characters; skipping the demo account."
        )
        return
    if services.users.exists("alex"):
        return
    services.users.create_user("alex", password)
    profile, notes = demo_profile()
    services.profiles.put_profile("alex", profile)
    for note in notes:
        services.profiles.add_note("alex", note)
    logger.info("Created the local 'alex' demo account.")


def build_services(
    settings: Settings,
    *,
    llm: ChatLLM | None = None,
    retriever: Retriever | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Services:
    """Construct storage, clients and limiters from configuration."""
    database = Database(settings.resolve(settings.database_path))
    database.initialise()
    cipher = build_cipher(
        settings.data_encryption_keys.get_secret_value() if settings.data_encryption_keys else None
    )
    if retriever is None:
        embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_url,
            timeout=settings.embedding_timeout_seconds,
            http_client=http_client,
        )
        retriever = ChromaRetriever(
            settings.resolve(settings.chroma_dir), embeddings.aembed_query, embeddings.model
        )
    if llm is None:
        llm = OllamaChatClient(
            base_url=settings.ollama_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_num_ctx,
            num_predict=settings.llm_num_predict,
            http_client=http_client,
        )
    return Services(
        settings=settings,
        database=database,
        users=UserStore(database, timedelta(hours=settings.session_ttl_hours)),
        profiles=ProfileStore(database, cipher),
        conversations=ConversationStore(database, cipher),
        retriever=retriever,
        llm=llm,
        lessons=load_lessons(LESSONS_DIR),
        auth_limiter=RateLimiter(settings.auth_rate_limit_per_minute),
        chat_limiter=RateLimiter(settings.chat_rate_limit_per_minute),
        search_limiter=RateLimiter(settings.search_rate_limit_per_minute),
        chat_gate=ConcurrencyGate(
            settings.max_concurrent_chats, settings.max_concurrent_chats_per_user
        ),
        http_client=http_client,
    )


def create_app(
    settings: Settings | None = None,
    *,
    llm: ChatLLM | None = None,
    retriever: Retriever | None = None,
) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or get_settings()
    _configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        http_client = None
        if llm is None or retriever is None:
            http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    settings.llm_read_timeout_seconds,
                    connect=settings.llm_connect_timeout_seconds,
                ),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        services = build_services(settings, llm=llm, retriever=retriever, http_client=http_client)
        _seed_demo_user(services)
        app.state.services = services
        logger.info(
            "NetValor API started (environment=%s, drafts=%s, docs=%s).",
            settings.environment,
            settings.allow_draft,
            settings.show_docs,
        )
        try:
            yield
        finally:
            if http_client is not None:
                await http_client.aclose()

    app = FastAPI(
        title="NetValor API",
        description=(
            "Educational UK financial-literacy API. Calculations are estimates for "
            "learning purposes, not financial advice."
        ),
        version="0.2.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.show_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.show_docs else None,
    )
    install_error_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,  # Bearer tokens, not cookies.
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )
    app.add_middleware(RequestGuardMiddleware, max_body_bytes=settings.max_request_bytes)

    @app.get("/api/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        """Liveness: the process is running."""
        return HealthResponse(status="ok")

    @app.get("/api/health/ready", response_model=ReadinessResponse, tags=["health"])
    def ready(services: ServicesDep) -> ReadinessResponse:
        """Readiness: storage is reachable and lessons are loaded."""
        database_ok = services.database.ping()
        return ReadinessResponse(
            status="ok" if database_ok else "degraded",
            database=database_ok,
            lessons=len(services.lessons),
        )

    for router in (auth.router, profiles.router, chat.router, lessons.router, calc.router):
        app.include_router(router)
    return app


_default_app: FastAPI | None = None


def __getattr__(name: str) -> FastAPI:
    # `uvicorn app.main:app` builds the app lazily, so importing this module
    # (for example in tests) doesn't read the environment or create files.
    global _default_app
    if name == "app":
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(name)
