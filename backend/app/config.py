"""Typed application configuration loaded from environment variables and `.env`.

All settings have secure defaults. `Settings.validate_for_environment` refuses
combinations that would be unsafe in production (for example draft lessons,
seeded demo accounts, wildcard CORS or unencrypted personal data), so a
development convenience cannot silently reach a production deployment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """NetValor backend settings. Field names map to upper-case env vars."""

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    docs_enabled: bool | None = Field(
        default=None,
        description="Expose /docs and /openapi.json. Defaults to off in production.",
    )

    # --- Storage -----------------------------------------------------------
    database_path: Path = Path("data/netvalor.db")
    data_encryption_keys: SecretStr | None = Field(
        default=None,
        description=(
            "Comma-separated Fernet keys. The first encrypts new data; the rest "
            "can still decrypt (key rotation). Required in production."
        ),
    )

    # --- Authentication ----------------------------------------------------
    session_ttl_hours: int = Field(default=24 * 7, ge=1, le=24 * 90)
    allow_registration: bool = True
    seed_demo_user: bool = False
    demo_user_password: SecretStr | None = None

    # --- HTTP / abuse prevention --------------------------------------------
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    max_request_bytes: int = Field(default=64 * 1024, ge=1024)
    auth_rate_limit_per_minute: int = Field(default=10, ge=1)
    chat_rate_limit_per_minute: int = Field(default=20, ge=1)
    search_rate_limit_per_minute: int = Field(default=60, ge=1)
    max_concurrent_chats: int = Field(default=4, ge=1)
    max_concurrent_chats_per_user: int = Field(default=1, ge=1)

    # --- Ollama ------------------------------------------------------------
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b-q4_K_M"
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_num_ctx: int = Field(default=8192, ge=512)
    llm_num_predict: int = Field(default=700, ge=16, le=8192)
    llm_connect_timeout_seconds: float = Field(default=5.0, gt=0)
    llm_read_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_answer_chars: int = Field(default=6000, ge=200)
    embedding_model: str = "embeddinggemma-2:270m"
    embedding_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Retrieval ---------------------------------------------------------
    chroma_dir: Path = Path("data/chroma")
    allow_draft: bool = False
    retrieval_k: int = Field(default=4, ge=1, le=10)
    min_retrieval_score: float = Field(default=0.35, ge=-1, le=1)
    max_history_messages: int = Field(default=6, ge=0, le=40)
    max_history_chars: int = Field(default=6000, ge=0)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("ollama_url")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_URL must start with http:// or https://")
        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_for_environment(self) -> Settings:
        """Reject configurations that are unsafe for the selected environment."""
        if self.environment != "production":
            return self
        problems = []
        if self.allow_draft:
            problems.append("ALLOW_DRAFT must be false: draft lessons are not fact-reviewed.")
        if self.seed_demo_user:
            problems.append("SEED_DEMO_USER must be false in production.")
        if "*" in self.cors_origins:
            problems.append("CORS_ORIGINS must list explicit origins, not '*'.")
        if any(origin.startswith("http://") for origin in self.cors_origins):
            problems.append("CORS_ORIGINS must use https:// origins in production.")
        if self.data_encryption_keys is None:
            problems.append("DATA_ENCRYPTION_KEYS is required to encrypt personal data.")
        if self.docs_enabled:
            problems.append("DOCS_ENABLED should be false in production.")
        if problems:
            raise ValueError("Unsafe production configuration: " + " ".join(problems))
        return self

    @property
    def show_docs(self) -> bool:
        if self.docs_enabled is not None:
            return self.docs_enabled
        return self.environment != "production"

    def resolve(self, path: Path) -> Path:
        """Resolve a configured relative path against the backend directory."""
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def allowed_review_statuses(self) -> frozenset[str]:
        return frozenset({"approved", "draft"} if self.allow_draft else {"approved"})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""
    return Settings()
