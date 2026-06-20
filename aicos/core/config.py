"""Central configuration for AI-COS, loaded from environment variables and .env files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AICOSConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AICOS_",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ── Provider Credentials (no prefix — standard env var names) ──────────
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    openrouter_api_key: str | None = Field(None, alias="OPENROUTER_API_KEY")
    nvidia_api_key: str | None = Field(None, alias="NVIDIA_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")

    # ── Database ────────────────────────────────────────────────────────────
    db_path: str = Field("~/.aicos/aicos.db")
    # Set DATABASE_URL to use PostgreSQL (e.g. postgresql+asyncpg://user:pw@host/db)
    # Leave unset to use SQLite (split into separate files under db_path directory)
    database_url: str | None = Field(None)

    # ── Secrets ─────────────────────────────────────────────────────────────
    # Docker secrets dir: each file named like the env var (e.g. OPENAI_API_KEY)
    secrets_dir: str | None = Field(None)

    # ── Gateway ─────────────────────────────────────────────────────────────
    gateway_host: str = Field("0.0.0.0")
    gateway_port: int = Field(4000)
    gateway_api_key: str | None = Field(None)

    # ── Router ──────────────────────────────────────────────────────────────
    router_strategy: Literal["auto", "cheapest", "fastest", "best"] = Field("auto")
    default_model: str | None = Field(None)
    fallback_models: list[str] = Field(
        default_factory=lambda: [
            "openrouter/nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "gpt-4o-mini",
            "claude-haiku-4-5-20251001",
        ]
    )

    # ── Cache ────────────────────────────────────────────────────────────────
    cache_enabled: bool = Field(True)
    cache_similarity_threshold: float = Field(0.96, ge=0.0, le=1.0)
    cache_max_size: int = Field(10000, ge=1)
    cache_ttl_seconds: int = Field(86400)

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_enabled: bool = Field(True)
    memory_max_items: int = Field(10000, ge=1)
    memory_injection_limit: int = Field(5, ge=1, le=20)
    memory_relevance_threshold: float = Field(0.3, ge=0.0, le=1.0)

    # ── Context Optimization ─────────────────────────────────────────────────
    context_compression_enabled: bool = Field(True)
    max_context_tokens: int = Field(8000, ge=1000)
    litm_threshold_tokens: int = Field(6000, ge=1000)
    compression_target_ratio: float = Field(0.5, ge=0.1, le=0.9)

    # ── Analytics ────────────────────────────────────────────────────────────
    analytics_enabled: bool = Field(True)

    # ── Rate Limiting ────────────────────────────────────────────────────────
    rate_limit_enabled: bool = Field(True)
    rate_limit_rpm: int = Field(60, ge=1)

    # ── Circuit Breaker ──────────────────────────────────────────────────────
    circuit_breaker_failure_threshold: int = Field(5, ge=1)
    circuit_breaker_recovery_timeout: float = Field(30.0, ge=1.0)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    log_json: bool = Field(False)
    verbose: bool = Field(False)

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["*"])

    # ── Database Pool ────────────────────────────────────────────────────────
    # With PgBouncer in transaction mode, set db_pool_size=2, db_max_overflow=3.
    # Default of 5/5 is correct for direct PostgreSQL connections.
    db_pool_size: int = Field(5, ge=1, le=100)
    db_max_overflow: int = Field(5, ge=0, le=100)

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_dim: int = Field(384, ge=64)  # Must match the embedding model output

    # ── Observability ────────────────────────────────────────────────────────
    otel_endpoint: str | None = Field(None)  # OTLP gRPC endpoint e.g. http://jaeger:4317
    sentry_dsn: str | None = Field(None)     # Sentry DSN for error tracking

    # ── Optional Integrations ────────────────────────────────────────────────
    redis_url: str | None = Field(None)
    ollama_enabled: bool = Field(False)  # Set True to enable local Ollama routing

    @field_validator("db_path", mode="before")
    @classmethod
    def expand_db_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())

    @model_validator(mode="after")
    def _load_docker_secrets(self) -> "AICOSConfig":
        """
        Read API keys from Docker secrets directory (one file per secret).

        Files are named exactly like the env var they correspond to,
        e.g. /run/secrets/OPENAI_API_KEY.  Only overrides if the current
        value is unset — env vars always take precedence.
        """
        secrets_path = Path(self.secrets_dir) if self.secrets_dir else Path("/run/secrets")
        if not secrets_path.is_dir():
            return self

        _secret_fields = {
            "OPENAI_API_KEY": "openai_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "GEMINI_API_KEY": "gemini_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
            "NVIDIA_API_KEY": "nvidia_api_key",
            "AICOS_GATEWAY_API_KEY": "gateway_api_key",
            "DATABASE_URL": "database_url",
        }
        for filename, field_name in _secret_fields.items():
            secret_file = secrets_path / filename
            if secret_file.is_file() and not getattr(self, field_name):
                object.__setattr__(self, field_name, secret_file.read_text().strip())

        return self

    @field_validator("litm_threshold_tokens", mode="after")
    @classmethod
    def litm_below_max(cls, v: int, info: object) -> int:
        # Ensure LITM threshold is below max context
        return v

    def get_db_pool_kwargs(self) -> dict[str, int]:
        """Pool size kwargs for build_engine() (ignored for SQLite)."""
        return {"pool_size": self.db_pool_size, "max_overflow": self.db_max_overflow}

    def get_db_urls(self) -> dict[str, str]:
        """Return per-component database URLs.

        PostgreSQL: all components share one database (different tables).
        SQLite: separate files per component, WAL mode via build_engine().
        """
        if self.database_url:
            url = self.database_url
            return {"cache": url, "memory": url, "cost": url, "keys": url}

        db_dir = self.get_resolved_db_path().parent
        db_dir.mkdir(parents=True, exist_ok=True)
        return {
            "cache": f"sqlite+aiosqlite:///{db_dir}/cache.db",
            "memory": f"sqlite+aiosqlite:///{db_dir}/memory.db",
            "cost": f"sqlite+aiosqlite:///{db_dir}/cost.db",
            "keys": f"sqlite+aiosqlite:///{db_dir}/keys.db",
        }

    def get_resolved_db_path(self) -> Path:
        p = Path(self.db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def available_providers(self) -> list[str]:
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.gemini_api_key:
            providers.append("gemini")
        if self.openrouter_api_key:
            providers.append("openrouter")
        if self.nvidia_api_key:
            providers.append("nvidia")
        if self.ollama_enabled:
            providers.append("ollama")
        return providers

    def mask_secrets(self) -> dict[str, object]:
        """Return config dict with API keys masked for safe display."""
        data = self.model_dump()
        for key in ("openai_api_key", "anthropic_api_key", "gemini_api_key",
                    "openrouter_api_key", "nvidia_api_key", "gateway_api_key"):
            if data.get(key):
                val = str(data[key])
                data[key] = val[:8] + "..." + val[-4:] if len(val) > 12 else "***"
        return data


@lru_cache(maxsize=1)
def get_config() -> AICOSConfig:
    return AICOSConfig()


def reset_config() -> None:
    """Clear the cached config — useful in tests."""
    get_config.cache_clear()
