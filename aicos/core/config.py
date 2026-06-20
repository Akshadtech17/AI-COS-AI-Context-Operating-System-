"""Central configuration for AI-COS, loaded from environment variables and .env files."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
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

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    verbose: bool = Field(False)

    # ── Optional Integrations ────────────────────────────────────────────────
    redis_url: str | None = Field(None)

    @field_validator("db_path", mode="before")
    @classmethod
    def expand_db_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())

    @field_validator("litm_threshold_tokens", mode="after")
    @classmethod
    def litm_below_max(cls, v: int, info: object) -> int:
        # Ensure LITM threshold is below max context
        return v

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
        providers.append("ollama")  # Always available if running locally
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
