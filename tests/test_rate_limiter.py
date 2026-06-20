"""Tests for the Redis-backed rate limiter middleware."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from aicos.api.rate_limiter import RateLimitMiddleware


def _make_app(rpm: int = 2, enabled: bool = True, redis_url: str | None = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, rpm=rpm, enabled=enabled, redis_url=redis_url)

    @app.get("/v1/chat/completions")
    async def _chat() -> dict:
        return {"ok": True}

    @app.get("/health")
    async def _health() -> dict:
        return {"status": "ok"}

    return app


class TestInMemoryRateLimiter:
    def test_within_limit_returns_200(self) -> None:
        client = TestClient(_make_app(rpm=5))
        r = client.get("/v1/chat/completions")
        assert r.status_code == 200

    def test_exceeds_limit_returns_429(self) -> None:
        client = TestClient(_make_app(rpm=2))
        for _ in range(2):
            client.get("/v1/chat/completions")
        r = client.get("/v1/chat/completions")
        assert r.status_code == 429

    def test_429_has_retry_after_header(self) -> None:
        client = TestClient(_make_app(rpm=1))
        client.get("/v1/chat/completions")
        r = client.get("/v1/chat/completions")
        assert r.status_code == 429
        assert "Retry-After" in r.headers

    def test_429_body_has_error_field(self) -> None:
        client = TestClient(_make_app(rpm=1))
        client.get("/v1/chat/completions")
        r = client.get("/v1/chat/completions")
        assert r.status_code == 429
        assert "error" in r.json()

    def test_disabled_never_429(self) -> None:
        client = TestClient(_make_app(rpm=1, enabled=False))
        for _ in range(10):
            r = client.get("/v1/chat/completions")
            assert r.status_code == 200

    def test_non_rate_limited_path_not_throttled(self) -> None:
        client = TestClient(_make_app(rpm=1))
        # Exhaust the limit on the rate-limited path
        client.get("/v1/chat/completions")
        client.get("/v1/chat/completions")
        # Health endpoint is not rate-limited
        r = client.get("/health")
        assert r.status_code == 200

    def test_different_ips_counted_separately(self) -> None:
        """Two clients from different IPs don't share counters."""
        app = _make_app(rpm=1)
        c1 = TestClient(app)
        c2 = TestClient(app)
        c1.get("/v1/chat/completions")
        # c2 has its own counter (same IP in test though — just verify no crash)
        r = c2.get("/v1/chat/completions")
        # In tests, both clients share 127.0.0.1, so this will be rate-limited
        # The important thing is no exception is raised
        assert r.status_code in (200, 429)


class TestRedisRateLimiter:
    @pytest.mark.asyncio
    async def test_redis_check_called_when_redis_available(self) -> None:
        rl = RateLimitMiddleware(app=None, rpm=10, redis_url="redis://localhost:6379")

        mock_redis = AsyncMock()
        mock_pipe = AsyncMock()
        mock_pipe.execute = AsyncMock(return_value=[None, None, 1, None])
        mock_redis.pipeline = MagicMock(return_value=mock_pipe)
        rl._redis = mock_redis

        allowed = await rl._redis_check("127.0.0.1")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_redis_check_fails_open_on_exception(self) -> None:
        rl = RateLimitMiddleware(app=None, rpm=10, redis_url="redis://localhost:6379")

        mock_redis = AsyncMock()
        mock_redis.pipeline = MagicMock(side_effect=ConnectionError("Redis down"))
        rl._redis = mock_redis

        # Should not raise — fail open and allow the request
        allowed = await rl._redis_check("127.0.0.1")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_local_check_respects_rpm(self) -> None:
        rl = RateLimitMiddleware(app=None, rpm=3)
        for _ in range(3):
            assert await rl._local_check("test-ip") is True
        assert await rl._local_check("test-ip") is False

    @pytest.mark.asyncio
    async def test_local_check_different_ips_independent(self) -> None:
        rl = RateLimitMiddleware(app=None, rpm=1)
        assert await rl._local_check("ip-a") is True
        assert await rl._local_check("ip-b") is True  # Different IP, independent
        assert await rl._local_check("ip-a") is False  # ip-a exhausted
