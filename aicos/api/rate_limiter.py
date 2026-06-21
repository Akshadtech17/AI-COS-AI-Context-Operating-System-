"""
Distributed rate limiter middleware.

When Redis is configured (AICOS_REDIS_URL set): uses a Redis sorted-set sliding
window. Each client IP gets a key; members are timestamps. Expired members are
pruned on every request.

When Redis is NOT configured: falls back to an in-process sliding window.
Note: in-process rate limiting does NOT work correctly with multiple workers —
each process has its own counter. Configure Redis for multi-worker deployments.

Returns HTTP 429 with a Retry-After header on limit exceeded.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from aicos.core.logging import get_logger

log = get_logger("api.rate_limiter")

# Only rate-limit these paths (all others pass through)
_RATE_LIMITED_PATHS = {"/v1/chat/completions", "/v1/memory"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.

    Redis mode: O(log N) per request, shared across workers.
    In-memory mode: O(N) per request (N = requests in window), single-process only.
    """

    def __init__(
        self,
        app,
        rpm: int = 60,
        enabled: bool = True,
        redis_url: str | None = None,
    ) -> None:
        super().__init__(app)
        self._rpm = rpm
        self._enabled = enabled
        self._window = 60.0  # 1 minute
        self._redis: object | None = None
        self._redis_url = redis_url
        self._local: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def _ensure_redis(self) -> None:
        if self._redis_url and self._redis is None:
            try:
                import redis.asyncio as aioredis  # type: ignore[import]

                self._redis = await aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                )
                log.info("Rate limiter connected to Redis")
            except Exception as exc:
                log.warning(
                    "Redis unavailable for rate limiting, falling back to in-process",
                    extra={"error": str(exc)},
                )
                self._redis_url = None  # Stop retrying

    async def _redis_check(self, key: str) -> bool:
        try:
            r = self._redis
            now = time.time()
            window_start = now - self._window
            full_key = f"rl:{key}"

            pipe = r.pipeline()  # type: ignore[union-attr]
            pipe.zremrangebyscore(full_key, 0, window_start)
            pipe.zadd(full_key, {str(now): now})
            pipe.zcard(full_key)
            pipe.expire(full_key, int(self._window) + 1)
            results = await pipe.execute()

            count: int = results[2]
            return count <= self._rpm
        except Exception as exc:
            log.warning(
                "Redis rate limit check failed, allowing request", extra={"error": str(exc)}
            )
            return True  # Fail open on Redis error

    async def _local_check(self, key: str) -> bool:
        now = time.time()
        cutoff = now - self._window
        async with self._lock:
            timestamps = self._local[key]
            # Prune expired entries
            self._local[key] = [t for t in timestamps if t > cutoff]
            if len(self._local[key]) >= self._rpm:
                return False
            self._local[key].append(now)
            return True

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enabled or request.url.path not in _RATE_LIMITED_PATHS:
            return await call_next(request)

        await self._ensure_redis()

        client_ip = (request.client.host if request.client else None) or "unknown"
        allowed = (
            await self._redis_check(client_ip)
            if self._redis
            else await self._local_check(client_ip)
        )

        if not allowed:
            log.warning("Rate limit exceeded", extra={"ip": client_ip, "path": request.url.path})
            return JSONResponse(
                status_code=429,
                content={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}},
                headers={"Retry-After": str(int(self._window))},
            )

        return await call_next(request)
