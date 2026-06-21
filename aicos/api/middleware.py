"""
Request ID middleware — stamps every request with a correlation ID.

Reads X-Request-ID or X-Correlation-ID from the incoming request header
(or generates a new UUID). Stores the ID in a ContextVar so structured
log calls anywhere in the same async task automatically include it.
The ID is also echoed back in the response header.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from aicos.core.logging import get_logger, set_request_id

log = get_logger("api.middleware")


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-Correlation-ID")
            or str(uuid.uuid4())
        )
        set_request_id(request_id)

        t0 = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - t0) * 1000

        response.headers["X-Request-ID"] = request_id

        log.info(
            "HTTP %s %s %d",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response
