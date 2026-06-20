"""
OpenAI-compatible HTTP gateway built on FastAPI.

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible chat (streaming + non-streaming)
  GET  /v1/models            — List available models
  GET  /metrics              — Prometheus metrics
  GET  /ready                — Kubernetes readiness probe (fast, no external calls)
  GET  /live                 — Kubernetes liveness probe (always 200 if process alive)
  GET  /health               — Deep health check (probes each provider, ~5 s)
  GET  /stats                — JSON stats overview
  POST /v1/memory            — Store a memory
  GET  /v1/memory/search     — Search memories
  DELETE /v1/memory/{id}     — Delete a memory
  POST /v1/keys              — Create API key  (master key required)
  GET  /v1/keys              — List API keys   (master key required)
  DELETE /v1/keys/{id}       — Revoke API key  (master key required)
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

_DASHBOARD_HTML = Path(__file__).parent / "templates" / "dashboard.html"
_VERSION = "0.5.0"

from aicos.analytics.cost_tracker import CostTracker
from aicos.analytics.metrics import get_metrics
from aicos.api.middleware import RequestIDMiddleware
from aicos.api.rate_limiter import RateLimitMiddleware
from aicos.auth.api_keys import APIKey, APIKeyStore
from aicos.cache.semantic_cache import SemanticCache
from aicos.cache.sqlite_cache import SQLiteCache
from aicos.context.compressor import ContextCompressor
from aicos.context.history_manager import HistoryManager
from aicos.core.circuit_breaker import CircuitBreakerRegistry
from aicos.core.config import AICOSConfig, get_config
from aicos.core.gateway import AIGateway, GatewayRequest
from aicos.core.logging import configure_logging, get_logger
from aicos.core.router import MODEL_REGISTRY, ModelRouter
from aicos.core.telemetry import configure_telemetry
from aicos.memory.embeddings import EmbeddingEngine
from aicos.memory.memory_store import MemoryStore
from aicos.memory.retrieval import MemoryRetriever
from aicos.providers.base import StreamChunk

log = get_logger("api.routes")


# ── Request/Response Models ───────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str | list[Any]


class ChatCompletionRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage]
    max_tokens: int = Field(4096, ge=1, le=128_000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False
    session_id: Optional[str] = None
    skip_cache: bool = False
    skip_memory: bool = False
    skip_compression: bool = False


class MemoryStoreRequest(BaseModel):
    content: str
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=20)
    threshold: float = Field(0.3, ge=0.0, le=1.0)


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    scopes: list[str] = Field(default_factory=lambda: ["chat", "memory"])


# ── Application singletons ────────────────────────────────────────────────────

_gateway: AIGateway | None = None
_memory_store: MemoryStore | None = None
_key_store: APIKeyStore | None = None


async def _build_gateway(cfg: AICOSConfig) -> tuple[AIGateway, MemoryStore]:
    from aicos.providers.anthropic_provider import AnthropicProvider
    from aicos.providers.gemini_provider import GeminiProvider
    from aicos.providers.openai_provider import OpenAIProvider

    providers: dict[str, Any] = {}
    if cfg.openai_api_key:
        providers["openai"] = OpenAIProvider(api_key=cfg.openai_api_key)
    if cfg.anthropic_api_key:
        providers["anthropic"] = AnthropicProvider(api_key=cfg.anthropic_api_key)
    if cfg.gemini_api_key:
        providers["gemini"] = GeminiProvider(api_key=cfg.gemini_api_key)
    if cfg.openrouter_api_key:
        providers["openrouter"] = OpenAIProvider(
            api_key=cfg.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    if cfg.nvidia_api_key:
        providers["nvidia"] = OpenAIProvider(
            api_key=cfg.nvidia_api_key,
            base_url="https://integrate.api.nvidia.com/v1",
        )

    db_urls = cfg.get_db_urls()
    embedding_engine = EmbeddingEngine()

    sqlite_cache = SQLiteCache(
        database_url=db_urls["cache"],
        max_size=cfg.cache_max_size,
        ttl_seconds=cfg.cache_ttl_seconds,
    )
    await sqlite_cache.initialize()

    semantic_cache = SemanticCache(
        sqlite_cache=sqlite_cache,
        embedding_engine=embedding_engine,
        threshold=cfg.cache_similarity_threshold,
    )

    memory_store = MemoryStore(
        database_url=db_urls["memory"],
        embedding_engine=embedding_engine,
        max_items=cfg.memory_max_items,
    )
    await memory_store.initialize()

    memory_retriever = MemoryRetriever(
        store=memory_store,
        top_k=cfg.memory_injection_limit,
        threshold=cfg.memory_relevance_threshold,
    )

    compressor = ContextCompressor(max_tokens=cfg.max_context_tokens)
    history_manager = HistoryManager(
        compressor=compressor,
        max_tokens=cfg.max_context_tokens,
        litm_threshold_tokens=cfg.litm_threshold_tokens,
    )

    router = ModelRouter(cfg, embedding_engine=embedding_engine)

    cost_tracker = CostTracker(database_url=db_urls["cost"])
    await cost_tracker.initialize()

    circuit_breakers = CircuitBreakerRegistry(
        failure_threshold=cfg.circuit_breaker_failure_threshold,
        recovery_timeout=cfg.circuit_breaker_recovery_timeout,
    )

    gateway = AIGateway(
        config=cfg,
        router=router,
        providers=providers,
        semantic_cache=semantic_cache if cfg.cache_enabled else None,
        memory_retriever=memory_retriever if cfg.memory_enabled else None,
        history_manager=history_manager if cfg.context_compression_enabled else None,
        cost_tracker=cost_tracker,
        circuit_breakers=circuit_breakers,
    )

    log.info(
        "Gateway initialised",
        extra={
            "providers": list(providers.keys()),
            "cache": cfg.cache_enabled,
            "memory": cfg.memory_enabled,
            "db_mode": "postgresql" if cfg.database_url else "sqlite+wal",
        },
    )

    return gateway, memory_store


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(config: AICOSConfig | None = None) -> FastAPI:
    cfg = config or get_config()

    configure_logging(level=cfg.log_level, json_format=getattr(cfg, "log_json", False))
    configure_telemetry(
        otel_endpoint=getattr(cfg, "otel_endpoint", None),
        sentry_dsn=getattr(cfg, "sentry_dsn", None),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global _gateway, _memory_store, _key_store

        from aicos.core.database import build_engine
        from aicos.db.migrations import run_migrations

        # Run schema migrations before anything else opens connections
        db_urls = cfg.get_db_urls()
        _migration_engine = build_engine(db_urls["keys"])
        await run_migrations(_migration_engine)
        await _migration_engine.dispose()

        _gateway, _memory_store = await _build_gateway(cfg)

        _key_store = APIKeyStore(database_url=db_urls["keys"])
        await _key_store.initialize()

        log.info("AI-COS gateway started", extra={"version": _VERSION})
        yield

        if _memory_store:
            await _memory_store.close()
        if _key_store:
            await _key_store.close()
        if _gateway:
            await _gateway.close()
        _gateway = None
        _memory_store = None
        _key_store = None
        log.info("AI-COS gateway stopped")

    _NVIDIA_FAVICON = "https://www.nvidia.com/favicon.ico"

    app = FastAPI(
        title="AI-COS Gateway",
        description="OpenAI-compatible AI gateway with memory, caching, and routing",
        version=_VERSION,
        lifespan=lifespan,
        docs_url=None,
    )

    # ── Middleware (order matters: outermost runs first) ──────────────────
    app.add_middleware(
        RateLimitMiddleware,
        rpm=cfg.rate_limit_rpm,
        enabled=cfg.rate_limit_enabled,
        redis_url=cfg.redis_url,
    )
    app.add_middleware(RequestIDMiddleware)

    _wildcard_cors = cfg.cors_allowed_origins == ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_allowed_origins,
        allow_credentials=not _wildcard_cors,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Auth helpers ──────────────────────────────────────────────────────

    api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

    async def _check_api_key(
        authorization: str | None = Security(api_key_header),
    ) -> None:
        """Allow access with master key OR any valid per-user key."""
        if not cfg.gateway_api_key:
            return  # Open access — dev / single-user mode

        if not authorization:
            raise HTTPException(status_code=401, detail="API key required")

        token = authorization.removeprefix("Bearer ").strip()

        if token == cfg.gateway_api_key:
            return  # Master key

        if _key_store:
            key = await _key_store.validate(token)
            if key:
                return  # Valid per-user key

        raise HTTPException(status_code=403, detail="Invalid API key")

    def _check_master_key(
        authorization: str | None = Security(api_key_header),
    ) -> None:
        """Key management — master key only."""
        if not cfg.gateway_api_key:
            raise HTTPException(status_code=501, detail="Master key not configured")
        if not authorization:
            raise HTTPException(status_code=401, detail="Authorization required")
        token = authorization.removeprefix("Bearer ").strip()
        if token != cfg.gateway_api_key:
            raise HTTPException(status_code=403, detail="Master key required")

    def _get_gateway() -> AIGateway:
        if _gateway is None:
            raise HTTPException(status_code=503, detail="Gateway not initialized")
        return _gateway

    # ── Static routes ─────────────────────────────────────────────────────

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> RedirectResponse:
        return RedirectResponse(url=_NVIDIA_FAVICON)

    @app.get("/docs", include_in_schema=False)
    async def swagger_ui() -> Any:
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title="AI-COS Gateway",
            swagger_favicon_url=_NVIDIA_FAVICON,
        )

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> FileResponse:
        return FileResponse(_DASHBOARD_HTML)

    # ── Health & Observability ────────────────────────────────────────────

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        """
        Kubernetes readiness probe — sub-millisecond, no external calls.
        Returns 200 only after the gateway has fully initialised.
        Use this as the readinessProbe target in pod specs.
        """
        if _gateway is None:
            raise HTTPException(status_code=503, detail="Gateway not ready")
        return {"status": "ready"}

    @app.get("/live")
    async def live() -> dict[str, str]:
        """
        Kubernetes liveness probe — always 200 if the process is alive.
        Use this as the livenessProbe target in pod specs.
        """
        return {"status": "alive", "version": _VERSION}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """
        Deep health check — probes each configured provider (5-second timeout).
        Returns status='ok' if at least one provider responds, 'degraded' otherwise.
        Use this for monitoring dashboards, not for Kubernetes probes.
        """
        provider_health: dict[str, str] = {}
        gw = _gateway

        if gw is not None and isinstance(getattr(gw, "_providers", None), dict):
            async def _probe(name: str, provider: Any) -> tuple[str, str]:
                try:
                    ok = await asyncio.wait_for(provider.is_available(), timeout=5.0)
                    return name, "ok" if ok else "degraded"
                except asyncio.TimeoutError:
                    return name, "timeout"
                except Exception:
                    return name, "error"

            results = await asyncio.gather(
                *[_probe(n, p) for n, p in gw._providers.items()]
            )
            provider_health = dict(results)

        any_ok = any(v == "ok" for v in provider_health.values())
        status = "ok" if (any_ok or not provider_health) else "degraded"

        circuit_status = (
            gw._breakers.all_status() if gw and hasattr(gw, "_breakers") else []
        )

        return {
            "status": status,
            "version": _VERSION,
            "providers": provider_health,
            "circuits": circuit_status,
            "cache_enabled": cfg.cache_enabled,
            "memory_enabled": cfg.memory_enabled,
            "db_mode": "postgresql" if cfg.database_url else "sqlite+wal",
        }

    @app.get("/metrics")
    async def metrics() -> Any:
        return StreamingResponse(
            iter([get_metrics().to_prometheus()]),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return get_metrics().to_dict()

    # ── Chat Completions ──────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request,
        body: ChatCompletionRequest,
        _auth: None = Security(_check_api_key),
    ) -> Any:
        gateway = _get_gateway()
        messages = [m.model_dump() for m in body.messages]
        model = None if body.model in ("auto", "") else body.model

        gateway_request = GatewayRequest(
            messages=messages,
            model=model,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            stream=body.stream,
            session_id=body.session_id or str(uuid.uuid4()),
            skip_cache=body.skip_cache,
            skip_memory=body.skip_memory,
            skip_compression=body.skip_compression,
        )

        if body.stream:
            return EventSourceResponse(
                _stream_response(gateway, gateway_request, model or "auto"),
                media_type="text/event-stream",
            )

        response = await gateway.process(gateway_request)
        return _format_completion_response(response, body.messages)

    async def _stream_response(
        gateway: AIGateway,
        request: GatewayRequest,
        model: str,
    ) -> AsyncIterator[str]:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        try:
            async for chunk in gateway.stream(request):
                if not chunk.delta and not chunk.finish_reason:
                    continue

                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": chunk.model or model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.delta} if chunk.delta else {},
                            "finish_reason": chunk.finish_reason,
                        }
                    ],
                }
                yield json.dumps(data)

            yield "[DONE]"

        except Exception as exc:
            log.error("Streaming error", extra={"error": str(exc)}, exc_info=True)
            yield json.dumps({
                "error": {
                    "message": str(exc),
                    "type": "stream_error",
                    "code": "provider_error",
                }
            })

    def _format_completion_response(
        response: Any,
        original_messages: list[ChatMessage],
    ) -> dict[str, Any]:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": response.input_tokens,
                "completion_tokens": response.output_tokens,
                "total_tokens": response.input_tokens + response.output_tokens,
            },
            "aicos": {
                "cache_hit": response.cache_hit,
                "cache_type": response.cache_hit_type,
                "task_type": response.task_type,
                "tokens_before_compression": response.tokens_before_compression,
                "tokens_after_compression": response.tokens_after_compression,
                "compression_savings_pct": round(
                    (1 - response.tokens_after_compression / max(response.tokens_before_compression, 1)) * 100,
                    1,
                ),
                "memories_injected": response.memories_injected,
                "cost_usd": response.cost_usd,
                "latency_ms": round(response.latency_ms, 2),
                "routing": response.routing_reason,
            },
        }

    # ── Models ────────────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        available = cfg.available_providers()
        models = [
            {
                "id": model_id,
                "object": "model",
                "provider": spec.provider,
                "tier": spec.tier,
                "capabilities": list(spec.capabilities),
                "max_tokens": spec.max_tokens,
                "cost_per_1m_input_usd": spec.input_cost_per_1m,
                "cost_per_1m_output_usd": spec.output_cost_per_1m,
            }
            for model_id, spec in MODEL_REGISTRY.items()
            if spec.provider in available
        ]
        return {"object": "list", "data": models}

    # ── Memory API ────────────────────────────────────────────────────────

    @app.post("/v1/memory")
    async def store_memory(
        request: MemoryStoreRequest,
        _auth: None = Security(_check_api_key),
    ) -> dict[str, Any]:
        if _memory_store is None:
            raise HTTPException(status_code=503, detail="Memory store not initialized")
        memory_id = await _memory_store.store(
            content=request.content,
            tags=request.tags,
            metadata=request.metadata,
        )
        get_metrics().memory_stored.inc()
        return {"id": memory_id, "status": "stored"}

    @app.get("/v1/memory/search")
    async def search_memory(
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> dict[str, Any]:
        if _memory_store is None:
            raise HTTPException(status_code=503, detail="Memory store not initialized")
        results = await _memory_store.search(query, top_k=top_k, threshold=threshold)
        return {
            "query": query,
            "results": [
                {
                    "id": item.id,
                    "content": item.content,
                    "score": round(score, 4),
                    "tags": item.tag_list,
                    "created_at": item.created_at.isoformat(),
                }
                for item, score in results
            ],
        }

    @app.delete("/v1/memory/{memory_id}")
    async def delete_memory(
        memory_id: int,
        _auth: None = Security(_check_api_key),
    ) -> dict[str, Any]:
        if _memory_store is None:
            raise HTTPException(status_code=503, detail="Memory store not initialized")
        deleted = await _memory_store.forget(memory_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        return {"id": memory_id, "status": "deleted"}

    # ── API Key Management ────────────────────────────────────────────────

    @app.post("/v1/keys", status_code=201)
    async def create_api_key(
        body: CreateKeyRequest,
        _auth: None = Security(_check_master_key),
    ) -> dict[str, Any]:
        """
        Create a new per-user API key.
        The plaintext key is returned ONCE and never stored — save it immediately.
        """
        if _key_store is None:
            raise HTTPException(status_code=503, detail="Key store not initialized")
        plaintext, key = await _key_store.create_key(name=body.name, scopes=body.scopes)
        log.info("API key created", extra={"key_id": key.id, "key_name": key.name})
        return {
            "key": plaintext,
            "id": key.id,
            "name": key.name,
            "prefix": key.prefix,
            "scopes": key.scopes,
            "created_at": key.created_at.isoformat(),
            "warning": "Store this key securely — it will not be shown again.",
        }

    @app.get("/v1/keys")
    async def list_api_keys(
        _auth: None = Security(_check_master_key),
    ) -> dict[str, Any]:
        if _key_store is None:
            raise HTTPException(status_code=503, detail="Key store not initialized")
        keys = await _key_store.list_keys()
        return {
            "keys": [
                {
                    "id": k.id,
                    "name": k.name,
                    "prefix": k.prefix,
                    "scopes": k.scopes,
                    "created_at": k.created_at.isoformat(),
                    "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                }
                for k in keys
            ]
        }

    @app.delete("/v1/keys/{key_id}")
    async def revoke_api_key(
        key_id: int,
        _auth: None = Security(_check_master_key),
    ) -> dict[str, Any]:
        if _key_store is None:
            raise HTTPException(status_code=503, detail="Key store not initialized")
        revoked = await _key_store.revoke(key_id)
        if not revoked:
            raise HTTPException(status_code=404, detail=f"Key {key_id} not found")
        log.info("API key revoked", extra={"key_id": key_id})
        return {"id": key_id, "status": "revoked"}

    return app
