"""
OpenAI-compatible HTTP gateway built on FastAPI.

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible chat (streaming + non-streaming)
  GET  /v1/models            — List available models
  GET  /metrics              — Prometheus metrics
  GET  /health               — Health check
  GET  /stats                — JSON stats overview
  POST /v1/memory            — Store a memory
  GET  /v1/memory/search     — Search memories
  DELETE /v1/memory/{id}     — Delete a memory
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from aicos.analytics.cost_tracker import CostTracker
from aicos.analytics.metrics import get_metrics
from aicos.cache.semantic_cache import SemanticCache
from aicos.cache.sqlite_cache import SQLiteCache
from aicos.context.compressor import ContextCompressor
from aicos.context.history_manager import HistoryManager
from aicos.core.config import AICOSConfig, get_config
from aicos.core.gateway import AIGateway, GatewayRequest
from aicos.core.router import MODEL_REGISTRY, ModelRouter
from aicos.memory.embeddings import EmbeddingEngine
from aicos.memory.memory_store import MemoryStore
from aicos.memory.retrieval import MemoryRetriever
from aicos.providers.base import StreamChunk


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


# ── Gateway singleton ─────────────────────────────────────────────────────────

_gateway: AIGateway | None = None
_memory_store: MemoryStore | None = None


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

    db_path = cfg.get_resolved_db_path()
    embedding_engine = EmbeddingEngine()

    sqlite_cache = SQLiteCache(
        db_path=db_path.parent / "cache.db",
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
        db_path=db_path.parent / "memory.db",
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

    router = ModelRouter(cfg)
    gateway = AIGateway(
        config=cfg,
        router=router,
        providers=providers,
        semantic_cache=semantic_cache if cfg.cache_enabled else None,
        memory_retriever=memory_retriever if cfg.memory_enabled else None,
        history_manager=history_manager if cfg.context_compression_enabled else None,
        cost_tracker=CostTracker(),
    )

    return gateway, memory_store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _gateway, _memory_store
    cfg = get_config()
    _gateway, _memory_store = await _build_gateway(cfg)
    yield
    if _memory_store:
        await _memory_store.close()


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(config: AICOSConfig | None = None) -> FastAPI:
    cfg = config or get_config()

    limiter = Limiter(key_func=get_remote_address)

    _NVIDIA_FAVICON = "https://www.nvidia.com/favicon.ico"

    app = FastAPI(
        title="AI-COS Gateway",
        description="OpenAI-compatible AI gateway with memory, caching, and routing",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,  # replaced by custom endpoint below
    )

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

    app.state.limiter = limiter

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

    def _check_api_key(authorization: str | None = Security(api_key_header)) -> None:
        if not cfg.gateway_api_key:
            return
        if not authorization:
            raise HTTPException(status_code=401, detail="API key required")
        token = authorization.removeprefix("Bearer ").strip()
        if token != cfg.gateway_api_key:
            raise HTTPException(status_code=403, detail="Invalid API key")

    def _get_gateway() -> AIGateway:
        if _gateway is None:
            raise HTTPException(status_code=503, detail="Gateway not initialized")
        return _gateway

    # ── Health & Metrics ──────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.1.0",
            "providers": cfg.available_providers(),
            "cache_enabled": cfg.cache_enabled,
            "memory_enabled": cfg.memory_enabled,
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
        request: ChatCompletionRequest,
        _auth: None = Security(_check_api_key),
    ) -> Any:
        gateway = _get_gateway()
        messages = [m.model_dump() for m in request.messages]
        model = None if request.model in ("auto", "") else request.model

        gateway_request = GatewayRequest(
            messages=messages,
            model=model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream,
            session_id=request.session_id or str(uuid.uuid4()),
            skip_cache=request.skip_cache,
            skip_memory=request.skip_memory,
            skip_compression=request.skip_compression,
        )

        if request.stream:
            return EventSourceResponse(
                _stream_response(gateway, gateway_request, model or "auto"),
                media_type="text/event-stream",
            )

        response = await gateway.process(gateway_request)
        return _format_completion_response(response, request.messages)

    async def _stream_response(
        gateway: AIGateway,
        request: GatewayRequest,
        model: str,
    ) -> AsyncIterator[str]:
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

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

    return app
