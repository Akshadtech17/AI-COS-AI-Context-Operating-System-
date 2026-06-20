"""
AI Gateway — the core processing pipeline.

Every request flows through:
  Cache lookup → Memory injection → Context optimization →
  Model routing → LLM call → Analytics recording → Cache store

Implements OpenAI-compatible request/response models.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from aicos.analytics.cost_tracker import CostTracker
from aicos.analytics.metrics import get_metrics
from aicos.cache.semantic_cache import SemanticCache
from aicos.context.compressor import ContextCompressor, count_message_tokens
from aicos.context.history_manager import HistoryManager
from aicos.core.circuit_breaker import CircuitBreakerRegistry
from aicos.core.config import AICOSConfig
from aicos.core.logging import get_logger, set_session_id
from aicos.core.router import ModelRouter, RoutingDecision
from aicos.memory.retrieval import MemoryRetriever
from aicos.providers.base import BaseProvider, ProviderResponse, StreamChunk

log = get_logger("core.gateway")


@dataclass
class GatewayRequest:
    messages: list[dict[str, Any]]
    model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    stream: bool = False
    session_id: str | None = None
    skip_cache: bool = False
    skip_memory: bool = False
    skip_compression: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GatewayResponse:
    content: str
    model: str
    provider: str
    task_type: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cache_hit: bool
    cache_hit_type: str | None
    tokens_before_compression: int
    tokens_after_compression: int
    memories_injected: int
    routing_reason: str


class AIGateway:
    """
    Central request dispatcher. Wires together all AI-COS subsystems.
    Thread-safe; intended to be used as a singleton.
    """

    def __init__(
        self,
        config: AICOSConfig,
        router: ModelRouter,
        providers: dict[str, BaseProvider],
        semantic_cache: SemanticCache | None = None,
        memory_retriever: MemoryRetriever | None = None,
        history_manager: HistoryManager | None = None,
        cost_tracker: CostTracker | None = None,
        circuit_breakers: CircuitBreakerRegistry | None = None,
    ) -> None:
        self._config = config
        self._router = router
        self._providers = providers
        self._cache = semantic_cache
        self._memory = memory_retriever
        self._history = history_manager
        self._cost_tracker = cost_tracker or CostTracker()
        self._breakers = circuit_breakers or CircuitBreakerRegistry()
        self._compressor = ContextCompressor(
            max_tokens=config.max_context_tokens,
            model="gpt-4o-mini",
        )
        self._metrics = get_metrics()

    async def close(self) -> None:
        if self._cost_tracker:
            await self._cost_tracker.close()

    def _context_hash(self, messages: list[dict[str, Any]]) -> str:
        """Hash conversation context for cache keying."""
        serialized = str([(m.get("role"), m.get("content", "")[:200]) for m in messages[:-1]])
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def _get_provider(self, provider_name: str) -> BaseProvider:
        if provider_name in self._providers:
            return self._providers[provider_name]
        # Try any available provider as fallback
        if self._providers:
            return next(iter(self._providers.values()))
        raise RuntimeError("No providers configured. Set an API key in .env")

    async def process(self, request: GatewayRequest) -> GatewayResponse:
        """
        Full synchronous request pipeline.
        Raises on unrecoverable errors after all fallbacks are exhausted.
        """
        t_start = time.perf_counter()
        metrics = self._metrics

        if request.session_id:
            set_session_id(request.session_id)

        log.info(
            "Gateway request",
            extra={
                "n_messages": len(request.messages),
                "model_hint": request.model,
                "skip_cache": request.skip_cache,
            },
        )

        # ── Step 1: Routing ───────────────────────────────────────────────
        decision = self._router.select_model(
            messages=request.messages,
            override_model=request.model,
        )

        # ── Step 2: Memory injection ──────────────────────────────────────
        messages = list(request.messages)
        memories_injected = 0

        if self._memory and self._config.memory_enabled and not request.skip_memory:
            t_mem = time.perf_counter()
            messages = await self._memory.inject_into_messages(messages)
            mem_count = len(messages) - len(request.messages)
            memories_injected = max(0, mem_count)
            metrics.record_stage_latency(
                "memory_retrieval", (time.perf_counter() - t_mem) * 1000
            )
            metrics.memory_retrieved.inc(memories_injected)

        # ── Step 3: Context optimization ──────────────────────────────────
        tokens_before = count_message_tokens(messages)
        tokens_after = tokens_before

        if self._config.context_compression_enabled and not request.skip_compression:
            t_compress = time.perf_counter()

            if self._history:
                history_state = await self._history.process(messages)
                messages = history_state.messages
            else:
                result = self._compressor.compress(messages)
                messages = result.messages

            tokens_after = count_message_tokens(messages)
            compress_ms = (time.perf_counter() - t_compress) * 1000
            metrics.record_stage_latency("context_compression", compress_ms)

            if tokens_before != tokens_after:
                metrics.record_compression(tokens_before, tokens_after)

        # ── Step 4: Cache lookup ──────────────────────────────────────────
        cache_hit = False
        cache_hit_type: str | None = None
        last_user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_content = str(m.get("content", ""))
                break

        if self._cache and self._config.cache_enabled and not request.skip_cache:
            t_cache = time.perf_counter()
            context_hash = self._context_hash(messages)
            cache_result = await self._cache.get(last_user_content, context_hash)
            cache_ms = (time.perf_counter() - t_cache) * 1000
            metrics.record_stage_latency("cache_lookup", cache_ms)

            if cache_result:
                total_ms = (time.perf_counter() - t_start) * 1000
                log.info(
                    "Cache hit",
                    extra={
                        "hit_type": cache_result.hit_type,
                        "similarity": round(cache_result.similarity, 4),
                        "latency_ms": round(total_ms, 2),
                        "model": decision.model,
                    },
                )
                savings = self._cost_tracker.compute_savings(
                    decision.model,
                    tokens_after,
                    cache_result.source_entry.output_tokens if cache_result.source_entry else 100,
                )
                metrics.cache_hits.inc()
                metrics.record_cost_saved(savings)

                return GatewayResponse(
                    content=cache_result.response,
                    model=decision.model,
                    provider=decision.provider,
                    task_type=decision.task_type.value,
                    input_tokens=tokens_after,
                    output_tokens=cache_result.source_entry.output_tokens if cache_result.source_entry else 0,
                    cost_usd=0.0,
                    latency_ms=total_ms,
                    cache_hit=True,
                    cache_hit_type=cache_result.hit_type,
                    tokens_before_compression=tokens_before,
                    tokens_after_compression=tokens_after,
                    memories_injected=memories_injected,
                    routing_reason=decision.reasoning,
                )
            else:
                metrics.cache_misses.inc()

        # ── Step 5: LLM call with failover ────────────────────────────────
        t_llm = time.perf_counter()
        provider_response: ProviderResponse | None = None
        used_model = decision.model
        used_provider = decision.provider

        models_to_try = [decision.model] + decision.fallback_models

        for model_attempt in models_to_try:
            spec_model = model_attempt
            from aicos.core.router import MODEL_REGISTRY
            spec = MODEL_REGISTRY.get(model_attempt)
            if not spec:
                continue

            provider = self._providers.get(spec.provider)
            if not provider:
                continue

            breaker = self._breakers.get(spec.provider)
            if not breaker.can_attempt():
                log.warning(
                    "Circuit open — skipping provider",
                    extra={"provider": spec.provider, "model": model_attempt},
                )
                if model_attempt == models_to_try[-1]:
                    raise RuntimeError(
                        f"All providers circuit-open or failed. Last: {spec.provider}"
                    )
                continue

            log.info(
                "LLM call",
                extra={"provider": spec.provider, "model": model_attempt},
            )
            try:
                provider_response = await provider.complete(
                    messages=messages,
                    model=model_attempt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    **request.extra,
                )
                breaker.record_success()
                used_model = model_attempt
                used_provider = spec.provider
                break
            except Exception as e:
                breaker.record_failure()
                log.warning(
                    "Provider failed, trying fallback",
                    extra={
                        "provider": spec.provider,
                        "model": model_attempt,
                        "circuit_state": breaker.state.value,
                        "error": str(e),
                    },
                )
                if model_attempt == models_to_try[-1]:
                    log.error(
                        "All providers exhausted",
                        extra={"last_error": str(e)},
                    )
                    raise RuntimeError(
                        f"All providers failed. Last error: {e}"
                    ) from e
                continue

        if not provider_response:
            raise RuntimeError("No provider could handle this request")

        llm_ms = (time.perf_counter() - t_llm) * 1000
        metrics.record_stage_latency("llm_call", llm_ms)

        # ── Step 6: Cost tracking ─────────────────────────────────────────
        cost_record = self._cost_tracker.record(
            model=used_model,
            input_tokens=provider_response.input_tokens or tokens_after,
            output_tokens=provider_response.output_tokens,
            task_type=decision.task_type.value,
            session_id=request.session_id,
        )

        # ── Step 7: Cache store ───────────────────────────────────────────
        if self._cache and self._config.cache_enabled and not request.skip_cache:
            context_hash = self._context_hash(request.messages)
            await self._cache.set(
                prompt=last_user_content,
                response=provider_response.content,
                context_hash=context_hash,
                model=used_model,
                input_tokens=provider_response.input_tokens or tokens_after,
                output_tokens=provider_response.output_tokens,
            )

        total_ms = (time.perf_counter() - t_start) * 1000
        metrics.record_stage_latency("gateway_overhead", total_ms - llm_ms)

        log.info(
            "Request complete",
            extra={
                "provider": used_provider,
                "model": used_model,
                "latency_ms": round(total_ms, 2),
                "llm_ms": round(llm_ms, 2),
                "tokens_in": provider_response.input_tokens,
                "tokens_out": provider_response.output_tokens,
                "cost_usd": cost_record.cost_usd,
                "memories_injected": memories_injected,
            },
        )

        return GatewayResponse(
            content=provider_response.content,
            model=used_model,
            provider=used_provider,
            task_type=decision.task_type.value,
            input_tokens=provider_response.input_tokens or tokens_after,
            output_tokens=provider_response.output_tokens,
            cost_usd=cost_record.cost_usd,
            latency_ms=total_ms,
            cache_hit=False,
            cache_hit_type=None,
            tokens_before_compression=tokens_before,
            tokens_after_compression=tokens_after,
            memories_injected=memories_injected,
            routing_reason=decision.reasoning,
        )

    async def stream(
        self, request: GatewayRequest
    ) -> AsyncIterator[StreamChunk]:
        """
        Streaming pipeline — same pre-processing as process(), but returns
        an async iterator of StreamChunk tokens for SSE delivery.
        Cache lookup still applies; hits return in a single synthetic chunk.
        """
        # Pre-process messages (same as process())
        decision = self._router.select_model(
            messages=request.messages,
            override_model=request.model,
        )
        messages = list(request.messages)

        if self._memory and self._config.memory_enabled and not request.skip_memory:
            messages = await self._memory.inject_into_messages(messages)

        if self._config.context_compression_enabled and not request.skip_compression:
            if self._history:
                state = await self._history.process(messages)
                messages = state.messages
            else:
                messages = self._compressor.compress(messages).messages

        last_user_content = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user_content = str(m.get("content", ""))
                break

        # Cache check
        if self._cache and self._config.cache_enabled and not request.skip_cache:
            context_hash = self._context_hash(messages)
            cache_result = await self._cache.get(last_user_content, context_hash)
            if cache_result:
                self._metrics.cache_hits.inc()
                yield StreamChunk(
                    delta=cache_result.response,
                    model=decision.model,
                    finish_reason="stop",
                )
                return

        # Streaming LLM call with failover
        models_to_try = [decision.model] + decision.fallback_models
        collected_chunks: list[str] = []
        final_input_tokens = 0
        final_output_tokens = 0

        for model_attempt in models_to_try:
            from aicos.core.router import MODEL_REGISTRY
            spec = MODEL_REGISTRY.get(model_attempt)
            if not spec:
                continue
            provider = self._providers.get(spec.provider)
            if not provider:
                continue

            breaker = self._breakers.get(spec.provider)
            if not breaker.can_attempt():
                if model_attempt == models_to_try[-1]:
                    raise RuntimeError(f"All providers circuit-open. Last: {spec.provider}")
                continue

            try:
                async for chunk in provider.stream(
                    messages=messages,
                    model=model_attempt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                ):
                    if chunk.delta:
                        collected_chunks.append(chunk.delta)
                    if chunk.input_tokens:
                        final_input_tokens = chunk.input_tokens
                    if chunk.output_tokens:
                        final_output_tokens = chunk.output_tokens
                    yield chunk
                breaker.record_success()
                break
            except Exception:
                breaker.record_failure()
                if model_attempt == models_to_try[-1]:
                    raise
                continue

        # Cache the complete response
        if self._cache and self._config.cache_enabled and not request.skip_cache and collected_chunks:
            full_response = "".join(collected_chunks)
            context_hash = self._context_hash(request.messages)
            await self._cache.set(
                prompt=last_user_content,
                response=full_response,
                context_hash=context_hash,
                model=decision.model,
                input_tokens=final_input_tokens,
                output_tokens=final_output_tokens,
            )

        self._cost_tracker.record(
            model=decision.model,
            input_tokens=final_input_tokens,
            output_tokens=final_output_tokens,
            task_type=decision.task_type.value,
            session_id=request.session_id,
        )
