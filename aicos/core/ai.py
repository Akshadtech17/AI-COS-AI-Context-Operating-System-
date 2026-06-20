"""
AI — the primary public interface for AI-COS.

Provides the simplest possible DX:

    from aicos import AI

    ai = AI()
    response = ai.chat("Build a SaaS startup")
    ai.remember("User is an AI researcher")
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

from aicos.analytics.cost_tracker import CostTracker
from aicos.cache.semantic_cache import SemanticCache
from aicos.cache.sqlite_cache import SQLiteCache
from aicos.context.compressor import ContextCompressor
from aicos.context.history_manager import HistoryManager
from aicos.context.summarizer import ConversationSummarizer
from aicos.core.config import AICOSConfig, get_config
from aicos.core.gateway import AIGateway, GatewayRequest, GatewayResponse
from aicos.core.router import ModelRouter
from aicos.memory.embeddings import EmbeddingEngine
from aicos.memory.memory_store import MemoryStore
from aicos.memory.retrieval import MemoryRetriever
from aicos.providers.base import BaseProvider


class AI:
    """
    AI-COS primary interface.

    Instantiate once and reuse across your application.
    All subsystems are lazy-initialized on first call.
    """

    def __init__(
        self,
        config: AICOSConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        self._config = config or get_config()
        self._session_id = session_id
        self._conversation: list[dict[str, Any]] = []
        self._initialized = False
        self._gateway: AIGateway | None = None
        self._memory_store: MemoryStore | None = None
        self._cost_tracker = CostTracker()

    # ── Initialization ────────────────────────────────────────────────────

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        cfg = self._config
        db_path = cfg.get_resolved_db_path()

        # Build providers
        providers = self._build_providers()

        # Embedding engine (shared)
        embedding_engine = EmbeddingEngine()

        # Cache
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

        # Memory
        memory_store = MemoryStore(
            db_path=db_path.parent / "memory.db",
            embedding_engine=embedding_engine,
            max_items=cfg.memory_max_items,
        )
        await memory_store.initialize()
        self._memory_store = memory_store

        memory_retriever = MemoryRetriever(
            store=memory_store,
            top_k=cfg.memory_injection_limit,
            threshold=cfg.memory_relevance_threshold,
        )

        # Context management
        compressor = ContextCompressor(
            max_tokens=cfg.max_context_tokens,
        )
        router = ModelRouter(cfg)

        # Build summarizer if we have any provider
        summarizer: ConversationSummarizer | None = None
        if providers:
            summarizer = ConversationSummarizer(
                llm_caller=self._llm_call_simple,
                model="gpt-4o-mini" if "openai" in providers else next(iter(providers.keys())),
            )

        history_manager = HistoryManager(
            compressor=compressor,
            summarizer=summarizer,
            max_tokens=cfg.max_context_tokens,
            litm_threshold_tokens=cfg.litm_threshold_tokens,
        )

        self._gateway = AIGateway(
            config=cfg,
            router=router,
            providers=providers,
            semantic_cache=semantic_cache if cfg.cache_enabled else None,
            memory_retriever=memory_retriever if cfg.memory_enabled else None,
            history_manager=history_manager if cfg.context_compression_enabled else None,
            cost_tracker=self._cost_tracker,
        )

        self._initialized = True

    def _build_providers(self) -> dict[str, BaseProvider]:
        from aicos.providers.anthropic_provider import AnthropicProvider
        from aicos.providers.gemini_provider import GeminiProvider
        from aicos.providers.openai_provider import OpenAIProvider

        providers: dict[str, BaseProvider] = {}
        cfg = self._config

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

        return providers

    async def _llm_call_simple(
        self, messages: list[dict[str, Any]], model: str, max_tokens: int = 500
    ) -> str:
        """Lightweight LLM call used internally (e.g., by summarizer)."""
        await self._ensure_initialized()
        assert self._gateway is not None
        request = GatewayRequest(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            skip_cache=True,
            skip_memory=True,
            skip_compression=True,
        )
        response = await self._gateway.process(request)
        return response.content

    # ── Public API ────────────────────────────────────────────────────────

    def chat(
        self,
        message: str,
        system: str | None = None,
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Send a chat message and receive a response.

        Conversation history is maintained automatically.
        Memory, caching, and context optimization are applied transparently.

        Args:
            message: User message
            system: Optional system prompt (only applied if conversation is new)
            model: Override model selection
            stream: If True, returns an iterator of text chunks

        Returns:
            Response string (or iterator if stream=True)
        """
        return asyncio.get_event_loop().run_until_complete(
            self.achat(message, system=system, model=model, stream=stream, **kwargs)
        )

    async def achat(
        self,
        message: str,
        system: str | None = None,
        model: str | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> str:
        """Async version of chat()."""
        await self._ensure_initialized()
        assert self._gateway is not None

        # Initialize conversation with system prompt
        if system and not self._conversation:
            self._conversation.append({"role": "system", "content": system})

        self._conversation.append({"role": "user", "content": message})

        request = GatewayRequest(
            messages=self._conversation.copy(),
            model=model,
            session_id=self._session_id,
            **kwargs,
        )

        if stream:
            raise ValueError("Use astream() for streaming")

        response: GatewayResponse = await self._gateway.process(request)
        self._conversation.append({"role": "assistant", "content": response.content})

        return response.content

    async def astream(
        self,
        message: str,
        system: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response tokens as an async iterator."""
        await self._ensure_initialized()
        assert self._gateway is not None

        if system and not self._conversation:
            self._conversation.append({"role": "system", "content": system})

        self._conversation.append({"role": "user", "content": message})

        request = GatewayRequest(
            messages=self._conversation.copy(),
            model=model,
            stream=True,
            session_id=self._session_id,
            **kwargs,
        )

        collected = []
        async for chunk in self._gateway.stream(request):
            if chunk.delta:
                collected.append(chunk.delta)
                yield chunk.delta

        self._conversation.append({"role": "assistant", "content": "".join(collected)})

    def remember(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Store a memory for future retrieval.

        Args:
            content: The information to remember
            tags: Optional tags for filtering
            metadata: Optional structured metadata

        Returns:
            Memory ID
        """
        return asyncio.get_event_loop().run_until_complete(
            self.aremember(content, tags=tags, metadata=metadata)
        )

    async def aremember(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        await self._ensure_initialized()
        assert self._memory_store is not None
        memory_id = await self._memory_store.store(content, tags=tags, metadata=metadata)
        from aicos.analytics.metrics import get_metrics
        get_metrics().memory_stored.inc()
        return memory_id

    def forget(self, memory_id: int) -> bool:
        """Delete a stored memory by ID."""
        return asyncio.get_event_loop().run_until_complete(self.aforget(memory_id))

    async def aforget(self, memory_id: int) -> bool:
        await self._ensure_initialized()
        assert self._memory_store is not None
        return await self._memory_store.forget(memory_id)

    def search_memory(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Search stored memories by semantic similarity."""
        return asyncio.get_event_loop().run_until_complete(
            self.asearch_memory(query, top_k=top_k, threshold=threshold)
        )

    async def asearch_memory(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        await self._ensure_initialized()
        assert self._memory_store is not None
        results = await self._memory_store.search(query, top_k=top_k, threshold=threshold)
        return [
            {
                "id": item.id,
                "content": item.content,
                "score": round(score, 4),
                "tags": item.tag_list,
                "created_at": item.created_at.isoformat(),
            }
            for item, score in results
        ]

    def clear_history(self) -> None:
        """Clear conversation history (memory is preserved)."""
        self._conversation = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Read-only view of conversation history."""
        return list(self._conversation)

    @property
    def cost_summary(self) -> dict[str, Any]:
        """Return session cost summary."""
        return self._cost_tracker.session_summary()

    @property
    def metrics(self) -> dict[str, Any]:
        """Return session metrics."""
        from aicos.analytics.metrics import get_metrics
        return get_metrics().to_dict()
