"""Shared fixtures for AI-COS test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio

from aicos.cache.semantic_cache import SemanticCache
from aicos.cache.sqlite_cache import SQLiteCache
from aicos.core.config import AICOSConfig
from aicos.core.gateway import AIGateway
from aicos.core.router import ModelRouter
from aicos.memory.embeddings import EmbeddingEngine
from aicos.memory.memory_store import MemoryStore
from aicos.memory.retrieval import MemoryRetriever
from aicos.providers.base import BaseProvider, ProviderResponse, StreamChunk

# ── Config ────────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def config(tmp_dir: Path) -> AICOSConfig:
    return AICOSConfig(
        openai_api_key="sk-test-openai",
        anthropic_api_key=None,
        gemini_api_key=None,
        openrouter_api_key="sk-or-test-key",
        nvidia_api_key=None,
        db_path=str(tmp_dir / "aicos.db"),
        cache_enabled=True,
        memory_enabled=True,
        context_compression_enabled=True,
        analytics_enabled=True,
        router_strategy="auto",
        cache_similarity_threshold=0.96,
        max_context_tokens=4000,
        litm_threshold_tokens=3000,
    )


# ── Mock Provider ─────────────────────────────────────────────────────────────


class MockProvider(BaseProvider):
    def __init__(self, response_text: str = "Mock response") -> None:
        self._response_text = response_text
        self.call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    async def is_available(self) -> bool:
        return True

    async def complete(
        self, messages, model, max_tokens=4096, temperature=0.7, **kwargs
    ) -> ProviderResponse:
        self.call_count += 1
        return ProviderResponse(
            content=self._response_text,
            model=model,
            input_tokens=sum(len(str(m.get("content", "")).split()) for m in messages),
            output_tokens=len(self._response_text.split()),
            finish_reason="stop",
            raw={},
        )

    async def stream(
        self, messages, model, max_tokens=4096, temperature=0.7, **kwargs
    ) -> AsyncIterator[StreamChunk]:
        words = self._response_text.split()
        for word in words:
            yield StreamChunk(delta=word + " ", model=model)
        yield StreamChunk(
            delta="", model=model, finish_reason="stop", input_tokens=10, output_tokens=len(words)
        )


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


# ── Embedding Engine ──────────────────────────────────────────────────────────


@pytest.fixture
def embedding_engine() -> EmbeddingEngine:
    return EmbeddingEngine()


# ── Cache ─────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def sqlite_cache(tmp_dir: Path) -> AsyncIterator[SQLiteCache]:
    cache = SQLiteCache(database_url=f"sqlite+aiosqlite:///{tmp_dir}/cache.db", max_size=100)
    await cache.initialize()
    yield cache
    await cache.close()


@pytest_asyncio.fixture
async def semantic_cache(
    sqlite_cache: SQLiteCache, embedding_engine: EmbeddingEngine
) -> SemanticCache:
    return SemanticCache(
        sqlite_cache=sqlite_cache,
        embedding_engine=embedding_engine,
        threshold=0.96,
    )


# ── Memory ────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def memory_store(
    tmp_dir: Path, embedding_engine: EmbeddingEngine
) -> AsyncIterator[MemoryStore]:
    store = MemoryStore(
        database_url=f"sqlite+aiosqlite:///{tmp_dir}/memory.db",
        embedding_engine=embedding_engine,
        max_items=100,
    )
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def memory_retriever(memory_store: MemoryStore) -> MemoryRetriever:
    return MemoryRetriever(store=memory_store, top_k=5, threshold=0.3)


# ── Gateway ───────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def gateway(
    config: AICOSConfig,
    mock_provider: MockProvider,
    semantic_cache: SemanticCache,
    memory_retriever: MemoryRetriever,
) -> AIGateway:
    router = ModelRouter(config)
    return AIGateway(
        config=config,
        router=router,
        providers={"openai": mock_provider, "openrouter": mock_provider},
        semantic_cache=semantic_cache,
        memory_retriever=memory_retriever,
    )
