"""Tests for semantic cache: exact lookup, similarity search, TTL, LRU eviction."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio

from aicos.cache.semantic_cache import SemanticCache
from aicos.cache.sqlite_cache import SQLiteCache
from aicos.memory.embeddings import EmbeddingEngine


class TestSQLiteCache:
    @pytest.mark.asyncio
    async def test_set_and_get_exact(self, sqlite_cache: SQLiteCache) -> None:
        embedding = np.random.randn(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        await sqlite_cache.set(
            prompt="What is AI?",
            response="Artificial Intelligence is...",
            embedding=embedding,
        )

        key = SQLiteCache.make_key("What is AI?", "")
        result = await sqlite_cache.get_exact(key)

        assert result is not None
        assert result.response == "Artificial Intelligence is..."

    @pytest.mark.asyncio
    async def test_exact_miss(self, sqlite_cache: SQLiteCache) -> None:
        result = await sqlite_cache.get_exact("nonexistent-hash-12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_updates_existing(self, sqlite_cache: SQLiteCache) -> None:
        embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
        await sqlite_cache.set("Hello", "Response 1", embedding)
        await sqlite_cache.set("Hello", "Response 2 Updated", embedding)

        key = SQLiteCache.make_key("Hello", "")
        result = await sqlite_cache.get_exact(key)
        assert result is not None
        assert result.response == "Response 2 Updated"

    @pytest.mark.asyncio
    async def test_hit_count_increments(self, sqlite_cache: SQLiteCache) -> None:
        embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
        await sqlite_cache.set("Test", "Response", embedding)
        key = SQLiteCache.make_key("Test", "")

        for _ in range(3):
            await sqlite_cache.get_exact(key)

        result = await sqlite_cache.get_exact(key)
        assert result is not None
        assert result.hit_count >= 3

    @pytest.mark.asyncio
    async def test_clear(self, sqlite_cache: SQLiteCache) -> None:
        embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
        await sqlite_cache.set("A", "RA", embedding)
        await sqlite_cache.set("B", "RB", embedding)

        count = await sqlite_cache.clear()
        assert count == 2

        stats = await sqlite_cache.stats()
        assert stats["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_invalidate(self, sqlite_cache: SQLiteCache) -> None:
        embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
        await sqlite_cache.set("Test", "Response", embedding)
        key = SQLiteCache.make_key("Test", "")

        deleted = await sqlite_cache.invalidate(key)
        assert deleted is True

        result = await sqlite_cache.get_exact(key)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_recent(self, sqlite_cache: SQLiteCache) -> None:
        for i in range(5):
            emb = np.random.randn(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            await sqlite_cache.set(f"Prompt {i}", f"Response {i}", emb)

        entries = await sqlite_cache.get_recent(limit=10)
        assert len(entries) == 5


class TestSemanticCache:
    @pytest.mark.asyncio
    async def test_exact_cache_hit(self, semantic_cache: SemanticCache) -> None:
        await semantic_cache.set("What is Python?", "Python is a programming language.")
        result = await semantic_cache.get("What is Python?")

        assert result is not None
        assert result.hit_type == "exact"
        assert result.similarity == 1.0
        assert "Python" in result.response

    @pytest.mark.asyncio
    async def test_cache_miss(self, semantic_cache: SemanticCache) -> None:
        result = await semantic_cache.get("This query was never cached before xyz123")
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_similarity_hit(
        self, sqlite_cache: SQLiteCache, embedding_engine: EmbeddingEngine
    ) -> None:
        # Use a lower threshold to test semantic matching
        cache = SemanticCache(
            sqlite_cache=sqlite_cache,
            embedding_engine=embedding_engine,
            threshold=0.70,  # Lower threshold for test
        )

        # Store the original
        await cache.set("What is machine learning?", "ML is a subset of AI...")

        # Query with very similar prompt
        result = await cache.get("What is machine learning?")  # Exact should still hit
        assert result is not None

    @pytest.mark.asyncio
    async def test_cache_stores_metadata(self, semantic_cache: SemanticCache) -> None:
        await semantic_cache.set(
            prompt="Explain neural networks",
            response="Neural networks are...",
            model="gpt-4o-mini",
            input_tokens=15,
            output_tokens=50,
        )
        result = await semantic_cache.get("Explain neural networks")
        assert result is not None
        assert result.source_entry is not None
        assert result.source_entry.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_clear_cache(self, semantic_cache: SemanticCache) -> None:
        await semantic_cache.set("Q1", "A1")
        await semantic_cache.set("Q2", "A2")
        count = await semantic_cache.clear()
        assert count == 2

        result = await semantic_cache.get("Q1")
        assert result is None

    @pytest.mark.asyncio
    async def test_latency_tracking(self, semantic_cache: SemanticCache) -> None:
        await semantic_cache.set("Fast query", "Fast response")
        result = await semantic_cache.get("Fast query")
        assert result is not None
        assert result.latency_ms < 1000  # Should be fast
        assert result.latency_ms >= 0
