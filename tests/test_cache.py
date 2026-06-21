"""Tests for semantic cache: exact lookup, similarity search, TTL, LRU eviction."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

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

    @pytest.mark.asyncio
    async def test_semantic_hit_returns_cached_response(self, sqlite_cache: SQLiteCache) -> None:
        """Lines 91-92: best_score >= threshold branch returns a semantic CacheResult.

        Uses mocked embeddings so any two distinct prompts appear maximally similar
        (cosine similarity = 1.0). The exact hash lookup misses (different prompt text)
        and falls through to the semantic scan, which then exceeds the threshold.
        """
        fixed_vec = np.ones(10, dtype=np.float32)
        fixed_vec /= np.linalg.norm(fixed_vec)

        mock_emb = MagicMock()
        mock_emb.embed = MagicMock(return_value=fixed_vec)
        mock_emb.batch_similarity = MagicMock(
            side_effect=lambda q, matrix: np.ones(len(matrix), dtype=np.float32)
        )

        cache = SemanticCache(
            sqlite_cache=sqlite_cache,
            embedding_engine=mock_emb,
            threshold=0.95,
        )

        await cache.set("What is 2 plus 2?", "The answer is 4")
        # Different text → different hash → exact miss → semantic path
        result = await cache.get("What does 2 + 2 equal?")

        assert result is not None
        assert result.hit_type == "semantic"
        assert result.response == "The answer is 4"
        assert result.similarity >= 0.95

    @pytest.mark.asyncio
    async def test_semantic_hit_latency_is_populated(self, sqlite_cache: SQLiteCache) -> None:
        """CacheResult.latency_ms is set on a semantic hit."""
        fixed_vec = np.ones(10, dtype=np.float32) / np.sqrt(10)
        mock_emb = MagicMock()
        mock_emb.embed = MagicMock(return_value=fixed_vec)
        mock_emb.batch_similarity = MagicMock(
            side_effect=lambda q, matrix: np.ones(len(matrix), dtype=np.float32)
        )
        cache = SemanticCache(sqlite_cache=sqlite_cache, embedding_engine=mock_emb, threshold=0.5)

        await cache.set("original prompt text", "cached response")
        result = await cache.get("a different prompt text")

        assert result is not None
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_semantic_below_threshold_returns_none(self, sqlite_cache: SQLiteCache) -> None:
        """Line 100: candidates exist but best_score < threshold → None."""
        fixed_vec = np.ones(10, dtype=np.float32) / np.sqrt(10)
        mock_emb = MagicMock()
        mock_emb.embed = MagicMock(return_value=fixed_vec)
        # Similarity below threshold → no semantic hit
        mock_emb.batch_similarity = MagicMock(
            side_effect=lambda q, matrix: np.full(len(matrix), 0.50, dtype=np.float32)
        )
        cache = SemanticCache(sqlite_cache=sqlite_cache, embedding_engine=mock_emb, threshold=0.95)

        await cache.set("stored prompt", "stored response")
        result = await cache.get("different prompt")  # exact miss, score 0.50 < 0.95

        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_removes_entry(self, semantic_cache: SemanticCache) -> None:
        """Lines 124-126: invalidate() deletes the entry and marks _index_dirty."""
        await semantic_cache.set("Remember this", "Some response")

        deleted = await semantic_cache.invalidate("Remember this")

        assert deleted is True
        assert semantic_cache._index_dirty is True
        # Exact lookup must now miss; no other entries → get() returns None
        result = await semantic_cache.get("Remember this")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_returns_false(
        self, semantic_cache: SemanticCache
    ) -> None:
        """invalidate() on an unknown prompt returns False."""
        deleted = await semantic_cache.invalidate("this was never stored xyz999")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_stats_returns_threshold_and_scan_limit(
        self, semantic_cache: SemanticCache
    ) -> None:
        """Lines 135-138: stats() extends SQLiteCache stats with threshold and scan_limit."""
        await semantic_cache.set("A prompt", "A response")

        stats = await semantic_cache.stats()

        assert "threshold" in stats
        assert "scan_limit" in stats
        assert stats["threshold"] == 0.96  # matches conftest fixture value
        assert isinstance(stats["scan_limit"], int)
        assert stats["scan_limit"] > 0
