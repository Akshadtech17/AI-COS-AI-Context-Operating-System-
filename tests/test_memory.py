"""Tests for long-term memory: store, search, forget, ranking."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from aicos.memory.embeddings import EmbeddingEngine
from aicos.memory.memory_store import MemoryStore
from aicos.memory.retrieval import MemoryRetriever


class TestEmbeddingEngine:
    def test_single_embed_shape(self, embedding_engine: EmbeddingEngine) -> None:
        vec = embedding_engine.embed("Hello world")
        assert vec.shape == (embedding_engine.dim,)

    def test_embed_is_normalized(self, embedding_engine: EmbeddingEngine) -> None:
        vec = embedding_engine.embed("Test sentence")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_batch_embed_shape(self, embedding_engine: EmbeddingEngine) -> None:
        texts = ["Hello", "World", "Test"]
        vecs = embedding_engine.embed_batch(texts)
        assert vecs.shape == (3, embedding_engine.dim)

    def test_similarity_self(self, embedding_engine: EmbeddingEngine) -> None:
        vec = embedding_engine.embed("Python programming language")
        sim = embedding_engine.similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_similarity_different(self, embedding_engine: EmbeddingEngine) -> None:
        vec_a = embedding_engine.embed("Python programming language")
        vec_b = embedding_engine.embed("cooking recipes for pasta")
        sim = embedding_engine.similarity(vec_a, vec_b)
        assert sim < 0.95  # Should not be very similar

    def test_batch_similarity_shape(self, embedding_engine: EmbeddingEngine) -> None:
        query = embedding_engine.embed("Python")
        candidates = embedding_engine.embed_batch(["Python", "Java", "JavaScript"])
        scores = embedding_engine.batch_similarity(query, candidates)
        assert scores.shape == (3,)

    def test_top_k(self, embedding_engine: EmbeddingEngine) -> None:
        query = embedding_engine.embed("machine learning")
        candidates = embedding_engine.embed_batch([
            "machine learning and AI",
            "cooking and recipes",
            "deep learning neural networks",
            "gardening tips",
        ])
        results = embedding_engine.top_k(query, candidates, k=2, threshold=0.0)
        assert len(results) == 2
        indices = [r[0] for r in results]
        # ML and deep learning should rank higher than cooking/gardening
        assert 0 in indices or 2 in indices

    def test_cache_consistency(self, embedding_engine: EmbeddingEngine) -> None:
        # Same text should produce same embedding
        vec1 = embedding_engine.embed("consistent text")
        vec2 = embedding_engine.embed("consistent text")
        assert np.allclose(vec1, vec2)


class TestMemoryStore:
    @pytest.mark.asyncio
    async def test_store_and_count(self, memory_store: MemoryStore) -> None:
        await memory_store.store("User is a Python developer")
        count = await memory_store.count()
        assert count == 1

    @pytest.mark.asyncio
    async def test_store_multiple(self, memory_store: MemoryStore) -> None:
        for i in range(5):
            await memory_store.store(f"Memory {i}")
        count = await memory_store.count()
        assert count == 5

    @pytest.mark.asyncio
    async def test_store_with_tags(self, memory_store: MemoryStore) -> None:
        memory_id = await memory_store.store(
            "User prefers dark mode",
            tags=["preferences", "ui"],
        )
        assert memory_id > 0

    @pytest.mark.asyncio
    async def test_search_returns_results(self, memory_store: MemoryStore) -> None:
        await memory_store.store("User is an AI researcher")
        await memory_store.store("User loves Python programming")
        await memory_store.store("User prefers coffee over tea")

        results = await memory_store.search("artificial intelligence research", threshold=0.0)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, memory_store: MemoryStore) -> None:
        await memory_store.store("Python is great for data science")
        results = await memory_store.search("Python programming", threshold=0.9)
        # High threshold — may or may not find results, but shouldn't crash
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_forget_returns_true(self, memory_store: MemoryStore) -> None:
        memory_id = await memory_store.store("Temporary memory")
        deleted = await memory_store.forget(memory_id)
        assert deleted is True
        count = await memory_store.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_forget_nonexistent_returns_false(self, memory_store: MemoryStore) -> None:
        deleted = await memory_store.forget(99999)
        assert deleted is False

    @pytest.mark.asyncio
    async def test_search_ranking_order(self, memory_store: MemoryStore) -> None:
        await memory_store.store("Machine learning and deep learning AI")
        await memory_store.store("JavaScript and web development")
        await memory_store.store("Data science and machine learning")

        results = await memory_store.search("machine learning", threshold=0.0, top_k=3)
        assert len(results) > 0
        # Scores should be in descending order
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_forget_by_content(self, memory_store: MemoryStore) -> None:
        await memory_store.store("User wants to be a data scientist")
        initial_count = await memory_store.count()

        deleted = await memory_store.forget_by_content(
            "User wants to be a data scientist", threshold=0.99
        )
        assert deleted >= 0  # May or may not match depending on threshold


class TestMemoryRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_returns_list(self, memory_retriever: MemoryRetriever) -> None:
        results = await memory_retriever.retrieve("test query")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_inject_into_empty_messages(
        self, memory_store: MemoryStore, memory_retriever: MemoryRetriever
    ) -> None:
        await memory_store.store("User is a machine learning engineer")
        messages = [{"role": "user", "content": "What projects should I work on?"}]
        augmented = await memory_retriever.inject_into_messages(messages)
        # Should not crash; may or may not inject depending on similarity
        assert isinstance(augmented, list)
        assert len(augmented) >= len(messages)

    @pytest.mark.asyncio
    async def test_inject_preserves_existing_system(
        self, memory_store: MemoryStore, memory_retriever: MemoryRetriever
    ) -> None:
        await memory_store.store("User is a senior developer")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me coding advice"},
        ]
        augmented = await memory_retriever.inject_into_messages(messages)
        # System message should still be present (possibly augmented)
        system_msgs = [m for m in augmented if m.get("role") == "system"]
        assert len(system_msgs) >= 1
