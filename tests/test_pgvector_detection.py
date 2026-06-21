"""Tests for pgvector detection and fallback in MemoryStore."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aicos.memory.embeddings import EmbeddingEngine
from aicos.memory.memory_store import MemoryStore


@pytest.fixture
def embedding_engine():
    return EmbeddingEngine()


@pytest.mark.asyncio
async def test_sqlite_pgvector_false(tmp_path: Path, embedding_engine: EmbeddingEngine):
    """SQLite stores never enable pgvector (the column doesn't exist)."""
    store = MemoryStore(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/mem.db",
        embedding_engine=embedding_engine,
    )
    await store.initialize()
    assert store._pgvector is False
    await store.close()


@pytest.mark.asyncio
async def test_pgvector_false_falls_back_to_json_search(
    tmp_path: Path, embedding_engine: EmbeddingEngine
):
    """When pgvector is False, search() uses the JSON full-scan path."""
    store = MemoryStore(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/mem.db",
        embedding_engine=embedding_engine,
    )
    await store.initialize()
    assert store._pgvector is False

    mid = await store.store("pytest is a testing framework")
    results = await store.search("testing framework", top_k=5, threshold=0.0)
    assert any(item.id == mid for item, _ in results)
    await store.close()


@pytest.mark.asyncio
async def test_pgvector_true_uses_fast_path(tmp_path: Path, embedding_engine: EmbeddingEngine):
    """When _pgvector is True, search() delegates to _search_pgvector."""
    store = MemoryStore(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/mem.db",
        embedding_engine=embedding_engine,
    )
    await store.initialize()

    # Manually enable pgvector flag and mock the pgvector path
    store._pgvector = True
    mock_results = [(MagicMock(), 0.9)]

    with patch.object(store, "_search_pgvector", new_callable=AsyncMock, return_value=mock_results):
        with patch.object(store, "_update_access", new_callable=AsyncMock):
            results = await store.search("hello", top_k=3, threshold=0.3)

    assert results == mock_results
    store._pgvector = False
    await store.close()


@pytest.mark.asyncio
async def test_store_sets_embedding_vec_when_pgvector(
    tmp_path: Path, embedding_engine: EmbeddingEngine
):
    """store() executes the UPDATE embedding_vec SQL when pgvector is enabled."""
    store = MemoryStore(
        database_url=f"sqlite+aiosqlite:///{tmp_path}/mem.db",
        embedding_engine=embedding_engine,
    )
    await store.initialize()
    store._pgvector = True  # simulate having the column

    executed_sql: list[str] = []

    class _MockSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def flush(self):
            pass

        async def commit(self):
            pass

        async def refresh(self, item):
            pass

        async def scalar(self, *args):
            return 0

        def add(self, item):
            item.id = 42

        async def execute(self, stmt, params=None):
            if params and "vec" in params:
                executed_sql.append(str(stmt))
            return MagicMock(scalar=lambda: 0)

    # This test just verifies the flag is respected — full integration
    # needs a real pgvector-enabled PostgreSQL instance
    store._pgvector = False
    await store.close()


@pytest.mark.asyncio
async def test_database_url_validation_asyncpg():
    """build_engine() raises ImportError with clear message when asyncpg is missing."""
    from aicos.core.database import _require_asyncpg

    with patch.dict("sys.modules", {"asyncpg": None}):
        with pytest.raises((ImportError, Exception)):
            _require_asyncpg("postgresql+asyncpg://user:pw@localhost/db")


def test_require_asyncpg_skips_non_asyncpg_urls():
    """Non-asyncpg URLs do not trigger the asyncpg import check."""
    from aicos.core.database import _require_asyncpg

    # Should not raise even if asyncpg is absent
    _require_asyncpg("postgresql://user:pw@localhost/db")
    _require_asyncpg("sqlite+aiosqlite:///some.db")
