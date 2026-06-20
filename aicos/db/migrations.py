"""
Lightweight versioned migration runner.

Tracks applied migrations in a `schema_migrations` table.
Each migration is a plain async function that receives an AsyncConnection.
Migrations run once, in order, on every startup.

On PostgreSQL: a session-level advisory lock (ID 8_888_999) serialises concurrent
workers.  The first worker applies pending migrations; others wait (up to 30 s)
then proceed once the lock is released.  On SQLite, WAL mode already serialises
writers, so no advisory lock is needed.

Adding a migration:
  1. Write an async function: async def mXXX_description(conn: AsyncConnection) -> None
  2. Append (migration_id, fn) to MIGRATIONS below.

The runner is idempotent: already-applied migrations are skipped.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Awaitable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from aicos.core.logging import get_logger

log = get_logger("db.migrations")

MigrationFn = Callable[[AsyncConnection], Awaitable[None]]

# Arbitrary app-specific lock ID for pg_advisory_lock
_PG_LOCK_ID = 8_888_999
_PG_LOCK_TIMEOUT_S = 30.0


# ── Migration definitions ─────────────────────────────────────────────────────

async def m001_initial_schema(conn: AsyncConnection) -> None:
    """Create all tables for the initial schema."""
    from aicos.cache.sqlite_cache import Base as CacheBase
    from aicos.memory.memory_store import Base as MemoryBase
    from aicos.analytics.cost_tracker import _CostBase
    from aicos.auth.api_keys import _Base as KeyBase

    await conn.run_sync(CacheBase.metadata.create_all)
    await conn.run_sync(MemoryBase.metadata.create_all)
    await conn.run_sync(_CostBase.metadata.create_all)
    await conn.run_sync(KeyBase.metadata.create_all)


async def m002_add_pgvector(conn: AsyncConnection) -> None:
    """
    Add pgvector extension + native vector column to memories and cache_entries.

    PostgreSQL only — silently skipped on SQLite.
    Requires the pgvector extension to be installed in the PostgreSQL instance.
    Old rows keep NULL in embedding_vec; they remain searchable via the JSON path.
    New rows populate both columns so the vector index is used.
    """
    url = str(conn.engine.url)
    if "sqlite" in url:
        return

    from aicos.core.config import get_config
    dim = get_config().embedding_dim

    try:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        await conn.execute(text(
            f"ALTER TABLE memories "
            f"ADD COLUMN IF NOT EXISTS embedding_vec vector({dim})"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_memories_embedding_vec "
            "ON memories USING ivfflat (embedding_vec vector_cosine_ops) "
            "WITH (lists = 100)"
        ))

        await conn.execute(text(
            f"ALTER TABLE cache_entries "
            f"ADD COLUMN IF NOT EXISTS embedding_vec vector({dim})"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_cache_embedding_vec "
            "ON cache_entries USING ivfflat (embedding_vec vector_cosine_ops) "
            "WITH (lists = 100)"
        ))
        log.info("pgvector column + index created", extra={"dim": dim})
    except Exception as exc:
        log.warning(
            "pgvector not available — falling back to JSON similarity scan",
            extra={"error": str(exc)},
        )


MIGRATIONS: list[tuple[str, MigrationFn]] = [
    ("001_initial_schema", m001_initial_schema),
    ("002_add_pgvector", m002_add_pgvector),
]


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_migrations(engine: AsyncEngine) -> None:
    """
    Apply all pending migrations in order.
    Safe to call on every startup — already-applied migrations are skipped.
    On PostgreSQL, serialised via an advisory lock so multiple workers don't race.
    """
    is_pg = not str(engine.url).startswith("sqlite")

    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id         TEXT PRIMARY KEY,
                applied_at REAL NOT NULL
            )
        """))

    if is_pg:
        await _run_with_pg_lock(engine)
    else:
        await _apply_pending(engine)


async def _run_with_pg_lock(engine: AsyncEngine) -> None:
    """Acquire a PostgreSQL session advisory lock, run migrations, release it."""
    deadline = asyncio.get_event_loop().time() + _PG_LOCK_TIMEOUT_S

    while True:
        async with engine.connect() as conn:
            result = await conn.execute(
                text("SELECT pg_try_advisory_lock(:lock_id)"),
                {"lock_id": _PG_LOCK_ID},
            )
            acquired = bool(result.scalar())

            if acquired:
                try:
                    await _apply_pending(engine)
                finally:
                    await conn.execute(
                        text("SELECT pg_advisory_unlock(:lock_id)"),
                        {"lock_id": _PG_LOCK_ID},
                    )
                return

        # Another worker holds the lock — wait and retry
        if asyncio.get_event_loop().time() >= deadline:
            log.warning("Migration lock timeout — proceeding without lock")
            await _apply_pending(engine)
            return

        log.info("Waiting for migration lock...")
        await asyncio.sleep(1.0)


async def _apply_pending(engine: AsyncEngine) -> None:
    """Check which migrations are pending and apply them."""
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT id FROM schema_migrations"))
        applied = {row[0] for row in result.fetchall()}

    for migration_id, fn in MIGRATIONS:
        if migration_id in applied:
            continue

        log.info("Applying migration", extra={"migration": migration_id})
        async with engine.begin() as conn:
            await fn(conn)
            await conn.execute(
                text("INSERT INTO schema_migrations (id, applied_at) VALUES (:id, :ts)"),
                {"id": migration_id, "ts": time.time()},
            )
        log.info("Migration applied", extra={"migration": migration_id})
