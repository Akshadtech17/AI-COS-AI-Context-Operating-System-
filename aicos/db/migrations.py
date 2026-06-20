"""
Lightweight versioned migration runner.

Tracks applied migrations in a `schema_migrations` table.
Each migration is a plain async function that receives an AsyncConnection.
Migrations run once, in order, on every startup.

Adding a migration:
  1. Write an async function: async def mXXX_description(conn: AsyncConnection) -> None
  2. Append (migration_id, fn) to MIGRATIONS below.

The runner is idempotent: already-applied migrations are skipped.
"""

from __future__ import annotations

from typing import Callable, Awaitable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from aicos.core.logging import get_logger

log = get_logger("db.migrations")

MigrationFn = Callable[[AsyncConnection], Awaitable[None]]


# ── Migration definitions ─────────────────────────────────────────────────────

async def m001_initial_schema(conn: AsyncConnection) -> None:
    """
    Create all tables for the initial schema.
    All SQLAlchemy ORM Base.metadata.create_all calls happen here so that
    subsequent migrations can ALTER TABLE safely.
    """
    from aicos.cache.sqlite_cache import Base as CacheBase
    from aicos.memory.memory_store import Base as MemoryBase
    from aicos.analytics.cost_tracker import _CostBase
    from aicos.auth.api_keys import _Base as KeyBase

    await conn.run_sync(CacheBase.metadata.create_all)
    await conn.run_sync(MemoryBase.metadata.create_all)
    await conn.run_sync(_CostBase.metadata.create_all)
    await conn.run_sync(KeyBase.metadata.create_all)


MIGRATIONS: list[tuple[str, MigrationFn]] = [
    ("001_initial_schema", m001_initial_schema),
    # ("002_add_cost_index", m002_add_cost_index),  # example
]


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_migrations(engine: AsyncEngine) -> None:
    """
    Apply all pending migrations in order.
    Safe to call on every startup — already-applied migrations are skipped.
    """
    async with engine.begin() as conn:
        # Ensure the migrations tracking table exists
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                id         TEXT PRIMARY KEY,
                applied_at REAL NOT NULL
            )
        """))

        # Find already-applied migrations
        result = await conn.execute(text("SELECT id FROM schema_migrations"))
        applied = {row[0] for row in result.fetchall()}

    # Apply each pending migration in its own transaction
    for migration_id, fn in MIGRATIONS:
        if migration_id in applied:
            continue

        log.info("Applying migration", extra={"migration": migration_id})
        async with engine.begin() as conn:
            await fn(conn)
            import time
            await conn.execute(
                text("INSERT INTO schema_migrations (id, applied_at) VALUES (:id, :ts)"),
                {"id": migration_id, "ts": time.time()},
            )
        log.info("Migration applied", extra={"migration": migration_id})
