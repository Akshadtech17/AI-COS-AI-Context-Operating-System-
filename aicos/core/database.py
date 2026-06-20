"""
Async engine factory — the single place that decides SQLite vs PostgreSQL behaviour.

SQLite (dev / single-server):
  - WAL mode: readers never block writers, multiple workers can read concurrently
  - busy_timeout 5 s: writers queue instead of failing immediately
  - synchronous=NORMAL: safe with WAL (only fsync at checkpoint, not every commit)

PostgreSQL / PgBouncer (production / multi-worker):
  - Validates asyncpg is installed at engine-creation time (clear error, not at query time)
  - pool_size and max_overflow are configurable (reduce when sitting behind PgBouncer)
  - pool_recycle=1800 s: recycle stale connections before the server drops them
  - pool_pre_ping: discard dead connections before handing them to application code

PgBouncer guidance:
  In transaction-pooling mode, set db_pool_size=2 and db_max_overflow=3 in config.
  PgBouncer then handles multiplexing; each worker store uses only 2-5 connections
  at peak, so 4 workers × 4 stores × 5 = 80 app→PgBouncer connections map to
  ~20 real PostgreSQL connections.
"""

from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def build_engine(
    database_url: str,
    pool_size: int = 5,
    max_overflow: int = 5,
) -> AsyncEngine:
    """Return a configured async engine for the given URL."""
    if database_url.startswith("sqlite"):
        engine = create_async_engine(
            database_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine.sync_engine, "connect")
        def _set_wal_pragmas(dbapi_conn, _record: object) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA wal_autocheckpoint=1000")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

        return engine

    # PostgreSQL — validate asyncpg is installed early so the error is actionable
    _require_asyncpg(database_url)

    return create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=30,
        pool_recycle=1800,
    )


def _require_asyncpg(database_url: str) -> None:
    """Fail fast with a clear message if asyncpg isn't installed."""
    if "asyncpg" not in database_url:
        return
    try:
        import asyncpg  # noqa: F401
    except ImportError:
        raise ImportError(
            "asyncpg is required for PostgreSQL support but is not installed.\n"
            "Fix: pip install 'aicos[postgres]'  or  pip install asyncpg\n"
            f"DATABASE_URL was: {database_url}"
        ) from None


def sqlite_url(path: str) -> str:
    """Convert an absolute file path to a sqlite+aiosqlite:// URL."""
    return f"sqlite+aiosqlite:///{path}"


def is_sqlite(database_url: str) -> bool:
    return database_url.startswith("sqlite")
