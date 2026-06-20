"""
Async engine factory — the single place that decides SQLite vs PostgreSQL behaviour.

SQLite (dev / single-server):
  - WAL mode: readers never block writers, multiple workers can read concurrently
  - busy_timeout 5 s: writers queue instead of failing immediately
  - synchronous=NORMAL: safe with WAL (only fsync at checkpoint, not every commit)

PostgreSQL (production / multi-worker):
  - asyncpg driver, pool_size=10, max_overflow=20
  - pool_recycle=1800 s: recycle stale connections before the server drops them
"""

from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


def build_engine(database_url: str) -> AsyncEngine:
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

    # PostgreSQL / other — requires asyncpg in the environment
    return create_async_engine(
        database_url,
        echo=False,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1800,
    )


def sqlite_url(path: str) -> str:
    """Convert an absolute file path to a sqlite+aiosqlite:// URL."""
    return f"sqlite+aiosqlite:///{path}"
