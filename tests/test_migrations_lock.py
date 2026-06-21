"""Tests for migration runner — advisory lock and idempotency."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aicos.db.migrations import _PG_LOCK_ID, MIGRATIONS


@pytest.mark.asyncio
async def test_run_migrations_sqlite_skips_lock(tmp_path):
    """SQLite path goes straight to _apply_pending, no advisory lock."""
    from aicos.core.database import build_engine
    from aicos.db.migrations import run_migrations

    engine = build_engine(f"sqlite+aiosqlite:///{tmp_path}/mig.db")
    try:
        with patch("aicos.db.migrations._apply_pending", new_callable=AsyncMock) as mock_apply:
            with patch("aicos.db.migrations._run_with_pg_lock", new_callable=AsyncMock) as mock_pg:
                await run_migrations(engine)
                mock_apply.assert_awaited_once()
                mock_pg.assert_not_awaited()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_apply_pending_idempotent(tmp_path):
    """Calling _apply_pending twice only applies each migration once."""
    from sqlalchemy import text

    from aicos.core.database import build_engine

    engine = build_engine(f"sqlite+aiosqlite:///{tmp_path}/mig2.db")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id TEXT PRIMARY KEY, applied_at REAL NOT NULL
                )
            """)
            )

        with patch(
            "aicos.db.migrations.MIGRATIONS",
            [
                ("test_001", AsyncMock()),
                ("test_002", AsyncMock()),
            ],
        ):
            from aicos.db import migrations as mig_module

            # First run — both applied
            await mig_module._apply_pending(engine)
            # Second run — none applied (already in DB)
            await mig_module._apply_pending(engine)

        # Check that test_001 and test_002 are in schema_migrations
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT id FROM schema_migrations"))
            applied = {row[0] for row in result.fetchall()}
        assert "test_001" in applied
        assert "test_002" in applied
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_pg_lock_id_is_stable():
    """The advisory lock ID must not change between releases (would cause deadlock on upgrade)."""
    assert _PG_LOCK_ID == 8_888_999


@pytest.mark.asyncio
async def test_migration_list_grows_monotonically():
    """Migrations are always in ascending order — inserting in the middle is forbidden."""
    ids = [mid for mid, _ in MIGRATIONS]
    assert ids == sorted(ids), f"Migrations out of order: {ids}"


@pytest.mark.asyncio
async def test_run_with_pg_lock_timeout_falls_through(tmp_path):
    """When lock timeout is reached, migrations run without the lock rather than hanging."""
    import aicos.db.migrations as mig_module
    from aicos.core.database import build_engine
    from aicos.db.migrations import _run_with_pg_lock

    engine = build_engine(f"sqlite+aiosqlite:///{tmp_path}/lock_timeout.db")

    applied = []

    async def _fake_apply(eng):
        applied.append(True)

    # Simulate lock never being acquired by forcing timeout immediately
    original_timeout = mig_module._PG_LOCK_TIMEOUT_S
    try:
        mig_module._PG_LOCK_TIMEOUT_S = -1.0  # negative → timeout immediately
        with patch("aicos.db.migrations._apply_pending", side_effect=_fake_apply):
            with patch("aicos.db.migrations.asyncio") as mock_asyncio:
                mock_loop = MagicMock()
                mock_loop.time.return_value = 9999.0  # already past deadline
                mock_asyncio.get_event_loop.return_value = mock_loop
                mock_asyncio.sleep = AsyncMock()

                # Build a mock engine that returns a mock connection
                mock_engine = MagicMock()
                mock_conn_ctx = AsyncMock()
                mock_conn_ctx.__aenter__ = AsyncMock(
                    return_value=MagicMock(
                        execute=AsyncMock(
                            return_value=MagicMock(scalar=MagicMock(return_value=False))
                        )
                    )
                )
                mock_conn_ctx.__aexit__ = AsyncMock()
                mock_engine.connect.return_value = mock_conn_ctx
                mock_engine.url = "postgresql+asyncpg://test"

                await _run_with_pg_lock(mock_engine)
                assert applied, "Should have fallen through to _apply_pending on timeout"
    finally:
        mig_module._PG_LOCK_TIMEOUT_S = original_timeout
        await engine.dispose()
