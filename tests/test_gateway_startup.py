"""Integration test for the real _build_gateway startup path.

Every route test mocks _build_gateway, so these tests exercise the actual
production startup path: DB creation, embedding engine, semantic cache,
memory store, router — all wired together against a real (temp) SQLite DB.
"""

from __future__ import annotations

import pytest

from aicos.core.config import AICOSConfig


@pytest.fixture
def startup_config(tmp_path):
    return AICOSConfig(
        openai_api_key="sk-test-openai",
        anthropic_api_key=None,
        gemini_api_key=None,
        openrouter_api_key=None,
        nvidia_api_key=None,
        db_path=str(tmp_path / "aicos.db"),
        cache_enabled=True,
        memory_enabled=True,
        context_compression_enabled=True,
        rate_limit_enabled=False,
    )


async def _cleanup(gateway, memory_store) -> None:
    await memory_store.close()
    if gateway._cache is not None:
        await gateway._cache._cache.close()
    await gateway.close()  # disposes cost_tracker's SQLAlchemy engine


class TestBuildGateway:
    @pytest.mark.asyncio
    async def test_returns_gateway_and_memory_store(self, startup_config) -> None:
        from aicos.api.routes import _build_gateway

        gateway, memory_store = await _build_gateway(startup_config)
        try:
            assert gateway is not None
            assert memory_store is not None
        finally:
            await _cleanup(gateway, memory_store)

    @pytest.mark.asyncio
    async def test_gateway_has_provider(self, startup_config) -> None:
        from aicos.api.routes import _build_gateway

        gateway, memory_store = await _build_gateway(startup_config)
        try:
            assert "openai" in gateway._providers
        finally:
            await _cleanup(gateway, memory_store)

    @pytest.mark.asyncio
    async def test_gateway_has_cache(self, startup_config) -> None:
        from aicos.api.routes import _build_gateway

        gateway, memory_store = await _build_gateway(startup_config)
        try:
            assert gateway._cache is not None
        finally:
            await _cleanup(gateway, memory_store)

    @pytest.mark.asyncio
    async def test_gateway_has_memory_retriever(self, startup_config) -> None:
        from aicos.api.routes import _build_gateway

        gateway, memory_store = await _build_gateway(startup_config)
        try:
            assert gateway._memory is not None
        finally:
            await _cleanup(gateway, memory_store)

    @pytest.mark.asyncio
    async def test_gateway_has_router(self, startup_config) -> None:
        from aicos.api.routes import _build_gateway

        gateway, memory_store = await _build_gateway(startup_config)
        try:
            assert gateway._router is not None
        finally:
            await _cleanup(gateway, memory_store)

    @pytest.mark.asyncio
    async def test_gateway_has_history_manager(self, startup_config) -> None:
        from aicos.api.routes import _build_gateway

        gateway, memory_store = await _build_gateway(startup_config)
        try:
            assert gateway._history is not None
        finally:
            await _cleanup(gateway, memory_store)

    @pytest.mark.asyncio
    async def test_cache_disabled(self, tmp_path) -> None:
        from aicos.api.routes import _build_gateway

        cfg = AICOSConfig(
            openai_api_key="sk-test",
            db_path=str(tmp_path / "aicos.db"),
            cache_enabled=False,
            memory_enabled=True,
            context_compression_enabled=False,
        )
        gateway, memory_store = await _build_gateway(cfg)
        try:
            assert gateway._cache is None
        finally:
            await _cleanup(gateway, memory_store)

    @pytest.mark.asyncio
    async def test_memory_disabled(self, tmp_path) -> None:
        from aicos.api.routes import _build_gateway

        cfg = AICOSConfig(
            openai_api_key="sk-test",
            db_path=str(tmp_path / "aicos.db"),
            cache_enabled=True,
            memory_enabled=False,
            context_compression_enabled=False,
        )
        gateway, memory_store = await _build_gateway(cfg)
        try:
            assert gateway._memory is None
        finally:
            await _cleanup(gateway, memory_store)

    @pytest.mark.asyncio
    async def test_no_providers_when_no_keys(self, tmp_path) -> None:
        from aicos.api.routes import _build_gateway

        cfg = AICOSConfig(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            nvidia_api_key=None,
            db_path=str(tmp_path / "aicos.db"),
            cache_enabled=False,
            memory_enabled=False,
            context_compression_enabled=False,
        )
        gateway, memory_store = await _build_gateway(cfg)
        try:
            assert gateway._providers == {}
        finally:
            await _cleanup(gateway, memory_store)

    @pytest.mark.asyncio
    async def test_multiple_providers(self, tmp_path) -> None:
        from aicos.api.routes import _build_gateway

        cfg = AICOSConfig(
            openai_api_key="sk-openai-test",
            openrouter_api_key="sk-or-test",
            anthropic_api_key=None,
            gemini_api_key=None,
            nvidia_api_key=None,
            db_path=str(tmp_path / "aicos.db"),
            cache_enabled=False,
            memory_enabled=False,
            context_compression_enabled=False,
        )
        gateway, memory_store = await _build_gateway(cfg)
        try:
            assert "openai" in gateway._providers
            assert "openrouter" in gateway._providers
        finally:
            await _cleanup(gateway, memory_store)
