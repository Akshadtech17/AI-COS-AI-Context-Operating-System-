"""Tests for the AI public interface (aicos.core.ai)."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aicos.core.ai import AI
from aicos.core.config import AICOSConfig
from aicos.providers.base import ProviderResponse, StreamChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ai_config(tmp_path):
    return AICOSConfig(
        openrouter_api_key="sk-or-test-key",
        openai_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        nvidia_api_key=None,
        db_path=str(tmp_path / "aicos.db"),
        cache_enabled=True,
        memory_enabled=True,
        context_compression_enabled=True,
        max_context_tokens=4000,
        litm_threshold_tokens=3000,
    )


@pytest.fixture
def mock_prov():
    from tests.conftest import MockProvider
    return MockProvider("AI says hello")


@pytest.fixture
def ai(ai_config, mock_prov):
    instance = AI(config=ai_config)
    instance._build_providers = MagicMock(return_value={"openrouter": mock_prov})
    return instance


# ── Initialization ────────────────────────────────────────────────────────────

class TestAIInitialization:
    @pytest.mark.asyncio
    async def test_lazy_init_only_once(self, ai) -> None:
        await ai._ensure_initialized()
        await ai._ensure_initialized()  # second call is a no-op
        assert ai._initialized is True
        assert ai._gateway is not None

    @pytest.mark.asyncio
    async def test_build_providers_called_once(self, ai) -> None:
        await ai._ensure_initialized()
        ai._build_providers.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_store_created(self, ai) -> None:
        await ai._ensure_initialized()
        assert ai._memory_store is not None


# ── Chat ──────────────────────────────────────────────────────────────────────

class TestAIChat:
    @pytest.mark.asyncio
    async def test_achat_returns_string(self, ai) -> None:
        result = await ai.achat("Hello")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_achat_with_system_prompt(self, ai) -> None:
        result = await ai.achat("Hello", system="You are a helpful assistant.")
        assert isinstance(result, str)
        assert ai._conversation[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_achat_system_only_set_once(self, ai) -> None:
        await ai.achat("First message", system="Be concise.")
        await ai.achat("Second message", system="Ignore this system")
        # System message should appear exactly once
        system_msgs = [m for m in ai._conversation if m["role"] == "system"]
        assert len(system_msgs) == 1

    @pytest.mark.asyncio
    async def test_achat_builds_history(self, ai) -> None:
        await ai.achat("First")
        await ai.achat("Second")
        user_msgs = [m for m in ai._conversation if m["role"] == "user"]
        assert len(user_msgs) == 2

    @pytest.mark.asyncio
    async def test_achat_stream_raises(self, ai) -> None:
        with pytest.raises(ValueError, match="astream"):
            await ai.achat("Hello", stream=True)

    @pytest.mark.asyncio
    async def test_achat_with_model_override(self, ai) -> None:
        result = await ai.achat("Hello", model="gpt-4o-mini")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_astream_yields_tokens(self, ai) -> None:
        tokens = []
        async for token in ai.astream("Stream this"):
            tokens.append(token)
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.asyncio
    async def test_astream_with_system(self, ai) -> None:
        tokens = []
        async for token in ai.astream("Hello", system="Be brief."):
            tokens.append(token)
        assert ai._conversation[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_astream_appends_to_history(self, ai) -> None:
        async for _ in ai.astream("Hello"):
            pass
        assistant_msgs = [m for m in ai._conversation if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1


# ── Memory ────────────────────────────────────────────────────────────────────

class TestAIMemory:
    @pytest.mark.asyncio
    async def test_aremember_returns_id(self, ai) -> None:
        memory_id = await ai.aremember("Python is a great language")
        assert isinstance(memory_id, int)
        assert memory_id > 0

    @pytest.mark.asyncio
    async def test_aremember_with_tags(self, ai) -> None:
        memory_id = await ai.aremember("User likes Python", tags=["preferences"])
        assert memory_id > 0

    @pytest.mark.asyncio
    async def test_aforget_existing(self, ai) -> None:
        memory_id = await ai.aremember("Forget me")
        deleted = await ai.aforget(memory_id)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_aforget_nonexistent(self, ai) -> None:
        deleted = await ai.aforget(999999)
        assert deleted is False

    @pytest.mark.asyncio
    async def test_asearch_memory_returns_list(self, ai) -> None:
        await ai.aremember("Machine learning is fascinating")
        results = await ai.asearch_memory("machine learning")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_asearch_memory_structure(self, ai) -> None:
        await ai.aremember("The user is a Python developer", tags=["profile"])
        results = await ai.asearch_memory("Python developer", top_k=3)
        if results:
            r = results[0]
            assert "id" in r
            assert "content" in r
            assert "score" in r
            assert "tags" in r
            assert "created_at" in r


# ── History ───────────────────────────────────────────────────────────────────

class TestAIHistory:
    @pytest.mark.asyncio
    async def test_history_empty_initially(self, ai) -> None:
        assert ai.history == []

    @pytest.mark.asyncio
    async def test_history_after_chat(self, ai) -> None:
        await ai.achat("Hello")
        assert len(ai.history) >= 2  # user + assistant

    @pytest.mark.asyncio
    async def test_clear_history(self, ai) -> None:
        await ai.achat("Hello")
        ai.clear_history()
        assert ai.history == []

    @pytest.mark.asyncio
    async def test_memory_persists_after_clear(self, ai) -> None:
        await ai.aremember("Persistent fact")
        ai.clear_history()
        results = await ai.asearch_memory("Persistent fact")
        assert isinstance(results, list)  # memory store untouched


# ── Internal helpers ─────────────────────────────────────────────────────────

class TestAIInternals:
    @pytest.mark.asyncio
    async def test_llm_call_simple_returns_string(self, ai) -> None:
        await ai._ensure_initialized()
        result = await ai._llm_call_simple(
            [{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_llm_call_simple_with_max_tokens(self, ai) -> None:
        await ai._ensure_initialized()
        result = await ai._llm_call_simple(
            [{"role": "user", "content": "Brief answer please"}],
            model="gpt-4o-mini",
            max_tokens=100,
        )
        assert isinstance(result, str)


# ── Analytics ─────────────────────────────────────────────────────────────────

class TestAIAnalytics:
    @pytest.mark.asyncio
    async def test_cost_summary(self, ai) -> None:
        await ai.achat("Hello")
        summary = ai.cost_summary
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_metrics(self, ai) -> None:
        await ai.achat("Hello")
        m = ai.metrics
        assert isinstance(m, dict)


# ── Provider building ─────────────────────────────────────────────────────────

class TestBuildProviders:
    def test_build_providers_openrouter_key_present(self, ai_config) -> None:
        # ai_config has openrouter_api_key set
        with patch("openai.AsyncOpenAI"):
            instance = AI(config=ai_config)
            providers = instance._build_providers()
        assert "openrouter" in providers

    def test_build_providers_nvidia(self, tmp_path) -> None:
        cfg = AICOSConfig(
            nvidia_api_key="nvapi-test",
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            db_path=str(tmp_path / "aicos.db"),
        )
        with patch("openai.AsyncOpenAI"):
            instance = AI(config=cfg)
            providers = instance._build_providers()
        assert "nvidia" in providers
        assert "openai" not in providers

    def test_build_providers_openrouter(self, tmp_path) -> None:
        cfg = AICOSConfig(
            openrouter_api_key="sk-or-test",
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            nvidia_api_key=None,
            db_path=str(tmp_path / "aicos.db"),
        )
        with patch("openai.AsyncOpenAI"):
            instance = AI(config=cfg)
            providers = instance._build_providers()
        assert "openrouter" in providers

    def test_build_providers_no_keys_excludes_openai(self, tmp_path) -> None:
        cfg = AICOSConfig(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            nvidia_api_key=None,
            db_path=str(tmp_path / "aicos.db"),
        )
        instance = AI(config=cfg)
        providers = instance._build_providers()
        assert "openai" not in providers
        assert "anthropic" not in providers
        assert "nvidia" not in providers
