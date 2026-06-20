"""Tests for OpenAI, Anthropic, and Gemini providers (all mocked — no live API calls)."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aicos.providers.base import ProviderResponse, StreamChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_openai_response(content: str = "Hello!", model: str = "gpt-4o-mini") -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.choices[0].finish_reason = "stop"
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.model = model
    resp.model_dump.return_value = {"id": "test"}
    return resp


async def _openai_chunk_stream() -> AsyncIterator[Any]:
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "Hello"
    chunk.choices[0].finish_reason = None
    chunk.model = "gpt-4o-mini"
    yield chunk

    usage_chunk = MagicMock()
    usage_chunk.choices = []
    usage_chunk.usage = MagicMock()
    usage_chunk.usage.prompt_tokens = 10
    usage_chunk.usage.completion_tokens = 5
    yield usage_chunk


# ── OpenAI Provider ───────────────────────────────────────────────────────────

class TestOpenAIProvider:
    @pytest.fixture
    def provider(self):
        with patch("openai.AsyncOpenAI"):
            from aicos.providers.openai_provider import OpenAIProvider
            p = OpenAIProvider(api_key="sk-test")
            p._client = AsyncMock()
            return p

    def test_name(self, provider) -> None:
        assert provider.name == "openai"

    @pytest.mark.asyncio
    async def test_is_available_true(self, provider) -> None:
        provider._client.models.list = AsyncMock(return_value=[])
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false(self, provider) -> None:
        provider._client.models.list = AsyncMock(side_effect=Exception("unreachable"))
        assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_complete_returns_response(self, provider) -> None:
        provider._client.chat.completions.create = AsyncMock(
            return_value=_make_openai_response("Hi there!")
        )
        messages = [{"role": "user", "content": "Hello"}]
        result = await provider.complete(messages, model="gpt-4o-mini")

        assert isinstance(result, ProviderResponse)
        assert result.content == "Hi there!"
        assert result.model == "gpt-4o-mini"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_complete_empty_content(self, provider) -> None:
        resp = _make_openai_response()
        resp.choices[0].message.content = None
        provider._client.chat.completions.create = AsyncMock(return_value=resp)
        result = await provider.complete([{"role": "user", "content": "Hi"}], model="gpt-4o-mini")
        assert result.content == ""

    @pytest.mark.asyncio
    async def test_complete_no_usage(self, provider) -> None:
        resp = _make_openai_response()
        resp.usage = None
        provider._client.chat.completions.create = AsyncMock(return_value=resp)
        result = await provider.complete([{"role": "user", "content": "Hi"}], model="gpt-4o-mini")
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider) -> None:
        provider._client.chat.completions.create = AsyncMock(
            return_value=_openai_chunk_stream()
        )
        messages = [{"role": "user", "content": "Hello"}]
        chunks = []
        async for chunk in provider.stream(messages, model="gpt-4o-mini"):
            chunks.append(chunk)

        assert any(c.delta == "Hello" for c in chunks)

    @pytest.mark.asyncio
    async def test_stream_usage_chunk(self, provider) -> None:
        provider._client.chat.completions.create = AsyncMock(
            return_value=_openai_chunk_stream()
        )
        chunks = []
        async for chunk in provider.stream([{"role": "user", "content": "Hi"}], model="gpt-4o-mini"):
            chunks.append(chunk)
        usage_chunks = [c for c in chunks if c.input_tokens and c.input_tokens > 0]
        assert len(usage_chunks) == 1
        assert usage_chunks[0].input_tokens == 10

    @pytest.mark.asyncio
    async def test_stream_with_base_url(self) -> None:
        with patch("openai.AsyncOpenAI") as mock_cls:
            from aicos.providers.openai_provider import OpenAIProvider
            OpenAIProvider(api_key="sk-test", base_url="https://openrouter.ai/api/v1")
            mock_cls.assert_called_once_with(
                api_key="sk-test", base_url="https://openrouter.ai/api/v1"
            )


# ── Anthropic Provider ────────────────────────────────────────────────────────

class TestAnthropicProvider:
    @pytest.fixture
    def provider(self):
        with patch("anthropic.AsyncAnthropic"):
            from aicos.providers.anthropic_provider import AnthropicProvider
            p = AnthropicProvider(api_key="sk-ant-test")
            p._client = AsyncMock()
            return p

    def test_name(self, provider) -> None:
        assert provider.name == "anthropic"

    @pytest.mark.asyncio
    async def test_is_available_true(self, provider) -> None:
        provider._client.models.list = AsyncMock(return_value=[])
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false(self, provider) -> None:
        provider._client.models.list = AsyncMock(side_effect=Exception("blocked"))
        assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_complete_returns_response(self, provider) -> None:
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Claude says hi")]
        mock_resp.model = "claude-haiku-4-5-20251001"
        mock_resp.usage.input_tokens = 8
        mock_resp.usage.output_tokens = 4
        mock_resp.stop_reason = "end_turn"
        mock_resp.model_dump.return_value = {}
        provider._client.messages.create = AsyncMock(return_value=mock_resp)

        result = await provider.complete(
            [{"role": "user", "content": "Hello"}],
            model="claude-haiku-4-5-20251001",
        )

        assert isinstance(result, ProviderResponse)
        assert result.content == "Claude says hi"
        assert result.input_tokens == 8
        assert result.finish_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_complete_separates_system_messages(self, provider) -> None:
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="ok")]
        mock_resp.model = "claude-haiku-4-5-20251001"
        mock_resp.usage.input_tokens = 5
        mock_resp.usage.output_tokens = 2
        mock_resp.stop_reason = "end_turn"
        mock_resp.model_dump.return_value = {}
        provider._client.messages.create = AsyncMock(return_value=mock_resp)

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        await provider.complete(messages, model="claude-haiku-4-5-20251001")

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "Be concise."
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    @pytest.mark.asyncio
    async def test_complete_no_stop_reason(self, provider) -> None:
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="hi")]
        mock_resp.model = "claude-haiku-4-5-20251001"
        mock_resp.usage.input_tokens = 3
        mock_resp.usage.output_tokens = 1
        mock_resp.stop_reason = None
        mock_resp.model_dump.return_value = {}
        provider._client.messages.create = AsyncMock(return_value=mock_resp)

        result = await provider.complete(
            [{"role": "user", "content": "Hi"}], model="claude-haiku-4-5-20251001"
        )
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider) -> None:
        mock_final = MagicMock()
        mock_final.model = "claude-haiku-4-5-20251001"
        mock_final.usage.input_tokens = 10
        mock_final.usage.output_tokens = 6
        mock_final.stop_reason = "end_turn"

        async def fake_text_stream():
            yield "Hello"
            yield " world"

        mock_stream_obj = MagicMock()
        mock_stream_obj.text_stream = fake_text_stream()
        mock_stream_obj.get_final_message = AsyncMock(return_value=mock_final)

        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_stream_obj)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        provider._client.messages.stream = MagicMock(return_value=mock_ctx)

        chunks = []
        async for chunk in provider.stream(
            [{"role": "user", "content": "Hi"}], model="claude-haiku-4-5-20251001"
        ):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if c.delta]
        assert any("Hello" in c.delta for c in text_chunks)
        final_chunk = chunks[-1]
        assert final_chunk.input_tokens == 10
        assert final_chunk.finish_reason == "end_turn"


# ── Gemini Provider ───────────────────────────────────────────────────────────

class TestGeminiProvider:
    @pytest.fixture
    def provider(self):
        mock_genai = MagicMock()
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            from aicos.providers.gemini_provider import GeminiProvider
            p = GeminiProvider(api_key="test-key")
            p._genai = mock_genai
            return p

    def test_name(self, provider) -> None:
        assert provider.name == "gemini"

    def test_extract_model_name(self, provider) -> None:
        assert provider._extract_model_name("gemini/gemini-2.0-flash") == "gemini-2.0-flash"
        assert provider._extract_model_name("gemini-2.0-flash") == "gemini-2.0-flash"

    def test_convert_messages_mixed(self, provider) -> None:
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Thanks"},
        ]
        system, history = provider._convert_messages(messages)
        assert "Be helpful." in system
        assert any(h["role"] == "user" for h in history)
        assert any(h["role"] == "model" for h in history)

    def test_convert_messages_no_system(self, provider) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        system, history = provider._convert_messages(messages)
        assert system == ""
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_is_available_true(self, provider) -> None:
        provider._genai.list_models.return_value = []
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false(self, provider) -> None:
        provider._genai.list_models.side_effect = Exception("no access")
        assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_complete_returns_response(self, provider) -> None:
        mock_response = MagicMock()
        mock_response.text = "Gemini says hello"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        provider._genai.GenerativeModel.return_value = mock_model
        provider._genai.GenerationConfig.return_value = MagicMock()

        result = await provider.complete(
            [{"role": "user", "content": "Hello"}],
            model="gemini/gemini-2.0-flash",
        )

        assert isinstance(result, ProviderResponse)
        assert result.content == "Gemini says hello"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    @pytest.mark.asyncio
    async def test_complete_no_usage_metadata(self, provider) -> None:
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage_metadata = None

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        provider._genai.GenerativeModel.return_value = mock_model
        provider._genai.GenerationConfig.return_value = MagicMock()

        result = await provider.complete(
            [{"role": "user", "content": "Hi"}],
            model="gemini/gemini-2.0-flash",
        )
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @pytest.mark.asyncio
    async def test_complete_with_system_message(self, provider) -> None:
        mock_response = MagicMock()
        mock_response.text = "ok"
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 2

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = mock_response
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        provider._genai.GenerativeModel.return_value = mock_model
        provider._genai.GenerationConfig.return_value = MagicMock()

        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        result = await provider.complete(messages, model="gemini/gemini-2.0-flash")
        assert result.content == "ok"
        # System instruction passed to GenerativeModel
        provider._genai.GenerativeModel.assert_called_once()
        call_kwargs = provider._genai.GenerativeModel.call_args.kwargs
        assert call_kwargs.get("system_instruction") == "Be concise."

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider) -> None:
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk1.usage_metadata = None

        chunk2 = MagicMock()
        chunk2.text = " world"
        chunk2.usage_metadata.prompt_token_count = 8
        chunk2.usage_metadata.candidates_token_count = 4

        mock_chat = MagicMock()
        mock_chat.send_message.return_value = [chunk1, chunk2]
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        provider._genai.GenerativeModel.return_value = mock_model
        provider._genai.GenerationConfig.return_value = MagicMock()

        chunks = []
        async for chunk in provider.stream(
            [{"role": "user", "content": "Hi"}], model="gemini/gemini-2.0-flash"
        ):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if c.delta]
        assert any("Hello" in c.delta for c in text_chunks)
        final = chunks[-1]
        assert final.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_no_last_user_message(self, provider) -> None:
        mock_chat = MagicMock()
        mock_chat.send_message.return_value = []
        mock_model = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        provider._genai.GenerativeModel.return_value = mock_model
        provider._genai.GenerationConfig.return_value = MagicMock()

        chunks = []
        async for chunk in provider.stream(
            [{"role": "assistant", "content": "I said something"}],
            model="gemini/gemini-2.0-flash",
        ):
            chunks.append(chunk)

        assert chunks[-1].finish_reason == "stop"
