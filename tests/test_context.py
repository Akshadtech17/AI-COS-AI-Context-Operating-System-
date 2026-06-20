"""Tests for context compression, token counting, and LITM solver."""

from __future__ import annotations

import pytest

from aicos.context.compressor import (
    ContextCompressor,
    CompressionResult,
    count_tokens,
    count_message_tokens,
)
from aicos.context.history_manager import HistoryManager
from aicos.context.compressor import count_message_tokens as _cmtokens


class TestTokenCounting:
    def test_count_tokens_string(self) -> None:
        count = count_tokens("Hello world")
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_empty(self) -> None:
        count = count_tokens("")
        assert count == 0

    def test_count_tokens_longer_is_more(self) -> None:
        short = count_tokens("Hello")
        long = count_tokens("Hello world, this is a longer sentence with more words")
        assert long > short

    def test_count_message_tokens(self) -> None:
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help?"},
        ]
        total = count_message_tokens(messages)
        assert total > 0
        assert isinstance(total, int)

    def test_count_message_tokens_empty_list(self) -> None:
        count = count_message_tokens([])
        assert count >= 0


class TestContextCompressor:
    @pytest.fixture
    def compressor(self) -> ContextCompressor:
        return ContextCompressor(max_tokens=500, preserve_last_turns=2)

    def test_short_conversation_unchanged(self, compressor: ContextCompressor) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = compressor.compress(messages)
        assert result.tokens_before == result.tokens_after
        assert result.compression_ratio == 0.0

    def test_long_conversation_compressed(self, compressor: ContextCompressor) -> None:
        # Build a conversation that exceeds the budget
        long_content = " ".join(["word"] * 200)
        messages = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
            {"role": "user", "content": "Short final message"},
        ]
        result = compressor.compress(messages, budget=200)
        assert result.tokens_after <= result.tokens_before

    def test_system_message_preserved(self, compressor: ContextCompressor) -> None:
        system_content = "You are an expert assistant with specialized knowledge."
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": " ".join(["word"] * 300)},
        ]
        result = compressor.compress(messages, budget=100)
        # System message must be in result
        system_msgs = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == system_content

    def test_compression_result_fields(self, compressor: ContextCompressor) -> None:
        messages = [{"role": "user", "content": " ".join(["word"] * 100)}]
        result = compressor.compress(messages)
        assert isinstance(result, CompressionResult)
        assert result.tokens_before >= 0
        assert result.tokens_after >= 0
        assert 0.0 <= result.compression_ratio <= 1.0

    def test_tokens_saved_calculation(self, compressor: ContextCompressor) -> None:
        messages = [{"role": "user", "content": " ".join(["word"] * 200)}]
        result = compressor.compress(messages, budget=50)
        assert result.tokens_saved == result.tokens_before - result.tokens_after

    def test_code_block_preserved(self, compressor: ContextCompressor) -> None:
        code = "```python\ndef my_function():\n    return 42\n```"
        filler = " ".join(["padding word"] * 100)
        messages = [
            {"role": "user", "content": f"{filler}\n\n{code}"},
        ]
        result = compressor.compress(messages, budget=100)
        # Code block should survive in at least one message
        all_content = " ".join(str(m.get("content", "")) for m in result.messages)
        assert "def my_function" in all_content or "my_function" in all_content

    def test_compress_preserves_last_turns(self, compressor: ContextCompressor) -> None:
        long = " ".join(["word"] * 200)
        final_user = "This is the final important question"
        final_assistant = "This is the final important answer"
        messages = [
            {"role": "user", "content": long},
            {"role": "assistant", "content": long},
            {"role": "user", "content": long},
            {"role": "assistant", "content": long},
            {"role": "user", "content": final_user},
            {"role": "assistant", "content": final_assistant},
        ]
        result = compressor.compress(messages, budget=150)
        all_content = " ".join(str(m.get("content", "")) for m in result.messages)
        assert "final important" in all_content


class TestHistoryManager:
    @pytest.fixture
    def history_manager(self) -> HistoryManager:
        # preserve_last_turns=1 so compression can fire on small test conversations
        compressor = ContextCompressor(max_tokens=500, preserve_last_turns=1)
        return HistoryManager(
            compressor=compressor,
            summarizer=None,  # No LLM summarizer in unit tests
            max_tokens=500,
            litm_threshold_tokens=300,
            preserve_turns=2,
        )

    @pytest.mark.asyncio
    async def test_short_history_unchanged(self, history_manager: HistoryManager) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        state = await history_manager.process(messages)
        assert len(state.messages) >= 2
        assert state.litm_applied is False

    @pytest.mark.asyncio
    async def test_long_history_compressed(self, history_manager: HistoryManager) -> None:
        # Use short but multi-sentence paragraphs so extractive compression can work.
        # Each paragraph fits within the preserve budget so the middle section can be
        # compressed meaningfully.
        para = (
            "Transformers use self-attention to relate tokens. "
            "This enables parallel processing across the sequence. "
            "Positional encodings preserve order information. "
        )
        # Middle messages need to exceed max_tokens (500) so final compression fires
        messages = [
            {"role": "user", "content": para * 15},      # middle (~225 tokens)
            {"role": "assistant", "content": para * 15}, # middle (~225 tokens)
            {"role": "user", "content": para},            # preserved
            {"role": "assistant", "content": para},       # preserved
            {"role": "user", "content": "Final question"},
            {"role": "assistant", "content": "Final answer"},
        ]
        original_tokens = _cmtokens(messages)
        # Total should be >> max_tokens (500) to trigger compression
        assert original_tokens > 500, "Test data must exceed max_tokens to exercise compression"
        state = await history_manager.process(messages)
        assert state.total_tokens < original_tokens

    @pytest.mark.asyncio
    async def test_system_message_preserved_in_litm(self, history_manager: HistoryManager) -> None:
        system = "You are a specialized assistant."
        long = " ".join(["word"] * 80)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": long},
            {"role": "assistant", "content": long},
            {"role": "user", "content": long},
            {"role": "assistant", "content": long},
            {"role": "user", "content": "Final"},
        ]
        state = await history_manager.process(messages)
        system_msgs = [m for m in state.messages if m.get("role") == "system"]
        assert any(system in str(m.get("content", "")) for m in system_msgs)

    def test_truncate_to_budget(self, history_manager: HistoryManager) -> None:
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": " ".join(["w"] * 100)},
            {"role": "assistant", "content": " ".join(["w"] * 100)},
            {"role": "user", "content": "Final"},
        ]
        truncated = history_manager.truncate_to_budget(messages, budget=50)
        from aicos.context.compressor import count_message_tokens
        assert count_message_tokens(truncated) <= 80  # Some tolerance
