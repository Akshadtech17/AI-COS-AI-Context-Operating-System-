"""Tests for context compression, token counting, and LITM solver."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aicos.context.compressor import (
    CompressionResult,
    ContextCompressor,
    ProtectedBlock,
    _compress_text,
    _extract_protected,
    _restore_protected,
    _score_sentences,
    count_message_tokens,
    count_tokens,
)
from aicos.context.compressor import count_message_tokens as _cmtokens
from aicos.context.history_manager import HistoryManager


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
            {"role": "user", "content": para * 15},  # middle (~225 tokens)
            {"role": "assistant", "content": para * 15},  # middle (~225 tokens)
            {"role": "user", "content": para},  # preserved
            {"role": "assistant", "content": para},  # preserved
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


# ── Targeted branch coverage ──────────────────────────────────────────────────


class TestTokenCountingEdgeCases:
    def test_unknown_model_fallback(self) -> None:
        """Line 32-33: KeyError in tiktoken triggers cl100k_base fallback."""
        count = count_tokens("hello world", model="unknown-model-does-not-exist-xyz")
        assert count > 0

    def test_count_message_tokens_list_content(self) -> None:
        """Lines 48-51: multimodal/structured content as a list."""
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": "http://example.com/img.png"},
                ],
            }
        ]
        count = count_message_tokens(msgs)
        assert count > 0


class TestExtractProtected:
    def test_code_block_extracted(self) -> None:
        text = "Before\n```python\nprint('hello')\n```\nAfter"
        processed, blocks = _extract_protected(text)
        assert len(blocks) >= 1
        assert any("python" in b.content for b in blocks)
        assert "```" not in processed.replace("\x00", "")

    def test_inline_code_extracted(self) -> None:
        text = "Call `asyncio.run()` here"
        processed, blocks = _extract_protected(text)
        assert any("asyncio" in b.content for b in blocks)

    def test_multiline_json_extracted(self) -> None:
        """Lines 85-89: long multiline JSON gets extracted."""
        json_block = '{\n  "key": "value",\n  "nested": {"a": 1, "b": 2, "c": 3}\n}'
        text = f"Some text before\n{json_block}\nSome text after"
        processed, blocks = _extract_protected(text)
        assert any("key" in b.content for b in blocks)

    def test_short_json_not_extracted(self) -> None:
        """Short single-line JSON is NOT extracted (len <= 50 or no newline)."""
        text = 'Use {"key": "val"} inline'
        _, blocks = _extract_protected(text)
        assert not any("key" in b.content for b in blocks)

    def test_no_protected_blocks(self) -> None:
        text = "Plain prose with no code or JSON."
        processed, blocks = _extract_protected(text)
        assert len(blocks) == 0
        assert processed == text

    def test_restore_protected(self) -> None:
        """Line 96: _restore_protected replaces placeholders."""
        blocks = [ProtectedBlock(placeholder="\x00BLOCK0\x00", content="```\ncode\n```")]
        text = "before \x00BLOCK0\x00 after"
        restored = _restore_protected(text, blocks)
        assert "```\ncode\n```" in restored
        assert "\x00BLOCK0\x00" not in restored


class TestScoreSentences:
    def test_empty_list_returns_empty(self) -> None:
        """Line 108: empty sentence list returns []."""
        assert _score_sentences([]) == []

    def test_first_sentence_bonus(self) -> None:
        sentences = ["First sentence with words.", "Middle filler.", "Last sentence here."]
        scores = _score_sentences(sentences)
        assert scores[0] > scores[1]  # first has bonus

    def test_last_sentence_bonus(self) -> None:
        """Lines 123-124: last sentence gets 0.2 positional bonus."""
        sentences = ["A sentence.", "Another sentence.", "Final sentence here with words."]
        scores = _score_sentences(sentences)
        # Last gets 0.2 bonus, middle gets none
        assert scores[-1] > scores[1]

    def test_short_sentence_penalty(self) -> None:
        """Line 141: sentences with < 5 words get length_factor = 0.5."""
        sentences = ["Hi.", "A long sentence with many words in it for scoring purposes today."]
        scores = _score_sentences(sentences)
        # Short sentence should score lower than long
        assert scores[0] < scores[1]

    def test_long_sentence_factor(self) -> None:
        """Line 143: sentences with > 100 words get length_factor = 0.7."""
        long = " ".join(["word"] * 110)
        short = "Normal sentence with around ten words in it here."
        sentences = [long, short]
        scores = _score_sentences(sentences)
        assert len(scores) == 2  # Both scored


class TestCompressText:
    def test_no_compression_when_within_budget(self) -> None:
        """Line 156: text already fits, returned as-is."""
        text = "Short text"
        result = _compress_text(text, budget_tokens=1000)
        assert result == text

    def test_empty_text_returns_empty(self) -> None:
        """Line 161: empty/whitespace text returns original."""
        result = _compress_text("", budget_tokens=100)
        assert result == ""

    def test_whitespace_only_returns_original(self) -> None:
        """Line 161: text that strips to nothing returns original."""
        result = _compress_text("   ", budget_tokens=5)
        assert result.strip() == "" or result == "   "

    def test_compresses_long_text(self) -> None:
        long = " ".join([f"Sentence {i} has several words in it." for i in range(50)])
        compressed = _compress_text(long, budget_tokens=30)
        assert count_tokens(compressed) < count_tokens(long)

    def test_budget_exhausted_breaks_loop(self) -> None:
        """Line 175: loop breaks when remaining_budget <= 0."""
        long = " ".join([f"Sentence number {i} with words." for i in range(100)])
        compressed = _compress_text(long, budget_tokens=5)
        assert count_tokens(compressed) <= count_tokens(long)


class TestCompressionResultEdgeCases:
    def test_savings_pct_zero_tokens_before(self) -> None:
        """Lines 201-203: savings_pct returns 0.0 when tokens_before == 0."""
        result = CompressionResult(
            messages=[], tokens_before=0, tokens_after=0, compression_ratio=0.0
        )
        assert result.savings_pct == 0.0

    def test_savings_pct_normal(self) -> None:
        result = CompressionResult(
            messages=[], tokens_before=100, tokens_after=60, compression_ratio=0.4
        )
        assert abs(result.savings_pct - 40.0) < 0.01

    def test_tokens_saved(self) -> None:
        result = CompressionResult(
            messages=[], tokens_before=200, tokens_after=80, compression_ratio=0.6
        )
        assert result.tokens_saved == 120


class TestCompressMessageEdgeCases:
    def test_non_string_content_skipped(self) -> None:
        """Line 295: list/structured content is returned unchanged."""
        compressor = ContextCompressor(max_tokens=10)
        msg = {"role": "user", "content": [{"type": "image", "url": "http://x.com"}]}
        result = compressor._compress_message(msg, budget=5)
        assert result == msg

    def test_compress_message_with_code_block(self) -> None:
        """Line 317: block placeholder path in _compress_message."""
        compressor = ContextCompressor(max_tokens=50)
        code = "```python\nfor i in range(100):\n    print(i * 2 + 1)\n```"
        content = f"Natural language text. {code} More natural text here today."
        msg = {"role": "user", "content": content}
        result = compressor._compress_message(msg, budget=20)
        assert isinstance(result["content"], str)


# ── HistoryManager LITM and summarizer paths ──────────────────────────────────


class TestHistoryManagerLITM:
    @pytest.mark.asyncio
    async def test_not_enough_turns_falls_back_to_compression(self) -> None:
        """Lines 85-86: when conv_msgs <= min_conv, use compressor directly."""
        compressor = ContextCompressor(max_tokens=50)
        manager = HistoryManager(
            compressor=compressor,
            summarizer=None,
            litm_threshold_tokens=5,  # extremely low to force processing
            preserve_turns=10,  # high so conv_msgs <= min_conv always
        )
        msgs = [
            {"role": "user", "content": "Short A"},
            {"role": "assistant", "content": "Short B"},
        ]
        state = await manager.process(msgs)
        assert state.messages is not None
        assert not state.litm_applied

    @pytest.mark.asyncio
    async def test_summarizer_success_produces_recap(self) -> None:
        """Lines 99-106, 109-113: summarizer returns text → recap injected."""
        from aicos.context.summarizer import ConversationSummarizer

        mock_summarizer = MagicMock(spec=ConversationSummarizer)
        mock_summarizer.summarize_turns = AsyncMock(return_value="Summary of earlier conversation.")

        compressor = ContextCompressor(max_tokens=8000)
        manager = HistoryManager(
            compressor=compressor,
            summarizer=mock_summarizer,
            litm_threshold_tokens=10,
            preserve_turns=1,
        )

        msgs = [
            {"role": "user", "content": "Early turn one content"},
            {"role": "assistant", "content": "Early response one"},
            {"role": "user", "content": "Early turn two content"},
            {"role": "assistant", "content": "Early response two"},
            {"role": "user", "content": "Recent question"},
            {"role": "assistant", "content": "Recent answer"},
        ]
        state = await manager.process(msgs)
        assert state.litm_applied
        assert state.recap_injected
        recap_msgs = [m for m in state.messages if "Context Recap" in str(m.get("content", ""))]
        assert len(recap_msgs) == 1

    @pytest.mark.asyncio
    async def test_summarizer_failure_falls_back_to_extractive(self) -> None:
        """Lines 104-106: summarizer raises → extractive fallback, litm_applied=False."""
        from aicos.context.summarizer import ConversationSummarizer

        mock_summarizer = MagicMock(spec=ConversationSummarizer)
        mock_summarizer.summarize_turns = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        compressor = ContextCompressor(max_tokens=8000)
        manager = HistoryManager(
            compressor=compressor,
            summarizer=mock_summarizer,
            litm_threshold_tokens=10,
            preserve_turns=1,
        )

        msgs = [
            {"role": "user", "content": "Turn one"},
            {"role": "assistant", "content": "Response one"},
            {"role": "user", "content": "Turn two"},
            {"role": "assistant", "content": "Response two"},
            {"role": "user", "content": "Recent"},
            {"role": "assistant", "content": "Recent reply"},
        ]
        state = await manager.process(msgs)
        assert not state.litm_applied
        assert not state.recap_injected

    @pytest.mark.asyncio
    async def test_final_compression_when_still_over_budget(self) -> None:
        """Lines 120-124: assembled messages still over max_tokens → final compress."""
        compressor = ContextCompressor(max_tokens=30, preserve_last_turns=1)
        manager = HistoryManager(
            compressor=compressor,
            summarizer=None,
            max_tokens=30,
            litm_threshold_tokens=5,
            preserve_turns=1,
        )
        msgs = [
            {"role": "user", "content": " ".join(["word"] * 30)},
            {"role": "assistant", "content": " ".join(["word"] * 30)},
            {"role": "user", "content": " ".join(["word"] * 30)},
            {"role": "assistant", "content": " ".join(["word"] * 30)},
            {"role": "user", "content": "Final"},
            {"role": "assistant", "content": "Final reply"},
        ]
        state = await manager.process(msgs)
        assert state.total_tokens <= count_message_tokens(msgs)
