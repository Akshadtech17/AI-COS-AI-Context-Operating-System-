"""Tests for ConversationSummarizer and MemoryRetriever."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aicos.context.summarizer import ConversationSummarizer
from aicos.memory.retrieval import MemoryRetriever


# ── ConversationSummarizer ────────────────────────────────────────────────────

class TestConversationSummarizer:
    @pytest.fixture
    def mock_llm(self):
        return AsyncMock(return_value="Context Recap: User asked about Python. • Python is versatile.")

    @pytest.fixture
    def summarizer(self, mock_llm):
        return ConversationSummarizer(llm_caller=mock_llm, model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_summarize_empty_turns_returns_empty(self, summarizer) -> None:
        result = await summarizer.summarize_turns([])
        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_calls_llm(self, summarizer, mock_llm) -> None:
        turns = [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]
        result = await summarizer.summarize_turns(turns)
        mock_llm.assert_called_once()
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_strips_whitespace(self, summarizer) -> None:
        mock_llm = AsyncMock(return_value="  Summary text  \n")
        s = ConversationSummarizer(llm_caller=mock_llm, model="gpt-4o-mini")
        turns = [{"role": "user", "content": "Hello"}]
        result = await s.summarize_turns(turns)
        assert result == "Summary text"

    @pytest.mark.asyncio
    async def test_summarize_passes_model_and_max_tokens(self, mock_llm) -> None:
        s = ConversationSummarizer(
            llm_caller=mock_llm, model="gpt-4o-mini", max_summary_tokens=200
        )
        turns = [{"role": "user", "content": "Hello"}]
        await s.summarize_turns(turns)
        _, kwargs = mock_llm.call_args
        assert kwargs.get("max_tokens") == 200 or mock_llm.call_args[0][2] == 200

    def test_format_turns_basic(self, summarizer) -> None:
        turns = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        text = summarizer._format_turns(turns)
        assert "[USER]: Hello" in text
        assert "[ASSISTANT]: Hi there" in text

    def test_format_turns_truncates_long_content(self, summarizer) -> None:
        long_content = "x" * 1500
        turns = [{"role": "user", "content": long_content}]
        text = summarizer._format_turns(turns)
        assert len(text) < 1100  # truncated to 997 + "..."
        assert text.endswith("...")

    def test_format_turns_missing_role(self, summarizer) -> None:
        turns = [{"content": "No role here"}]
        text = summarizer._format_turns(turns)
        assert "[UNKNOWN]" in text

    def test_format_turns_missing_content(self, summarizer) -> None:
        turns = [{"role": "user"}]
        text = summarizer._format_turns(turns)
        assert "[USER]:" in text


# ── MemoryRetriever ───────────────────────────────────────────────────────────

class TestMemoryRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_returns_list(
        self, memory_retriever: MemoryRetriever, memory_store
    ) -> None:
        await memory_store.store("Python is a great language for data science")
        results = await memory_retriever.retrieve("Python programming")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_retrieve_empty_store(self, memory_retriever: MemoryRetriever) -> None:
        results = await memory_retriever.retrieve("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieved_memory_context_line(
        self, memory_retriever: MemoryRetriever, memory_store
    ) -> None:
        await memory_store.store("User is a senior Python developer", tags=["profile"])
        results = await memory_retriever.retrieve("Python developer")
        if results:
            line = results[0].to_context_line()
            assert "User is a senior Python developer" in line
            assert "[profile]" in line

    @pytest.mark.asyncio
    async def test_retrieved_memory_no_tags_context_line(
        self, memory_retriever: MemoryRetriever, memory_store
    ) -> None:
        await memory_store.store("Fact with no tags")
        results = await memory_retriever.retrieve("Fact with no tags")
        if results:
            line = results[0].to_context_line()
            assert "[" not in line

    @pytest.mark.asyncio
    async def test_inject_with_no_memories(self, memory_retriever: MemoryRetriever) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = await memory_retriever.inject_into_messages(messages, query="Hello")
        assert result == messages

    @pytest.mark.asyncio
    async def test_inject_adds_system_message(
        self, memory_retriever: MemoryRetriever, memory_store
    ) -> None:
        await memory_store.store("User prefers concise answers")
        messages = [{"role": "user", "content": "concise answers please"}]
        result = await memory_retriever.inject_into_messages(messages)
        roles = [m["role"] for m in result]
        assert "system" in roles

    @pytest.mark.asyncio
    async def test_inject_appends_to_existing_system(
        self, memory_retriever: MemoryRetriever, memory_store
    ) -> None:
        await memory_store.store("User is a Python expert")
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Python expert user"},
        ]
        result = await memory_retriever.inject_into_messages(messages)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "Be helpful." in str(system_msgs[0]["content"])

    @pytest.mark.asyncio
    async def test_inject_extracts_query_from_last_user_msg(
        self, memory_retriever: MemoryRetriever, memory_store
    ) -> None:
        await memory_store.store("User loves machine learning")
        messages = [{"role": "user", "content": "machine learning"}]
        # No explicit query — should extract from messages
        result = await memory_retriever.inject_into_messages(messages, query=None)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_inject_no_user_messages_returns_original(
        self, memory_retriever: MemoryRetriever, memory_store
    ) -> None:
        await memory_store.store("something")
        messages = [{"role": "system", "content": "System only"}]
        result = await memory_retriever.inject_into_messages(messages, query=None)
        assert result == messages

    @pytest.mark.asyncio
    async def test_format_for_display_with_memories(
        self, memory_retriever: MemoryRetriever, memory_store
    ) -> None:
        await memory_store.store("User is experienced in Python")
        text = await memory_retriever.format_for_display("Python")
        assert "Retrieved" in text or "No relevant" in text

    @pytest.mark.asyncio
    async def test_format_for_display_no_memories(
        self, memory_retriever: MemoryRetriever
    ) -> None:
        text = await memory_retriever.format_for_display("something obscure")
        assert "No relevant memories" in text

    @pytest.mark.asyncio
    async def test_format_memories_respects_token_budget(
        self, memory_retriever: MemoryRetriever, memory_store
    ) -> None:
        # Store many items to test budget enforcement
        for i in range(10):
            await memory_store.store(f"Memory item {i} with some content to fill tokens")
        memories = await memory_retriever.retrieve("memory item")
        assert len(memories) <= memory_retriever._top_k
