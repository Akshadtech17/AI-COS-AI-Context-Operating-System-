"""
History Manager with Lost-in-the-Middle (LITM) solver.

Problem: LLMs have poor recall for information in the "middle" of long contexts.
Solution: When conversation exceeds threshold, compress the middle into a recap
and keep: [System] → [Recap] → [Last N turns].

This ensures the most relevant context is always at the beginning and end of
the context window, where LLM attention is strongest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aicos.context.compressor import ContextCompressor, count_message_tokens
from aicos.context.summarizer import ConversationSummarizer


@dataclass
class HistoryState:
    messages: list[dict[str, Any]]
    total_tokens: int
    litm_applied: bool = False
    compression_applied: bool = False
    recap_injected: bool = False


class HistoryManager:
    """
    Manages conversation history with automatic LITM resolution and compression.

    Pipeline (triggered when context exceeds litm_threshold_tokens):
    1. Separate system messages from conversation
    2. Keep last `preserve_turns` turns verbatim
    3. Take middle section (everything before the preserved turns)
    4. Summarize the middle section via LLM
    5. Inject summary as a synthetic system/user message
    6. If still over budget, apply extractive compression to remaining messages
    """

    RECAP_ROLE = "system"
    RECAP_PREFIX = "📋 **Context Recap** (earlier conversation compressed):\n"

    def __init__(
        self,
        compressor: ContextCompressor,
        summarizer: ConversationSummarizer | None = None,
        max_tokens: int = 8000,
        litm_threshold_tokens: int = 6000,
        preserve_turns: int = 3,
        model: str = "gpt-4o-mini",
    ) -> None:
        self._compressor = compressor
        self._summarizer = summarizer
        self._max_tokens = max_tokens
        self._litm_threshold = litm_threshold_tokens
        self._preserve_turns = preserve_turns
        self._model = model

    async def process(
        self, messages: list[dict[str, Any]]
    ) -> HistoryState:
        """
        Process messages and apply LITM + compression as needed.
        Always returns a message list safe to send to an LLM.
        """
        token_count = count_message_tokens(messages, self._model)

        if token_count <= self._litm_threshold:
            return HistoryState(
                messages=messages,
                total_tokens=token_count,
            )

        # Apply LITM
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        min_conv = self._preserve_turns * 2  # user + assistant pairs

        if len(conv_msgs) <= min_conv:
            # Not enough turns to apply LITM — just compress
            result = self._compressor.compress(messages, budget=self._max_tokens)
            return HistoryState(
                messages=result.messages,
                total_tokens=result.tokens_after,
                compression_applied=result.tokens_saved > 0,
            )

        preserved = conv_msgs[-min_conv:]
        middle = conv_msgs[:-min_conv]

        recap_text: str | None = None
        litm_applied = False

        if self._summarizer and middle:
            try:
                summary = await self._summarizer.summarize_turns(middle)
                if summary:
                    recap_text = self.RECAP_PREFIX + summary
                    litm_applied = True
            except Exception:
                # Summarizer failed — fall back to extractive compression
                pass

        if recap_text:
            recap_msg: dict[str, Any] = {
                "role": self.RECAP_ROLE,
                "content": recap_text,
            }
            assembled = system_msgs + [recap_msg] + preserved
        else:
            # Extractive fallback: compress middle without LLM
            result = self._compressor.compress(middle, budget=1000)
            assembled = system_msgs + result.messages + preserved

        # Final compression pass if still over budget
        final_tokens = count_message_tokens(assembled, self._model)
        if final_tokens > self._max_tokens:
            compressed = self._compressor.compress(assembled, budget=self._max_tokens)
            assembled = compressed.messages
            final_tokens = compressed.tokens_after

        return HistoryState(
            messages=assembled,
            total_tokens=final_tokens,
            litm_applied=litm_applied,
            compression_applied=True,
            recap_injected=recap_text is not None,
        )

    def truncate_to_budget(
        self, messages: list[dict[str, Any]], budget: int
    ) -> list[dict[str, Any]]:
        """
        Hard truncation as last resort — removes oldest non-system messages.
        Should only be called if process() still exceeds budget.
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        while conv_msgs and count_message_tokens(system_msgs + conv_msgs, self._model) > budget:
            conv_msgs.pop(0)

        return system_msgs + conv_msgs
