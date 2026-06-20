"""
Conversation summarizer — generates LLM-powered abstractive summaries
of conversation segments for the LITM solver.
"""

from __future__ import annotations

from typing import Any


SUMMARIZE_SYSTEM_PROMPT = """\
You are a precise conversation summarizer.
Summarize the provided conversation turns into a dense, factual summary.

Requirements:
- Preserve ALL key facts, decisions, code snippets, and conclusions
- Use bullet points for clarity
- Be maximally concise — every word must earn its place
- Do not add interpretation or opinions
- Format: "Context Recap: ..." followed by bullet points"""


class ConversationSummarizer:
    """
    Generates abstractive summaries of conversation segments.
    Used by HistoryManager to compress the 'middle' section of long conversations.
    """

    def __init__(
        self,
        llm_caller: Any,  # callable: async (messages) -> str
        model: str = "gpt-4o-mini",
        max_summary_tokens: int = 500,
    ) -> None:
        self._llm = llm_caller
        self._model = model
        self._max_summary_tokens = max_summary_tokens

    async def summarize_turns(
        self, turns: list[dict[str, Any]]
    ) -> str:
        """
        Generate a summary of conversation turns.

        Args:
            turns: List of message dicts with role/content

        Returns:
            Compact text summary suitable for context injection
        """
        if not turns:
            return ""

        conversation_text = self._format_turns(turns)

        messages = [
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Summarize this conversation segment "
                    f"(max {self._max_summary_tokens} tokens):\n\n"
                    f"{conversation_text}"
                ),
            },
        ]

        summary = await self._llm(messages, model=self._model, max_tokens=self._max_summary_tokens)
        return summary.strip()

    def _format_turns(self, turns: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for turn in turns:
            role = str(turn.get("role", "unknown")).upper()
            content = str(turn.get("content", ""))
            if len(content) > 1000:
                content = content[:997] + "..."
            lines.append(f"[{role}]: {content}")
        return "\n\n".join(lines)
