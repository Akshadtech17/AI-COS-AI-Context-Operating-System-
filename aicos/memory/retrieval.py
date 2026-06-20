"""
Memory retrieval module — fetches relevant memories and formats them for
injection into LLM context windows.
"""

from __future__ import annotations

from dataclasses import dataclass

from aicos.memory.memory_store import MemoryItem, MemoryStore


@dataclass
class RetrievedMemory:
    id: int
    content: str
    score: float
    tags: list[str]

    def to_context_line(self) -> str:
        tag_str = f" [{', '.join(self.tags)}]" if self.tags else ""
        return f"- {self.content}{tag_str}"


class MemoryRetriever:
    """
    Retrieves and ranks memories for context injection.

    Injection strategy:
    - Retrieve top-K memories by composite score
    - Format as a structured memory block
    - Insert before the last user message for maximum attention
    """

    MEMORY_SYSTEM_PREFIX = (
        "The following memories are relevant to this conversation. "
        "Use them to provide personalized and contextually aware responses:\n"
    )

    def __init__(
        self,
        store: MemoryStore,
        top_k: int = 5,
        threshold: float = 0.3,
        max_injection_tokens: int = 500,
    ) -> None:
        self._store = store
        self._top_k = top_k
        self._threshold = threshold
        self._max_injection_tokens = max_injection_tokens

    async def retrieve(
        self,
        query: str,
        tags: list[str] | None = None,
    ) -> list[RetrievedMemory]:
        """Retrieve relevant memories for the given query."""
        results = await self._store.search(
            query=query,
            top_k=self._top_k,
            threshold=self._threshold,
            tags=tags,
        )
        return [
            RetrievedMemory(
                id=item.id,
                content=item.content,
                score=score,
                tags=item.tag_list,
            )
            for item, score in results
        ]

    async def inject_into_messages(
        self,
        messages: list[dict[str, object]],
        query: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, object]]:
        """
        Inject relevant memories into the messages list.

        Injects a system-role memory block before the first non-system message.
        If a system message already exists, appends memory context to it.
        """
        if not query:
            # Extract query from last user message
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if not user_msgs:
                return messages
            query = str(user_msgs[-1].get("content", ""))

        memories = await self.retrieve(query, tags=tags)
        if not memories:
            return messages

        memory_text = self._format_memories(memories)
        augmented = list(messages)

        system_idx = next(
            (i for i, m in enumerate(augmented) if m.get("role") == "system"), None
        )

        if system_idx is not None:
            existing = str(augmented[system_idx].get("content", ""))
            augmented[system_idx] = {
                **augmented[system_idx],
                "content": existing + "\n\n" + memory_text,
            }
        else:
            memory_message: dict[str, object] = {
                "role": "system",
                "content": memory_text,
            }
            augmented.insert(0, memory_message)

        return augmented

    def _format_memories(self, memories: list[RetrievedMemory]) -> str:
        lines = [self.MEMORY_SYSTEM_PREFIX]
        token_budget = self._max_injection_tokens
        for mem in memories:
            line = mem.to_context_line()
            estimated_tokens = len(line.split()) + 2
            if token_budget - estimated_tokens < 0:
                break
            lines.append(line)
            token_budget -= estimated_tokens

        return "\n".join(lines)

    async def format_for_display(self, query: str) -> str:
        """Return a human-readable summary of retrieved memories."""
        memories = await self.retrieve(query)
        if not memories:
            return "No relevant memories found."

        lines = [f"Retrieved {len(memories)} memories:"]
        for i, mem in enumerate(memories, 1):
            score_pct = f"{mem.score * 100:.1f}%"
            lines.append(f"  {i}. [{score_pct}] {mem.content[:100]}")
        return "\n".join(lines)
