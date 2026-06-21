"""Base provider interface and shared data models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class StreamChunk:
    delta: str
    model: str
    finish_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class BaseProvider(ABC):
    """Abstract base class for LLM provider implementations."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse: ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]: ...

    @abstractmethod
    async def is_available(self) -> bool: ...
