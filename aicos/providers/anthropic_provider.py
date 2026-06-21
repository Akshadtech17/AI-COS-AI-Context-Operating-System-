"""Anthropic Claude provider with streaming support and retry logic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from aicos.providers.base import BaseProvider, ProviderResponse, StreamChunk


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    async def is_available(self) -> bool:
        try:
            # Lightweight check — list models endpoint
            await self._client.models.list()
            return True
        except Exception:
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> ProviderResponse:
        # Anthropic separates system from conversation messages
        system_parts = [m for m in messages if m.get("role") == "system"]
        conv_parts = [m for m in messages if m.get("role") != "system"]
        system_text = "\n\n".join(str(m.get("content", "")) for m in system_parts)

        response = await self._client.messages.create(
            model=model,
            system=system_text or "You are a helpful assistant.",
            messages=[  # type: ignore[arg-type]
                {"role": m["role"], "content": m.get("content", "")} for m in conv_parts
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        content = "".join(block.text for block in response.content if hasattr(block, "text"))

        return ProviderResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason or "stop",
            raw=response.model_dump(),
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        system_parts = [m for m in messages if m.get("role") == "system"]
        conv_parts = [m for m in messages if m.get("role") != "system"]
        system_text = "\n\n".join(str(m.get("content", "")) for m in system_parts)

        async with self._client.messages.stream(
            model=model,
            system=system_text or "You are a helpful assistant.",
            messages=[  # type: ignore[arg-type]
                {"role": m["role"], "content": m.get("content", "")} for m in conv_parts
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        ) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta=text, model=model)

            final = await stream.get_final_message()
            yield StreamChunk(
                delta="",
                model=final.model,
                finish_reason=final.stop_reason or "stop",
                input_tokens=final.usage.input_tokens,
                output_tokens=final.usage.output_tokens,
            )
