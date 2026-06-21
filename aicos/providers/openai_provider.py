"""OpenAI provider using the official openai SDK with retry logic."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from aicos.providers.base import BaseProvider, ProviderResponse, StreamChunk


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        import openai

        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "openai"

    async def is_available(self) -> bool:
        try:
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
        response = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        choice = response.choices[0]
        usage = response.usage

        return ProviderResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
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
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        async for chunk in stream:
            if not chunk.choices:
                # Usage chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    yield StreamChunk(
                        delta="",
                        model=model,
                        finish_reason="stop",
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                continue

            choice = chunk.choices[0]
            delta = choice.delta.content or ""
            finish_reason = choice.finish_reason

            yield StreamChunk(
                delta=delta,
                model=chunk.model or model,
                finish_reason=finish_reason,
            )
