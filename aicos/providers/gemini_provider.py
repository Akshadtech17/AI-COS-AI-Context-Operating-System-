"""Google Gemini provider via the google-generativeai SDK."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from aicos.providers.base import BaseProvider, ProviderResponse, StreamChunk


class GeminiProvider(BaseProvider):
    def __init__(self, api_key: str) -> None:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._genai = genai
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "gemini"

    async def is_available(self) -> bool:
        try:
            list(self._genai.list_models())
            return True
        except Exception:
            return False

    def _extract_model_name(self, model: str) -> str:
        # Strip "gemini/" prefix used in routing
        return model.replace("gemini/", "")

    def _convert_messages(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Convert OpenAI-format messages to Gemini format."""
        system_text = ""
        history: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))

            if role == "system":
                system_text += content + "\n"
            elif role == "user":
                history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                history.append({"role": "model", "parts": [content]})

        return system_text.strip(), history

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
        import asyncio

        model_name = self._extract_model_name(model)
        system_text, history = self._convert_messages(messages)

        genai_model = self._genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_text or None,
        )

        generation_config = self._genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # Separate last user message from history
        if history and history[-1]["role"] == "user":
            last_user_msg = history[-1]["parts"][0]
            chat_history = history[:-1]
        else:
            last_user_msg = ""
            chat_history = history

        chat = genai_model.start_chat(history=chat_history)

        response = await asyncio.to_thread(
            chat.send_message,
            last_user_msg,
            generation_config=generation_config,
        )

        usage = response.usage_metadata
        return ProviderResponse(
            content=response.text,
            model=model_name,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            finish_reason="stop",
            raw={"text": response.text},
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        import asyncio

        model_name = self._extract_model_name(model)
        system_text, history = self._convert_messages(messages)

        genai_model = self._genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_text or None,
        )

        generation_config = self._genai.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        if history and history[-1]["role"] == "user":
            last_user_msg = history[-1]["parts"][0]
            chat_history = history[:-1]
        else:
            last_user_msg = ""
            chat_history = history

        chat = genai_model.start_chat(history=chat_history)

        response_iter = await asyncio.to_thread(
            chat.send_message,
            last_user_msg,
            generation_config=generation_config,
            stream=True,
        )

        total_in = 0
        total_out = 0

        for chunk in response_iter:
            if chunk.text:
                yield StreamChunk(delta=chunk.text, model=model_name)
            if chunk.usage_metadata:
                total_in = chunk.usage_metadata.prompt_token_count or 0
                total_out = chunk.usage_metadata.candidates_token_count or 0

        yield StreamChunk(
            delta="",
            model=model_name,
            finish_reason="stop",
            input_tokens=total_in,
            output_tokens=total_out,
        )
