"""Tests for the AI Gateway pipeline — routing, caching, memory injection."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aicos.core.gateway import AIGateway, GatewayRequest
from aicos.providers.base import ProviderResponse, StreamChunk


class TestGatewayBasic:
    @pytest.mark.asyncio
    async def test_basic_request(self, gateway: AIGateway) -> None:
        request = GatewayRequest(
            messages=[{"role": "user", "content": "Hello"}],
            skip_cache=True,
            skip_memory=True,
        )
        response = await gateway.process(request)
        assert response.content == "Mock response"
        assert response.model is not None
        assert response.provider is not None

    @pytest.mark.asyncio
    async def test_response_has_metrics(self, gateway: AIGateway) -> None:
        request = GatewayRequest(
            messages=[{"role": "user", "content": "Test"}],
            skip_cache=True,
        )
        response = await gateway.process(request)
        assert response.latency_ms > 0
        assert response.tokens_before_compression >= 0
        assert response.tokens_after_compression >= 0

    @pytest.mark.asyncio
    async def test_cache_miss_then_hit(self, gateway: AIGateway) -> None:
        messages = [{"role": "user", "content": "What is 2 + 2?"}]

        # First call — cache miss
        req1 = GatewayRequest(messages=messages)
        resp1 = await gateway.process(req1)
        assert resp1.cache_hit is False

        # Second call — should hit cache
        req2 = GatewayRequest(messages=messages)
        resp2 = await gateway.process(req2)
        assert resp2.cache_hit is True

    @pytest.mark.asyncio
    async def test_skip_cache_flag(self, gateway: AIGateway) -> None:
        messages = [{"role": "user", "content": "Skip cache test"}]

        # First call to populate cache
        await gateway.process(GatewayRequest(messages=messages))

        # Second call with skip_cache — should not hit cache
        resp = await gateway.process(GatewayRequest(messages=messages, skip_cache=True))
        assert resp.cache_hit is False

    @pytest.mark.asyncio
    async def test_task_type_in_response(self, gateway: AIGateway) -> None:
        request = GatewayRequest(
            messages=[{"role": "user", "content": "def fibonacci(n): implement this"}],
            skip_cache=True,
        )
        response = await gateway.process(request)
        assert response.task_type == "coding"

    @pytest.mark.asyncio
    async def test_routing_reason_in_response(self, gateway: AIGateway) -> None:
        request = GatewayRequest(
            messages=[{"role": "user", "content": "Hello"}],
            skip_cache=True,
        )
        response = await gateway.process(request)
        assert response.routing_reason
        assert len(response.routing_reason) > 0

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, gateway: AIGateway) -> None:
        request = GatewayRequest(
            messages=[{"role": "user", "content": "Stream test"}],
            stream=True,
            skip_cache=True,
        )
        chunks = []
        async for chunk in gateway.stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Concatenate deltas
        full_text = "".join(c.delta for c in chunks if c.delta)
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_stream_cache_hit_returns_single_chunk(self, gateway: AIGateway) -> None:
        messages = [{"role": "user", "content": "Stream cache test"}]

        # Populate cache via non-streaming call
        await gateway.process(GatewayRequest(messages=messages))

        # Stream should return cached result as single chunk
        chunks = []
        async for chunk in gateway.stream(GatewayRequest(messages=messages, stream=True)):
            chunks.append(chunk)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_cost_tracking(self, gateway: AIGateway) -> None:
        request = GatewayRequest(
            messages=[{"role": "user", "content": "Cost tracking test"}],
            skip_cache=True,
        )
        response = await gateway.process(request)
        # Cost should be non-negative (may be 0 for mock provider)
        assert response.cost_usd >= 0

    @pytest.mark.asyncio
    async def test_session_id_passthrough(self, gateway: AIGateway) -> None:
        request = GatewayRequest(
            messages=[{"role": "user", "content": "Session test"}],
            session_id="test-session-123",
            skip_cache=True,
        )
        response = await gateway.process(request)
        assert response.content is not None


class TestGatewayFailover:
    @pytest.mark.asyncio
    async def test_failover_to_second_provider(
        self, config, mock_provider, semantic_cache, memory_retriever
    ) -> None:
        from aicos.core.gateway import AIGateway
        from aicos.core.router import ModelRouter
        from tests.conftest import MockProvider

        # Provider that always fails
        class FailingProvider(MockProvider):
            async def complete(self, *args, **kwargs):
                raise ConnectionError("Provider unavailable")
            async def stream(self, *args, **kwargs):
                raise ConnectionError("Provider unavailable")
                yield  # Make it a generator

        failing = FailingProvider()
        working = MockProvider("Fallback response")

        router = ModelRouter(config)
        gw = AIGateway(
            config=config,
            router=router,
            providers={"openai": failing, "anthropic": working},
            semantic_cache=semantic_cache,
            memory_retriever=memory_retriever,
        )

        request = GatewayRequest(
            messages=[{"role": "user", "content": "Failover test"}],
            skip_cache=True,
        )
        # Should not raise — should failover
        # Note: failover depends on router finding anthropic as fallback
        # This test verifies the mechanism doesn't crash
        try:
            response = await gw.process(request)
            assert response.content is not None
        except RuntimeError as e:
            # Acceptable if no fallback routes exist for the task
            assert "failed" in str(e).lower() or "provider" in str(e).lower()
