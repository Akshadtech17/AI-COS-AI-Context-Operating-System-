"""Tests for the agent framework — data classes, Tool schema, and BaseAgent basics."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aicos.agents.base_agent import AgentResult, BaseAgent, MaxStepsExceeded, Tool
from aicos.core.gateway import AIGateway


# ── Tool ─────────────────────────────────────────────────────────────────────

class TestTool:
    @pytest.mark.asyncio
    async def test_tool_schema_basic(self) -> None:
        async def my_func(x: str) -> str:
            return x

        tool = Tool(name="my_tool", description="Does something", func=my_func)
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "my_tool"
        assert schema["function"]["description"] == "Does something"

    def test_tool_schema_with_params(self) -> None:
        async def noop() -> None:
            pass

        params = {"type": "object", "properties": {"q": {"type": "string"}}}
        tool = Tool(name="search", description="Search", func=noop, parameters=params)
        schema = tool.to_openai_schema()
        assert schema["function"]["parameters"] == params

    def test_tool_schema_empty_params(self) -> None:
        async def noop() -> None:
            pass

        tool = Tool(name="t", description="d", func=noop)
        schema = tool.to_openai_schema()
        assert schema["function"]["parameters"] == {"type": "object", "properties": {}}


# ── AgentResult ───────────────────────────────────────────────────────────────

class TestAgentResult:
    def test_agent_result_success(self) -> None:
        result = AgentResult(
            output="Done",
            structured={"key": "value"},
            steps=3,
            tool_calls=[],
            tokens_used=100,
            success=True,
        )
        assert result.success is True
        assert result.error is None
        assert result.steps == 3

    def test_agent_result_failure(self) -> None:
        result = AgentResult(
            output="",
            structured={},
            steps=1,
            tool_calls=[],
            tokens_used=50,
            success=False,
            error="API error",
        )
        assert result.success is False
        assert result.error == "API error"


# ── MaxStepsExceeded ──────────────────────────────────────────────────────────

class TestMaxStepsExceeded:
    def test_is_exception(self) -> None:
        exc = MaxStepsExceeded("too many steps")
        assert isinstance(exc, Exception)
        assert str(exc) == "too many steps"


# ── BaseAgent ─────────────────────────────────────────────────────────────────

class TestBaseAgent:
    @pytest.fixture
    def mock_gateway(self, mock_provider):
        from aicos.analytics.cost_tracker import CostTracker
        from aicos.core.config import AICOSConfig
        from aicos.core.router import ModelRouter
        cfg = MagicMock(spec=AICOSConfig)
        cfg.router_strategy = "auto"
        cfg.default_model = None
        cfg.fallback_models = []
        cfg.available_providers.return_value = ["openai"]
        router = MagicMock()
        router.select_model.return_value = MagicMock(
            model="gpt-4o-mini", provider="openai",
            task_type=MagicMock(value="simple"), reasoning="mock"
        )
        gw = MagicMock(spec=AIGateway)
        return gw

    def test_base_agent_init(self, mock_gateway) -> None:
        agent = BaseAgent(gateway=mock_gateway, model="gpt-4o-mini", max_steps=5)
        assert agent._model == "gpt-4o-mini"
        assert agent._max_steps == 5
        assert agent._tool_map == {}

    def test_base_agent_default_system_prompt(self, mock_gateway) -> None:
        agent = BaseAgent(gateway=mock_gateway)
        assert "helpful" in agent.system_prompt.lower()

    @pytest.mark.asyncio
    async def test_run_gateway_error_returns_failure(self, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(side_effect=RuntimeError("gateway down"))
        agent = BaseAgent(gateway=mock_gateway, max_steps=1)
        result = await agent.run("Do something")
        assert result.success is False
        assert "gateway down" in result.error

    @pytest.mark.asyncio
    async def test_run_with_context(self, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(side_effect=RuntimeError("stop"))
        agent = BaseAgent(gateway=mock_gateway, max_steps=1)
        result = await agent.run("task", context={"key": "value"})
        assert result.success is False
