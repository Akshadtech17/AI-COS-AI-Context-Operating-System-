"""Tests for StartupAgent and CodingAgent."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aicos.agents.base_agent import AgentResult, BaseAgent, MaxStepsExceeded, Tool
from aicos.agents.coding_agent import CodingAgent
from aicos.agents.startup_agent import StartupAgent
from aicos.core.gateway import GatewayResponse


# ── Helpers ───────────────────────────────────────────────────────────────────

def _json_response(data: Any) -> GatewayResponse:
    return GatewayResponse(
        content=json.dumps(data),
        model="gpt-4o-mini",
        provider="openai",
        task_type="analysis",
        input_tokens=50,
        output_tokens=100,
        cost_usd=0.001,
        latency_ms=200.0,
        cache_hit=False,
        cache_hit_type=None,
        tokens_before_compression=50,
        tokens_after_compression=50,
        memories_injected=0,
        routing_reason="test",
    )


def _text_response(text: str) -> GatewayResponse:
    return GatewayResponse(
        content=text,
        model="gpt-4o-mini",
        provider="openai",
        task_type="simple",
        input_tokens=20,
        output_tokens=30,
        cost_usd=0.0005,
        latency_ms=100.0,
        cache_hit=False,
        cache_hit_type=None,
        tokens_before_compression=20,
        tokens_after_compression=20,
        memories_injected=0,
        routing_reason="test",
    )


@pytest.fixture
def mock_gateway():
    gw = MagicMock()
    gw.process = AsyncMock(return_value=_json_response({"result": "test_output"}))
    return gw


# ── StartupAgent ──────────────────────────────────────────────────────────────

class TestStartupAgentInit:
    def test_creates_tools(self, mock_gateway) -> None:
        agent = StartupAgent(gateway=mock_gateway)
        assert len(agent.tools) == 5

    def test_tool_names(self, mock_gateway) -> None:
        agent = StartupAgent(gateway=mock_gateway)
        names = {t.name for t in agent.tools}
        assert "market_research" in names
        assert "competitive_analysis" in names
        assert "pricing_strategy" in names
        assert "branding_suggestions" in names
        assert "financial_projections" in names

    def test_tool_map_built(self, mock_gateway) -> None:
        agent = StartupAgent(gateway=mock_gateway)
        assert "market_research" in agent._tool_map
        assert isinstance(agent._tool_map["market_research"], Tool)

    def test_max_steps_set(self, mock_gateway) -> None:
        agent = StartupAgent(gateway=mock_gateway)
        assert agent._max_steps == 10

    def test_temperature_set(self, mock_gateway) -> None:
        agent = StartupAgent(gateway=mock_gateway)
        assert agent._temperature == 0.4

    def test_custom_model(self, mock_gateway) -> None:
        agent = StartupAgent(gateway=mock_gateway, model="gpt-4o")
        assert agent._model == "gpt-4o"


class TestStartupAgentTools:
    @pytest.fixture
    def agent(self, mock_gateway) -> StartupAgent:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({
                "tam_usd": 1_000_000_000,
                "growth_rate_yoy_pct": 25,
                "key_trends": ["AI adoption"],
                "market_maturity": "growing",
            })
        )
        return StartupAgent(gateway=mock_gateway)

    @pytest.mark.asyncio
    async def test_market_research_returns_dict(self, agent) -> None:
        result = await agent._market_research("AI infrastructure")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_market_research_calls_gateway(self, agent, mock_gateway) -> None:
        await agent._market_research("SaaS")
        mock_gateway.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_market_research_with_geography(self, agent) -> None:
        result = await agent._market_research("fintech", geography="US")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_competitive_analysis_returns_dict(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({"competitors": [], "market_gaps": []})
        )
        result = await agent._competitive_analysis("AI tools")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_competitive_analysis_with_competitors(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({"competitors": [{"name": "OpenAI"}]})
        )
        result = await agent._competitive_analysis("LLM APIs", competitors=["OpenAI", "Anthropic"])
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_pricing_strategy_returns_dict(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({"model": "freemium", "tiers": []})
        )
        result = await agent._pricing_strategy("SaaS", "SMB")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_branding_suggestions_returns_dict(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({"name_suggestions": [], "tagline": "Test"})
        )
        result = await agent._branding_suggestions("AI assistant")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_branding_with_values(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({"tagline": "Build the future"})
        )
        result = await agent._branding_suggestions("startup", values=["trust", "speed"])
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_financial_projections_returns_dict(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({"year_1": {"arr_usd": 100_000}})
        )
        result = await agent._financial_projections("subscription", 10, 99.0)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_call_llm_handles_invalid_json(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_text_response("This is not JSON at all")
        )
        result = await agent._call_llm("Some prompt")
        assert isinstance(result, dict)
        assert "raw" in result

    @pytest.mark.asyncio
    async def test_call_llm_strips_markdown_fences(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_text_response('```json\n{"key": "value"}\n```')
        )
        result = await agent._call_llm("Some prompt")
        assert result.get("key") == "value"


# ── CodingAgent ───────────────────────────────────────────────────────────────

class TestCodingAgentInit:
    def test_creates_tools(self, mock_gateway) -> None:
        agent = CodingAgent(gateway=mock_gateway)
        assert len(agent.tools) == 4

    def test_tool_names(self, mock_gateway) -> None:
        agent = CodingAgent(gateway=mock_gateway)
        names = {t.name for t in agent.tools}
        assert "generate_code" in names
        assert "review_code" in names
        assert "generate_tests" in names
        assert "plan_architecture" in names

    def test_max_steps_set(self, mock_gateway) -> None:
        agent = CodingAgent(gateway=mock_gateway)
        assert agent._max_steps == 8

    def test_low_temperature(self, mock_gateway) -> None:
        agent = CodingAgent(gateway=mock_gateway)
        assert agent._temperature == 0.2


class TestCodingAgentTools:
    @pytest.fixture
    def agent(self, mock_gateway) -> CodingAgent:
        return CodingAgent(gateway=mock_gateway)

    @pytest.mark.asyncio
    async def test_generate_code_returns_dict(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({
                "code": "def hello(): pass",
                "filename": "hello.py",
                "dependencies": [],
            })
        )
        result = await agent._generate_code("hello function", "python")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_code_with_requirements(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({"code": "...", "filename": "app.py"})
        )
        result = await agent._generate_code(
            "FastAPI endpoint", "python",
            requirements=["async", "Pydantic validation"],
            constraints=["no external deps"],
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_review_code_returns_dict(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({
                "overall_rating": 8,
                "summary": "Clean code",
                "bugs": [],
                "security_issues": [],
            })
        )
        result = await agent._review_code("def foo(): pass", "python")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_review_code_with_focus(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({"overall_rating": 9, "bugs": []})
        )
        result = await agent._review_code("x = 1", "python", focus=["security", "bugs"])
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_tests_returns_dict(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({
                "test_code": "def test_foo(): pass",
                "framework": "pytest",
                "test_count": 5,
            })
        )
        result = await agent._generate_tests("def foo(): return 1", "python")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_generate_tests_default_framework(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(return_value=_json_response({"test_code": "..."}))
        result = await agent._generate_tests("code", "javascript")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_plan_architecture_returns_dict(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_json_response({
                "architecture_style": "microservices",
                "components": [],
            })
        )
        result = await agent._plan_architecture("E-commerce platform")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_plan_architecture_with_scale(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(return_value=_json_response({"architecture_style": "monolith"}))
        result = await agent._plan_architecture("Blog", scale="prototype")
        assert isinstance(result, dict)

    def test_default_framework_python(self, agent) -> None:
        assert agent._default_framework("python") == "pytest"

    def test_default_framework_javascript(self, agent) -> None:
        assert agent._default_framework("javascript") == "jest"

    def test_default_framework_go(self, agent) -> None:
        assert agent._default_framework("go") == "testing"

    def test_default_framework_unknown(self, agent) -> None:
        result = agent._default_framework("cobol")
        assert "framework" in result.lower()

    @pytest.mark.asyncio
    async def test_run_code_task_sets_language(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_text_response('{"output": "done"}')
        )
        # run() will hit max_steps with a mock that doesn't return the right format,
        # so we just verify run_code_task sets context correctly
        agent.run = AsyncMock(return_value=AgentResult(
            output="done", structured={}, steps=1,
            tool_calls=[], tokens_used=100, success=True,
        ))
        result = await agent.run_code_task("write a sort function", language="typescript")
        agent.run.assert_called_once()
        call_kwargs = agent.run.call_args[1]
        assert call_kwargs["context"]["preferred_language"] == "typescript"

    @pytest.mark.asyncio
    async def test_call_llm_handles_json_in_codeblock(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_text_response('```json\n{"generated": true}\n```')
        )
        result = await agent._call_llm("prompt")
        assert result.get("generated") is True

    @pytest.mark.asyncio
    async def test_call_llm_handles_invalid_json(self, agent, mock_gateway) -> None:
        mock_gateway.process = AsyncMock(
            return_value=_text_response("here is some plain text")
        )
        result = await agent._call_llm("prompt")
        assert "raw" in result


# ── Tool schema ───────────────────────────────────────────────────────────────

class TestToolSchema:
    def test_openai_schema_structure(self, mock_gateway) -> None:
        agent = CodingAgent(gateway=mock_gateway)
        tool = agent._tool_map["generate_code"]
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "generate_code"
        assert "parameters" in schema["function"]

    def test_all_tools_have_valid_schema(self, mock_gateway) -> None:
        agent = StartupAgent(gateway=mock_gateway)
        for tool in agent.tools:
            schema = tool.to_openai_schema()
            assert schema["type"] == "function"
            assert schema["function"]["name"] == tool.name
            assert "description" in schema["function"]
