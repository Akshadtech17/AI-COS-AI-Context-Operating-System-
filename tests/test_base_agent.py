"""Tests for BaseAgent — execution loop, tool dispatch, and output parsing.

These tests exercise the code paths in base_agent.py that were previously
uncovered (lines 128-230): the ReAct run() loop, _extract_tool_calls(),
_execute_tool(), and _parse_structured_output().
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aicos.agents.base_agent import BaseAgent, MaxStepsExceededError, Tool
from aicos.core.gateway import GatewayResponse

# ── Helpers ───────────────────────────────────────────────────────────────────


def _resp(content: str, input_tokens: int = 10, output_tokens: int = 20) -> GatewayResponse:
    return GatewayResponse(
        content=content,
        model="gpt-4o-mini",
        provider="openai",
        task_type="agent",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=0.001,
        latency_ms=100.0,
        cache_hit=False,
        cache_hit_type=None,
        tokens_before_compression=input_tokens,
        tokens_after_compression=input_tokens,
        memories_injected=0,
        routing_reason="test",
    )


def _raw_resp(content: str, tool_calls: list[dict]) -> MagicMock:
    """Simulate an LLM response that carries OpenAI-style tool_calls in .raw."""
    mock = MagicMock(spec=GatewayResponse)
    mock.content = content
    mock.input_tokens = 10
    mock.output_tokens = 10
    mock.raw = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {})),
                            }
                        }
                        for tc in tool_calls
                    ]
                }
            }
        ]
    }
    return mock


# ── Concrete test agent ───────────────────────────────────────────────────────


class _AddAgent(BaseAgent):
    """Minimal concrete agent used throughout these tests."""

    system_prompt = "You are a calculator agent."

    def __init__(self, gateway: MagicMock, **kwargs) -> None:
        self.tools = [
            Tool(
                name="add",
                description="Add two numbers",
                func=self._add,
                parameters={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            ),
            Tool(
                name="exploding_tool",
                description="Always raises",
                func=self._explode,
            ),
        ]
        super().__init__(gateway, **kwargs)

    async def _add(self, a: float = 0, b: float = 0) -> dict:
        return {"result": a + b}

    async def _explode(self) -> None:
        raise RuntimeError("boom")


@pytest.fixture
def mock_gw() -> MagicMock:
    gw = MagicMock()
    gw.process = AsyncMock(return_value=_resp('{"done": true}'))
    return gw


@pytest.fixture
def agent(mock_gw: MagicMock) -> _AddAgent:
    return _AddAgent(gateway=mock_gw)


# ── run() loop — lines 128–165 ────────────────────────────────────────────────


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_immediate_final_answer(self, agent, mock_gw) -> None:
        """No tool calls → final answer on step 1."""
        mock_gw.process = AsyncMock(return_value=_resp('{"answer": 42}'))
        result = await agent.run("What is 6 × 7?")
        assert result.success is True
        assert result.steps == 1
        assert result.structured == {"answer": 42}
        assert result.tool_calls == []
        assert result.error is None

    @pytest.mark.asyncio
    async def test_tokens_accumulated_single_step(self, agent, mock_gw) -> None:
        """tokens_used = input + output for the single response."""
        mock_gw.process = AsyncMock(
            return_value=_resp('{"ok": true}', input_tokens=7, output_tokens=13)
        )
        result = await agent.run("hi")
        assert result.tokens_used == 20

    @pytest.mark.asyncio
    async def test_json_tool_call_then_final_answer(self, agent, mock_gw) -> None:
        """JSON-fallback tool call → tool executes → final answer."""
        tool_resp = _resp('{"tool": "add", "args": {"a": 3, "b": 4}}', 10, 10)
        final_resp = _resp('{"result": 7}', 15, 5)
        mock_gw.process = AsyncMock(side_effect=[tool_resp, final_resp])

        result = await agent.run("Add 3 and 4")
        assert result.success is True
        assert result.steps == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "add"
        assert result.tool_calls[0]["args"] == {"a": 3, "b": 4}
        assert result.tool_calls[0]["step"] == 0

    @pytest.mark.asyncio
    async def test_tokens_accumulated_across_steps(self, agent, mock_gw) -> None:
        """tokens_used sums across the tool-call step and the final step."""
        tool_resp = _resp(
            '{"tool": "add", "args": {"a": 1, "b": 1}}', input_tokens=10, output_tokens=10
        )
        final_resp = _resp('{"result": 2}', input_tokens=15, output_tokens=5)
        mock_gw.process = AsyncMock(side_effect=[tool_resp, final_resp])

        result = await agent.run("compute")
        assert result.tokens_used == 40  # (10+10) + (15+5)

    @pytest.mark.asyncio
    async def test_tool_result_injected_into_next_request(self, agent, mock_gw) -> None:
        """After a tool call the result message is included in the next request."""
        tool_resp = _resp('{"tool": "add", "args": {"a": 2, "b": 3}}')
        final_resp = _resp("done")
        mock_gw.process = AsyncMock(side_effect=[tool_resp, final_resp])

        await agent.run("compute 2+3")

        second_call_messages = mock_gw.process.call_args_list[1][0][0].messages
        tool_msg = next(
            m for m in second_call_messages if "Tool 'add' returned" in m.get("content", "")
        )
        assert '"result": 5' in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_openai_format_tool_call(self, agent, mock_gw) -> None:
        """OpenAI raw tool_calls format triggers the tool dispatch path."""
        mock_raw = _raw_resp("calling add", [{"name": "add", "arguments": {"a": 1, "b": 9}}])
        final_resp = _resp('{"answer": 10}')
        mock_gw.process = AsyncMock(side_effect=[mock_raw, final_resp])

        result = await agent.run("use openai format")
        assert result.success is True
        assert result.tool_calls[0]["name"] == "add"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_step(self, agent, mock_gw) -> None:
        """Multiple tool_calls in a single response are all dispatched."""
        mock_raw = _raw_resp(
            "two calls",
            [
                {"name": "add", "arguments": {"a": 1, "b": 2}},
                {"name": "add", "arguments": {"a": 3, "b": 4}},
            ],
        )
        final_resp = _resp('{"done": true}')
        mock_gw.process = AsyncMock(side_effect=[mock_raw, final_resp])

        result = await agent.run("two operations")
        assert result.success is True
        assert len(result.tool_calls) == 2

    @pytest.mark.asyncio
    async def test_max_steps_exceeded(self, mock_gw) -> None:
        """When every step returns a tool call, MaxStepsExceededError is raised."""
        agent = _AddAgent(gateway=mock_gw, max_steps=2)
        always_tool = _resp('{"tool": "add", "args": {"a": 1, "b": 1}}')
        mock_gw.process = AsyncMock(return_value=always_tool)

        with pytest.raises(MaxStepsExceededError):
            await agent.run("loop forever")

    @pytest.mark.asyncio
    async def test_gateway_exception_returns_failure(self, agent, mock_gw) -> None:
        """If the gateway raises, run() returns AgentResult(success=False)."""
        mock_gw.process = AsyncMock(side_effect=RuntimeError("API unavailable"))

        result = await agent.run("will fail")
        assert result.success is False
        assert result.error == "API unavailable"
        assert result.steps == 0
        assert result.tokens_used == 0
        assert result.output == ""

    @pytest.mark.asyncio
    async def test_context_appended_to_system_prompt(self, agent, mock_gw) -> None:
        """Extra context is JSON-serialised and appended to the system message."""
        mock_gw.process = AsyncMock(return_value=_resp('{"ok": true}'))
        context = {"user": "Alice", "mode": "test"}

        await agent.run("do something", context=context)

        first_request = mock_gw.process.call_args_list[0][0][0]
        system_content = first_request.messages[0]["content"]
        assert "Alice" in system_content
        assert "Context" in system_content

    @pytest.mark.asyncio
    async def test_no_context_leaves_system_prompt_unchanged(self, agent, mock_gw) -> None:
        """When context=None the system prompt is used verbatim."""
        mock_gw.process = AsyncMock(return_value=_resp('{"ok": true}'))
        await agent.run("task")

        first_request = mock_gw.process.call_args_list[0][0][0]
        assert first_request.messages[0]["content"] == agent.system_prompt


# ── _extract_tool_calls — lines 167–200 ──────────────────────────────────────


class TestExtractToolCalls:
    def test_openai_format_single_tool(self, agent) -> None:
        """lines 174-185: OpenAI choices → message → tool_calls."""
        raw = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}
                        ]
                    }
                }
            ]
        }
        result = agent._extract_tool_calls("", raw)
        assert len(result) == 1
        assert result[0]["name"] == "add"
        assert result[0]["arguments"] == {"a": 1, "b": 2}

    def test_openai_format_multiple_tools(self, agent) -> None:
        """Multiple tool_calls parsed from OpenAI format."""
        raw = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"function": {"name": "add", "arguments": '{"a": 1, "b": 1}'}},
                            {"function": {"name": "add", "arguments": '{"a": 2, "b": 2}'}},
                        ]
                    }
                }
            ]
        }
        result = agent._extract_tool_calls("", raw)
        assert len(result) == 2

    def test_openai_format_empty_tool_calls_falls_through(self, agent) -> None:
        """Empty tool_calls list → falls through to content-based fallback."""
        raw = {"choices": [{"message": {"tool_calls": []}}]}
        result = agent._extract_tool_calls("plain answer", raw)
        assert result == []

    def test_json_content_with_tool_key(self, agent) -> None:
        """lines 188-196: raw JSON body with 'tool' key."""
        content = '{"tool": "add", "args": {"a": 5, "b": 6}}'
        result = agent._extract_tool_calls(content, {})
        assert len(result) == 1
        assert result[0]["name"] == "add"
        assert result[0]["arguments"] == {"a": 5, "b": 6}

    def test_markdown_json_with_tool_key(self, agent) -> None:
        """```json block containing a tool call."""
        content = '```json\n{"tool": "add", "args": {"a": 10, "b": 20}}\n```'
        result = agent._extract_tool_calls(content, {})
        assert len(result) == 1
        assert result[0]["name"] == "add"

    def test_json_without_tool_key_returns_empty(self, agent) -> None:
        """Valid JSON but no 'tool' key → not a tool call."""
        result = agent._extract_tool_calls('{"answer": 42}', {})
        assert result == []

    def test_invalid_json_fallback_returns_empty(self, agent) -> None:
        """lines 197-198: JSONDecodeError caught, returns []."""
        result = agent._extract_tool_calls("{not valid json}", {})
        assert result == []

    def test_plain_text_returns_empty(self, agent) -> None:
        """line 200: plain prose → no tool calls."""
        result = agent._extract_tool_calls("The answer is 42.", {})
        assert result == []

    def test_empty_raw_and_no_json_content(self, agent) -> None:
        """Both raw={} and non-JSON content → empty."""
        result = agent._extract_tool_calls("Sure, I can help with that.", {})
        assert result == []

    def test_openai_format_missing_arguments_defaults_to_empty(self, agent) -> None:
        """Missing 'arguments' key in function → defaults to '{}'."""
        raw = {"choices": [{"message": {"tool_calls": [{"function": {"name": "add"}}]}}]}
        result = agent._extract_tool_calls("", raw)
        assert result[0]["arguments"] == {}


# ── _execute_tool — lines 202–212 ────────────────────────────────────────────


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_successful_tool_execution(self, agent) -> None:
        """lines 208-210: known tool called with correct args."""
        result = await agent._execute_tool("add", {"a": 3.0, "b": 4.0})
        assert result == {"result": 7.0}

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_dict(self, agent) -> None:
        """lines 205-206: unknown tool name → {"error": ...}."""
        result = await agent._execute_tool("nonexistent", {"x": 1})
        assert "error" in result
        assert "Unknown tool" in result["error"]
        assert "nonexistent" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_exception_returns_error_dict(self, agent) -> None:
        """lines 211-212: tool raises → error dict, does not propagate."""
        result = await agent._execute_tool("exploding_tool", {})
        assert "error" in result
        assert "exploding_tool" in result["error"]
        assert "boom" in result["error"]

    @pytest.mark.asyncio
    async def test_tool_with_no_args(self, agent) -> None:
        """Tool called with empty args dict."""
        result = await agent._execute_tool("add", {})
        assert result == {"result": 0}  # defaults a=0, b=0


# ── _parse_structured_output — lines 214–230 ─────────────────────────────────


class TestParseStructuredOutput:
    def test_markdown_json_block(self, agent) -> None:
        """lines 217-222: extracts JSON from ```json ... ``` fence."""
        content = '```json\n{"status": "done", "count": 3}\n```'
        result = agent._parse_structured_output(content)
        assert result == {"status": "done", "count": 3}

    def test_raw_json(self, agent) -> None:
        """lines 224-228: parses plain JSON string."""
        result = agent._parse_structured_output('{"key": "value", "num": 99}')
        assert result == {"key": "value", "num": 99}

    def test_plain_text_fallback(self, agent) -> None:
        """line 230: non-JSON → {"output": <content>}."""
        text = "The capital of France is Paris."
        result = agent._parse_structured_output(text)
        assert result == {"output": text}

    def test_invalid_markdown_json_falls_to_raw_attempt(self, agent) -> None:
        """lines 221-222: bad JSON inside fence → falls through to raw attempt."""
        content = "```json\n{invalid}\n```"
        result = agent._parse_structured_output(content)
        assert "output" in result  # ends up in fallback

    def test_empty_string_fallback(self, agent) -> None:
        """Empty content → fallback with empty output."""
        result = agent._parse_structured_output("")
        assert result == {"output": ""}

    def test_nested_json(self, agent) -> None:
        """Deeply nested JSON parsed correctly."""
        data = {"a": {"b": {"c": [1, 2, 3]}}}
        result = agent._parse_structured_output(json.dumps(data))
        assert result == data

    def test_markdown_fence_with_trailing_text(self, agent) -> None:
        """Text after the closing fence is ignored."""
        content = '```json\n{"answer": 42}\n```\nSome trailing explanation.'
        result = agent._parse_structured_output(content)
        assert result == {"answer": 42}
