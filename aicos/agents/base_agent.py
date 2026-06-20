"""
Base Agent Framework — tool-calling agent with structured output.

Architecture:
  1. ReAct loop: think → act → observe → repeat until done
  2. Tool definitions registered as Python functions with type hints
  3. Structured output via JSON schema validation
  4. Hard step limit to prevent runaway execution
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from aicos.core.gateway import AIGateway, GatewayRequest


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Awaitable[Any]]
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }


@dataclass
class AgentResult:
    output: str
    structured: dict[str, Any]
    steps: int
    tool_calls: list[dict[str, Any]]
    tokens_used: int
    success: bool
    error: str | None = None


class MaxStepsExceeded(Exception):
    pass


class BaseAgent:
    """
    Tool-calling LLM agent.

    Subclass this and override:
    - system_prompt: str — agent persona and instructions
    - tools: list[Tool] — available tools

    Then call agent.run(task) to execute.
    """

    system_prompt: str = "You are a helpful AI agent."
    tools: list[Tool] = []

    def __init__(
        self,
        gateway: AIGateway,
        model: str | None = None,
        max_steps: int = 15,
        temperature: float = 0.3,
    ) -> None:
        self._gateway = gateway
        self._model = model
        self._max_steps = max_steps
        self._temperature = temperature
        self._tool_map = {t.name: t for t in self.tools}

    async def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        """
        Execute the agent on a task.

        Args:
            task: Natural language task description
            context: Optional additional context injected into system prompt

        Returns:
            AgentResult with output, structured data, and execution stats
        """
        system = self.system_prompt
        if context:
            system += f"\n\nContext:\n{json.dumps(context, indent=2)}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

        tool_schemas = [t.to_openai_schema() for t in self.tools]
        all_tool_calls: list[dict[str, Any]] = []
        total_tokens = 0

        for step in range(self._max_steps):
            request = GatewayRequest(
                messages=messages,
                model=self._model,
                temperature=self._temperature,
                skip_cache=True,
                skip_compression=False,
                extra={"tools": tool_schemas, "tool_choice": "auto"} if tool_schemas else {},
            )

            try:
                response = await self._gateway.process(request)
            except Exception as e:
                return AgentResult(
                    output="",
                    structured={},
                    steps=step,
                    tool_calls=all_tool_calls,
                    tokens_used=total_tokens,
                    success=False,
                    error=str(e),
                )

            total_tokens += response.input_tokens + response.output_tokens

            # Check for tool calls in the raw response
            raw = getattr(response, "raw", {}) or {}
            tool_calls_raw = self._extract_tool_calls(response.content, raw)

            if tool_calls_raw:
                # Execute tools
                messages.append({"role": "assistant", "content": response.content})

                for tc in tool_calls_raw:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})

                    all_tool_calls.append({"name": tool_name, "args": tool_args, "step": step})

                    tool_result = await self._execute_tool(tool_name, tool_args)
                    messages.append({
                        "role": "user",
                        "content": f"Tool '{tool_name}' returned:\n{json.dumps(tool_result, indent=2)}",
                    })
            else:
                # Final answer — try to parse as JSON
                output = response.content.strip()
                structured = self._parse_structured_output(output)

                return AgentResult(
                    output=output,
                    structured=structured,
                    steps=step + 1,
                    tool_calls=all_tool_calls,
                    tokens_used=total_tokens,
                    success=True,
                )

        raise MaxStepsExceeded(
            f"Agent '{self.__class__.__name__}' exceeded {self._max_steps} steps"
        )

    def _extract_tool_calls(
        self,
        content: str,
        raw: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract tool calls from LLM response (OpenAI format or JSON fallback)."""
        # OpenAI-style tool_calls in raw response
        choices = raw.get("choices", [])
        if choices:
            choice = choices[0]
            tool_calls = choice.get("message", {}).get("tool_calls", [])
            if tool_calls:
                return [
                    {
                        "name": tc["function"]["name"],
                        "arguments": json.loads(tc["function"].get("arguments", "{}")),
                    }
                    for tc in tool_calls
                ]

        # Fallback: parse JSON tool call from content
        if "```json" in content or content.strip().startswith("{"):
            try:
                parsed = json.loads(
                    content.split("```json")[-1].split("```")[0].strip()
                    if "```json" in content
                    else content
                )
                if "tool" in parsed:
                    return [{"name": parsed["tool"], "arguments": parsed.get("args", {})}]
            except (json.JSONDecodeError, KeyError):
                pass

        return []

    async def _execute_tool(self, name: str, args: dict[str, Any]) -> Any:
        """Execute a registered tool, catching errors gracefully."""
        tool = self._tool_map.get(name)
        if not tool:
            return {"error": f"Unknown tool: {name}"}

        try:
            result = await tool.func(**args)
            return result
        except Exception as e:
            return {"error": f"Tool '{name}' failed: {e}"}

    def _parse_structured_output(self, content: str) -> dict[str, Any]:
        """Attempt to parse JSON from the model's final response."""
        # Try to extract JSON from markdown code block
        if "```json" in content:
            try:
                json_str = content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                pass

        # Try raw JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        return {"output": content}
