"""
CodingAgent — specialized software engineering agent.

Capabilities:
  - Code generation (Python, JS, TypeScript, Go, etc.)
  - Code review and refactoring
  - Bug analysis and fixes
  - Architecture planning
  - Test generation
  - Documentation generation
"""

from __future__ import annotations

import json
from typing import Any

from aicos.agents.base_agent import AgentResult, BaseAgent, Tool
from aicos.core.gateway import AIGateway, GatewayRequest

CODING_SYSTEM_PROMPT = """\
You are a senior software engineer and architect with 15+ years of experience.

Your principles:
- Write production-quality code that could ship today
- Prefer simplicity and readability over cleverness
- Always include error handling for external calls
- Write self-documenting code with meaningful names
- Consider security implications
- Follow the language's idiomatic style

For each task:
1. Analyze requirements carefully
2. Plan the implementation
3. Write clean, working code
4. Explain key decisions
5. Identify potential issues

Output structured JSON with code and metadata."""


class CodingAgent(BaseAgent):
    system_prompt = CODING_SYSTEM_PROMPT
    tools: list[Tool] = []

    def __init__(self, gateway: AIGateway, model: str | None = None) -> None:
        super().__init__(gateway=gateway, model=model, max_steps=8, temperature=0.2)
        self.tools = self._build_tools()
        self._tool_map = {t.name: t for t in self.tools}

    def _build_tools(self) -> list[Tool]:
        return [
            Tool(
                name="generate_code",
                description="Generate code for a specific component or feature",
                func=self._generate_code,
                parameters={
                    "type": "object",
                    "properties": {
                        "component": {"type": "string", "description": "What to generate"},
                        "language": {"type": "string", "description": "Programming language"},
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Functional requirements",
                        },
                        "constraints": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Technical constraints",
                        },
                    },
                    "required": ["component", "language"],
                },
            ),
            Tool(
                name="review_code",
                description="Review code for bugs, security issues, and improvements",
                func=self._review_code,
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to review"},
                        "language": {"type": "string"},
                        "focus": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["bugs", "security", "performance", "style", "all"],
                            },
                            "default": ["all"],
                        },
                    },
                    "required": ["code", "language"],
                },
            ),
            Tool(
                name="generate_tests",
                description="Generate unit tests for code",
                func=self._generate_tests,
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to test"},
                        "language": {"type": "string"},
                        "framework": {"type": "string", "description": "Test framework"},
                        "coverage_targets": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["code", "language"],
                },
            ),
            Tool(
                name="plan_architecture",
                description="Plan system architecture for a software project",
                func=self._plan_architecture,
                parameters={
                    "type": "object",
                    "properties": {
                        "project": {"type": "string"},
                        "scale": {
                            "type": "string",
                            "enum": ["prototype", "startup", "enterprise"],
                        },
                        "requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["project"],
                },
            ),
        ]

    async def _call_llm(self, prompt: str) -> Any:
        messages = [
            {
                "role": "system",
                "content": ("You are an expert software engineer. Respond with valid JSON only."),
            },
            {"role": "user", "content": prompt},
        ]
        request = GatewayRequest(
            messages=messages,
            temperature=0.1,
            skip_cache=False,
            skip_memory=True,
        )
        response = await self._gateway.process(request)
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw": response.content}

    async def _generate_code(
        self,
        component: str,
        language: str,
        requirements: list[str] | None = None,
        constraints: list[str] | None = None,
    ) -> dict[str, Any]:
        req_str = "\n".join(f"- {r}" for r in (requirements or []))
        con_str = "\n".join(f"- {c}" for c in (constraints or []))

        return await self._call_llm(
            f"""Generate production-quality {language} code for: {component}

Requirements:
{req_str or "- General purpose, clean implementation"}

Constraints:
{con_str or "- Standard best practices"}

Respond with:
{{
  "code": "<complete working code>",
  "filename": "<suggested filename>",
  "dependencies": ["<package>@<version>"],
  "usage_example": "<usage snippet>",
  "design_decisions": ["<key decision and rationale>"],
  "known_limitations": ["<limitation>"],
  "estimated_complexity": "O(...)"
}}"""
        )

    async def _review_code(
        self,
        code: str,
        language: str,
        focus: list[str] | None = None,
    ) -> dict[str, Any]:
        focus_str = ", ".join(focus or ["all"])
        return await self._call_llm(
            f"""Review this {language} code focusing on: {focus_str}

```{language}
{code}
```

Respond with:
{{
  "overall_rating": <1-10>,
  "summary": "<brief assessment>",
  "bugs": [
    {{"severity": "critical|high|medium|low", "line": <n>, "issue": "...", "fix": "..."}}
  ],
  "security_issues": [
    {{"severity": "...", "type": "...", "description": "...", "fix": "..."}}
  ],
  "performance_issues": [
    {{"description": "...", "impact": "...", "fix": "..."}}
  ],
  "style_issues": [
    {{"description": "...", "suggestion": "..."}}
  ],
  "positive_aspects": ["..."],
  "refactored_code": "<improved version if significant changes needed, else null>"
}}"""
        )

    async def _generate_tests(
        self,
        code: str,
        language: str,
        framework: str = "",
        coverage_targets: list[str] | None = None,
    ) -> dict[str, Any]:
        fw = framework or self._default_framework(language)
        targets = "\n".join(f"- {t}" for t in (coverage_targets or []))
        default_targets = "- Happy path\n- Edge cases\n- Error conditions"
        targets_text = targets or default_targets
        return await self._call_llm(
            f"""Generate comprehensive {fw} tests for this {language} code:

```{language}
{code}
```

Coverage targets:
{targets_text}

Respond with:
{{
  "test_code": "<complete test file>",
  "framework": "{fw}",
  "test_count": <number>,
  "coverage_estimate_pct": <number>,
  "test_cases": [
    {{"name": "...", "type": "unit|integration|edge_case", "description": "..."}}
  ],
  "setup_instructions": "..."
}}"""
        )

    async def _plan_architecture(
        self,
        project: str,
        scale: str = "startup",
        requirements: list[str] | None = None,
    ) -> dict[str, Any]:
        req_str = "\n".join(f"- {r}" for r in (requirements or []))
        return await self._call_llm(
            f"""Design system architecture for: {project}
Scale: {scale}
Requirements:
{req_str or "- Standard web application"}

Respond with:
{{
  "architecture_style": "microservices|monolith|serverless|event-driven",
  "components": [
    {{"name": "...", "responsibility": "...", "technology": "...", "rationale": "..."}}
  ],
  "data_stores": [
    {{"type": "...", "technology": "...", "use_case": "..."}}
  ],
  "api_design": {{"style": "REST|GraphQL|gRPC", "authentication": "..."}},
  "infrastructure": {{
    "cloud": "...",
    "containers": true/false,
    "cdn": true/false,
    "estimated_monthly_cost_usd": <number>
  }},
  "scalability_plan": "...",
  "tech_stack": {{"frontend": "...", "backend": "...", "database": "...", "cache": "..."}},
  "development_phases": [
    {{"phase": 1, "name": "...", "duration_weeks": <n>, "deliverables": ["..."]}}
  ]
}}"""
        )

    @staticmethod
    def _default_framework(language: str) -> str:
        defaults = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "go": "testing",
            "rust": "cargo test",
            "java": "junit",
            "ruby": "rspec",
        }
        return defaults.get(language.lower(), "appropriate testing framework")

    async def run_code_task(
        self,
        task: str,
        language: str = "python",
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Convenience method that pre-sets language context."""
        ctx = context or {}
        ctx["preferred_language"] = language
        return await self.run(task, context=ctx)
