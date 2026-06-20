"""
StartupAgent — orchestrates multi-step startup analysis using specialized tools.

Capabilities:
  - Market research
  - Competitive analysis
  - Pricing strategy
  - Branding suggestions
  - Financial projections
  - Go-to-market planning

Output: Structured JSON startup blueprint.
"""

from __future__ import annotations

import json
from typing import Any

from aicos.agents.base_agent import BaseAgent, Tool
from aicos.core.gateway import AIGateway, GatewayRequest


STARTUP_SYSTEM_PROMPT = """\
You are an elite startup advisor and business analyst with expertise in:
- Market sizing and opportunity assessment
- Competitive intelligence and positioning
- Pricing strategy and revenue modeling
- Brand architecture and positioning
- Financial modeling and projections
- Go-to-market strategy

Your analysis must be data-driven, specific, and actionable.
Always use the available tools to gather structured information before synthesizing.
Conclude with a comprehensive JSON startup blueprint.

Output format (final answer must be valid JSON):
{
  "startup_name": "...",
  "tagline": "...",
  "market": {...},
  "competitors": [...],
  "pricing": {...},
  "branding": {...},
  "financials": {...},
  "gtm_strategy": {...},
  "risk_factors": [...],
  "recommended_next_steps": [...]
}"""


class StartupAgent(BaseAgent):
    system_prompt = STARTUP_SYSTEM_PROMPT
    tools: list[Tool] = []

    def __init__(self, gateway: AIGateway, model: str | None = None) -> None:
        super().__init__(gateway=gateway, model=model, max_steps=10, temperature=0.4)
        # Register tools as instance methods (tools reference self)
        self.tools = self._build_tools()
        self._tool_map = {t.name: t for t in self.tools}

    def _build_tools(self) -> list[Tool]:
        return [
            Tool(
                name="market_research",
                description="Research market size, trends, and opportunity for a given niche",
                func=self._market_research,
                parameters={
                    "type": "object",
                    "properties": {
                        "niche": {"type": "string", "description": "The market niche to research"},
                        "geography": {
                            "type": "string",
                            "description": "Target geography (global, US, EU, etc.)",
                            "default": "global",
                        },
                    },
                    "required": ["niche"],
                },
            ),
            Tool(
                name="competitive_analysis",
                description="Analyze competitors and identify gaps and positioning opportunities",
                func=self._competitive_analysis,
                parameters={
                    "type": "object",
                    "properties": {
                        "space": {"type": "string", "description": "The competitive space"},
                        "competitors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Known competitor names",
                        },
                    },
                    "required": ["space"],
                },
            ),
            Tool(
                name="pricing_strategy",
                description="Develop a pricing strategy with tiers and revenue projections",
                func=self._pricing_strategy,
                parameters={
                    "type": "object",
                    "properties": {
                        "product_type": {
                            "type": "string",
                            "description": "SaaS, marketplace, consumer app, etc.",
                        },
                        "target_segment": {"type": "string"},
                        "market_size_usd": {"type": "number"},
                    },
                    "required": ["product_type", "target_segment"],
                },
            ),
            Tool(
                name="branding_suggestions",
                description="Generate brand identity, name suggestions, and positioning statements",
                func=self._branding_suggestions,
                parameters={
                    "type": "object",
                    "properties": {
                        "concept": {"type": "string", "description": "Core startup concept"},
                        "values": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Core brand values",
                        },
                    },
                    "required": ["concept"],
                },
            ),
            Tool(
                name="financial_projections",
                description="Build 3-year financial projections with key assumptions",
                func=self._financial_projections,
                parameters={
                    "type": "object",
                    "properties": {
                        "model_type": {
                            "type": "string",
                            "description": "subscription, transaction, usage, etc.",
                        },
                        "initial_customers": {"type": "integer"},
                        "avg_revenue_per_user": {"type": "number"},
                        "growth_rate_monthly": {"type": "number"},
                    },
                    "required": ["model_type", "initial_customers", "avg_revenue_per_user"],
                },
            ),
        ]

    async def _call_llm(self, prompt: str, output_hint: str = "") -> dict[str, Any]:
        """Internal LLM call for tool execution."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise business analyst. "
                    "Respond ONLY with valid JSON. No markdown, no explanations. "
                    f"{output_hint}"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        request = GatewayRequest(
            messages=messages,
            temperature=0.3,
            skip_cache=False,
            skip_memory=True,
        )
        response = await self._gateway.process(request)
        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw": response.content}

    async def _market_research(self, niche: str, geography: str = "global") -> dict[str, Any]:
        return await self._call_llm(
            f"""Research the {niche} market ({geography}).
Provide:
{{
  "tam_usd": <total addressable market in USD>,
  "sam_usd": <serviceable addressable market>,
  "som_usd": <serviceable obtainable market>,
  "growth_rate_yoy_pct": <annual growth rate>,
  "key_trends": [<3-5 trends>],
  "pain_points": [<top 3 pain points>],
  "buying_triggers": [<what drives purchase decisions>],
  "market_maturity": "emerging|growing|mature|declining"
}}""",
            "Output must be a JSON object with market research data.",
        )

    async def _competitive_analysis(
        self, space: str, competitors: list[str] | None = None
    ) -> dict[str, Any]:
        comp_list = ", ".join(competitors) if competitors else "identify the top players"
        return await self._call_llm(
            f"""Analyze the competitive landscape for {space}.
Competitors to analyze: {comp_list}
Provide:
{{
  "competitors": [
    {{
      "name": "...",
      "positioning": "...",
      "pricing_model": "...",
      "strengths": ["..."],
      "weaknesses": ["..."],
      "estimated_arr_usd": <number or null>
    }}
  ],
  "market_gaps": ["<unserved needs>"],
  "winning_position": "<recommended differentiation strategy>",
  "moat_opportunities": ["<potential competitive moats>"]
}}""",
        )

    async def _pricing_strategy(
        self,
        product_type: str,
        target_segment: str,
        market_size_usd: float = 0,
    ) -> dict[str, Any]:
        return await self._call_llm(
            f"""Design a pricing strategy for a {product_type} targeting {target_segment}.
Market size: ${market_size_usd:,.0f}
Provide:
{{
  "model": "freemium|subscription|usage|one-time|hybrid",
  "tiers": [
    {{
      "name": "...",
      "price_usd_monthly": <number>,
      "features": ["..."],
      "target": "..."
    }}
  ],
  "free_tier": true/false,
  "annual_discount_pct": <number>,
  "ltv_estimate_usd": <number>,
  "cac_target_usd": <number>,
  "ltv_cac_ratio": <number>,
  "pricing_rationale": "..."
}}""",
        )

    async def _branding_suggestions(
        self, concept: str, values: list[str] | None = None
    ) -> dict[str, Any]:
        value_list = ", ".join(values) if values else "innovation, trust, simplicity"
        return await self._call_llm(
            f"""Create a brand identity for: {concept}
Core values: {value_list}
Provide:
{{
  "name_suggestions": [
    {{"name": "...", "rationale": "...", "domain_availability": "likely|check"}}
  ],
  "tagline": "...",
  "brand_voice": "...",
  "color_palette": {{
    "primary": "<hex>",
    "secondary": "<hex>",
    "accent": "<hex>"
  }},
  "typography_style": "...",
  "positioning_statement": "For [target], [brand] is the [category] that [key benefit] because [reason to believe].",
  "brand_archetype": "..."
}}""",
        )

    async def _financial_projections(
        self,
        model_type: str,
        initial_customers: int,
        avg_revenue_per_user: float,
        growth_rate_monthly: float = 0.10,
    ) -> dict[str, Any]:
        return await self._call_llm(
            f"""Build 3-year financial projections:
Model: {model_type}
Initial customers: {initial_customers}
ARPU: ${avg_revenue_per_user}/month
Monthly growth rate: {growth_rate_monthly * 100:.0f}%

Provide:
{{
  "year_1": {{
    "customers_eoy": <number>,
    "arr_usd": <number>,
    "gross_margin_pct": <number>,
    "burn_usd": <monthly burn>,
    "runway_months": <number>
  }},
  "year_2": {{ ... }},
  "year_3": {{ ... }},
  "break_even_month": <month number from start>,
  "total_funding_needed_usd": <number>,
  "key_assumptions": ["..."],
  "unit_economics": {{
    "cac_usd": <number>,
    "ltv_usd": <number>,
    "payback_months": <number>,
    "gross_margin_pct": <number>
  }}
}}""",
        )
