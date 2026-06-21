"""
Model Router — classifies tasks and selects optimal LLM based on cost, latency,
and capability. Uses embedding-based zero-shot classification with regex fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aicos.core.config import AICOSConfig


class TaskType(StrEnum):
    SIMPLE = "simple"
    CODING = "coding"
    VISION = "vision"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    AGENT = "agent"


@dataclass
class ModelSpec:
    model_id: str
    provider: str
    input_cost_per_1m: float  # USD per 1M input tokens
    output_cost_per_1m: float  # USD per 1M output tokens
    max_tokens: int
    capabilities: set[str]
    avg_latency_ms: int
    tier: str  # free | cheap | mid | premium | local


# ── Model Registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, ModelSpec] = {
    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4o-mini": ModelSpec(
        model_id="gpt-4o-mini",
        provider="openai",
        input_cost_per_1m=0.15,
        output_cost_per_1m=0.60,
        max_tokens=128_000,
        capabilities={"text", "code", "vision", "json"},
        avg_latency_ms=800,
        tier="cheap",
    ),
    "gpt-4o": ModelSpec(
        model_id="gpt-4o",
        provider="openai",
        input_cost_per_1m=2.50,
        output_cost_per_1m=10.00,
        max_tokens=128_000,
        capabilities={"text", "code", "vision", "json", "reasoning"},
        avg_latency_ms=1500,
        tier="premium",
    ),
    "o1-mini": ModelSpec(
        model_id="o1-mini",
        provider="openai",
        input_cost_per_1m=3.00,
        output_cost_per_1m=12.00,
        max_tokens=128_000,
        capabilities={"text", "code", "reasoning", "math"},
        avg_latency_ms=5000,
        tier="mid",
    ),
    # ── Anthropic ─────────────────────────────────────────────────────────────
    "claude-haiku-4-5-20251001": ModelSpec(
        model_id="claude-haiku-4-5-20251001",
        provider="anthropic",
        input_cost_per_1m=0.25,
        output_cost_per_1m=1.25,
        max_tokens=200_000,
        capabilities={"text", "code", "json"},
        avg_latency_ms=600,
        tier="cheap",
    ),
    "claude-sonnet-4-6": ModelSpec(
        model_id="claude-sonnet-4-6",
        provider="anthropic",
        input_cost_per_1m=3.00,
        output_cost_per_1m=15.00,
        max_tokens=200_000,
        capabilities={"text", "code", "vision", "json", "reasoning", "analysis"},
        avg_latency_ms=1200,
        tier="mid",
    ),
    "claude-opus-4-8": ModelSpec(
        model_id="claude-opus-4-8",
        provider="anthropic",
        input_cost_per_1m=15.00,
        output_cost_per_1m=75.00,
        max_tokens=200_000,
        capabilities={"text", "code", "vision", "json", "reasoning", "analysis", "agent"},
        avg_latency_ms=3000,
        tier="premium",
    ),
    # ── Google Gemini ─────────────────────────────────────────────────────────
    "gemini/gemini-2.0-flash": ModelSpec(
        model_id="gemini/gemini-2.0-flash",
        provider="gemini",
        input_cost_per_1m=0.10,
        output_cost_per_1m=0.40,
        max_tokens=1_000_000,
        capabilities={"text", "code", "vision", "json"},
        avg_latency_ms=700,
        tier="cheap",
    ),
    "gemini/gemini-1.5-pro": ModelSpec(
        model_id="gemini/gemini-1.5-pro",
        provider="gemini",
        input_cost_per_1m=1.25,
        output_cost_per_1m=5.00,
        max_tokens=2_000_000,
        capabilities={"text", "code", "vision", "json", "reasoning"},
        avg_latency_ms=1500,
        tier="mid",
    ),
    # ── NVIDIA Nemotron Ultra (free) ──────────────────────────────────────────
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": ModelSpec(
        model_id="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        provider="nvidia",
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        max_tokens=128_000,
        capabilities={"text", "code", "reasoning", "analysis", "json", "vision"},
        avg_latency_ms=2500,
        tier="free",
    ),
    "openrouter/nvidia/llama-3.1-nemotron-ultra-253b-v1": ModelSpec(
        model_id="nvidia/llama-3.1-nemotron-ultra-253b-v1",
        provider="openrouter",
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        max_tokens=128_000,
        capabilities={"text", "code", "reasoning", "analysis", "json", "vision"},
        avg_latency_ms=2500,
        tier="free",
    ),
    # ── Ollama (local) ────────────────────────────────────────────────────────
    "ollama/llama3.2": ModelSpec(
        model_id="ollama/llama3.2",
        provider="ollama",
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        max_tokens=128_000,
        capabilities={"text", "code"},
        avg_latency_ms=2000,
        tier="local",
    ),
    "ollama/codellama": ModelSpec(
        model_id="ollama/codellama",
        provider="ollama",
        input_cost_per_1m=0.0,
        output_cost_per_1m=0.0,
        max_tokens=16_000,
        capabilities={"code"},
        avg_latency_ms=3000,
        tier="local",
    ),
}

# Task → preferred capability
TASK_CAPABILITY_MAP: dict[TaskType, list[str]] = {
    TaskType.SIMPLE: ["text"],
    TaskType.CODING: ["code"],
    TaskType.VISION: ["vision"],
    TaskType.REASONING: ["reasoning"],
    TaskType.CREATIVE: ["text"],
    TaskType.ANALYSIS: ["analysis"],
    TaskType.AGENT: ["agent"],
}

# Task → preferred tier order (free = Nemotron, always preferred)
TASK_TIER_PREFERENCE: dict[TaskType, list[str]] = {
    TaskType.SIMPLE: ["free", "cheap", "local", "mid"],
    TaskType.CODING: ["free", "mid", "cheap", "premium"],
    TaskType.VISION: ["free", "mid", "premium"],
    TaskType.REASONING: ["free", "premium", "mid"],
    TaskType.CREATIVE: ["free", "mid", "cheap"],
    TaskType.ANALYSIS: ["free", "premium", "mid"],
    TaskType.AGENT: ["free", "premium", "mid"],
}

# ── Regex patterns (fallback classifier) ─────────────────────────────────────
_CODE_RE = re.compile(
    r"\bdef\s+\w+|\bclass\s+\w+|\bimport\s+\w+|"
    r"\bfunction\s+\w+|\bconst\s+|\bvar\s+|\blet\s+|"
    r"```[\w]*\n|<code>|</code>|"
    r"\bSQL\b|\bSELECT\b|\bINSERT\b|\bCREATE TABLE\b|"
    r"\bpip install\b|\bnpm install\b|\bcargo add\b|"
    r"\bdebug\b|\brefactor\b|\bimplement\b|\bwrite.*code\b|"
    r"\b\.py\b|\b\.js\b|\b\.ts\b|\b\.go\b|\b\.rs\b",
    re.IGNORECASE,
)
_VISION_RE = re.compile(
    r"\bimage\b|\bphoto\b|\bpicture\b|\bscreenshot\b|\bdiagram\b|"
    r"\bdescribe.*image\b|\bwhat.*see\b|\banalyze.*image\b",
    re.IGNORECASE,
)
_REASONING_RE = re.compile(
    r"\banalyze\b|\bcompare\b|\bevaluate\b|\bstrategize\b|"
    r"\barchitect\b|\bdesign.*system\b|\btrade.?off\b|"
    r"\bpros.*cons\b|\bmath\b|\bprove\b|\bderive\b",
    re.IGNORECASE,
)
_ANALYSIS_RE = re.compile(
    r"\bmarket research\b|\bcompetitor\b|\bfinancial\b|\bstrategy\b|"
    r"\bbusiness plan\b|\bROI\b|\bmetrics\b|\breport\b|"
    r"\binsight\b|\btrend\b|\bforecast\b",
    re.IGNORECASE,
)
_CREATIVE_RE = re.compile(
    r"\bwrite.*story\b|\bwrite.*poem\b|\bcreative\b|\bblog post\b|"
    r"\bmarketing copy\b|\btag.*line\b|\bslogan\b|\bbranding\b",
    re.IGNORECASE,
)
_AGENT_RE = re.compile(
    r"\bagent\b|\borchestrate\b|\bmulti.?step\b|\bworkflow\b|"
    r"\bautonomous\b|\bautomate\b",
    re.IGNORECASE,
)


# ── Embedding-based zero-shot classifier ──────────────────────────────────────


class EmbeddingTaskClassifier:
    """
    Zero-shot task classifier using prototype embeddings.

    For each TaskType, a centroid is computed from representative phrases.
    A new message is classified to the nearest centroid by cosine similarity.
    More robust than regex for paraphrasing and natural language variation.
    """

    _PROTOTYPES: dict[str, list[str]] = {
        "coding": [
            "write a Python function",
            "implement this algorithm",
            "debug this code",
            "refactor the class",
            "write unit tests",
            "create a REST API endpoint",
            "fix the bug",
            "write SQL query",
            "npm install package",
            "pip install library",
            "write a script",
        ],
        "reasoning": [
            "analyze the trade-offs",
            "compare these approaches",
            "evaluate this design decision",
            "prove this theorem",
            "what are the pros and cons",
            "design the system architecture",
            "logical reasoning step by step",
            "math problem solution",
        ],
        "analysis": [
            "market research report",
            "competitor analysis",
            "business strategy evaluation",
            "financial analysis",
            "ROI calculation",
            "data insights",
            "trend forecast",
            "performance metrics review",
        ],
        "creative": [
            "write a short story",
            "marketing copy for product",
            "write a blog post",
            "generate creative ideas",
            "write a poem",
            "branding tagline",
            "narrative writing",
        ],
        "vision": [
            "describe what is in this image",
            "what do you see in the photo",
            "analyze this picture",
            "read the text in the diagram",
            "identify objects in the screenshot",
        ],
        "agent": [
            "automate this multi-step workflow",
            "orchestrate these tasks",
            "autonomous execution of the process",
            "run the agent workflow",
            "coordinate multiple steps automatically",
        ],
        "simple": [
            "what is the capital city",
            "who is this person",
            "when did this event happen",
            "how many items are there",
            "tell me briefly",
            "quick question about",
            "simple definition of",
        ],
    }

    _TYPE_MAP: dict[str, TaskType] = {
        "coding": TaskType.CODING,
        "reasoning": TaskType.REASONING,
        "analysis": TaskType.ANALYSIS,
        "creative": TaskType.CREATIVE,
        "vision": TaskType.VISION,
        "agent": TaskType.AGENT,
        "simple": TaskType.SIMPLE,
    }

    def __init__(self, embedding_engine: Any) -> None:
        import numpy as np

        self._engine = embedding_engine
        self._centroids: dict[TaskType, Any] = {}
        for key, phrases in self._PROTOTYPES.items():
            vecs = self._engine.embed_batch(phrases)
            centroid = vecs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            self._centroids[self._TYPE_MAP[key]] = centroid / norm if norm > 0 else centroid

    def classify(self, text: str) -> TaskType:
        import numpy as np

        query = self._engine.embed(text[:600])
        scores = {t: float(np.dot(query, c)) for t, c in self._centroids.items()}
        return max(scores, key=lambda t: scores[t])


# ── Routing decision ──────────────────────────────────────────────────────────


@dataclass
class RoutingDecision:
    model: str
    provider: str
    task_type: TaskType
    estimated_cost_per_1k: float
    reasoning: str
    fallback_models: list[str] = field(default_factory=list)


class ModelRouter:
    def __init__(
        self,
        config: AICOSConfig,
        embedding_engine: Any | None = None,
    ) -> None:
        self.config = config
        self._available_models: dict[str, ModelSpec] = {}
        self._classifier: EmbeddingTaskClassifier | None = (
            EmbeddingTaskClassifier(embedding_engine) if embedding_engine else None
        )
        self._refresh_available()

    def _refresh_available(self) -> None:
        providers = set(self.config.available_providers())
        self._available_models = {
            mid: spec for mid, spec in MODEL_REGISTRY.items() if spec.provider in providers
        }

    def classify_task(self, messages: list[dict[str, object]]) -> TaskType:
        text = " ".join(
            m["content"] if isinstance(m.get("content"), str) else ""
            for m in messages
            if m.get("role") in ("user", "system")
        )

        # Prefer embedding classifier when available
        if self._classifier and text.strip():
            return self._classifier.classify(text)

        # Regex fallback
        scores: dict[TaskType, int] = {t: 0 for t in TaskType}
        if _CODE_RE.search(text):
            scores[TaskType.CODING] += 3
        if _VISION_RE.search(text):
            scores[TaskType.VISION] += 5
        if _REASONING_RE.search(text):
            scores[TaskType.REASONING] += 2
        if _ANALYSIS_RE.search(text):
            scores[TaskType.ANALYSIS] += 2
        if _CREATIVE_RE.search(text):
            scores[TaskType.CREATIVE] += 2
        if _AGENT_RE.search(text):
            scores[TaskType.AGENT] += 3

        if len(text.split()) < 30 and max(scores.values()) == 0:
            return TaskType.SIMPLE

        best = max(scores, key=lambda t: scores[t])
        return best if scores[best] > 0 else TaskType.SIMPLE

    def select_model(
        self,
        messages: list[dict[str, object]],
        task_type: TaskType | None = None,
        override_model: str | None = None,
    ) -> RoutingDecision:
        self._refresh_available()

        if override_model:
            spec = MODEL_REGISTRY.get(override_model)
            if spec:
                return RoutingDecision(
                    model=override_model,
                    provider=spec.provider,
                    task_type=task_type or TaskType.SIMPLE,
                    estimated_cost_per_1k=spec.input_cost_per_1m / 1000,
                    reasoning=f"Explicit model override: {override_model}",
                )

        if self.config.default_model and self.config.router_strategy != "auto":
            model_id = self.config.default_model
            spec = MODEL_REGISTRY.get(model_id)
            if spec:
                return RoutingDecision(
                    model=model_id,
                    provider=spec.provider,
                    task_type=task_type or TaskType.SIMPLE,
                    estimated_cost_per_1k=spec.input_cost_per_1m / 1000,
                    reasoning=f"Config default: {model_id}",
                )

        if not task_type:
            task_type = self.classify_task(messages)

        if not self._available_models:
            raise RuntimeError("No AI providers configured. Set at least one API key in .env")

        strategy = self.config.router_strategy
        candidates = list(self._available_models.values())

        required_caps = set(TASK_CAPABILITY_MAP.get(task_type, ["text"]))
        capable = [m for m in candidates if required_caps.issubset(m.capabilities)]
        if not capable:
            capable = candidates

        if strategy == "cheapest":
            chosen = min(capable, key=lambda m: m.input_cost_per_1m)
        elif strategy == "fastest":
            chosen = min(capable, key=lambda m: m.avg_latency_ms)
        elif strategy == "best":
            tier_order = {"premium": 0, "free": 1, "mid": 1, "cheap": 2, "local": 3}
            chosen = min(capable, key=lambda m: tier_order.get(m.tier, 99))
        else:  # auto
            tier_pref = TASK_TIER_PREFERENCE.get(task_type, ["free", "mid", "cheap"])
            tier_map: dict[str, list[ModelSpec]] = {}
            for m in capable:
                tier_map.setdefault(m.tier, []).append(m)
            chosen = None
            for tier in tier_pref:
                if tier in tier_map:
                    chosen = min(tier_map[tier], key=lambda m: m.input_cost_per_1m)
                    break
            if chosen is None:
                chosen = min(capable, key=lambda m: m.input_cost_per_1m)

        fallbacks = [
            m.model_id
            for m in sorted(capable, key=lambda m: m.input_cost_per_1m)
            if m.model_id != chosen.model_id
        ][:3]
        for fb in self.config.fallback_models:
            if fb not in fallbacks and fb != chosen.model_id:
                fallbacks.append(fb)

        return RoutingDecision(
            model=chosen.model_id,
            provider=chosen.provider,
            task_type=task_type,
            estimated_cost_per_1k=chosen.input_cost_per_1m / 1000,
            reasoning=(
                f"Strategy={strategy}, task={task_type.value}, "
                f"tier={chosen.tier}, cost=${chosen.input_cost_per_1m}/1M"
            ),
            fallback_models=fallbacks,
        )

    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        spec = MODEL_REGISTRY.get(model_id)
        if not spec:
            return 0.0
        return (
            input_tokens * spec.input_cost_per_1m / 1_000_000
            + output_tokens * spec.output_cost_per_1m / 1_000_000
        )
