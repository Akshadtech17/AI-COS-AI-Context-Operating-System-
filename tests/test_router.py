"""Tests for the model router."""

from __future__ import annotations

import pytest

from aicos.core.config import AICOSConfig
from aicos.core.router import ModelRouter, TaskType


@pytest.fixture
def router(config: AICOSConfig) -> ModelRouter:
    return ModelRouter(config)


class TestTaskClassification:
    def test_simple_question(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "What is the capital of France?"}]
        task = router.classify_task(messages)
        assert task == TaskType.SIMPLE

    def test_coding_task_python(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "def fibonacci(n): write this function"}]
        task = router.classify_task(messages)
        assert task == TaskType.CODING

    def test_coding_task_class(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "class UserService: implement this"}]
        task = router.classify_task(messages)
        assert task == TaskType.CODING

    def test_coding_task_import(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "import pandas and write a CSV parser"}]
        task = router.classify_task(messages)
        assert task == TaskType.CODING

    def test_vision_task(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Describe what is in this image"}]
        task = router.classify_task(messages)
        assert task == TaskType.VISION

    def test_reasoning_task(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Analyze the trade-offs between microservices and monolith architectures"}]
        task = router.classify_task(messages)
        assert task == TaskType.REASONING

    def test_analysis_task(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Do market research on the competitor landscape for B2B SaaS"}]
        task = router.classify_task(messages)
        assert task == TaskType.ANALYSIS

    def test_creative_task(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Write a creative blog post about AI"}]
        task = router.classify_task(messages)
        assert task == TaskType.CREATIVE

    def test_code_block_detection(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Fix this:\n```python\nfor i in range(n)\n```"}]
        task = router.classify_task(messages)
        assert task == TaskType.CODING


class TestModelSelection:
    def test_returns_routing_decision(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        decision = router.select_model(messages)
        assert decision.model
        assert decision.provider
        assert decision.task_type
        assert isinstance(decision.estimated_cost_per_1k, float)

    def test_override_model(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        model = "openrouter/nvidia/llama-3.1-nemotron-ultra-253b-v1"
        decision = router.select_model(messages, override_model=model)
        assert decision.model == model
        assert "override" in decision.reasoning.lower()

    def test_cheapest_strategy(self, config: AICOSConfig) -> None:
        config.router_strategy = "cheapest"
        router = ModelRouter(config)
        messages = [{"role": "user", "content": "Hello"}]
        decision = router.select_model(messages)
        # Cheapest model should be selected
        assert decision.model is not None
        assert decision.estimated_cost_per_1k >= 0

    def test_fallback_models_included(self, router: ModelRouter) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        decision = router.select_model(messages)
        assert isinstance(decision.fallback_models, list)

    def test_no_providers_raises(self) -> None:
        from unittest.mock import patch
        cfg = AICOSConfig(
            openai_api_key=None,
            anthropic_api_key=None,
            gemini_api_key=None,
            openrouter_api_key=None,
            db_path="~/.aicos/test.db",
        )
        router = ModelRouter(cfg)
        messages = [{"role": "user", "content": "Hello"}]
        # Patch _refresh_available so it leaves _available_models empty
        with patch.object(router, "_refresh_available"):
            router._available_models = {}
            with pytest.raises(RuntimeError, match="No AI providers"):
                router.select_model(messages)


class TestCostEstimation:
    def test_known_model_cost(self, router: ModelRouter) -> None:
        # Nemotron is free
        cost = router.estimate_cost(
            "openrouter/nvidia/llama-3.1-nemotron-ultra-253b-v1",
            input_tokens=1000,
            output_tokens=100,
        )
        assert cost == 0.0

    def test_unknown_model_cost(self, router: ModelRouter) -> None:
        cost = router.estimate_cost("unknown-model", input_tokens=1000, output_tokens=100)
        assert cost == 0.0

    def test_zero_tokens(self, router: ModelRouter) -> None:
        cost = router.estimate_cost(
            "openrouter/nvidia/llama-3.1-nemotron-ultra-253b-v1",
            input_tokens=0,
            output_tokens=0,
        )
        assert cost == 0.0
