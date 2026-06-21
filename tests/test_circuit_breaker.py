"""Tests for the per-provider circuit breaker."""

from __future__ import annotations

import time

from aicos.core.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitState


class TestCircuitBreakerStates:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker(provider="openai")
        assert cb.state is CircuitState.CLOSED
        assert cb.can_attempt() is True

    def test_single_failure_stays_closed(self) -> None:
        cb = CircuitBreaker(provider="openai", failure_threshold=3)
        cb.record_failure()
        assert cb.state is CircuitState.CLOSED
        assert cb.can_attempt() is True

    def test_reaches_threshold_opens(self) -> None:
        cb = CircuitBreaker(provider="openai", failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state is CircuitState.OPEN
        assert cb.can_attempt() is False

    def test_open_blocks_immediately(self) -> None:
        cb = CircuitBreaker(provider="openai", failure_threshold=1)
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        # All subsequent calls blocked without any waiting
        for _ in range(5):
            assert cb.can_attempt() is False

    def test_open_transitions_to_half_open_after_timeout(self) -> None:
        cb = CircuitBreaker(provider="openai", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        assert cb.state is CircuitState.OPEN

        time.sleep(0.1)

        assert cb.can_attempt() is True  # triggers HALF_OPEN transition
        assert cb.state is CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self) -> None:
        cb = CircuitBreaker(provider="openai", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        cb.can_attempt()  # transitions to HALF_OPEN

        cb.record_success()
        assert cb.state is CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.can_attempt() is True

    def test_half_open_failure_reopens_circuit(self) -> None:
        cb = CircuitBreaker(provider="openai", failure_threshold=1, recovery_timeout=0.05)
        cb.record_failure()
        time.sleep(0.1)
        cb.can_attempt()  # transitions to HALF_OPEN

        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        assert cb.can_attempt() is False

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker(provider="openai", failure_threshold=5)
        for _ in range(3):
            cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state is CircuitState.CLOSED

    def test_status_dict_keys(self) -> None:
        cb = CircuitBreaker(provider="anthropic", failure_threshold=3)
        status = cb.status()
        assert status["provider"] == "anthropic"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["seconds_open"] is None

    def test_status_reports_time_open(self) -> None:
        cb = CircuitBreaker(provider="openai", failure_threshold=1, recovery_timeout=60.0)
        cb.record_failure()
        status = cb.status()
        assert status["state"] == "open"
        assert isinstance(status["seconds_open"], float)
        assert float(str(status["seconds_open"])) >= 0


class TestCircuitBreakerRegistry:
    def test_creates_breaker_on_demand(self) -> None:
        registry = CircuitBreakerRegistry()
        breaker = registry.get("openai")
        assert isinstance(breaker, CircuitBreaker)
        assert breaker.provider == "openai"

    def test_same_provider_returns_same_instance(self) -> None:
        registry = CircuitBreakerRegistry()
        b1 = registry.get("openai")
        b2 = registry.get("openai")
        assert b1 is b2

    def test_different_providers_independent(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=1)
        registry.get("openai").record_failure()
        assert registry.get("openai").state is CircuitState.OPEN
        assert registry.get("anthropic").state is CircuitState.CLOSED

    def test_all_status_returns_list(self) -> None:
        registry = CircuitBreakerRegistry()
        registry.get("openai")
        registry.get("anthropic")
        statuses = registry.all_status()
        assert len(statuses) == 2
        providers = {s["provider"] for s in statuses}
        assert providers == {"openai", "anthropic"}

    def test_custom_threshold_propagates(self) -> None:
        registry = CircuitBreakerRegistry(failure_threshold=2, recovery_timeout=10.0)
        cb = registry.get("gemini")
        assert cb.failure_threshold == 2
        assert cb.recovery_timeout == 10.0
