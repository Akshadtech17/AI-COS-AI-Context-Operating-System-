"""
Per-provider circuit breaker.

State machine:
  CLOSED   — normal; failures accumulate toward threshold
  OPEN     — provider blocked; new calls fail immediately
             after recovery_timeout → transitions to HALF_OPEN
  HALF_OPEN — one probe call allowed
             success → CLOSED (reset count)
             failure → OPEN (reset timer)

Usage:
    registry = CircuitBreakerRegistry()
    breaker = registry.get("openai")

    if not breaker.can_attempt():
        raise ProviderCircuitOpen("openai circuit is open")
    try:
        result = await provider.complete(...)
        breaker.record_success()
    except Exception:
        breaker.record_failure()
        raise
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a call is rejected because the circuit is open."""


@dataclass
class CircuitBreaker:
    provider: str
    failure_threshold: int = 5
    recovery_timeout: float = 30.0

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _failure_count: int = field(default=0, init=False, repr=False)
    _opened_at: float | None = field(default=None, init=False, repr=False)

    def can_attempt(self) -> bool:
        """Return True if a call should be attempted."""
        if self._state is CircuitState.CLOSED:
            return True

        if self._state is CircuitState.OPEN:
            elapsed = time.monotonic() - (self._opened_at or 0)
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                return True
            return False

        # HALF_OPEN: allow one probe through
        return True

    def record_success(self) -> None:
        """Call after a provider call succeeds."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self) -> None:
        """Call after a provider call fails."""
        self._failure_count += 1
        if (
            self._state is CircuitState.HALF_OPEN
            or self._failure_count >= self.failure_threshold
        ):
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def status(self) -> dict[str, object]:
        elapsed = None
        if self._opened_at is not None:
            elapsed = round(time.monotonic() - self._opened_at, 1)
        return {
            "provider": self.provider,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "seconds_open": elapsed,
        }

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count


class CircuitBreakerRegistry:
    """Holds one CircuitBreaker per provider name."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, provider: str) -> CircuitBreaker:
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(
                provider=provider,
                failure_threshold=self._failure_threshold,
                recovery_timeout=self._recovery_timeout,
            )
        return self._breakers[provider]

    def all_status(self) -> list[dict[str, object]]:
        return [b.status() for b in self._breakers.values()]
