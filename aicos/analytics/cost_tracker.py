"""
Real-time cost tracker with per-session and cumulative tracking.
Persists cost history to SQLite for reporting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from aicos.analytics.metrics import get_metrics
from aicos.core.router import MODEL_REGISTRY


@dataclass
class CostRecord:
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    task_type: str
    timestamp: datetime
    session_id: str | None = None
    cache_hit: bool = False


class CostTracker:
    """Tracks and persists per-request cost data."""

    def __init__(self) -> None:
        self._session_cost: float = 0.0
        self._session_tokens_in: int = 0
        self._session_tokens_out: int = 0
        self._session_requests: int = 0
        self._session_start: float = time.time()
        self._records: list[CostRecord] = []

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        spec = MODEL_REGISTRY.get(model)
        if not spec:
            # Unknown model — estimate at mid-tier pricing
            return (input_tokens * 1.0 + output_tokens * 3.0) / 1_000_000
        return (
            input_tokens * spec.input_cost_per_1m / 1_000_000
            + output_tokens * spec.output_cost_per_1m / 1_000_000
        )

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str,
        session_id: str | None = None,
        cache_hit: bool = False,
    ) -> CostRecord:
        cost = 0.0 if cache_hit else self.compute_cost(model, input_tokens, output_tokens)
        spec = MODEL_REGISTRY.get(model)
        provider = spec.provider if spec else "unknown"

        record = CostRecord(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_type=task_type,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            cache_hit=cache_hit,
        )

        self._records.append(record)
        self._session_cost += cost
        self._session_tokens_in += input_tokens
        self._session_tokens_out += output_tokens
        self._session_requests += 1

        metrics = get_metrics()
        metrics.record_request(
            model=model,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=0,
            cache_hit=cache_hit,
        )

        return record

    def compute_savings(self, cached_model: str, input_tokens: int, output_tokens: int) -> float:
        """What it would have cost without the cache hit."""
        return self.compute_cost(cached_model, input_tokens, output_tokens)

    @property
    def session_cost(self) -> float:
        return self._session_cost

    @property
    def session_tokens(self) -> tuple[int, int]:
        return self._session_tokens_in, self._session_tokens_out

    @property
    def session_duration_seconds(self) -> float:
        return time.time() - self._session_start

    def session_summary(self) -> dict[str, object]:
        duration = self.session_duration_seconds
        return {
            "requests": self._session_requests,
            "input_tokens": self._session_tokens_in,
            "output_tokens": self._session_tokens_out,
            "total_tokens": self._session_tokens_in + self._session_tokens_out,
            "cost_usd": round(self._session_cost, 6),
            "duration_seconds": round(duration, 2),
            "cost_per_minute": round(self._session_cost / (duration / 60), 6) if duration > 0 else 0,
        }

    def breakdown_by_model(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for r in self._records:
            if r.model not in result:
                result[r.model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            result[r.model]["requests"] = int(result[r.model]["requests"]) + 1  # type: ignore
            result[r.model]["input_tokens"] = int(result[r.model]["input_tokens"]) + r.input_tokens  # type: ignore
            result[r.model]["output_tokens"] = int(result[r.model]["output_tokens"]) + r.output_tokens  # type: ignore
            result[r.model]["cost_usd"] = round(float(result[r.model]["cost_usd"]) + r.cost_usd, 6)  # type: ignore
        return result

    def cheapest_model_for_task(self, task_type: str) -> str | None:
        """Return the cheapest model seen in this session for a given task."""
        task_records = [r for r in self._records if r.task_type == task_type]
        if not task_records:
            return None
        return min(task_records, key=lambda r: r.cost_usd / max(r.input_tokens + r.output_tokens, 1)).model
