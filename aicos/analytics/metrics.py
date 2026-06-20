"""
Prometheus-compatible metrics collector for AI-COS.
Tracks requests, tokens, costs, latency, and cache performance.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from threading import Lock
from typing import Iterator


@dataclass
class Counter:
    _value: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> float:
        return self._value


@dataclass
class Histogram:
    """Simple histogram with fixed buckets for latency tracking."""
    buckets: list[float] = field(default_factory=lambda: [
        5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000
    ])
    _counts: dict[float, int] = field(default_factory=dict)
    _sum: float = 0.0
    _count: int = 0
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self._counts = {b: 0 for b in self.buckets}
        self._counts[float("inf")] = 0

    def observe(self, value_ms: float) -> None:
        with self._lock:
            self._sum += value_ms
            self._count += 1
            for b in self.buckets:
                if value_ms <= b:
                    self._counts[b] += 1
            self._counts[float("inf")] += 1

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def total(self) -> float:
        return self._sum

    def p50(self) -> float:
        return self._percentile(0.50)

    def p95(self) -> float:
        return self._percentile(0.95)

    def p99(self) -> float:
        return self._percentile(0.99)

    def _percentile(self, p: float) -> float:
        if not self._count:
            return 0.0
        target = self._count * p
        cumulative = 0
        prev_b = 0.0
        for b in sorted(self._counts):
            cumulative += self._counts[b]
            if cumulative >= target:
                return b
            prev_b = b
        return prev_b


class MetricsCollector:
    """Thread-safe Prometheus-compatible metrics collector."""

    def __init__(self) -> None:
        # Request counters
        self.requests_total: Counter = Counter()
        self.requests_by_model: defaultdict[str, Counter] = defaultdict(Counter)
        self.requests_by_task: defaultdict[str, Counter] = defaultdict(Counter)
        self.request_errors: Counter = Counter()

        # Token tracking
        self.tokens_input_total: Counter = Counter()
        self.tokens_output_total: Counter = Counter()
        self.tokens_by_model: defaultdict[str, dict[str, Counter]] = defaultdict(
            lambda: {"input": Counter(), "output": Counter()}
        )

        # Cost tracking
        self.cost_total_usd: Counter = Counter()
        self.cost_by_model: defaultdict[str, Counter] = defaultdict(Counter)
        self.cost_saved_usd: Counter = Counter()  # Via cache hits

        # Cache metrics
        self.cache_hits: Counter = Counter()
        self.cache_misses: Counter = Counter()
        self.cache_semantic_hits: Counter = Counter()

        # Latency histograms (milliseconds)
        self.latency_total_ms: Histogram = Histogram()
        self.latency_by_stage: dict[str, Histogram] = {
            "cache_lookup": Histogram(),
            "context_compression": Histogram(),
            "memory_retrieval": Histogram(),
            "llm_call": Histogram(),
            "gateway_overhead": Histogram(),
        }

        # Memory metrics
        self.memory_stored: Counter = Counter()
        self.memory_retrieved: Counter = Counter()

        # Compression metrics
        self.tokens_before_compression: Counter = Counter()
        self.tokens_after_compression: Counter = Counter()

        self._start_time: float = time.time()

    def record_request(
        self,
        model: str,
        task_type: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        cache_hit: bool = False,
        error: bool = False,
    ) -> None:
        self.requests_total.inc()
        self.requests_by_model[model].inc()
        self.requests_by_task[task_type].inc()

        if error:
            self.request_errors.inc()
            return

        self.tokens_input_total.inc(input_tokens)
        self.tokens_output_total.inc(output_tokens)
        self.tokens_by_model[model]["input"].inc(input_tokens)
        self.tokens_by_model[model]["output"].inc(output_tokens)

        self.cost_total_usd.inc(cost_usd)
        self.cost_by_model[model].inc(cost_usd)

        self.latency_total_ms.observe(latency_ms)
        self.latency_by_stage["llm_call"].observe(latency_ms)

        if cache_hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()

    def record_stage_latency(self, stage: str, latency_ms: float) -> None:
        if stage in self.latency_by_stage:
            self.latency_by_stage[stage].observe(latency_ms)

    def record_compression(self, tokens_before: int, tokens_after: int) -> None:
        self.tokens_before_compression.inc(tokens_before)
        self.tokens_after_compression.inc(tokens_after)

    def record_cost_saved(self, usd: float) -> None:
        self.cost_saved_usd.inc(usd)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits.value + self.cache_misses.value
        return self.cache_hits.value / total if total else 0.0

    @property
    def compression_ratio(self) -> float:
        before = self.tokens_before_compression.value
        after = self.tokens_after_compression.value
        return 1.0 - (after / before) if before > 0 else 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def to_prometheus(self) -> str:
        """Render metrics in Prometheus text exposition format."""
        lines: list[str] = []

        def gauge(name: str, value: float, help_text: str, labels: str = "") -> None:
            lines.append(f"# HELP aicos_{name} {help_text}")
            lines.append(f"# TYPE aicos_{name} gauge")
            if labels:
                lines.append(f'aicos_{name}{{{labels}}} {value}')
            else:
                lines.append(f"aicos_{name} {value}")

        def counter(name: str, value: float, help_text: str, labels: str = "") -> None:
            lines.append(f"# HELP aicos_{name}_total {help_text}")
            lines.append(f"# TYPE aicos_{name}_total counter")
            if labels:
                lines.append(f'aicos_{name}_total{{{labels}}} {value}')
            else:
                lines.append(f"aicos_{name}_total {value}")

        counter("requests", self.requests_total.value, "Total LLM requests")
        counter("request_errors", self.request_errors.value, "Total request errors")
        counter("tokens_input", self.tokens_input_total.value, "Total input tokens processed")
        counter("tokens_output", self.tokens_output_total.value, "Total output tokens generated")
        counter("cache_hits", self.cache_hits.value, "Total semantic cache hits")
        counter("cache_misses", self.cache_misses.value, "Total cache misses")
        counter("cost_usd", self.cost_total_usd.value, "Total cost in USD")
        counter("cost_saved_usd", self.cost_saved_usd.value, "Total cost saved via cache in USD")
        counter("memory_stored", self.memory_stored.value, "Total memories stored")
        counter("memory_retrieved", self.memory_retrieved.value, "Total memories retrieved")

        gauge("cache_hit_rate", round(self.cache_hit_rate, 4), "Cache hit rate (0-1)")
        gauge("compression_ratio", round(self.compression_ratio, 4), "Token compression ratio (0-1)")
        gauge("uptime_seconds", round(self.uptime_seconds, 2), "Uptime in seconds")

        gauge(
            "latency_mean_ms",
            round(self.latency_total_ms.mean, 2),
            "Mean request latency in milliseconds",
        )
        gauge(
            "latency_p95_ms",
            round(self.latency_total_ms.p95(), 2),
            "P95 request latency in milliseconds",
        )
        gauge(
            "latency_p99_ms",
            round(self.latency_total_ms.p99(), 2),
            "P99 request latency in milliseconds",
        )

        for stage, hist in self.latency_by_stage.items():
            gauge(
                f"stage_latency_mean_ms",
                round(hist.mean, 2),
                f"Mean {stage} latency",
                labels=f'stage="{stage}"',
            )

        for model, cnt in self.requests_by_model.items():
            counter(
                "requests_by_model",
                cnt.value,
                "Requests per model",
                labels=f'model="{model}"',
            )

        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        tokens_saved = max(
            0,
            self.tokens_before_compression.value - self.tokens_after_compression.value,
        )
        hit_rate_pct = round(self.cache_hit_rate * 100, 1)
        compression_ratio_pct = round(self.compression_ratio * 100, 1)
        task_counts = {t: c.value for t, c in self.requests_by_task.items()}
        memories = self.memory_stored.value

        return {
            "requests": {
                "total": self.requests_total.value,
                "errors": self.request_errors.value,
                "by_model": {m: c.value for m, c in self.requests_by_model.items()},
                "by_task": task_counts,
                "by_task_type": task_counts,  # dashboard alias
            },
            "tokens": {
                "input_total": self.tokens_input_total.value,
                "output_total": self.tokens_output_total.value,
            },
            "cost": {
                "total_usd": round(self.cost_total_usd.value, 6),
                "saved_usd": round(self.cost_saved_usd.value, 6),
                "by_model": {m: round(c.value, 6) for m, c in self.cost_by_model.items()},
            },
            "cache": {
                "hits": self.cache_hits.value,
                "misses": self.cache_misses.value,
                "hit_rate": round(self.cache_hit_rate, 4),
                "hit_rate_pct": hit_rate_pct,  # dashboard alias (0-100)
                "semantic_hits": self.cache_semantic_hits.value,
            },
            "latency": {
                "mean_ms": round(self.latency_total_ms.mean, 2),
                "p50_ms": round(self.latency_total_ms.p50(), 2),
                "p95_ms": round(self.latency_total_ms.p95(), 2),
                "p99_ms": round(self.latency_total_ms.p99(), 2),
            },
            "compression": {
                "tokens_saved": tokens_saved,
                "ratio": round(self.compression_ratio, 4),
            },
            # dashboard reads stats.context.*
            "context": {
                "tokens_saved": tokens_saved,
                "compression_ratio_pct": compression_ratio_pct,
            },
            "memory": {
                "stored": memories,
                "total_stored": memories,  # dashboard alias
                "retrieved": self.memory_retrieved.value,
                "max_items": 10_000,
            },
            "uptime_seconds": round(self.uptime_seconds, 2),
        }


@lru_cache(maxsize=1)
def get_metrics() -> MetricsCollector:
    return MetricsCollector()
