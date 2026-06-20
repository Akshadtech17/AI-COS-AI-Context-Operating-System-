"""Tests for metrics collection and cost tracking."""

from __future__ import annotations

import pytest

from aicos.analytics.cost_tracker import CostTracker
from aicos.analytics.metrics import MetricsCollector, Counter, Histogram


class TestCounter:
    def test_initial_value(self) -> None:
        c = Counter()
        assert c.value == 0.0

    def test_increment_by_one(self) -> None:
        c = Counter()
        c.inc()
        assert c.value == 1.0

    def test_increment_by_amount(self) -> None:
        c = Counter()
        c.inc(5.5)
        assert c.value == 5.5

    def test_multiple_increments(self) -> None:
        c = Counter()
        c.inc(1)
        c.inc(2)
        c.inc(3)
        assert c.value == 6.0


class TestHistogram:
    def test_empty_histogram(self) -> None:
        h = Histogram()
        assert h.count == 0
        assert h.mean == 0.0

    def test_observe_updates_count(self) -> None:
        h = Histogram()
        h.observe(100.0)
        assert h.count == 1

    def test_mean_calculation(self) -> None:
        h = Histogram()
        h.observe(100.0)
        h.observe(200.0)
        assert h.mean == 150.0

    def test_percentiles(self) -> None:
        h = Histogram()
        for v in range(1, 101):
            h.observe(float(v))
        assert h.p50() > 0
        assert h.p95() >= h.p50()
        assert h.p99() >= h.p95()

    def test_total(self) -> None:
        h = Histogram()
        h.observe(10.0)
        h.observe(20.0)
        assert h.total == 30.0


class TestMetricsCollector:
    @pytest.fixture
    def collector(self) -> MetricsCollector:
        return MetricsCollector()

    def test_initial_state(self, collector: MetricsCollector) -> None:
        assert collector.requests_total.value == 0
        assert collector.tokens_input_total.value == 0
        assert collector.cost_total_usd.value == 0
        assert collector.cache_hit_rate == 0.0

    def test_record_request(self, collector: MetricsCollector) -> None:
        collector.record_request(
            model="gpt-4o-mini",
            task_type="coding",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            latency_ms=500.0,
        )
        assert collector.requests_total.value == 1
        assert collector.tokens_input_total.value == 100
        assert collector.tokens_output_total.value == 50
        assert collector.cost_total_usd.value == 0.001

    def test_cache_hit_rate(self, collector: MetricsCollector) -> None:
        collector.cache_hits.inc(3)
        collector.cache_misses.inc(1)
        assert collector.cache_hit_rate == 0.75

    def test_cache_hit_rate_zero_total(self, collector: MetricsCollector) -> None:
        assert collector.cache_hit_rate == 0.0

    def test_compression_ratio(self, collector: MetricsCollector) -> None:
        collector.record_compression(tokens_before=1000, tokens_after=400)
        assert abs(collector.compression_ratio - 0.6) < 1e-5

    def test_to_prometheus_format(self, collector: MetricsCollector) -> None:
        collector.requests_total.inc(5)
        prometheus = collector.to_prometheus()
        assert "aicos_requests_total" in prometheus
        assert "5" in prometheus

    def test_to_dict_structure(self, collector: MetricsCollector) -> None:
        data = collector.to_dict()
        assert "requests" in data
        assert "tokens" in data
        assert "cost" in data
        assert "cache" in data
        assert "latency" in data
        assert "uptime_seconds" in data

    def test_per_model_tracking(self, collector: MetricsCollector) -> None:
        collector.record_request(
            model="gpt-4o",
            task_type="reasoning",
            input_tokens=500,
            output_tokens=200,
            cost_usd=0.01,
            latency_ms=2000.0,
        )
        assert "gpt-4o" in collector.requests_by_model
        assert collector.requests_by_model["gpt-4o"].value == 1

    def test_stage_latency_recording(self, collector: MetricsCollector) -> None:
        collector.record_stage_latency("cache_lookup", 5.0)
        assert collector.latency_by_stage["cache_lookup"].count == 1
        assert collector.latency_by_stage["cache_lookup"].mean == 5.0

    def test_error_tracking(self, collector: MetricsCollector) -> None:
        collector.record_request(
            model="gpt-4o-mini",
            task_type="simple",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0,
            latency_ms=0,
            error=True,
        )
        assert collector.request_errors.value == 1


class TestCostTracker:
    @pytest.fixture
    def tracker(self) -> CostTracker:
        return CostTracker()

    def test_compute_cost_nemotron(self, tracker: CostTracker) -> None:
        # Nemotron Ultra is free — cost should be 0
        cost = tracker.compute_cost(
            "openrouter/nvidia/llama-3.1-nemotron-ultra-253b-v1",
            input_tokens=1_000_000,
            output_tokens=0,
        )
        assert cost == 0.0

    def test_compute_cost_unknown_model(self, tracker: CostTracker) -> None:
        cost = tracker.compute_cost("unknown-model", input_tokens=1000, output_tokens=100)
        assert cost > 0  # Falls back to mid-tier estimate

    def test_record_updates_session(self, tracker: CostTracker) -> None:
        tracker.record(
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=100,
            task_type="coding",
        )
        assert tracker.session_cost > 0
        assert tracker._session_requests == 1

    def test_cache_hit_is_free(self, tracker: CostTracker) -> None:
        record = tracker.record(
            model="gpt-4o",
            input_tokens=5000,
            output_tokens=500,
            task_type="analysis",
            cache_hit=True,
        )
        assert record.cost_usd == 0.0

    def test_session_summary(self, tracker: CostTracker) -> None:
        tracker.record("gpt-4o-mini", 100, 50, "simple")
        summary = tracker.session_summary()
        assert "cost_usd" in summary
        assert "requests" in summary
        assert summary["requests"] == 1

    def test_breakdown_by_model(self, tracker: CostTracker) -> None:
        tracker.record("gpt-4o-mini", 100, 50, "simple")
        tracker.record("gpt-4o", 200, 100, "reasoning")
        breakdown = tracker.breakdown_by_model()
        assert "gpt-4o-mini" in breakdown
        assert "gpt-4o" in breakdown
