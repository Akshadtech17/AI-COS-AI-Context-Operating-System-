"""Integration tests for HTTP routes (routes.py)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aicos.api.routes import create_app
from aicos.core.config import AICOSConfig
from aicos.core.gateway import GatewayResponse


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def route_config(tmp_path):
    return AICOSConfig(
        openai_api_key="sk-test-key",
        openrouter_api_key=None,
        anthropic_api_key=None,
        gemini_api_key=None,
        nvidia_api_key=None,
        db_path=str(tmp_path / "test.db"),
        cache_enabled=False,
        memory_enabled=False,
        context_compression_enabled=False,
        analytics_enabled=True,
        gateway_api_key=None,
    )


def _make_response(**overrides: Any) -> GatewayResponse:
    defaults = dict(
        content="Hello from AI-COS",
        model="gpt-4o-mini",
        provider="openai",
        task_type="simple",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.0001,
        latency_ms=120.0,
        cache_hit=False,
        cache_hit_type=None,
        tokens_before_compression=10,
        tokens_after_compression=10,
        memories_injected=0,
        routing_reason="strategy=auto, task=simple",
    )
    return GatewayResponse(**{**defaults, **overrides})


@pytest.fixture
def client(route_config):
    mock_gw = MagicMock()
    mock_gw.process = AsyncMock(return_value=_make_response())

    async def _fake_build(cfg):
        return mock_gw, None

    with patch("aicos.api.routes._build_gateway", _fake_build):
        app = create_app(config=route_config)
        with TestClient(app) as c:
            yield c, mock_gw


@pytest.fixture
def auth_client(tmp_path):
    cfg = AICOSConfig(
        openai_api_key="sk-test-key",
        db_path=str(tmp_path / "test.db"),
        cache_enabled=False,
        memory_enabled=False,
        context_compression_enabled=False,
        gateway_api_key="secret-key",
    )
    mock_gw = MagicMock()
    mock_gw.process = AsyncMock(return_value=_make_response())

    async def _fake_build(cfg):
        return mock_gw, None

    with patch("aicos.api.routes._build_gateway", _fake_build):
        app = create_app(config=cfg)
        with TestClient(app) as c:
            yield c


# ── Health & Meta ─────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self, client) -> None:
        c, _ = client
        r = c.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.2.0"

    def test_health_lists_providers(self, client) -> None:
        c, _ = client
        r = c.get("/health")
        assert "providers" in r.json()

    def test_health_shows_feature_flags(self, client) -> None:
        c, _ = client
        data = r.json() if (r := c.get("/health")) else {}
        assert "cache_enabled" in data
        assert "memory_enabled" in data


class TestStatsEndpoint:
    def test_stats_returns_dict(self, client) -> None:
        c, _ = client
        r = c.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert "requests" in data
        assert "cache" in data
        assert "latency" in data
        assert "cost" in data

    def test_stats_has_dashboard_fields(self, client) -> None:
        c, _ = client
        data = c.get("/stats").json()
        # Dashboard-specific aliases
        assert "by_task_type" in data["requests"]
        assert "hit_rate_pct" in data["cache"]
        assert "total_stored" in data["memory"]
        assert "context" in data
        assert "tokens_saved" in data["context"]
        assert "compression_ratio_pct" in data["context"]


class TestMetricsEndpoint:
    def test_metrics_returns_prometheus_format(self, client) -> None:
        c, _ = client
        r = c.get("/metrics")
        assert r.status_code == 200
        assert "aicos_" in r.text


# ── Dashboard ─────────────────────────────────────────────────────────────────

class TestDashboard:
    def test_root_serves_dashboard(self, client) -> None:
        c, _ = client
        r = c.get("/")
        assert r.status_code == 200
        assert "AI-COS" in r.text

    def test_dashboard_path_serves_dashboard(self, client) -> None:
        c, _ = client
        r = c.get("/dashboard")
        assert r.status_code == 200
        assert "AI-COS" in r.text

    def test_dashboard_is_html(self, client) -> None:
        c, _ = client
        r = c.get("/dashboard")
        assert "text/html" in r.headers["content-type"]


# ── Models ────────────────────────────────────────────────────────────────────

class TestModelsEndpoint:
    def test_list_models_returns_list(self, client) -> None:
        c, _ = client
        r = c.get("/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)

    def test_models_include_provider_info(self, client) -> None:
        c, _ = client
        data = c.get("/v1/models").json()
        if data["data"]:
            model = data["data"][0]
            assert "provider" in model
            assert "capabilities" in model
            assert "cost_per_1m_input_usd" in model


# ── Chat Completions ──────────────────────────────────────────────────────────

class TestChatCompletions:
    def test_basic_chat(self, client) -> None:
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "model": "auto",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello from AI-COS"

    def test_response_has_openai_fields(self, client) -> None:
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
        })
        data = r.json()
        assert "id" in data
        assert "choices" in data
        assert "usage" in data
        assert "aicos" in data

    def test_aicos_metadata_in_response(self, client) -> None:
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
        })
        aicos = r.json()["aicos"]
        assert "cache_hit" in aicos
        assert "task_type" in aicos
        assert "cost_usd" in aicos
        assert "latency_ms" in aicos

    def test_gateway_process_called(self, client) -> None:
        c, mock_gw = client
        c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
        })
        mock_gw.process.assert_called_once()

    def test_skip_cache_flag(self, client) -> None:
        c, mock_gw = client
        c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "skip_cache": True,
        })
        call_args = mock_gw.process.call_args[0][0]
        assert call_args.skip_cache is True

    def test_validation_rejects_invalid_temperature(self, client) -> None:
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 5.0,  # max is 2.0
        })
        assert r.status_code == 422

    def test_validation_rejects_missing_messages(self, client) -> None:
        c, _ = client
        r = c.post("/v1/chat/completions", json={"model": "auto"})
        assert r.status_code == 422

    def test_cache_hit_response(self, client) -> None:
        c, mock_gw = client
        mock_gw.process = AsyncMock(return_value=_make_response(
            cache_hit=True, cache_hit_type="semantic", cost_usd=0.0,
        ))
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert r.json()["aicos"]["cache_hit"] is True


# ── Auth ──────────────────────────────────────────────────────────────────────

class TestAuthentication:
    def test_no_key_required_without_config(self, client) -> None:
        c, _ = client
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
        })
        assert r.status_code == 200

    def test_missing_key_returns_401(self, auth_client) -> None:
        r = auth_client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "test"}],
        })
        assert r.status_code == 401

    def test_wrong_key_returns_403(self, auth_client) -> None:
        r = auth_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 403

    def test_correct_key_returns_200(self, auth_client) -> None:
        r = auth_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
            headers={"Authorization": "Bearer secret-key"},
        )
        assert r.status_code == 200


# ── CORS ──────────────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_headers_present(self, client) -> None:
        c, _ = client
        r = c.options("/v1/chat/completions", headers={"Origin": "http://localhost:3000"})
        assert r.headers.get("access-control-allow-origin") in ("*", "http://localhost:3000")

    def test_openapi_schema_accessible(self, client) -> None:
        c, _ = client
        r = c.get("/openapi.json")
        assert r.status_code == 200
        assert r.json()["info"]["version"] == "0.2.0"
