"""Integration tests for HTTP routes (routes.py)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aicos.api.routes import create_app
from aicos.core.config import AICOSConfig
from aicos.core.gateway import GatewayResponse
from aicos.providers.base import StreamChunk

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


def _mock_gateway() -> MagicMock:
    mock_gw = MagicMock()
    mock_gw.process = AsyncMock(return_value=_make_response())
    mock_gw.close = AsyncMock()
    return mock_gw


@pytest.fixture
def client(route_config):
    mock_gw = _mock_gateway()

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
    mock_gw = _mock_gateway()

    async def _fake_build(cfg):
        return mock_gw, None

    with patch("aicos.api.routes._build_gateway", _fake_build):
        app = create_app(config=cfg)
        with TestClient(app) as c:
            yield c


def _make_mock_memory_store() -> MagicMock:
    mock_item = MagicMock()
    mock_item.id = 1
    mock_item.content = "Test memory content"
    mock_item.tag_list = ["test", "demo"]
    mock_item.created_at = datetime(2026, 1, 1, 12, 0, 0)

    ms = MagicMock()
    ms.store = AsyncMock(return_value=1)
    ms.search = AsyncMock(return_value=[(mock_item, 0.95)])
    ms.forget = AsyncMock(return_value=True)
    ms.close = AsyncMock()
    return ms


@pytest.fixture
def client_with_memory(route_config):
    mock_gw = _mock_gateway()
    mock_ms = _make_mock_memory_store()

    async def _fake_build(cfg):
        return mock_gw, mock_ms

    with patch("aicos.api.routes._build_gateway", _fake_build):
        app = create_app(config=route_config)
        with TestClient(app) as c:
            yield c, mock_gw, mock_ms


@pytest.fixture
def rate_limited_client(tmp_path):
    cfg = AICOSConfig(
        openai_api_key="sk-test-key",
        db_path=str(tmp_path / "test.db"),
        cache_enabled=False,
        memory_enabled=False,
        context_compression_enabled=False,
        rate_limit_enabled=True,
        rate_limit_rpm=2,  # Very low for testing
    )
    mock_gw = _mock_gateway()

    async def _fake_build(cfg):
        return mock_gw, None

    with patch("aicos.api.routes._build_gateway", _fake_build):
        app = create_app(config=cfg)
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Health & Meta ─────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_ok(self, client) -> None:
        c, _ = client
        r = c.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.5.0"

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
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "auto",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello from AI-COS"

    def test_response_has_openai_fields(self, client) -> None:
        c, _ = client
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        data = r.json()
        assert "id" in data
        assert "choices" in data
        assert "usage" in data
        assert "aicos" in data

    def test_aicos_metadata_in_response(self, client) -> None:
        c, _ = client
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        aicos = r.json()["aicos"]
        assert "cache_hit" in aicos
        assert "task_type" in aicos
        assert "cost_usd" in aicos
        assert "latency_ms" in aicos

    def test_gateway_process_called(self, client) -> None:
        c, mock_gw = client
        c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        mock_gw.process.assert_called_once()

    def test_skip_cache_flag(self, client) -> None:
        c, mock_gw = client
        c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "skip_cache": True,
            },
        )
        call_args = mock_gw.process.call_args[0][0]
        assert call_args.skip_cache is True

    def test_validation_rejects_invalid_temperature(self, client) -> None:
        c, _ = client
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 5.0,  # max is 2.0
            },
        )
        assert r.status_code == 422

    def test_validation_rejects_missing_messages(self, client) -> None:
        c, _ = client
        r = c.post("/v1/chat/completions", json={"model": "auto"})
        assert r.status_code == 422

    def test_cache_hit_response(self, client) -> None:
        c, mock_gw = client
        mock_gw.process = AsyncMock(
            return_value=_make_response(
                cache_hit=True,
                cache_hit_type="semantic",
                cost_usd=0.0,
            )
        )
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert r.json()["aicos"]["cache_hit"] is True


# ── Auth ──────────────────────────────────────────────────────────────────────


class TestAuthentication:
    def test_no_key_required_without_config(self, client) -> None:
        c, _ = client
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert r.status_code == 200

    def test_missing_key_returns_401(self, auth_client) -> None:
        r = auth_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
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
        assert r.json()["info"]["version"] == "0.5.0"


# ── Streaming ─────────────────────────────────────────────────────────────────


class TestStreaming:
    @staticmethod
    async def _stream_chunks(request):
        yield StreamChunk(delta="Hello", model="gpt-4o-mini")
        yield StreamChunk(delta=" world", model="gpt-4o-mini")
        yield StreamChunk(
            delta="",
            model="gpt-4o-mini",
            finish_reason="stop",
            input_tokens=5,
            output_tokens=2,
        )

    def test_streaming_returns_200(self, client) -> None:
        c, mock_gw = client
        mock_gw.stream = self._stream_chunks
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        assert r.status_code == 200

    def test_streaming_content_type_is_sse(self, client) -> None:
        c, mock_gw = client
        mock_gw.stream = self._stream_chunks
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        assert "text/event-stream" in r.headers.get("content-type", "")

    def test_streaming_body_contains_done(self, client) -> None:
        c, mock_gw = client
        mock_gw.stream = self._stream_chunks
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        assert "[DONE]" in r.text

    def test_streaming_body_contains_json_chunks(self, client) -> None:
        c, mock_gw = client
        mock_gw.stream = self._stream_chunks
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        assert "chat.completion.chunk" in r.text

    def test_streaming_non_streaming_are_different(self, client) -> None:
        c, mock_gw = client
        mock_gw.stream = self._stream_chunks
        r_stream = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        r_sync = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )
        # Stream response contains SSE format, sync contains JSON object
        assert "[DONE]" in r_stream.text
        assert r_sync.json()["object"] == "chat.completion"


# ── Memory API ────────────────────────────────────────────────────────────────


class TestMemoryAPI:
    def test_store_memory_returns_id(self, client_with_memory) -> None:
        c, _, mock_ms = client_with_memory
        r = c.post("/v1/memory", json={"content": "Remember this fact"})
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == 1
        assert data["status"] == "stored"

    def test_store_memory_calls_store(self, client_with_memory) -> None:
        c, _, mock_ms = client_with_memory
        c.post("/v1/memory", json={"content": "Test memory", "tags": ["work"]})
        mock_ms.store.assert_called_once()
        call_kwargs = mock_ms.store.call_args[1]
        assert call_kwargs["content"] == "Test memory"
        assert call_kwargs["tags"] == ["work"]

    def test_store_memory_with_metadata(self, client_with_memory) -> None:
        c, _, mock_ms = client_with_memory
        r = c.post(
            "/v1/memory",
            json={
                "content": "Test",
                "tags": ["a", "b"],
                "metadata": {"source": "user"},
            },
        )
        assert r.status_code == 200

    def test_search_memory_returns_results(self, client_with_memory) -> None:
        c, _, mock_ms = client_with_memory
        r = c.get("/v1/memory/search?query=test")
        assert r.status_code == 200
        data = r.json()
        assert "query" in data
        assert "results" in data
        assert data["query"] == "test"

    def test_search_memory_result_structure(self, client_with_memory) -> None:
        c, _, mock_ms = client_with_memory
        r = c.get("/v1/memory/search?query=test")
        results = r.json()["results"]
        assert len(results) == 1
        result = results[0]
        assert result["id"] == 1
        assert result["content"] == "Test memory content"
        assert result["score"] == 0.95
        assert "tags" in result
        assert "created_at" in result

    def test_search_memory_custom_params(self, client_with_memory) -> None:
        c, _, mock_ms = client_with_memory
        c.get("/v1/memory/search?query=test&top_k=3&threshold=0.5")
        mock_ms.search.assert_called_once_with("test", top_k=3, threshold=0.5)

    def test_delete_memory_returns_status(self, client_with_memory) -> None:
        c, _, mock_ms = client_with_memory
        r = c.delete("/v1/memory/1")
        assert r.status_code == 200
        assert r.json()["status"] == "deleted"

    def test_delete_memory_not_found(self, client_with_memory) -> None:
        c, _, mock_ms = client_with_memory
        mock_ms.forget = AsyncMock(return_value=False)
        r = c.delete("/v1/memory/999")
        assert r.status_code == 404

    def test_memory_unavailable_returns_503(self, client) -> None:
        c, _ = client
        r = c.post("/v1/memory", json={"content": "test"})
        assert r.status_code == 503


# ── API Key Management ────────────────────────────────────────────────────────


@pytest.fixture
def key_mgmt_client(tmp_path):
    """Client with master key configured for key management tests."""
    cfg = AICOSConfig(
        openai_api_key="sk-test-key",
        db_path=str(tmp_path / "test.db"),
        cache_enabled=False,
        memory_enabled=False,
        context_compression_enabled=False,
        gateway_api_key="master-key-secret",
    )
    mock_gw = _mock_gateway()

    async def _fake_build(cfg):
        return mock_gw, None

    with patch("aicos.api.routes._build_gateway", _fake_build):
        app = create_app(config=cfg)
        with TestClient(app) as c:
            yield c


class TestAPIKeyManagement:
    def test_create_key_requires_master_key(self, key_mgmt_client) -> None:
        r = key_mgmt_client.post("/v1/keys", json={"name": "test-key"})
        assert r.status_code == 401

    def test_create_key_wrong_master_key_returns_403(self, key_mgmt_client) -> None:
        r = key_mgmt_client.post(
            "/v1/keys",
            json={"name": "test-key"},
            headers={"Authorization": "Bearer wrong-master"},
        )
        assert r.status_code == 403

    def test_create_key_returns_plaintext_once(self, key_mgmt_client) -> None:
        r = key_mgmt_client.post(
            "/v1/keys",
            json={"name": "my-service"},
            headers={"Authorization": "Bearer master-key-secret"},
        )
        assert r.status_code == 201
        data = r.json()
        assert "key" in data
        assert data["key"].startswith("aicos-")
        assert "id" in data
        assert data["name"] == "my-service"
        assert "prefix" in data
        assert "warning" in data

    def test_create_key_with_custom_scopes(self, key_mgmt_client) -> None:
        r = key_mgmt_client.post(
            "/v1/keys",
            json={"name": "read-only", "scopes": ["chat"]},
            headers={"Authorization": "Bearer master-key-secret"},
        )
        assert r.status_code == 201
        assert r.json()["scopes"] == ["chat"]

    def test_list_keys_requires_master_key(self, key_mgmt_client) -> None:
        r = key_mgmt_client.get("/v1/keys")
        assert r.status_code == 401

    def test_list_keys_shows_created_keys(self, key_mgmt_client) -> None:
        key_mgmt_client.post(
            "/v1/keys",
            json={"name": "service-a"},
            headers={"Authorization": "Bearer master-key-secret"},
        )
        r = key_mgmt_client.get(
            "/v1/keys",
            headers={"Authorization": "Bearer master-key-secret"},
        )
        assert r.status_code == 200
        keys = r.json()["keys"]
        assert len(keys) >= 1
        assert any(k["name"] == "service-a" for k in keys)

    def test_list_keys_does_not_expose_hash(self, key_mgmt_client) -> None:
        key_mgmt_client.post(
            "/v1/keys",
            json={"name": "service-b"},
            headers={"Authorization": "Bearer master-key-secret"},
        )
        r = key_mgmt_client.get(
            "/v1/keys",
            headers={"Authorization": "Bearer master-key-secret"},
        )
        for key in r.json()["keys"]:
            assert "key_hash" not in key
            assert "hash" not in key
            assert len(key.get("prefix", "")) <= 20

    def test_revoke_key_removes_from_list(self, key_mgmt_client) -> None:
        create_r = key_mgmt_client.post(
            "/v1/keys",
            json={"name": "to-revoke"},
            headers={"Authorization": "Bearer master-key-secret"},
        )
        key_id = create_r.json()["id"]

        revoke_r = key_mgmt_client.delete(
            f"/v1/keys/{key_id}",
            headers={"Authorization": "Bearer master-key-secret"},
        )
        assert revoke_r.status_code == 200
        assert revoke_r.json()["status"] == "revoked"

        list_r = key_mgmt_client.get(
            "/v1/keys",
            headers={"Authorization": "Bearer master-key-secret"},
        )
        keys = list_r.json()["keys"]
        assert not any(k["id"] == key_id for k in keys)

    def test_revoke_nonexistent_key_returns_404(self, key_mgmt_client) -> None:
        r = key_mgmt_client.delete(
            "/v1/keys/99999",
            headers={"Authorization": "Bearer master-key-secret"},
        )
        assert r.status_code == 404

    def test_per_user_key_grants_chat_access(self, key_mgmt_client) -> None:
        create_r = key_mgmt_client.post(
            "/v1/keys",
            json={"name": "user-key"},
            headers={"Authorization": "Bearer master-key-secret"},
        )
        user_key = create_r.json()["key"]

        r = key_mgmt_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {user_key}"},
        )
        assert r.status_code == 200

    def test_revoked_key_denied(self, key_mgmt_client) -> None:
        create_r = key_mgmt_client.post(
            "/v1/keys",
            json={"name": "temp-key"},
            headers={"Authorization": "Bearer master-key-secret"},
        )
        data = create_r.json()
        user_key = data["key"]
        key_id = data["id"]

        key_mgmt_client.delete(
            f"/v1/keys/{key_id}",
            headers={"Authorization": "Bearer master-key-secret"},
        )

        r = key_mgmt_client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {user_key}"},
        )
        assert r.status_code == 403


# ── Streaming Error Recovery ──────────────────────────────────────────────────


class TestStreamingErrorRecovery:
    @staticmethod
    async def _failing_stream(request):
        yield StreamChunk(delta="Starting...", model="gpt-4o-mini")
        raise RuntimeError("Provider connection lost")

    def test_streaming_error_returns_error_json(self, client) -> None:
        c, mock_gw = client
        mock_gw.stream = self._failing_stream
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        assert r.status_code == 200  # SSE stream starts 200 even on error
        assert "stream_error" in r.text or "provider_error" in r.text or "error" in r.text

    def test_streaming_error_does_not_return_500(self, client) -> None:
        c, mock_gw = client
        mock_gw.stream = self._failing_stream
        r = c.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            },
        )
        assert r.status_code != 500


# ── Rate Limiting ─────────────────────────────────────────────────────────────


class TestRateLimiting:
    def test_within_limit_succeeds(self, rate_limited_client) -> None:
        r = rate_limited_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert r.status_code == 200

    def test_exceeding_limit_returns_429(self, rate_limited_client) -> None:
        # Limit is 2/minute; 3rd request should be rate limited
        for _ in range(2):
            rate_limited_client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "test"}],
                },
            )
        r = rate_limited_client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert r.status_code == 429
