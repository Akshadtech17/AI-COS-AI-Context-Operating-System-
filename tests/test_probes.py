"""Tests for /ready and /live Kubernetes probes."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aicos.api.routes import create_app
from aicos.core.config import AICOSConfig


def _minimal_config(tmp_path) -> AICOSConfig:
    return AICOSConfig(
        openai_api_key="sk-test",
        db_path=str(tmp_path / "aicos.db"),
        rate_limit_enabled=False,
        log_json=False,
    )


def _mock_gateway():
    gw = MagicMock()
    gw.close = AsyncMock()
    gw._providers = {}
    gw._breakers = MagicMock()
    gw._breakers.all_status.return_value = []
    gw._cost_tracker = MagicMock()
    return gw


@pytest.fixture
def probe_client(tmp_path):
    cfg = _minimal_config(tmp_path)
    gw = _mock_gateway()
    ms = MagicMock()
    ms.close = AsyncMock()
    ks = MagicMock()
    ks.close = AsyncMock()
    ks.initialize = AsyncMock()

    async def _fake_build(c):
        return gw, ms

    # run_migrations and build_engine are local-imported inside lifespan;
    # patch them at their source modules so the local 'from … import' binds the mock.
    with (
        patch("aicos.api.routes._build_gateway", side_effect=_fake_build),
        patch("aicos.db.migrations.run_migrations", new_callable=AsyncMock),
        patch("aicos.core.database.build_engine", return_value=MagicMock(dispose=AsyncMock())),
        patch("aicos.api.routes.APIKeyStore", return_value=ks),
    ):
        app = create_app(config=cfg)
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


class TestLivenessProbe:
    def test_live_always_200(self, probe_client):
        resp = probe_client.get("/live")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"
        assert "version" in data

    def test_live_returns_version(self, probe_client):
        resp = probe_client.get("/live")
        assert resp.json()["version"] == "0.5.0"

    def test_live_no_auth_required(self, probe_client):
        resp = probe_client.get("/live")
        assert resp.status_code == 200


class TestReadinessProbe:
    def test_ready_200_when_gateway_initialised(self, probe_client):
        resp = probe_client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    def test_ready_503_when_gateway_none(self, tmp_path):
        """Returns 503 when _gateway is None (startup in progress or failed)."""
        import aicos.api.routes as routes_mod

        original = routes_mod._gateway
        routes_mod._gateway = None
        try:
            cfg = _minimal_config(tmp_path)
            ks = MagicMock()
            ks.close = AsyncMock()
            ks.initialize = AsyncMock()

            async def _hang(c):
                import asyncio
                await asyncio.sleep(9999)

            with (
                patch("aicos.api.routes._build_gateway", side_effect=_hang),
                patch("aicos.db.migrations.run_migrations", new_callable=AsyncMock),
                patch("aicos.core.database.build_engine", return_value=MagicMock(dispose=AsyncMock())),
                patch("aicos.api.routes.APIKeyStore", return_value=ks),
            ):
                app = create_app(config=cfg)
                # TestClient without 'with' doesn't start lifespan → _gateway stays None
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.get("/ready")
                assert resp.status_code == 503
        finally:
            routes_mod._gateway = original

    def test_probe_paths_no_auth_required(self, probe_client):
        resp_live = probe_client.get("/live")
        resp_ready = probe_client.get("/ready")
        assert resp_live.status_code == 200
        assert resp_ready.status_code == 200

    def test_probes_are_fast(self, probe_client):
        """Both probes complete without any external network calls."""
        start = time.perf_counter()
        probe_client.get("/live")
        probe_client.get("/ready")
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"Probes took {elapsed:.2f}s — should be sub-millisecond"

    def test_health_still_exists(self, probe_client):
        """/health deep probe still works alongside the new lightweight probes."""
        resp = probe_client.get("/health")
        assert resp.status_code == 200
        assert "status" in resp.json()
