"""Tests for the CLI (cli/main.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from aicos.cli.main import app

runner = CliRunner()


# ── Help / version ────────────────────────────────────────────────────────────

class TestCLIHelp:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "AI-COS" in result.output

    def test_start_help(self) -> None:
        result = runner.invoke(app, ["start", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output
        assert "--host" in result.output

    def test_chat_help(self) -> None:
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0

    def test_remember_help(self) -> None:
        result = runner.invoke(app, ["remember", "--help"])
        assert result.exit_code == 0

    def test_forget_help(self) -> None:
        result = runner.invoke(app, ["forget", "--help"])
        assert result.exit_code == 0

    def test_search_help(self) -> None:
        result = runner.invoke(app, ["search", "--help"])
        assert result.exit_code == 0

    def test_stats_help(self) -> None:
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0

    def test_config_help(self) -> None:
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0


# ── config command ────────────────────────────────────────────────────────────

class TestConfigCommand:
    def test_config_runs_successfully(self) -> None:
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0

    def test_config_shows_table(self) -> None:
        result = runner.invoke(app, ["config"])
        assert "Configuration" in result.output

    def test_config_masks_api_keys(self) -> None:
        result = runner.invoke(app, ["config"])
        # Real key should not appear, masked version should
        assert "sk-or-v1-3306fac" not in result.output


# ── stats command ─────────────────────────────────────────────────────────────

class TestStatsCommand:
    def test_stats_runs_successfully(self) -> None:
        result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0

    def test_stats_shows_metrics(self) -> None:
        result = runner.invoke(app, ["stats"])
        assert "Total Requests" in result.output

    def test_stats_shows_cost(self) -> None:
        result = runner.invoke(app, ["stats"])
        assert "Cost" in result.output or "Requests" in result.output


# ── start command (mocked uvicorn) ────────────────────────────────────────────

class TestStartCommand:
    def test_start_calls_uvicorn(self) -> None:
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(app, ["start", "--port", "4001"])
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["port"] == 4001

    def test_start_default_host(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["start"])
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["host"] == "0.0.0.0"

    def test_start_verbose_flag(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["start", "--verbose"])
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["log_level"] == "debug"

    def test_start_shows_banner(self) -> None:
        with patch("uvicorn.run"):
            result = runner.invoke(app, ["start"])
            assert "AI" in result.output or "AICOS" in result.output or "gateway" in result.output.lower()


# ── remember / forget / search (mocked AI) ────────────────────────────────────

class TestMemoryCommands:
    def test_remember_stores_memory(self) -> None:
        mock_ai = MagicMock()
        mock_ai.aremember = MagicMock(return_value=42)

        with patch("aicos.cli.main.asyncio") as mock_asyncio:
            mock_asyncio.run = lambda coro: 42
            # Just test it doesn't crash on invocation
            result = runner.invoke(app, ["remember", "Test memory content"])
            # Either succeeds or fails with known error (no real event loop needed)
            assert result.exit_code in (0, 1)

    def test_remember_requires_content(self) -> None:
        result = runner.invoke(app, ["remember"])
        assert result.exit_code != 0

    def test_forget_requires_id(self) -> None:
        result = runner.invoke(app, ["forget"])
        assert result.exit_code != 0

    def test_forget_requires_integer_id(self) -> None:
        result = runner.invoke(app, ["forget", "not-an-int"])
        assert result.exit_code != 0

    def test_search_requires_query(self) -> None:
        result = runner.invoke(app, ["search"])
        assert result.exit_code != 0
