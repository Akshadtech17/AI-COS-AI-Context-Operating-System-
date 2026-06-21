"""Tests for the CLI (cli/main.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from aicos.cli.main import app

runner = CliRunner(env={"NO_COLOR": "1"})


def _make_mock_ai(
    remember_id: int = 42,
    forget_result: bool = True,
    search_results: list | None = None,
    chat_response: str = "AI response text",
) -> MagicMock:
    ai = MagicMock()
    ai.aremember = AsyncMock(return_value=remember_id)
    ai.aforget = AsyncMock(return_value=forget_result)
    default_results = [
        {
            "id": 1,
            "content": "Test memory content here",
            "score": 0.95,
            "tags": ["test", "demo"],
            "created_at": "2026-01-01T00:00:00",
        }
    ]
    ai.asearch_memory = AsyncMock(
        return_value=default_results if search_results is None else search_results
    )
    ai.achat = AsyncMock(return_value=chat_response)
    ai.clear_history = MagicMock()
    ai.cost_summary = {"total_tokens": 100, "cost_usd": 0.001, "requests": 1}

    async def _astream(*args, **kwargs):
        yield "Hello"
        yield " World"

    ai.astream = _astream
    return ai


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


# ── start command ─────────────────────────────────────────────────────────────


class TestStartCommand:
    def test_start_calls_uvicorn(self) -> None:
        with patch("uvicorn.run") as mock_run:
            runner.invoke(app, ["start", "--port", "4001"])
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
            assert "AI" in result.output or "gateway" in result.output.lower()

    def test_start_no_providers_warning(self) -> None:
        """Warning panel shown when no API keys are configured."""
        with patch("uvicorn.run"):
            with patch("aicos.core.config.AICOSConfig.available_providers", return_value=[]):
                result = runner.invoke(app, ["start"])
        assert result.exit_code == 0
        assert (
            "Warning" in result.output
            or "warning" in result.output.lower()
            or "Notice" in result.output
        )


# ── remember command ──────────────────────────────────────────────────────────


class TestRememberCommand:
    def test_remember_success(self) -> None:
        mock_ai = _make_mock_ai(remember_id=42)
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["remember", "Important fact"])
        assert result.exit_code == 0
        assert "42" in result.output
        mock_ai.aremember.assert_called_once_with("Important fact", tags=[])

    def test_remember_with_tags(self) -> None:
        mock_ai = _make_mock_ai(remember_id=7)
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["remember", "Tagged fact", "--tags", "ai,research"])
        assert result.exit_code == 0
        assert "7" in result.output
        assert "ai" in result.output or "research" in result.output
        mock_ai.aremember.assert_called_once_with("Tagged fact", tags=["ai", "research"])

    def test_remember_requires_content(self) -> None:
        result = runner.invoke(app, ["remember"])
        assert result.exit_code != 0

    def test_remember_returns_id(self) -> None:
        mock_ai = _make_mock_ai(remember_id=99)
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["remember", "Another fact"])
        assert "99" in result.output


# ── forget command ────────────────────────────────────────────────────────────


class TestForgetCommand:
    def test_forget_success(self) -> None:
        mock_ai = _make_mock_ai(forget_result=True)
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["forget", "5"])
        assert result.exit_code == 0
        assert "5" in result.output
        mock_ai.aforget.assert_called_once_with(5)

    def test_forget_not_found(self) -> None:
        mock_ai = _make_mock_ai(forget_result=False)
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["forget", "999"])
        assert result.exit_code != 0
        assert "999" in result.output

    def test_forget_requires_id(self) -> None:
        result = runner.invoke(app, ["forget"])
        assert result.exit_code != 0

    def test_forget_requires_integer_id(self) -> None:
        result = runner.invoke(app, ["forget", "not-an-int"])
        assert result.exit_code != 0


# ── search command ────────────────────────────────────────────────────────────


class TestSearchCommand:
    def test_search_with_results(self) -> None:
        mock_ai = _make_mock_ai(
            search_results=[
                {
                    "id": 1,
                    "content": "Test memory content here",
                    "score": 0.95,
                    "tags": ["test"],
                    "created_at": "2026-01-01T00:00:00",
                }
            ]
        )
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["search", "test query"])
        assert result.exit_code == 0
        assert "Test memory content" in result.output
        mock_ai.asearch_memory.assert_called_once_with("test query", top_k=5, threshold=0.3)

    def test_search_no_results(self) -> None:
        mock_ai = _make_mock_ai(search_results=[])
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["search", "nothing here"])
        assert result.exit_code == 0
        assert "No memories" in result.output or "no memories" in result.output.lower()

    def test_search_custom_top_k(self) -> None:
        mock_ai = _make_mock_ai(search_results=[])
        with patch("aicos.AI", return_value=mock_ai):
            runner.invoke(app, ["search", "query", "--top-k", "10"])
        mock_ai.asearch_memory.assert_called_once_with("query", top_k=10, threshold=0.3)

    def test_search_custom_threshold(self) -> None:
        mock_ai = _make_mock_ai(search_results=[])
        with patch("aicos.AI", return_value=mock_ai):
            runner.invoke(app, ["search", "query", "--threshold", "0.8"])
        mock_ai.asearch_memory.assert_called_once_with("query", top_k=5, threshold=0.8)

    def test_search_requires_query(self) -> None:
        result = runner.invoke(app, ["search"])
        assert result.exit_code != 0

    def test_search_shows_score(self) -> None:
        mock_ai = _make_mock_ai(
            search_results=[
                {
                    "id": 2,
                    "content": "High score memory",
                    "score": 0.98,
                    "tags": [],
                    "created_at": "2026-01-01T00:00:00",
                }
            ]
        )
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["search", "high"])
        assert "0.98" in result.output or "High score" in result.output


# ── chat command ──────────────────────────────────────────────────────────────


class TestChatCommand:
    def test_chat_no_stream(self) -> None:
        mock_ai = _make_mock_ai(chat_response="Test AI reply")
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["chat", "Hello", "--no-stream"])
        assert result.exit_code == 0
        assert "Test AI reply" in result.output

    def test_chat_with_stream(self) -> None:
        mock_ai = _make_mock_ai()
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["chat", "Hello", "--stream"])
        assert result.exit_code == 0
        assert "Hello" in result.output or "World" in result.output

    def test_chat_shows_stats(self) -> None:
        mock_ai = _make_mock_ai(chat_response="Response")
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["chat", "Hi", "--no-stream", "--stats"])
        assert result.exit_code == 0
        assert "tokens" in result.output or "cost" in result.output

    def test_chat_no_stats(self) -> None:
        mock_ai = _make_mock_ai(chat_response="Response")
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["chat", "Hi", "--no-stream", "--no-stats"])
        assert result.exit_code == 0
        assert "tokens" not in result.output

    def test_chat_with_model(self) -> None:
        mock_ai = _make_mock_ai(chat_response="Model response")
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(app, ["chat", "Hello", "--model", "gpt-4o", "--no-stream"])
        assert result.exit_code == 0
        mock_ai.achat.assert_called_once()
        call_kwargs = mock_ai.achat.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    def test_chat_with_system_prompt(self) -> None:
        mock_ai = _make_mock_ai(chat_response="System response")
        with patch("aicos.AI", return_value=mock_ai):
            result = runner.invoke(
                app, ["chat", "Hello", "--system", "You are helpful", "--no-stream"]
            )
        assert result.exit_code == 0
        call_kwargs = mock_ai.achat.call_args[1]
        assert call_kwargs["system"] == "You are helpful"
