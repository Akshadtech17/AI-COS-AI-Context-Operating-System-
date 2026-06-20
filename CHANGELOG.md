# Changelog

All notable changes to AI-COS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.3.0] — 2026-06-20

### Added
- **Structured logging** — `aicos/core/logging.py`: JSON formatter (via `orjson`), human-readable
  color formatter for dev, `configure_logging()` called at startup, `get_logger()` for namespaced
  loggers. Every request logs gateway entry, cache hit/miss, LLM call, provider failure, and
  request complete with full cost + latency telemetry.
- **Request tracing** — `aicos/api/middleware.py`: `RequestIDMiddleware` stamps every request with
  `X-Request-ID` (accepted from header or generated as UUID4) and propagates it via `ContextVar`
  so all log lines within a request carry the same correlation ID.
- **Per-user API key management** — `aicos/auth/api_keys.py` + `/v1/keys` HTTP API:
  - `POST /v1/keys` (master key required) — creates a SHA-256–hashed key; plaintext shown once.
  - `GET /v1/keys` — lists active keys with prefix, scopes, last-used timestamp.
  - `DELETE /v1/keys/{id}` — revokes a key; immediate effect on next request.
  - Key format: `aicos-{urlsafe}-{hex40}` — prefix stored for safe display.
  - Auth chain: master key → per-user key store → 403. Open access when no key is configured.
- **Persistent cost tracking** — `CostTracker` now accepts `db_path`, creates `cost_records`
  table via SQLAlchemy async, persists every record fire-and-forget (never blocks the request
  path), and restores the last 1 000 records on restart.
- **Deep health check** — `/health` probes each configured provider with `asyncio.wait_for`
  (5-second timeout per provider), reports per-provider status (`ok` / `degraded` / `timeout`),
  and returns `status: "degraded"` if no provider responds.
- **Streaming error recovery** — `_stream_response` wraps the SSE generator in `try/except`
  and emits a structured JSON error event instead of dropping the connection silently.
- `AICOS_LOG_JSON` config field — set `true` for machine-readable JSON logs in production.

### Changed
- Version bumped `0.2.0 → 0.3.0` in `pyproject.toml`, `aicos/__init__.py`, and API `info.version`.
- Gateway logs structured events at `INFO` level throughout the pipeline (routing, cache,
  memory injection, LLM call, provider fallback, request complete).
- `CostTracker.__init__` signature: `db_path: Path | None = None` (backward-compatible —
  defaults to in-memory mode, matching prior behaviour for tests and CLI).

---

## [0.2.0] — 2026-05-15

### Added
- **Multi-provider routing** — ModelRouter with 11 models across OpenAI, Anthropic, Gemini,
  OpenRouter, NVIDIA, and Ollama. Strategies: `auto`, `cheapest`, `fastest`, `best`.
- **EmbeddingTaskClassifier** — zero-shot task classification via prototype centroids;
  auto-detects `sentence-transformers` (all-MiniLM-L6-v2), falls back to character n-gram hashing.
- **StartupAgent** — tool-calling agent with market research, competitive analysis, pricing,
  branding, and financial projection tools.
- **CodingAgent** — tool-calling agent with code generation, review, test generation,
  and architecture planning tools.
- **Live dashboard** — dark-mode web UI at `/` and `/dashboard`, auto-refreshing every 3s,
  showing provider status, cache hit rate, cost, latency, and task breakdown.
- **Docker support** — `Dockerfile`, `docker-compose.yml`, `.dockerignore`.
- **Rate limiting** — `slowapi` integration on `/v1/chat/completions`; returns HTTP 429
  when `AICOS_RATE_LIMIT_RPM` is exceeded.
- **Streaming tests** — SSE content-type, `[DONE]` sentinel, chunk format validated.
- **Memory HTTP API tests** — `POST/GET/DELETE /v1/memory` fully covered.
- **Agent tests** — 46 tests for StartupAgent and CodingAgent (all tools, JSON parsing,
  error handling, OpenAI schema validation).
- `py.typed` marker — package now declares PEP 561 type support.
- `.pre-commit-config.yaml` — ruff lint + format, standard file hygiene hooks.
- CI/CD via GitHub Actions — tests on Python 3.11 and 3.12, ruff lint job.

### Fixed
- `available_providers()` no longer unconditionally includes `"ollama"`.
  Ollama is opt-in via `AICOS_OLLAMA_ENABLED=true`.
- Dashboard JS field names aligned with `metrics.to_dict()` output (added alias
  fields: `by_task_type`, `hit_rate_pct`, `total_stored`, `context.tokens_saved`).
- `EmbeddingEngine` now uses real sentence-transformer embeddings when the library
  is installed, making semantic cache similarity genuinely semantic.
- `lifespan` moved inside `create_app` closure so test config is respected instead
  of always reading from environment.
- `@limiter.limit()` decorator now applied to the chat endpoint (was configured
  but never enforced in v0.1.0).

### Changed
- CLI banner updated to `v0.2.0`.
- `pyproject.toml` version bumped to `0.2.0`.
- Test coverage threshold raised to 70%; actual coverage: 88%.

---

## [0.1.0] — 2026-04-20

### Added
- **OpenAI-compatible gateway** — `POST /v1/chat/completions` with SSE streaming.
- **Semantic cache** — cosine similarity, SQLite backend, configurable threshold.
- **Long-term memory** — `MemoryStore` with composite relevance scoring.
- **Context compression** — extractive + LITM (Lost In The Middle) solver.
- **Analytics** — per-request cost, latency, token tracking via `MetricsStore`.
- **`AI` client** — high-level Python SDK (`ai.chat()`, `ai.remember()`, `ai.search_memory()`).
- **CLI** — `aicos start`, `aicos chat`, `aicos remember`, `aicos forget`, `aicos search`,
  `aicos stats`, `aicos config`.
- **Health and stats endpoints** — `GET /health`, `GET /stats`, `GET /metrics`.
- **Pydantic-settings config** — reads from `.env`, all options overridable via env vars.
- Initial test suite — 73.83% coverage.
