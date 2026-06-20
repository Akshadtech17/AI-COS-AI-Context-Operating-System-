# Changelog

All notable changes to AI-COS are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
