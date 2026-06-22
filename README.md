# AI-COS: AI Context Operating System — A Production-Grade Middleware Layer for Large Language Model Applications

---

## Abstract

AI-COS (AI Context Operating System) is an open-source, self-hosted middleware framework that sits between application code and large language model (LLM) providers. The system addresses the infrastructure concerns that arise when deploying LLM-powered applications at production scale: model selection, response caching, long-term conversational memory, token-budget management, cost tracking, and fault tolerance. AI-COS implements a multi-stage request pipeline that includes an embedding-based task classifier for intelligent model routing, a two-phase semantic cache using cosine similarity, a composite-scored memory retrieval engine, a TF-IDF context compressor, and a Lost-in-the-Middle (LITM) solver for conversation window management. The system exposes an OpenAI-compatible HTTP gateway (FastAPI + Uvicorn) and a Python SDK, supports six provider families (OpenAI, Anthropic, Google Gemini, NVIDIA, OpenRouter, Ollama), and ships with a production Docker Compose stack including nginx, PgBouncer, PostgreSQL with pgvector, Redis, Prometheus, and Grafana. At version 0.5.0 the test suite comprises 449 tests at 90% code coverage.

---

## I. Introduction

The widespread adoption of LLM APIs has introduced a class of cross-cutting infrastructure concerns that are orthogonal to application business logic but critical to production viability. Applications that call LLM providers directly face several practical challenges:

**Cost and latency unpredictability.** Every LLM call incurs variable token costs. Repeated or semantically near-identical queries sent separately each incur full cost and full round-trip latency (typically 500 ms–5 s), even when a cached response would be adequate.

**Context window limitations.** Models operate within fixed token budgets. Long multi-turn conversations overflow context windows, and naive truncation of earlier turns causes the model to lose critical context. The "Lost-in-the-Middle" phenomenon [6] further degrades recall for information placed in the middle of a long context.

**Provider lock-in and fragility.** Hardcoding a single provider couples application code to one vendor's availability, pricing, and capability profile. Provider outages or model deprecations require emergency code changes.

**Memory discontinuity.** Standard LLM APIs are stateless. Persistent facts — user preferences, project context, prior decisions — must be re-supplied in every prompt by application code, duplicating logic and consuming tokens.

**Operational invisibility.** Without dedicated instrumentation, token consumption, cost accumulation, latency distributions, and cache effectiveness are invisible to operators.

AI-COS resolves all five concerns as a single cohesive middleware layer. Application code issues a `chat()` call or an HTTP request; the system transparently handles routing, caching, memory injection, context optimization, cost recording, and failover before returning a response. The design goal is zero application-layer changes when adding or switching providers, strategies, or backends.

---

## II. Methodology

### A. Overall Architecture

The system is organized as a layered pipeline. Every inbound request passes through a fixed sequence of processing stages before reaching an LLM provider, and a complementary sequence of post-processing stages on the response path.

```
Inbound Request
  → Memory Injection      (top-K relevant long-term memories)
  → Context Compression   (TF-IDF extractive reduction)
  → LITM Solver           (conversation window management)
  → Semantic Cache Lookup (cosine similarity; hit → return immediately)
  → Model Router          (task classification → optimal model selection)
  → Circuit Breaker check (skip OPEN/failed providers)
  → LLM Call + Failover   (retry across provider fallback chain)
  → Cache Store           (async, non-blocking)
  → Cost Record           (async, fire-and-forget)
  → Outbound Response
```

All stages are asynchronous (Python `asyncio`). Cost recording and cache storage are fire-and-forget tasks that never add to response latency.

### B. Semantic Caching

The cache operates in two phases. Phase 1 is an exact-match lookup keyed by a SHA-256 hash of the prompt and model identifier, implemented as an indexed database read (target: < 5 ms). Phase 2, triggered only on exact-miss, computes cosine similarity between the incoming prompt's embedding and the embeddings of recent cache entries (target: < 20 ms, vectorized NumPy). A configurable similarity threshold (default 0.96) determines whether a semantic match is accepted. Cache entries carry per-entry TTL (default 86 400 s). In production, pgvector's IVFFlat approximate nearest-neighbor index replaces the NumPy scan for sub-millisecond retrieval at millions of entries.

### C. Long-Term Memory and Composite Scoring

Memories are stored as text with associated embedding vectors, tags, and metadata. Retrieval uses a composite relevance score:

```
score = cosine_similarity × 0.60
      + recency_decay      × 0.25   (exp(−age_days / 30))
      + access_frequency   × 0.15   (log(1 + count) / log(1 + max_count))
```

The formula balances semantic relevance (60%) with temporal freshness (25%) and reinforced-access frequency (15%). Retrieved memories are injected as a formatted block immediately before the final user message in the conversation history.

### D. Task-Adaptive Model Routing

The router classifies each request into one of seven task types: `simple`, `coding`, `vision`, `reasoning`, `creative`, `analysis`, or `agent`. Classification uses an embedding-based zero-shot classifier with prototype centroids. A regex-based fallback handles unambiguous signals (code syntax markers, vision keywords, agent orchestration vocabulary). The selected task type is combined with a routing strategy (`auto`, `cheapest`, `fastest`, `best`) and a fallback priority chain to choose a specific model from the model registry. The `auto` strategy routes low-complexity requests to free or cheap tiers and escalates complex tasks (reasoning, analysis, agent) to premium tiers.

### E. Context Compression and LITM Solver

The compressor applies extractive summarization to natural-language message content only. Code blocks, JSON objects, and XML are detected by pattern matching and preserved verbatim. Natural language is scored by TF-IDF sentence importance with positional bonuses for first and last sentences, yielding 40–80% token reduction in practice.

The LITM solver triggers when a conversation exceeds `AICOS_LITM_THRESHOLD_TOKENS` (default 6 000 tokens). It discards the middle turns, retains the most recent turns, and inserts an LLM-generated abstractive summary (recap) of the discarded segment between the system message and the retained turns.

### F. Fault Isolation and Rate Limiting

Each provider has an independent circuit breaker implementing the standard CLOSED → OPEN → HALF_OPEN state machine [12]. Consecutive failures beyond a threshold open the circuit; after a recovery timeout the breaker enters HALF_OPEN and allows one probe request. The gateway's failover mechanism skips OPEN providers and retries through the fallback chain.

Rate limiting uses a Redis sorted-set sliding-window algorithm shared across all workers. When Redis is unavailable the system falls back to an in-process list-of-timestamps implementation, maintaining correctness for single-worker deployments.

---

## III. Literature Survey

### A. LLM Proxy and Routing Frameworks

LiteLLM [2] provides a unified Python interface and proxy server for calling 100+ LLM APIs in OpenAI format. AI-COS depends on LiteLLM as a lower-level dependency and extends it with application-level concerns (memory, semantic cache, LITM solver, cost analytics) not addressed by LiteLLM itself. LocalAI and Ollama [10] serve local open-weight models; AI-COS integrates with Ollama-compatible endpoints as a zero-cost provider tier.

### B. Semantic Caching for LLMs

GPTCache [3] demonstrated semantic caching for LLM responses using embedding similarity, reporting significant latency and cost reductions for repeated or near-duplicate queries. AI-COS implements the same two-phase architecture (exact hash → cosine scan) and extends it with per-entry TTL, pgvector ANN indexing, and integration into a broader request pipeline.

### C. Retrieval-Augmented Generation

Lewis et al. [4] introduced Retrieval-Augmented Generation (RAG), combining a dense retrieval index with a generative model to inject relevant documents into prompts. AI-COS applies the same principle to conversational long-term memory: stored facts are retrieved by semantic similarity and injected into the prompt context, enabling persistent user- and project-level awareness across sessions.

### D. Sentence Embeddings

Reimers and Gurevych [5] introduced Sentence-BERT (SBERT), producing semantically meaningful fixed-size sentence embeddings via siamese BERT networks. AI-COS uses the `sentence-transformers` library (SBERT-derived models) when available for high-quality embeddings, and falls back to a deterministic character n-gram hash embedding (512-dimensional) when no model is installed, preserving functionality without requiring a GPU or large model download.

### E. Lost-in-the-Middle Problem

Liu et al. [6] demonstrated empirically that transformer-based language models exhibit significantly degraded recall for information placed in the middle of long contexts, even when that information is within the model's nominal context window. AI-COS's LITM solver directly addresses this finding by restructuring long conversations so that critical context appears at the beginning (as a recap) and end (as recent turns), minimizing the middle region.

### F. Vector Databases and Approximate Nearest-Neighbor Search

pgvector [7] extends PostgreSQL with a native vector type and IVFFlat and HNSW approximate nearest-neighbor indexes. AI-COS uses pgvector for both cache and memory search in production deployments, avoiding the operational overhead of a separate vector database while achieving sub-millisecond retrieval at scale.

### G. OpenAI Compatibility Standard

The OpenAI Chat Completions API [1] has emerged as a de-facto standard interface for LLM access. AI-COS exposes a `/v1/chat/completions` endpoint that is wire-compatible with the OpenAI API, allowing any existing OpenAI client library to target the AI-COS gateway without code changes.

---

## IV. Implementation

### A. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| HTTP Framework | FastAPI 0.115+, Uvicorn 0.32+ |
| Async I/O | Python `asyncio`, `aiohttp`, `aiosqlite` |
| ORM / Database | SQLAlchemy 2.0 (async), aiosqlite (dev), asyncpg (prod) |
| Vector Search | pgvector 0.3+ (PostgreSQL extension) |
| Embeddings | sentence-transformers 3.3+ (optional), hash fallback (built-in) |
| Numerics | NumPy 1.26+, tiktoken 0.8+ |
| LLM Providers | openai 1.55+, anthropic 0.40+, google-generativeai 0.8+, litellm 1.51+ |
| CLI | Typer 0.13+, Rich 13.9+ |
| Config | pydantic-settings 2.6+ |
| Caching | aiosqlite (dev), PostgreSQL + pgvector (prod) |
| Rate Limiting | redis 5.2+, hiredis 3.1+ |
| Retry Logic | tenacity 9.0+ |
| Serialization | orjson 3.10+ |
| Observability | Prometheus (custom collector), OpenTelemetry SDK 1.26+, Sentry SDK 2.16+ |
| Containerization | Docker, Docker Compose (dev + prod profiles) |
| Reverse Proxy | nginx (TLS 1.2/1.3, HSTS, SSE buffering disabled) |
| Connection Pool | PgBouncer (transaction mode) |
| Monitoring | Prometheus 2.x, Grafana (auto-provisioned datasource) |
| Testing | pytest 8.3+, pytest-asyncio, pytest-cov (449 tests, 90% coverage) |
| Linting | Ruff 0.8+, mypy 1.13+ |

### B. Package Structure

The `aicos` Python package is organized into ten subpackages, each with a single responsibility:

- **`aicos.core`** — Public SDK entry point (`AI` class), request pipeline orchestrator (`AIGateway`), model router, circuit breaker, database engine factory, configuration loader, structured logger, and telemetry initializer.
- **`aicos.providers`** — Abstract `BaseProvider` interface and concrete implementations for OpenAI/OpenRouter/NVIDIA (`openai_provider`), Anthropic (`anthropic_provider`), and Google Gemini (`gemini_provider`). Each implementation uses the vendor's official async SDK with tenacity-based exponential backoff (3 attempts).
- **`aicos.cache`** — `SQLiteCache` (SQLAlchemy-backed persistent store) and `SemanticCache` (two-phase lookup coordinator).
- **`aicos.memory`** — `EmbeddingEngine` (sentence-transformers or hash fallback), `MemoryStore` (SQLAlchemy ORM + pgvector ANN path), and `MemoryRetriever` (composite scorer and injector).
- **`aicos.context`** — `ContextCompressor` (TF-IDF extractive, code-preserving), `HistoryManager` (token budget + LITM trigger), and `ConversationSummarizer` (LLM-based abstractive summary).
- **`aicos.analytics`** — `CostTracker` (per-request DB-backed recording) and `MetricsCollector` (Prometheus-compatible counters and latency histograms).
- **`aicos.auth`** — `APIKeyStore` (SHA-256-hashed per-user keys with scopes, shown plaintext exactly once).
- **`aicos.api`** — FastAPI application factory (`create_app`), all HTTP endpoints, `RequestIDMiddleware` (X-Request-ID propagation via ContextVar), and `RateLimitMiddleware` (Redis sliding window with in-process fallback).
- **`aicos.agents`** — `BaseAgent` (ReAct tool-calling loop, max 15 steps), `StartupAgent` (market research, competitive analysis, pricing, branding, financials, GTM strategy), and `CodingAgent` (code generation, review, test generation, architecture planning).
- **`aicos.cli`** — Typer CLI with commands: `start`, `chat`, `remember`, `forget`, `search`, `stats`, `config`.
- **`aicos.db`** — Versioned migration runner with PostgreSQL advisory lock for safe concurrent multi-worker startup.

### C. Model Registry

The router maintains a static registry of 11 model specifications covering all supported providers. Each entry records cost per million input/output tokens, average latency, capability flags (vision, tool-calling, streaming), context window size, and tier label (free/cheap/mid/premium).

| Provider | Model ID | Tier | Input $/1M | Context |
|---|---|---|---|---|
| OpenAI | gpt-4o-mini | cheap | $0.15 | 128k |
| OpenAI | gpt-4o | premium | $2.50 | 128k |
| OpenAI | o1-mini | mid | $3.00 | 128k |
| Anthropic | claude-haiku-4-5 | cheap | $0.25 | 200k |
| Anthropic | claude-sonnet-4-6 | mid | $3.00 | 200k |
| Anthropic | claude-opus-4-8 | premium | $15.00 | 200k |
| Google | gemini-2.0-flash | cheap | $0.10 | 1M |
| Google | gemini-1.5-pro | mid | $1.25 | 2M |
| NVIDIA | llama-3.1-nemotron-ultra-253b | free | $0 | 128k |
| OpenRouter | (via NVIDIA) | free | $0 | 128k |
| Ollama | llama3.2, codellama | local | $0 | 128k |

### D. HTTP API

The FastAPI gateway is OpenAI-compatible. All endpoints:

| Method | Path | Function |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions (streaming + non-streaming, SSE) |
| `GET` | `/v1/models` | List available models |
| `POST` | `/v1/memory` | Store a long-term memory |
| `GET` | `/v1/memory/search` | Semantic memory search |
| `DELETE` | `/v1/memory/{id}` | Delete memory by ID |
| `POST` | `/v1/keys` | Create per-user API key (master key required) |
| `GET` | `/v1/keys` | List active keys |
| `DELETE` | `/v1/keys/{id}` | Revoke a key |
| `GET` | `/health` | Deep health check (probes each provider) |
| `GET` | `/ready` | Kubernetes readiness probe |
| `GET` | `/live` | Kubernetes liveness probe |
| `GET` | `/metrics` | Prometheus metrics export |
| `GET` | `/stats` | JSON cost and performance summary |
| `GET` | `/dashboard` | Web UI (dark-mode, live metrics) |

### E. Production Deployment

The production Docker Compose stack (`docker-compose.prod.yml`) starts seven services: nginx (TLS termination, HTTP→HTTPS redirect, TLS 1.2/1.3, HSTS), four AI-COS Uvicorn workers (4 workers each), PgBouncer (transaction-mode connection pooling, reducing 160 app connections to ~20 PostgreSQL connections), PostgreSQL 16 with the pgvector extension, Redis 7 (distributed rate limiting), Prometheus (metrics scraping from `/metrics`), and Grafana (auto-provisioned with Prometheus datasource). Kubernetes deployments use `/ready` as `readinessProbe` and `/live` as `livenessProbe`. API keys may be supplied as Docker Swarm/Compose secrets (read from `/run/secrets/`); environment variables always take precedence.

### F. Database and Migrations

All persistent state (cache entries, memory items, cost records, API keys) is stored in SQLAlchemy ORM models. A versioned migration runner (`aicos.db.migrations`) tracks applied migrations in a `schema_migrations` table and uses a PostgreSQL advisory lock (lock ID `8_888_999`) to serialize concurrent worker startup. Migration `m001` creates all base tables. Migration `m002` installs the pgvector extension and creates an IVFFlat cosine-distance index on the memory embedding column. SQLite uses WAL mode with `synchronous=NORMAL` and a 5-second busy timeout.

---

## V. Conclusion

AI-COS demonstrates that the cross-cutting infrastructure concerns of LLM application development — model routing, semantic caching, persistent memory, context window management, cost tracking, fault isolation, and observability — can be encapsulated in a single cohesive middleware layer. The layered pipeline design allows each concern to be developed, tested, and configured independently while composing cleanly at runtime. The OpenAI-compatible HTTP interface ensures that the system integrates with any existing LLM client library without code changes. The dual SQLite/PostgreSQL backend architecture provides a zero-configuration path for development and a horizontally scalable path for production with no application-level changes required. At version 0.5.0, the system achieves 90% test coverage across 449 tests, supports six provider families covering free through premium model tiers, and ships with a complete production infrastructure stack.

---

## VI. Future Scope

The following enhancements are planned or under investigation based on the current architecture:

**Streaming memory updates.** The current memory injection is computed before the LLM call. Future work will explore updating memory in real time as streaming tokens arrive, enabling higher-frequency memory capture without blocking response delivery.

**Multi-modal caching.** The semantic cache currently operates on text. Extending cache keys and similarity computation to cover image, audio, and document inputs would support multi-modal models (GPT-4o vision, Gemini 1.5 Pro) with equivalent cache efficiency.

**Adaptive compression threshold.** The TF-IDF compression ratio is currently fixed. A learned policy that adjusts aggressiveness based on observed model performance degradation would optimize the cost-quality trade-off per task type.

**HNSW indexing.** pgvector supports Hierarchical Navigable Small World (HNSW) indexing in addition to IVFFlat. Migrating the memory and cache indexes to HNSW would improve recall at high query rates and large dataset sizes.

**Agent orchestration layer.** The current `BaseAgent` implements a single-agent ReAct loop. A multi-agent orchestration layer would enable task decomposition, parallel sub-agent execution, and inter-agent communication, which is required for complex long-horizon tasks.

**Fine-tuned task classifier.** The current embedding classifier uses zero-shot prototype centroids. Training a lightweight fine-tuned classifier on labeled LLM task data would improve routing accuracy, particularly for ambiguous prompts at task-type boundaries.

**Cost prediction before routing.** Adding a pre-call cost estimate to the routing decision would allow hard per-request and per-session cost caps, improving budget predictability for multi-tenant deployments.

**Streaming rate-limit accounting.** The current rate limiter counts requests. Counting consumed tokens per minute (TPM) in addition to requests per minute (RPM) would allow finer-grained budget enforcement aligned with provider billing models.

---

## References

[1] OpenAI, "OpenAI API Reference — Chat Completions," OpenAI Platform Documentation, 2024. [Online]. Available: https://platform.openai.com/docs/api-reference/chat

[2] BerriAI, "LiteLLM: Call all LLM APIs using the OpenAI format," GitHub Repository, 2024. [Online]. Available: https://github.com/BerriAI/litellm

[3] Zhuang, S. et al., "GPTCache: A Data Store for Efficient LLM Responses," in *Proc. ACL 2023 Workshop on Retrieval-Augmented Models*, 2023.

[4] Lewis, P. et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 9459–9474, 2020.

[5] Reimers, N. and Gurevych, I., "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," in *Proc. Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 3982–3992, 2019.

[6] Liu, N. F. et al., "Lost in the Middle: How Language Models Use Long Contexts," *Transactions of the Association for Computational Linguistics*, vol. 12, pp. 157–173, 2024.

[7] pgvector Contributors, "pgvector: Open-source vector similarity search for Postgres," GitHub Repository, 2023. [Online]. Available: https://github.com/pgvector/pgvector

[8] Ramírez, S., "FastAPI: Modern, fast web framework for building APIs with Python," GitHub Repository, 2024. [Online]. Available: https://github.com/tiangolo/fastapi

[9] SQLAlchemy Contributors, "SQLAlchemy — The Database Toolkit for Python," Version 2.0, 2024. [Online]. Available: https://www.sqlalchemy.org/

[10] Jeffery Morgan, "Ollama: Get up and running with large language models locally," GitHub Repository, 2024. [Online]. Available: https://github.com/ollama/ollama

[11] Prometheus Authors, "Prometheus: Monitoring System and Time Series Database," Cloud Native Computing Foundation, 2024. [Online]. Available: https://prometheus.io/

[12] M. Fowler, "CircuitBreaker," martinfowler.com, 2014. [Online]. Available: https://martinfowler.com/bliki/CircuitBreaker.html

[13] Longa, F., "Tenacity: Retrying library for Python," GitHub Repository, 2024. [Online]. Available: https://github.com/jd/tenacity

[14] Anthropic, "Anthropic API Reference," Anthropic Documentation, 2024. [Online]. Available: https://docs.anthropic.com/

[15] Google, "Gemini API Reference," Google AI for Developers Documentation, 2024. [Online]. Available: https://ai.google.dev/api/

[16] OpenRouter, "OpenRouter: A unified interface for LLMs," OpenRouter Documentation, 2024. [Online]. Available: https://openrouter.ai/docs

[17] Bird, S., Klein, E. and Loper, E., *Natural Language Processing with Python*, O'Reilly Media, 2009. (TF-IDF foundations used in context compressor.)

[18] Devlin, J. et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," in *Proc. NAACL-HLT*, pp. 4171–4186, 2019. (Underlying architecture of sentence-transformers embeddings.)
