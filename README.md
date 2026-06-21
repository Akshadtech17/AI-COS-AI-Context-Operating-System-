# AI-COS — AI Context Operating System

> **The production-grade middleware layer between your application and any LLM.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/Akshadtech17/AI-COS-AI-Context-Operating-System-/actions/workflows/ci.yml/badge.svg)](https://github.com/Akshadtech17/AI-COS-AI-Context-Operating-System-/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-449%20passing-brightgreen.svg)](https://github.com/Akshadtech17/AI-COS-AI-Context-Operating-System-/actions)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen.svg)](https://github.com/Akshadtech17/AI-COS-AI-Context-Operating-System-/actions)
[![Version](https://img.shields.io/badge/version-0.5.0-blue.svg)](https://github.com/Akshadtech17/AI-COS-AI-Context-Operating-System-/blob/master/CHANGELOG.md)

AI-COS is a self-hosted AI infrastructure layer that sits between your applications and LLM providers. It handles model routing, semantic caching, long-term memory, context compression, cost tracking, and production observability — so your application code stays clean.

```python
from aicos import AI

ai = AI()
response = ai.chat("Build a SaaS startup")
# Routing, caching, memory, compression, cost tracking — all automatic.
```

---

## Quick Start

**You need one API key from any provider** — OpenRouter is the easiest (100+ models, free tier available).

### Option 1 — pip install

```bash
pip install aicos
```

Create a `.env` file with at least one API key:

```env
# Pick any one (or more):
OPENROUTER_API_KEY=sk-or-...      # 100+ models — openrouter.ai
OPENAI_API_KEY=sk-...             # openai.com
ANTHROPIC_API_KEY=sk-ant-...      # anthropic.com
GEMINI_API_KEY=...                # aistudio.google.com
```

Start the gateway:

```bash
aicos start
```

Open **http://localhost:4000/dashboard** — live dashboard with metrics, cost, and provider status.

### Option 2 — clone & run

```bash
git clone https://github.com/Akshadtech17/AI-COS-AI-Context-Operating-System-
cd AI-COS-AI-Context-Operating-System-
pip install -e ".[dev]"
cp .env.example .env   # fill in your API key(s)
aicos start
```

### Option 3 — Docker (zero Python setup)

```bash
git clone https://github.com/Akshadtech17/AI-COS-AI-Context-Operating-System-
cd AI-COS-AI-Context-Operating-System-
cp .env.example .env   # fill in your API key(s)
docker compose up
```

### Send your first request

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Or use the Python SDK:

```python
from aicos import AI

ai = AI()
print(ai.chat("Hello!"))

# Remember facts across conversations
ai.remember("User is building a SaaS product in Python")
print(ai.chat("What stack should I use?"))  # context is automatically injected

# Session stats
print(ai.cost_summary)   # {'total_cost_usd': 0.0001, 'total_tokens': 240}
print(ai.metrics)        # {'cache_hit_rate': 0.33, 'avg_latency_ms': 280}
```

Or point any OpenAI-compatible client at it:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:4000/v1", api_key="none")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Features

| Feature | Description |
|---|---|
| **Smart Model Router** | Embedding-based task classifier routes to the optimal model per request type (code, vision, reasoning, analysis, agent). Four strategies: `auto`, `cheapest`, `fastest`, `best`. |
| **Semantic Cache** | Cosine similarity cache — near-identical queries served in < 20 ms. SQLite (dev) or PostgreSQL + pgvector (prod) backend. |
| **Long-Term Memory** | `remember()` / `forget()` / `search_memory()` with composite relevance scoring. pgvector ANN index in production for sub-millisecond retrieval at any scale. |
| **Context Compression** | 40–80% token reduction preserving code blocks, JSON, and key facts via TF-IDF extractive summarisation. |
| **LITM Solver** | Lost-in-the-Middle fix: compresses conversation middle, keeps first + last turns, generates a recap. |
| **OpenAI-Compatible Gateway** | Drop-in `/v1/chat/completions` with SSE streaming, provider failover, and structured error events. |
| **Per-User API Keys** | SHA-256-hashed keys with scopes. Create, list, revoke via HTTP API. Master key required for management. |
| **Circuit Breaker** | Per-provider CLOSED → OPEN → HALF_OPEN state machine. Status visible in `/health`. |
| **Distributed Rate Limiting** | Redis sliding-window rate limiter shared across all workers. In-process fallback when Redis is absent. |
| **Persistent Cost Tracking** | Per-request cost, token, and latency records. Survives restarts. Prometheus metrics + JSON stats. |
| **Structured Logging** | JSON logs in production, colour text in dev. Every request stamped with `X-Request-ID` for correlation. |
| **Schema Migrations** | Versioned migration runner with PostgreSQL advisory lock — safe with multi-worker concurrent startup. |
| **TLS via nginx** | HTTP → HTTPS redirect, TLS 1.2/1.3, HSTS, SSE buffering disabled. Drop-in for Let's Encrypt. |
| **Docker Secrets** | API keys loaded from `/run/secrets/` (Docker Swarm / Compose secrets). Env vars always take precedence. |
| **PgBouncer Connection Pool** | Transaction-mode pooling reduces PostgreSQL connections from 160 to ~20 under a 4-worker deployment. |
| **pgvector ANN Search** | Native cosine-distance search via IVFFlat index. Scales memory and cache search to millions of entries. |
| **Kubernetes Probes** | `GET /ready` (fast, 503 during startup) and `GET /live` (always 200) — correct separation of concerns. |
| **Observability Stack** | Prometheus + Grafana pre-configured. Optional OpenTelemetry (OTLP gRPC) and Sentry integrations. |
| **Multi-Provider** | OpenAI, Anthropic, Gemini, NVIDIA Nemotron, OpenRouter (100+ models), Ollama. |
| **Agent Framework** | Tool-calling agents with `StartupAgent` (market research, pricing, branding) and `CodingAgent` (code gen, review, architecture). |
| **Web Dashboard** | Live dark-mode UI at `http://localhost:4000` — provider status, cache hit rate, cost, latency, task breakdown. |

---

## Architecture

```
Your Application
       │
       ▼
┌────────────────────────────────────────────────────────────────┐
│                     nginx (TLS termination)                     │
│                HTTP → HTTPS, TLS 1.2/1.3, HSTS                │
└───────────────────────────┬────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │   AI-COS Gateway (×4)      │   ← uvicorn workers
              │                           │
              │  RequestIDMiddleware       │   X-Request-ID tracing
              │  RateLimitMiddleware       │   Redis sliding window
              │                           │
              │  Memory Injection          │   top-K relevant memories
              │  Context Compression       │   40–80% token reduction
              │  LITM Solver              │   lost-in-the-middle fix
              │  Semantic Cache Lookup     │   cosine similarity < 20 ms
              │                           │
              │  Model Router              │   task → model → provider
              │  Circuit Breaker           │   per-provider fault isolation
              │                           │
              │  LLM Call + Failover       │
              │  Cost Tracking             │
              │  Cache Store               │
              └──┬──────────────────────┬─┘
                 │                      │
    ┌────────────┼──────────────┐       │
    ▼            ▼              ▼       ▼
┌────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
│OpenAI  │ │Anthropic │ │Gemini  │ │ NVIDIA / │
│        │ │          │ │        │ │OpenRouter│
└────────┘ └──────────┘ └────────┘ └──────────┘

         ┌─────────────────────────────────┐
         │        Data Layer               │
         │                                 │
         │  PgBouncer ← aicos workers      │
         │      └── PostgreSQL+pgvector    │  memories, cache, cost, keys
         │  Redis                          │  distributed rate limiting
         │                                 │
         │  Prometheus ← /metrics          │
         │  Grafana    ← Prometheus        │
         └─────────────────────────────────┘
```

### Request Pipeline

Every request flows in order:

```
Request
  → Memory Injection      (top-K memories from pgvector ANN index)
  → Context Compression   (TF-IDF extractive, LITM solver)
  → Semantic Cache Lookup (cosine similarity; cache hit → return in < 20 ms)
  → Model Router          (task classification → optimal model)
  → Circuit Breaker check (skip OPEN providers)
  → LLM Call + Failover   (retry across fallback chain)
  → Cache Store           (async, never blocks response)
  → Cost Record           (async, fire-and-forget)
  → Response
```

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env   # add at least one provider API key
docker compose up
```

- Dashboard: **http://localhost:4000**
- API: **http://localhost:4000/v1/chat/completions**

### pip

```bash
pip install "aicos[all]"
cp .env.example .env
aicos start
```

---

## Configuration

All settings are environment variables. Copy `.env.example` to `.env` and fill in your keys.

### Provider Keys

```env
# At least one required — use any combination
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...     # access 100+ models including free NVIDIA Nemotron
NVIDIA_API_KEY=nvapi-...         # direct NVIDIA API
```

### Core Settings

```env
# Router
AICOS_ROUTER_STRATEGY=auto        # auto | cheapest | fastest | best
AICOS_DEFAULT_MODEL=              # override auto-routing (e.g. gpt-4o)

# Cache
AICOS_CACHE_ENABLED=true
AICOS_CACHE_SIMILARITY_THRESHOLD=0.96   # 0.0–1.0, higher = stricter match
AICOS_CACHE_MAX_SIZE=10000
AICOS_CACHE_TTL_SECONDS=86400

# Memory
AICOS_MEMORY_ENABLED=true
AICOS_MEMORY_MAX_ITEMS=10000
AICOS_MEMORY_INJECTION_LIMIT=5         # memories injected per request
AICOS_MEMORY_RELEVANCE_THRESHOLD=0.3

# Context
AICOS_CONTEXT_COMPRESSION_ENABLED=true
AICOS_MAX_CONTEXT_TOKENS=8000
AICOS_LITM_THRESHOLD_TOKENS=6000

# Gateway
AICOS_GATEWAY_HOST=0.0.0.0
AICOS_GATEWAY_PORT=4000
AICOS_GATEWAY_API_KEY=                 # optional — leave unset for open access

# Logging
AICOS_LOG_LEVEL=INFO                   # DEBUG | INFO | WARNING | ERROR
AICOS_LOG_JSON=false                   # set true for structured JSON logs in prod

# Rate limiting
AICOS_RATE_LIMIT_ENABLED=true
AICOS_RATE_LIMIT_RPM=60

# Circuit breaker
AICOS_CIRCUIT_BREAKER_FAILURE_THRESHOLD=5    # consecutive failures to open
AICOS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT=30.0  # seconds before HALF_OPEN probe
```

### Database

```env
# Default: SQLite (WAL mode, separate file per component)
# Production: PostgreSQL + pgvector
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/aicos

# Connection pool (reduce when behind PgBouncer in transaction mode)
AICOS_DB_POOL_SIZE=5        # per-worker, per-store (default: 5)
AICOS_DB_MAX_OVERFLOW=5     # (default: 5; use 2/3 with PgBouncer)

# Embedding dimension — must match the embedding model
AICOS_EMBEDDING_DIM=384     # 384 for all-MiniLM-L6-v2, 512 for hash fallback
```

### Redis, Secrets & Observability

```env
# Redis — enables distributed rate limiting across workers
AICOS_REDIS_URL=redis://localhost:6379/0

# Docker secrets directory (one file per secret, named like env var)
AICOS_SECRETS_DIR=/run/secrets

# OpenTelemetry tracing (requires pip install 'aicos[otel]')
AICOS_OTEL_ENDPOINT=http://jaeger:4317

# Sentry error tracking (requires pip install 'aicos[sentry]')
AICOS_SENTRY_DSN=https://...@sentry.io/...
```

---

## Python SDK

```python
from aicos import AI

ai = AI()

# ── Chat ──────────────────────────────────────────────────────────────────────
response = ai.chat("What is the capital of France?")
print(response)  # "The capital of France is Paris."

# With system prompt
response = ai.chat(
    "What tech stack should I use?",
    system="You are a senior software architect.",
)

# Streaming
async for token in ai.astream("Write a haiku about distributed systems"):
    print(token, end="", flush=True)

# ── Memory ────────────────────────────────────────────────────────────────────
# Store memories (automatically injected into future chats)
ai.remember("User is building a B2B SaaS product", tags=["context", "project"])
ai.remember("User prefers Python over JavaScript", tags=["preferences"])
ai.remember("Team is 3 engineers, shipping in Q3", tags=["context"])

# Memories are automatically retrieved and injected into relevant future requests
response = ai.chat("What database should we use?")  # sees the project context

# Search memories
results = ai.search_memory("Python preferences", top_k=3)
for r in results:
    print(f"[{r['score']:.3f}] {r['content']}")

# Forget by ID
ai.forget(42)

# ── Model selection ───────────────────────────────────────────────────────────
# Override auto-routing
response = ai.chat("Review this code", model="claude-sonnet-4-6")

# ── Session stats ─────────────────────────────────────────────────────────────
print(ai.cost_summary)   # {'total_cost_usd': 0.0012, 'total_tokens': 840, ...}
print(ai.metrics)        # {'cache_hit_rate': 0.34, 'avg_latency_ms': 312, ...}

# ── Async usage ───────────────────────────────────────────────────────────────
import asyncio

async def main():
    ai = AI()
    response = await ai.achat("Hello, world!")
    memory_id = await ai.aremember("User said hello", tags=["session"])
    results = await ai.asearch_memory("greeting")

asyncio.run(main())
```

---

## OpenAI Drop-In

Point any OpenAI-compatible client at `http://localhost:4000`:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="your-aicos-key",  # or omit if no key configured
)

response = client.chat.completions.create(
    model="auto",   # AI-COS selects the optimal model automatically
    messages=[{"role": "user", "content": "Explain async/await in Python"}],
)
print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Write a sorting algorithm"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

```bash
# curl
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-aicos-key" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

Response includes AI-COS metadata:
```json
{
  "choices": [...],
  "aicos": {
    "cache_hit": false,
    "task_type": "simple",
    "tokens_before_compression": 12,
    "tokens_after_compression": 12,
    "compression_savings_pct": 0.0,
    "memories_injected": 2,
    "cost_usd": 0.0000015,
    "latency_ms": 312.4,
    "routing": "Strategy=auto, task=simple, tier=free, cost=$0.00/1M"
  }
}
```

---

## HTTP API Reference

### Chat

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming) |
| `GET` | `/v1/models` | List available models and providers |

### Memory

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/memory` | Store a memory |
| `GET` | `/v1/memory/search?query=...&top_k=5` | Semantic memory search |
| `DELETE` | `/v1/memory/{id}` | Delete a memory by ID |

### API Key Management (master key required)

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/keys` | Create a new per-user API key (shown once, plaintext never stored) |
| `GET` | `/v1/keys` | List all active keys (prefix + scopes + last-used timestamp) |
| `DELETE` | `/v1/keys/{id}` | Revoke a key immediately |

```bash
# Create a key
curl -X POST http://localhost:4000/v1/keys \
  -H "Authorization: Bearer $MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "production-app", "scopes": ["chat", "memory"]}'

# {"key": "aicos-abc123-...", "id": 1, "name": "production-app", ...}
# Save the key — it is shown exactly once.
```

### Observability

| Method | Path | Description |
|---|---|---|
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/stats` | JSON stats overview |
| `GET` | `/health` | Deep health check — probes each provider (~5 s) |
| `GET` | `/ready` | Kubernetes readiness probe — fast, 503 during startup |
| `GET` | `/live` | Kubernetes liveness probe — always 200 if process is running |

---

## Model Router

The router classifies each request's task type using an embedding-based classifier with regex fallback, then selects the optimal model based on strategy.

### Supported Models

| Provider | Model | Tier | Cost (in/out per 1M) | Context |
|---|---|---|---|---|
| **OpenAI** | gpt-4o-mini | cheap | $0.15 / $0.60 | 128k |
| **OpenAI** | gpt-4o | premium | $2.50 / $10.00 | 128k |
| **OpenAI** | o1-mini | mid | $3.00 / $12.00 | 128k |
| **Anthropic** | claude-haiku-4-5 | cheap | $0.25 / $1.25 | 200k |
| **Anthropic** | claude-sonnet-4-6 | mid | $3.00 / $15.00 | 200k |
| **Anthropic** | claude-opus-4-8 | premium | $15.00 / $75.00 | 200k |
| **Google** | gemini-2.0-flash | cheap | $0.10 / $0.40 | 1M |
| **Google** | gemini-1.5-pro | mid | $1.25 / $5.00 | 2M |
| **NVIDIA** | llama-3.1-nemotron-ultra-253b | **free** | $0 / $0 | 128k |
| **OpenRouter** | (via NVIDIA) | **free** | $0 / $0 | 128k |
| **Ollama** | llama3.2, codellama | **local** | $0 / $0 | 128k |

NVIDIA Nemotron Ultra is preferred by default — it's free, capable, and handles all task types.

### Task Classification

| Task Type | Detection Signals | Default Model Tier |
|---|---|---|
| `simple` | Short prompt, conversational | free → cheap |
| `coding` | `def`, `class`, code blocks, `import` | free → mid |
| `vision` | "image", "photo", "describe", "screenshot" | free → mid |
| `reasoning` | "analyze", "compare", "trade-off", "prove" | free → premium |
| `creative` | "write", "story", "poem", "creative" | free → mid |
| `analysis` | "market research", "financial", "strategy" | free → premium |
| `agent` | "orchestrate", "automate", "workflow", "plan" | free → premium |

### Router Strategies

```python
from aicos.core.config import AICOSConfig
from aicos import AI

# Cheapest model that can handle the task
ai = AI(config=AICOSConfig(router_strategy="cheapest"))

# Lowest average latency
ai = AI(config=AICOSConfig(router_strategy="fastest"))

# Highest-tier model regardless of cost
ai = AI(config=AICOSConfig(router_strategy="best"))

# Task-adaptive (default) — free/cheap for simple, premium for complex
ai = AI(config=AICOSConfig(router_strategy="auto"))
```

---

## Memory System

Memories are scored with a composite formula that balances semantic relevance, recency, and frequency:

```
score = cosine_similarity × 0.60
      + recency_decay      × 0.25     (exp(-age_days / 30))
      + access_frequency   × 0.15     (log(1 + count) / log(1 + max_count))
```

### Storage Backends

| Backend | When used | Search |
|---|---|---|
| SQLite + JSON | Dev / single-server | NumPy cosine scan (< 10k rows) |
| PostgreSQL + pgvector | Production | IVFFlat ANN index — sub-ms at any scale |

pgvector is enabled automatically when migration `002_add_pgvector` detects the PostgreSQL vector extension. Old rows remain searchable via the JSON path; new rows populate both columns.

```python
# Store with tags and metadata
ai.remember(
    "The payment service uses Stripe with SCA enabled",
    tags=["architecture", "payments"],
    metadata={"source": "design-doc", "version": 2},
)

# Semantic search
results = ai.search_memory("payment processing", top_k=5, threshold=0.3)

# Forget by ID
ai.forget(memory_id)
```

---

## Context Optimization

### Compression

Applied to natural language only — code and structured data are never modified:

| Content Type | Treatment |
|---|---|
| System messages | **Never modified** |
| Code blocks (` ``` `) | **Preserved verbatim** |
| JSON objects | **Preserved verbatim** |
| XML / HTML | **Preserved verbatim** |
| Natural language | **Compressed aggressively** — 40–80% reduction |

Algorithm: TF-IDF sentence scoring + positional bonuses (first/last sentences preserved).

### LITM Solver

When conversation exceeds `AICOS_LITM_THRESHOLD_TOKENS` (default: 6000):

```
Before: [System] [Turn 1] [Turn 2] [Turn 3] [Turn 4] [Turn 5] [Turn 6]
                          ← "lost in the middle" — model ignores these →

After:  [System] [Context Recap] [Turn 5] [Turn 6]
                       ↑
              LLM-generated summary of Turns 1-4
```

---

## Agent Framework

### StartupAgent

Full startup analysis with structured JSON output:

```python
from aicos import AI
from aicos.agents import StartupAgent

ai = AI()
await ai._ensure_initialized()

agent = StartupAgent(gateway=ai._gateway)
result = await agent.run("Build an AI-powered legal document analyzer for SMBs")

print(result.structured)
# {
#   "startup_name": "LegalLens AI",
#   "tagline": "Contract intelligence for growing businesses",
#   "market": {"tam_usd": 12_000_000_000, "yoy_growth_pct": 23, ...},
#   "competitors": [{"name": "ContractPodAi", "weakness": "enterprise-only"}, ...],
#   "pricing": {"tiers": [{"name": "Starter", "price_usd_monthly": 49}, ...]},
#   "branding": {"primary_color": "#1a1a2e", "tone": "professional"},
#   "financials": {"break_even_months": 14, "mrr_12m": 85_000}
# }
```

### CodingAgent

Software engineering agent with code generation, review, testing, and architecture tools:

```python
from aicos.agents import CodingAgent

agent = CodingAgent(gateway=ai._gateway)
result = await agent.run_code_task(
    "Build a rate limiter class with sliding window algorithm",
    language="python",
)
print(result.structured["code"])
print(result.structured["tests"])
print(result.structured["review"])
```

---

## CLI

```bash
# Start the gateway
aicos start
aicos start --port 4000 --host 0.0.0.0

# Interactive chat (conversation history maintained)
aicos chat
aicos chat "What is machine learning?" --model gpt-4o

# Memory management
aicos remember "I prefer type hints in Python" --tags "coding"
aicos search "Python preferences" --top-k 5
aicos forget 42

# Statistics
aicos stats

# Show resolved config
aicos config
```

---

## Production Deployment

### Single-server (SQLite, 1 worker)

```bash
cp .env.example .env
docker compose up -d
```

Suitable for: internal tools, side projects, single-team usage.

### Multi-worker (PostgreSQL + PgBouncer + Redis)

```bash
cp .env.example .env
# Set POSTGRES_PASSWORD, GRAFANA_PASSWORD, and all API keys in .env
docker compose -f docker-compose.prod.yml up -d
```

This starts:
- **nginx** — TLS termination on 80/443
- **4 AI-COS workers** — behind nginx
- **PgBouncer** — connection pooler in transaction mode
- **PostgreSQL 16 + pgvector** — shared persistence for all components
- **Redis 7** — distributed rate limiting
- **Prometheus** — metrics scraping at `:9090`
- **Grafana** — dashboards at `:3000` (auto-provisioned with Prometheus datasource)

### TLS / Let's Encrypt

Edit `nginx/nginx.conf` to add your domain and certificate paths:

```nginx
ssl_certificate     /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
```

Then uncomment the cert volume mount in `docker-compose.prod.yml`.

### Kubernetes

Use `/ready` as `readinessProbe` and `/live` as `livenessProbe`:

```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 4000
  initialDelaySeconds: 10
  periodSeconds: 5
  failureThreshold: 3

livenessProbe:
  httpGet:
    path: /live
    port: 4000
  initialDelaySeconds: 5
  periodSeconds: 10
```

### Docker Secrets

Mount secrets as files instead of env vars (Swarm / Compose secrets):

```yaml
secrets:
  OPENAI_API_KEY:
    file: ./secrets/openai_key.txt
  AICOS_GATEWAY_API_KEY:
    file: ./secrets/gateway_key.txt
```

AI-COS reads them from `/run/secrets/` automatically. Env vars always take precedence over secret files.

### Connection Pool Tuning (PgBouncer)

With PgBouncer in transaction mode, reduce pool sizes to avoid PostgreSQL connection exhaustion:

```env
AICOS_DB_POOL_SIZE=2
AICOS_DB_MAX_OVERFLOW=3
```

This gives: 4 workers × 4 stores × 5 connections = 80 app→PgBouncer connections → ~20 real PostgreSQL connections.

### Observability

```env
# OpenTelemetry traces to Jaeger / Grafana Tempo
AICOS_OTEL_ENDPOINT=http://jaeger:4317
# pip install 'aicos[otel]'

# Sentry error tracking
AICOS_SENTRY_DSN=https://...@sentry.io/...
# pip install 'aicos[sentry]'
```

---

## Installation

```bash
# Core only
pip install aicos

# With PostgreSQL support
pip install "aicos[postgres]"

# With pgvector (PostgreSQL vector search)
pip install "aicos[postgres,pgvector]"

# With Redis
pip install "aicos[redis]"

# With sentence-transformers (better embeddings, needed for pgvector dim=384)
pip install "aicos[embeddings]"

# With OpenTelemetry
pip install "aicos[otel]"

# With Sentry
pip install "aicos[sentry]"

# Everything
pip install "aicos[all]"
```

---

## Performance

| Operation | Target | Notes |
|---|---|---|
| Cache hit (exact) | < 5 ms | DB indexed |
| Cache hit (semantic) | < 20 ms | Cosine similarity |
| Memory search (pgvector) | < 5 ms | IVFFlat ANN index |
| Memory search (SQLite) | < 50 ms | NumPy scan, < 10k rows |
| Context compression | < 50 ms | Pure Python, TF-IDF |
| Gateway overhead | < 30 ms | Excluding LLM latency |
| Streaming | Real-time | SSE, no buffering |

---

## Development

```bash
git clone https://github.com/Akshadtech17/AI-COS-AI-Context-Operating-System-
cd AI-COS-AI-Context-Operating-System-
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=aicos

# Lint
ruff check aicos/

# Type check
mypy aicos/
```

### Project Structure

```
aicos/
├── aicos/
│   ├── api/
│   │   ├── middleware.py      # RequestIDMiddleware — X-Request-ID tracing
│   │   ├── rate_limiter.py    # RateLimitMiddleware — Redis + in-process sliding window
│   │   └── routes.py          # FastAPI app factory, all endpoints
│   ├── agents/
│   │   ├── base_agent.py      # Tool-calling agent base class
│   │   ├── startup_agent.py   # Market research, pricing, branding, financials
│   │   └── coding_agent.py    # Code gen, review, test gen, architecture
│   ├── analytics/
│   │   ├── cost_tracker.py    # Per-request cost + token tracking, DB-backed
│   │   └── metrics.py         # Prometheus metrics (latency histograms, counters)
│   ├── auth/
│   │   └── api_keys.py        # SHA-256-hashed per-user API key store
│   ├── cache/
│   │   ├── semantic_cache.py  # Cosine similarity cache (pgvector or NumPy)
│   │   └── sqlite_cache.py    # SQLAlchemy-backed cache store
│   ├── cli/
│   │   └── main.py            # Typer CLI (start, chat, remember, search, stats, config)
│   ├── context/
│   │   ├── compressor.py      # TF-IDF extractive compression
│   │   ├── history_manager.py # Token budget management + LITM trigger
│   │   └── summarizer.py      # LLM-based conversation summariser
│   ├── core/
│   │   ├── ai.py              # AI public interface (SDK entry point)
│   │   ├── circuit_breaker.py # CLOSED/OPEN/HALF_OPEN per-provider state machine
│   │   ├── config.py          # Pydantic-settings config with Docker secrets loader
│   │   ├── database.py        # Engine factory (SQLite WAL + PostgreSQL asyncpg)
│   │   ├── gateway.py         # Pipeline orchestrator (process + stream)
│   │   ├── logging.py         # JSON + colour formatters, ContextVar request ID
│   │   ├── router.py          # Task classifier + model selector
│   │   └── telemetry.py       # OpenTelemetry + Sentry initialisation
│   ├── db/
│   │   └── migrations.py      # Versioned migration runner + advisory lock
│   ├── memory/
│   │   ├── embeddings.py      # EmbeddingEngine (sentence-transformers or hash)
│   │   ├── memory_store.py    # SQLAlchemy store with pgvector ANN search path
│   │   └── retrieval.py       # Composite scorer, top-K retriever
│   └── providers/
│       ├── openai_provider.py  # OpenAI + OpenRouter + NVIDIA
│       ├── anthropic_provider.py
│       └── gemini_provider.py
├── docker-compose.yml          # Dev: single AI-COS + SQLite
├── docker-compose.prod.yml     # Prod: nginx + PgBouncer + PostgreSQL + Redis + Prometheus + Grafana
├── nginx/nginx.conf            # TLS termination config
├── monitoring/
│   ├── prometheus.yml          # Scrape config for aicos /metrics
│   └── grafana/provisioning/   # Auto-provisioned Prometheus datasource
└── tests/                      # 449 tests, 90% coverage
```

---

## Supported Providers

| Provider | Auth | Models | Notes |
|---|---|---|---|
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, o1-mini | Direct API |
| **Anthropic** | `ANTHROPIC_API_KEY` | claude-opus-4-8, claude-sonnet-4-6, claude-haiku-4-5 | Direct API |
| **Google** | `GEMINI_API_KEY` | gemini-2.0-flash, gemini-1.5-pro | Direct API |
| **NVIDIA** | `NVIDIA_API_KEY` | llama-3.1-nemotron-ultra-253b | Free tier available |
| **OpenRouter** | `OPENROUTER_API_KEY` | 100+ models incl. free Nemotron | Unified API |
| **Ollama** | none | Any local model | `AICOS_OLLAMA_ENABLED=true` |

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feat/my-feature`
3. Write tests for your changes
4. Run: `pytest tests/ && ruff check aicos/ && mypy aicos/`
5. Submit a PR

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built to be the infrastructure layer that every AI application deserves.*
