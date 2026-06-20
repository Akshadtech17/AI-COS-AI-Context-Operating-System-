# AI-COS — AI Context Operating System

> **The middleware layer between your app and any LLM.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-189%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-73%25-yellow.svg)]()

AI-COS is a self-hosted AI infrastructure layer that sits between your applications and LLM providers. It handles model routing, semantic caching, long-term memory, context compression, and observability — so your application code stays clean.

```python
from aicos import AI

ai = AI()
response = ai.chat("Build a SaaS startup")
# Routing, caching, memory, compression — all automatic.
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Smart Model Router** | Embedding-based task classifier routes to the best model per task (code, vision, reasoning, etc.) |
| **Semantic Cache** | Cosine similarity cache — identical or near-identical queries served in < 20ms |
| **Long-Term Memory** | `remember()` / `forget()` / `search_memory()` with composite relevance scoring |
| **Context Compression** | 40–80% token reduction preserving code blocks, JSON, and key facts |
| **LITM Solver** | Lost-in-the-Middle fix: compresses conversation middle, keeps recency and context |
| **OpenAI-Compatible Gateway** | Drop-in `/v1/chat/completions` with SSE streaming and provider failover |
| **Web Dashboard** | Live metrics, cache stats, provider status at `http://localhost:4000` |
| **Agent Framework** | Tool-calling agents with `StartupAgent` and `CodingAgent` built-in |
| **Observability** | Prometheus metrics, per-request cost tracking, latency histograms |
| **Multi-Provider** | OpenAI, Anthropic, Gemini, NVIDIA Nemotron, OpenRouter, Ollama |

---

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env   # add your API key(s)
docker compose up
```

Open **http://localhost:4000** for the live dashboard.
API at **http://localhost:4000/v1/chat/completions**.

### pip

```bash
pip install "aicos[all]"
cp .env.example .env
aicos start
```

### 1. Configure API Keys

```bash
cp .env.example .env
```

```env
# Use any combination — at least one required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=sk-or-...   # Access NVIDIA Nemotron Ultra (free)
NVIDIA_API_KEY=nvapi-...       # Direct NVIDIA API
```

### 2. Start the Gateway

```bash
aicos start --port 4000
```

### 3. Use the Python SDK

```python
from aicos import AI

ai = AI()

# Simple chat
response = ai.chat("What is the capital of France?")
print(response)  # "The capital of France is Paris."

# Store memories
ai.remember("User is building a B2B SaaS product", tags=["context", "project"])
ai.remember("User prefers Python over JavaScript")

# Memories are automatically injected into future chats
response = ai.chat("What tech stack should I use?")
# AI automatically considers stored memories

# Search memories
results = ai.search_memory("programming preferences")
for r in results:
    print(f"[{r['score']:.2f}] {r['content']}")

# Streaming
async for token in ai.astream("Write a haiku about AI"):
    print(token, end="", flush=True)
```

### 4. Use as OpenAI Drop-In

Point any OpenAI-compatible client at `http://localhost:4000`:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="auto",  # AI-COS selects optimal model
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

---

## Architecture

```
Application
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                      AI-COS Gateway                     │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  Memory  │  │  Cache   │  │ Context  │             │
│  │ Injector │  │ Lookup   │  │Compressor│             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       │              │              │                   │
│       └──────────────┼──────────────┘                   │
│                      │                                  │
│              ┌───────▼────────┐                         │
│              │ Model Router   │                         │
│              │ (task → model) │                         │
│              └───────┬────────┘                         │
│                      │                                  │
│         ┌────────────┼───────────┐                      │
│         ▼            ▼           ▼                      │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│   │  OpenAI  │ │Anthropic │ │  Gemini  │  + Ollama    │
│   └──────────┘ └──────────┘ └──────────┘              │
└─────────────────────────────────────────────────────────┘
    │                    │                    │
    ▼                    ▼                    ▼
┌──────────┐      ┌──────────┐       ┌──────────────┐
│  SQLite  │      │  SQLite  │       │  Analytics   │
│  Cache   │      │  Memory  │       │  (Prometheus)│
└──────────┘      └──────────┘       └──────────────┘
```

### Request Pipeline

Every request flows through this pipeline in order:

```
Request → Memory Injection → Context Compression → Cache Lookup →
Model Routing → LLM Call (w/ failover) → Cache Store → Response
```

**Cache hit:** Returns in < 20ms, cost = $0.

---

## Model Router

The router classifies tasks and selects the optimal model:

| Task Type | Detection | Model Tier |
|-----------|-----------|------------|
| `simple` | Short prompt, no keywords | Cheap (`gpt-4o-mini`, `claude-haiku`) |
| `coding` | `def`, `class`, code blocks, `import` | Mid (`claude-sonnet`, `gpt-4o`) |
| `vision` | "image", "photo", "describe" | Vision-capable models |
| `reasoning` | "analyze", "compare", "trade-off" | Premium (`gpt-4o`, `claude-opus`) |
| `analysis` | "market research", "financial", "strategy" | Premium |
| `agent` | "orchestrate", "automate", "workflow" | Premium |

Router strategies: `auto` (default), `cheapest`, `fastest`, `best`.

```python
from aicos.core.config import AICOSConfig
from aicos import AI

ai = AI(config=AICOSConfig(router_strategy="cheapest"))
```

---

## Memory System

Memories are stored with embeddings and retrieved by composite score:

```
score = cosine_similarity × 0.60 + recency_decay × 0.25 + access_frequency × 0.15
```

- **Recency decay**: `exp(-age_days / 30)` — fades over 30 days
- **Access frequency**: `log(1 + count) / log(1 + max_count)` — rewards used memories
- **Cosine similarity**: Feature-hash embeddings (512-d, no external model required)

```python
# Store with tags
ai.remember("User prefers async Python patterns", tags=["coding", "preferences"])

# Delete by ID
ai.forget(42)

# Semantic search
results = ai.search_memory("Python async preferences", top_k=3)
```

---

## Context Optimization

### Compression Rules

The compressor applies to natural language only:

| Content Type | Treatment |
|-------------|-----------|
| System messages | **Never modified** |
| Code blocks (` ``` `) | **Preserved verbatim** |
| JSON objects | **Preserved verbatim** |
| XML/HTML | **Preserved verbatim** |
| Natural language | **Compressed aggressively** (40–80% reduction) |

Algorithm: Extractive summarization via TF-IDF sentence scoring + positional bonuses.

### LITM Solver

When conversation exceeds `litm_threshold_tokens` (default: 6000):

```
Before: [System] [Turn 1] [Turn 2] [Turn 3] [Turn 4] [Turn 5] [Turn 6]
                                  ← "Lost in the Middle" zone →

After:  [System] [Context Recap] [Turn 5] [Turn 6]
                       ↑
                  LLM-generated summary of Turns 1-4
```

---

## Gateway API

### Chat Completions

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
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
    "latency_ms": 420.5,
    "routing": "Strategy=auto, task=simple, tier=cheap, cost=$0.15/1M"
  }
}
```

### Memory API

```bash
# Store memory
curl -X POST http://localhost:4000/v1/memory \
  -H "Content-Type: application/json" \
  -d '{"content": "User is an AI researcher", "tags": ["context"]}'

# Search memories
curl "http://localhost:4000/v1/memory/search?query=research&top_k=5"

# Delete memory
curl -X DELETE http://localhost:4000/v1/memory/42
```

### Metrics

```bash
curl http://localhost:4000/metrics   # Prometheus format
curl http://localhost:4000/stats     # JSON summary
curl http://localhost:4000/health    # Health check
```

---

## Agents

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
#   "market": {"tam_usd": 12000000000, ...},
#   "competitors": [...],
#   "pricing": {"tiers": [{"name": "Starter", "price_usd_monthly": 49}, ...]},
#   ...
# }
```

### CodingAgent

Software engineering agent with code review, generation, and architecture tools:

```python
agent = CodingAgent(gateway=ai._gateway)
result = await agent.run_code_task(
    "Build a rate limiter class with sliding window algorithm",
    language="python"
)
print(result.structured["code"])
```

---

## CLI Reference

```bash
# Start gateway
aicos start --port 4000 --host 0.0.0.0

# Interactive chat
aicos chat
aicos chat "What is machine learning?" --model gpt-4o

# Memory management
aicos remember "I prefer type hints in Python" --tags "coding"
aicos forget 42
aicos search "Python preferences" --top-k 5

# Statistics
aicos stats

# Configuration
aicos config
```

---

## Configuration

All settings via environment variables (prefix `AICOS_` for AI-COS settings):

```env
# Router
AICOS_ROUTER_STRATEGY=auto        # auto | cheapest | fastest | best
AICOS_DEFAULT_MODEL=gpt-4o-mini   # Override auto-routing

# Cache
AICOS_CACHE_ENABLED=true
AICOS_CACHE_SIMILARITY_THRESHOLD=0.96
AICOS_CACHE_MAX_SIZE=10000
AICOS_CACHE_TTL_SECONDS=86400

# Memory
AICOS_MEMORY_ENABLED=true
AICOS_MEMORY_MAX_ITEMS=10000
AICOS_MEMORY_INJECTION_LIMIT=5

# Context
AICOS_CONTEXT_COMPRESSION_ENABLED=true
AICOS_MAX_CONTEXT_TOKENS=8000
AICOS_LITM_THRESHOLD_TOKENS=6000

# Gateway
AICOS_GATEWAY_HOST=0.0.0.0
AICOS_GATEWAY_PORT=4000
AICOS_GATEWAY_API_KEY=          # Optional: require Bearer token
```

---

## Supported Providers

| Provider | Models | Cost |
|----------|--------|------|
| **OpenAI** | gpt-4o, gpt-4o-mini, o1, o1-mini | From $0.15/1M tokens |
| **Anthropic** | claude-opus-4, claude-sonnet-4, claude-haiku-4 | From $0.25/1M tokens |
| **Google** | gemini-2.0-flash, gemini-1.5-pro | From $0.10/1M tokens |
| **OpenRouter** | 100+ models | Varies |
| **Ollama** | Any local model | Free |

---

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Cache lookup (exact) | < 5ms | SQLite indexed |
| Cache lookup (semantic) | < 20ms | Vectorized NumPy |
| Context compression | < 50ms | Pure Python |
| Memory retrieval | < 50ms | Cosine similarity scan |
| Gateway overhead | < 30ms | Excluding LLM latency |
| Streaming | Real-time | SSE, no buffering |

---

## Development

```bash
# Clone and install
git clone https://github.com/aicos-dev/aicos
cd aicos
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v --cov=aicos

# Run type checker
python -m mypy aicos/

# Run linter
python -m ruff check aicos/
```

### Project Structure

```
aicos/
├── aicos/
│   ├── core/           # AI, Config, Router, Gateway
│   ├── memory/         # MemoryStore, Embeddings, Retrieval
│   ├── context/        # Compressor, Summarizer, HistoryManager
│   ├── cache/          # SemanticCache, SQLiteCache
│   ├── agents/         # BaseAgent, StartupAgent, CodingAgent
│   ├── analytics/      # MetricsCollector, CostTracker
│   ├── providers/      # OpenAI, Anthropic, Gemini
│   ├── api/            # FastAPI routes
│   └── cli/            # Typer CLI
├── tests/
├── pyproject.toml
└── README.md
```

---

## Roadmap

- [ ] Redis caching backend
- [ ] ChromaDB/FAISS vector memory
- [ ] Prompt injection detection
- [ ] A/B testing framework
- [ ] Multi-modal memory (images)
- [ ] Distributed gateway mode
- [ ] LiteLLM proxy integration
- [ ] Automatic model benchmarking

---

## Contributing

Contributions welcome. Please open an issue first for large changes.

1. Fork the repo
2. Create a branch: `git checkout -b feat/my-feature`
3. Write tests for your changes
4. Run: `pytest && ruff check && mypy aicos/`
5. Submit a PR

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built to be the infrastructure layer that every AI application deserves.*
