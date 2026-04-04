# Gemma 4 Local Inference Stack

**Three-tier private AI inference on a $599 Mac Mini M4 — accessible anywhere via Tailscale.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Models: Gemma 4](https://img.shields.io/badge/Models-Gemma_4-orange.svg)](https://huggingface.co/google/gemma-4-E2B-it)
[![MLX](https://img.shields.io/badge/Engine-MLX_0.31.2-green.svg)](https://github.com/ml-explore/mlx)

---

## What This Is

A production-ready, self-hosted AI inference stack using Google's Gemma 4 model family. Routes requests intelligently across three tiers based on task complexity. Exposes a single OpenAI-compatible API endpoint, accessible from any device on your Tailscale network.

**No API keys. No cloud bills. No data leaves your network.**

```
Client (any device on Tailscale)
  │
  ▼
FastAPI Gateway :8080  ←  OpenAI-compatible /v1/chat/completions
  │
  ├── Tier 1 Fast    :8082  Gemma 4 E2B  (MLX 4-bit)  0.27s classify
  ├── Tier 2 Primary :8083  Gemma 4 E4B  (MLX 4-bit)  1.57s summarize
  └── Tier 3 Heavy   :8081  Gemma 4 26B  (llama.cpp)  ~1.6 tok/s
```

---

## Measured Performance (Mac Mini M4, 16GB, macOS 26.3.1)

| Tier | Model | Engine | Speed | RAM | Latency |
|------|-------|--------|-------|-----|---------|
| Fast | E2B MLX 4-bit | MLX 0.31.2 | **126 tok/s** | 2.6 GB | **0.27s** classify |
| Primary | E4B MLX 4-bit | MLX 0.31.2 | **32 tok/s** | 4.3 GB | **1.57s** summarize |
| Heavy | 26B-A4B GGUF | llama.cpp CPU | **1.6 tok/s** | ~4 GB resident | 30-60s complex |

> **Fast and Primary tiers are 6-9× faster than the original plan's targets** thanks to MLX's native Metal GPU path on Apple Silicon. The heavy tier runs CPU-only on 16 GB (Metal OOM for a 16 GB model on a 16 GB machine). See [Hardware Notes](#hardware-notes).

---

## Cost Effectiveness

| Setup | Cost | E2B Speed | 26B Speed | Break-even vs Cloud |
|-------|------|-----------|-----------|---------------------|
| **Mac Mini M4 16GB** | **$599** | **126 tok/s** | 1.6 tok/s CPU | **~50 days** |
| Mac Mini M4 Pro 24GB | $1,399 | ~150 tok/s | ~12-20 tok/s GPU | ~116 days |
| Mac Studio M4 Max 64GB | $1,999 | ~160 tok/s | ~30-50 tok/s GPU | ~167 days |
| Cloud A100 (spot) | $1.50/hr | N/A | 80-120 tok/s | Never |

Break-even vs A100 cloud: `$599 ÷ ($1.50/hr × 8 hrs/day × 30 days) = ~1.66 months`
After break-even: **~$5/month electricity for unlimited private inference.**

---

## Quick Start

### Option A: Native macOS (Recommended — best performance)

```bash
# 1. Clone
git clone https://github.com/AlexiosBluffMara/gemma4-stack
cd gemma4-stack

# 2. Install dependencies (interactive — needs your password for Homebrew)
bash scripts/phase1_setup.sh

# 3. Start Fast + Primary tiers
bash scripts/start_fast.sh
bash scripts/start_primary.sh

# 4. Start gateway
bash scripts/start_gateway.sh

# 5. Verify
curl http://localhost:8080/health
# → {"tiers": {"fast": "ok", "primary": "ok", "heavy": "offline"}}

# 6. Test
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

### Option B: Docker (Cross-platform — Linux, Windows, macOS)

> **Note:** Docker runs CPU-only inference on all platforms (MLX requires native macOS + Apple Silicon). Expect ~15-25 tok/s for E2B/E4B vs 126 tok/s native.

```bash
# 1. Clone
git clone https://github.com/AlexiosBluffMara/gemma4-stack
cd gemma4-stack

# 2. Configure
cp docker/.env.example docker/.env
# Edit docker/.env if needed

# 3. Start always-on tiers (models auto-download on first start, ~17 GB)
cd docker
docker compose up -d fast primary gateway

# 4. Watch model download progress
docker compose logs -f fast

# 5. Verify (wait ~5 minutes for model downloads)
curl http://localhost:8080/health

# 6. (Optional) Start heavy tier
python3 scripts/download_heavy.py --dir ./models
docker compose --profile heavy up -d heavy
```

---

## Heavy Tier Setup

The 26B-A4B model is ~17 GB and must be downloaded separately:

```bash
# Install downloader deps
pip install huggingface-hub hf-transfer

# Download (~20-45 minutes, auto-resumes)
python3 scripts/download_heavy.py

# Native macOS — start heavy tier (stops Fast+Primary to free RAM)
bash scripts/start_heavy.sh

# Docker — mount models dir and start
docker compose --profile heavy up -d heavy
```

---

## Tailscale Mesh Access

Share the stack with any device on your Tailscale network:

### macOS Setup
```bash
# Install official Tailscale (Homebrew CLI alone won't work — needs the .pkg)
# Download: https://pkgs.tailscale.com/stable/#macos
# Install .pkg → approve Network Extension in System Settings

tailscale up

# Get your IP
tailscale ip -4
# → 100.x.x.x

# All tiers are now accessible at that IP:
curl http://100.x.x.x:8080/health
```

### Use from Any Device
```python
# Works from any laptop, phone, or server on your Tailscale network
from openai import OpenAI

client = OpenAI(
    base_url="http://100.x.x.x:8080",
    api_key="not-required"
)

response = client.chat.completions.create(
    model="auto",   # gateway auto-routes to correct tier
    messages=[{"role": "user", "content": "Summarize this PR diff: ..."}]
)
```

---

## API Reference

All endpoints are on the gateway at `:8080`.

### `POST /v1/chat/completions` — OpenAI-compatible
```json
{
  "messages": [{"role": "user", "content": "your message"}],
  "max_tokens": 512,
  "tier": "fast"   // optional: force "fast" | "primary" | "heavy"
}
```
Response includes `_routing` field: `{"tier": "primary", "latency_ms": 1640}`

### `POST /classify` — Fast classification only
```json
{"text": "Please fix the bug in the login module."}
```
→ `{"category": "request", "latency_ms": 270}`

### `POST /compress` — Primary summarization only
```json
{"text": "...", "words": 20}
```
→ `{"compressed": "...", "latency_ms": 1570}`

### `GET /health`
→ `{"tiers": {"fast": "ok", "primary": "ok", "heavy": "offline"}}`

### `GET /metrics`
→ Per-tier request counts and p50/p95 latencies

---

## Production Use Cases

### Tier 1 — Fast (0.27s, 126 tok/s) — Always On
| Use Case | How |
|----------|-----|
| Slack/Discord bot message triage | Webhook → `POST /classify` |
| Email ticket routing | Per-email → classify to queue |
| Git pre-commit hook | Reject vague commit messages in <0.5s |
| Intent detection | Classify voice transcription before routing |
| Spam / content moderation | Gate all submissions before storage |

### Tier 2 — Primary (1.57s, 32 tok/s) — Always On
| Use Case | How |
|----------|-----|
| PR diff summarizer | GitHub webhook → auto-summary on every PR |
| Meeting transcript digest | Daily cron → compress transcripts |
| Vector DB chunking | Pre-ingest pipeline → chunk summaries |
| Standup aggregator | Team async updates → single digest |
| Release notes drafts | CI trigger → changelog from commits |

### Tier 3 — Heavy (30-60s, 1.6 tok/s CPU) — On-Demand
| Use Case | How |
|----------|-----|
| Deep code review | Manual trigger or PR open webhook |
| Technical documentation | Post-deploy → draft API docs |
| Incident postmortem | Feed structured data → full postmortem |
| Architecture analysis | On request → trade-off analysis |
| Complex SQL generation | When Tier 2 fails → escalate |

---

## Hardware Notes

### Why 16 GB is Enough for Tiers 1 & 2
MLX runs E2B and E4B natively on the M4 GPU with 4-bit quantization, using only 2.6 GB and 4.3 GB respectively. Total footprint for both tiers + macOS ≈ 10 GB, leaving 6 GB headroom.

### Why 16 GB Limits the 26B Heavy Tier
The 26B-A4B GGUF is 16 GB on disk. llama.cpp's `--mmap` flag keeps only "hot" pages in RAM (the active expert weights, ~4-5 GB resident), with the rest paged from SSD on demand.

However, the M4's Metal GPU has a `recommendedMaxWorkingSetSize` of 12.7 GB. When llama.cpp tries to offload all 31 layers to Metal for compute, the combined weight buffers + KV cache + activations exceed this. Result: `kIOGPUCommandBufferCallbackErrorOutOfMemory`.

**Solution used:** `--gpu-layers 0` (CPU-only via Apple Accelerate/BLAS). Reliable at ~1.6 tok/s. Good enough for background/on-demand tasks.

**For GPU-accelerated 26B:** Requires 24 GB+ unified memory (Mac Mini M4 Pro ≥$1,399 or Mac Studio).

### mmap Explained
`mmap` maps a file's virtual address space into the process without copying it into RAM. The OS loads only the pages that are actively accessed (page fault on demand). For a MoE model:
- Total weights: 26B (16 GB on disk)
- Active per token: 4B (only the selected experts fire)
- RAM resident: ~4-5 GB at any time
- Rest: on SSD, fetched in milliseconds when needed (M4 SSD: ~7.5 GB/s)

---

## Architecture Details

```
Native macOS (fast path):          Docker (portable path):
────────────────────────           ───────────────────────
E2B: mlx_lm server :8082           E2B: ollama :8082 (port-mapped)
E4B: mlx_lm server :8083           E4B: ollama :8083 (port-mapped)
26B: llama-server  :8081           26B: llama.cpp:server :8081
GW:  uvicorn/fastapi :8080         GW:  uvicorn/fastapi :8080

LaunchAgents auto-start E2B,       docker compose manages lifecycle
E4B, gateway on login.             Heavy tier uses --profile heavy
```

---

## File Structure

```
gemma4-stack/
├── README.md                    ← This file
├── .gitignore
│
├── scripts/
│   ├── gateway.py               ← FastAPI auto-routing gateway
│   ├── local_llm.py             ← Python SDK for tier routing
│   ├── benchmark.py             ← Benchmark all active tiers
│   ├── health_check.sh          ← Check all services
│   ├── start_fast.sh            ← Start E2B (MLX, :8082)
│   ├── start_primary.sh         ← Start E4B (MLX, :8083)
│   ├── start_heavy.sh           ← Start 26B (llama.cpp, :8081)
│   ├── start_gateway.sh         ← Start FastAPI gateway (:8080)
│   ├── stop_*.sh                ← Stop individual tiers
│   ├── setup_tailscale.sh       ← Tailscale setup walkthrough
│   ├── download_heavy.py        ← Download 26B GGUF from HF
│   └── phase1_setup.sh          ← Full environment bootstrap
│
├── docker/
│   ├── docker-compose.yml       ← Orchestrate all tiers
│   ├── Dockerfile.gateway       ← Gateway container
│   ├── Dockerfile.fast          ← Fast tier (Ollama)
│   ├── Dockerfile.heavy         ← Heavy tier (llama.cpp)
│   └── .env.example             ← Environment config template
│
├── notebooks/
│   └── gemma4_local_inference.ipynb  ← Full implementation analysis
│
└── docs/
    └── IMPLEMENTATION_LOG.md    ← Plan vs reality, deviations, lessons
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gemma4 not supported` in mlx-lm | Install from GitHub: `pip install git+https://github.com/ml-explore/mlx-lm` |
| Metal `Insufficient Memory` on 26B | Use `start_heavy.sh cpu` for CPU-only mode |
| Tailscale daemon not running | Install official .pkg from pkgs.tailscale.com (Homebrew CLI alone is not enough) |
| Duplicate MLX server processes | `pkill -f "mlx_lm server"` then restart |
| llama-server `--n-parallel` error | Use `--parallel` (this version uses the short flag) |
| Docker: models not downloading | Check `docker compose logs fast` — Ollama pull may take 15-20 min |

---

## Implementation Notes

See [`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md) for the full plan-vs-reality log including:
- Why MLX was chosen over Ollama (5-8× faster, 55-63% less RAM)
- Why the 26B tier uses llama.cpp instead of MLX (OOM on 16 GB)
- Why mlx-lm needed GitHub main instead of PyPI (Gemma 4 support gap)
- Why Tailscale needs the official .pkg on macOS

---

## License

Models: Apache 2.0 (Google Gemma 4, Unsloth quantizations)
Code: Apache 2.0
