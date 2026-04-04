# Gemma 4 Inference Network

**Multi-device private AI — Mac Mini + MacBook Pro + public web + iPhone, all via MLX.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Models: Gemma 4](https://img.shields.io/badge/Models-Gemma_4-orange.svg)](https://huggingface.co/google/gemma-4-E2B-it)
[![MLX](https://img.shields.io/badge/Engine-MLX_0.31.2-green.svg)](https://github.com/ml-explore/mlx)

---

## What This Is

A multi-device AI inference network built on Google's Gemma 4 model family, running entirely on Apple Silicon via MLX. A dedicated Mac Mini serves lightweight models 24/7, a MacBook Pro M4 Max contributes a heavy-class model when online, and a Google Cloud proxy makes the whole thing accessible from the public internet or a native iPhone app.

**No API keys. No cloud GPU bills. No data leaves your hardware.**

```
PUBLIC INTERNET                       TAILSCALE MESH (private)
    │                                     │
    ▼                                     │
 GCP Cloud Run ──── Tailscale ────────────┤
 (proxy + web UI)   WireGuard             │
    │                                     │
    │         ┌───────────────────────────┤
    │         │                           │
    │    Mac Mini M4 16GB             MacBook Pro M4 Max
    │    (always-on)                  (intermittent)
    │    ├─ E2B MLX   :8082           ├─ 26B-A4B MLX :8084
    │    ├─ E4B MLX   :8083           └─ 256K context
    │    └─ Gateway   :8080               ~50-70 tok/s
    │       126 tok/s + 32 tok/s
    │                                     │
    │         ┌───────────────────────────┘
    │         │
    └────► iPhone App
           (public URL or Tailscale direct)
```

---

## Devices & Models

| Device | Role | Model | Params (active) | Quantization | Speed | Status |
|--------|------|-------|-----------------|-------------|-------|--------|
| Mac Mini M4 16GB | Fast tier | `mlx-community/gemma-4-e2b-it-4bit` | 5.1B (2.3B) | 4-bit MLX | **126 tok/s** | Always on |
| Mac Mini M4 16GB | Primary tier | `mlx-community/gemma-4-e4b-it-4bit` | 8.0B (4.5B) | 4-bit MLX | **32 tok/s** | Always on |
| MacBook Pro M4 Max | Heavy tier | `mlx-community/gemma-4-26b-a4b-it-4bit` | 26B (3.8B) | 4-bit MLX | **~50-70 tok/s** | When online |

### Why the 26B-A4B MoE on the MacBook Pro (not the 31B Dense)?

Both models fit on the M4 Max (15.6 GB vs 18.4 GB at 4-bit). The decision comes down to throughput vs peak quality:

| | 26B-A4B MoE | 31B Dense |
|---|---|---|
| **Total params** | 26B | 30.7B |
| **Active per token** | 3.8B | 30.7B (all) |
| **MLX 4-bit size** | 15.6 GB | 18.4 GB |
| **Estimated tok/s (M4 Max)** | **50-70** | 20-30 |
| **AIME 2026** | 88.3% | 89.2% |
| **LMArena score** | 1441 | 1452 |
| **Context window** | **256K** | 128K |

**The MoE wins for serving.** 97% of the dense model's quality at 2-3x the throughput. Since the MacBook Pro is intermittent (not always on), when it IS online you want maximum throughput for the queue of requests that accumulated. The MoE also supports 256K context — 2x the dense model — critical for document analysis. If you need peak quality for a single difficult task, the dense 31B can be swapped in via a one-line model change.

### Why NOT run E2B/E4B on the MacBook Pro too?

The Mac Mini already serves E2B and E4B 24/7. Running them on the MacBook Pro would duplicate effort on the same lightweight tasks. The MacBook Pro's M4 Max (40-core GPU, 273 GB/s bandwidth) is wasted on a 2.3B model. **Dedicate scarce intermittent compute to what the Mac Mini cannot do well: the heavy tier.**

---

## Architecture Deep Dive

### Unified Memory + MLX: Why This Works

Apple Silicon's unified memory architecture means CPU and GPU share the same physical DRAM — no PCIe bus copies. MLX exploits this with zero-copy Metal kernel dispatch:

```
Model weights (safetensors) → MLX array (unified memory)
                                       │
                             MLX lazy computation graph
                                       │
                           Metal kernel (MSL) dispatch
                                       │
                  GPU reads weights directly from same memory pool
                           (no staging buffer, no transfer)
                                       │
                              Token sampled → output
```

4-bit quantization compresses weights ~8x vs fp32, halving the effective bandwidth demand. On the Mac Mini's 120 GB/s unified bus, the E2B model (1.15 GB working set) generates at 126 tok/s. On the M4 Max's 273 GB/s bus, the 26B MoE (only 3.8B active = ~1.9 GB compute working set) is expected to hit 50-70 tok/s.

### Multi-Device Routing

```
Request → Gateway (Mac Mini :8080)
              │
              ├─ classify_request()
              │     │
              │     ├── greeting, simple question → Fast (E2B, local)
              │     ├── summarize, compress → Primary (E4B, local)
              │     └── code, analysis, complex → Heavy (26B, MacBook Pro)
              │
              ├─ if heavy tier offline → fallback to Primary (E4B)
              │                          with fallback note in response
              │
              └─ _routing: {"tier": "heavy", "device": "macbook-pro",
                            "latency_ms": 1420, "fallback": false}
```

The gateway runs a background health checker every 30 seconds. When the MacBook Pro connects to Tailscale, the heavy tier auto-discovers as online. When the MacBook Pro sleeps or disconnects, heavy requests gracefully fall back to E4B.

### Public Access via GCP Cloud Run

```
Browser (anywhere) → Cloud Run proxy (us-central1)
                         │
                         │ Tailscale sidecar (WireGuard)
                         │
                         └───→ Mac Mini 100.75.223.113:8080
                                    │
                                    ├── E2B (local)
                                    ├── E4B (local)
                                    └── 26B (→ MacBook Pro via Tailscale)
```

The Cloud Run service costs ~$0/month under free tier (2M requests/month). It adds rate limiting (30 req/min/IP) and optional API key auth.

---

## TurboQuant: KV Cache Compression (Future Optimization)

[TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) is a KV cache quantization technique from Google (ICLR 2026) that compresses the inference cache 3-5x with near-lossless quality. An [MLX implementation](https://github.com/rachittshah/mlx-turboquant) already exists.

### Why It Matters for This Stack

The KV cache is the second-largest memory consumer after model weights. At 16K tokens of context, the KV cache for the 26B model can exceed 4 GB. With TurboQuant:

| Scenario | Without TurboQuant | With TurboQuant (3-bit) | Savings |
|----------|-------------------|------------------------|---------|
| E4B @ 8K context | ~1.2 GB KV cache | ~260 MB | 4.6x |
| 26B @ 16K context | ~4.2 GB KV cache | ~910 MB | 4.6x |
| 26B @ 64K context | ~16.8 GB KV cache | ~3.6 GB | 4.6x |
| 26B @ 256K context | ~67 GB KV cache | ~14.5 GB | 4.6x |

**On the Mac Mini (16 GB):** TurboQuant enables long-context E4B inference that would otherwise OOM. A 32K-token summarization that requires ~2.4 GB KV cache drops to ~520 MB.

**On the MacBook Pro M4 Max (36 GB+):** TurboQuant unlocks the full 256K context window of the 26B model. Without compression, 256K context needs ~67 GB of KV cache — impossible even on 48 GB. With 3-bit TurboQuant, it drops to ~14.5 GB — fits comfortably alongside the 15.6 GB model weights.

### Integration Path

```bash
pip install mlx-turboquant   # or clone from GitHub

# Drop-in replacement in MLX model serving:
from mlx_turboquant.cache import TurboQuantKVCache
cache = [TurboQuantKVCache(bits=3, head_dim=128) for _ in range(num_layers)]
```

Benchmarks from [mlx-turboquant](https://github.com/rachittshah/mlx-turboquant) show **0.995+ cosine similarity** at 4-bit and **0.97+** at 3-bit — effectively lossless for real-world tasks.

### SwiftLM: Native Swift Alternative

[SwiftLM](https://github.com/SharpAI/SwiftLM) is a pure Swift/Metal inference server with TurboQuant built in. It eliminates Python's GIL overhead entirely, includes an iOS companion app ([SwiftBuddy](https://github.com/SharpAI/SwiftLM)), and supports SSD streaming for MoE expert weights. This is a potential future replacement for the `mlx_lm.server` backend that would also solve the iPhone app requirement natively.

---

## Quick Start

### Mac Mini (always-on server)

```bash
# 1. Clone
git clone https://github.com/AlexiosBluffMara/gemma4-stack
cd gemma4-stack

# 2. Bootstrap (Homebrew, Python 3.12, mlx-lm, download models)
bash scripts/phase1_setup.sh
source ~/.zprofile

# 3. Start everything
bash scripts/start_fast.sh       # E2B on :8082
bash scripts/start_primary.sh    # E4B on :8083
bash scripts/start_gateway.sh    # Gateway on :8080

# 4. Verify
curl http://localhost:8080/health
# → {"tiers": {"fast": "ok", "primary": "ok", "heavy": "offline"}}

# 5. Test
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

### MacBook Pro M4 Max (heavy tier)

```bash
# 1. Clone (same repo)
git clone https://github.com/AlexiosBluffMara/gemma4-stack
cd gemma4-stack

# 2. Setup (installs MLX, downloads 26B model — ~15.6 GB)
bash scripts/setup-macbook.sh

# 3. Set up Tailscale (so the Mac Mini can reach this machine)
bash scripts/setup_tailscale.sh

# 4. Start the heavy tier
bash scripts/start-macbook.sh
# → 26B-A4B serving on :8084

# 5. Tell the Mac Mini gateway where to find you
# On the Mac Mini, set the env var before starting the gateway:
export HEAVY_URL="http://<macbook-tailscale-ip>:8084"
bash scripts/start_gateway.sh
```

### Tailscale Mesh (both devices)

```bash
# On each device:
# 1. Install official .pkg from https://pkgs.tailscale.com/stable/#macos
# 2. Approve Network Extension in System Settings
# 3. Run:
bash scripts/setup_tailscale.sh
```

### Public Web Access (GCP Cloud Run)

```bash
# Deploy the proxy + web UI to Cloud Run
cd cloud
bash deploy.sh YOUR_GCP_PROJECT_ID

# The script prints your public URL:
# → https://gemma4-proxy-xxxxx-uc.a.run.app
# Anyone can now chat with your models from a browser.
```

---

## API Reference

All endpoints on the gateway at `:8080` (or through the Cloud Run proxy).

### `POST /v1/chat/completions` — OpenAI-compatible
```json
{
  "messages": [{"role": "user", "content": "your message"}],
  "max_tokens": 512,
  "temperature": 0.0,
  "tier": "heavy"
}
```
> `"tier"` is optional. Omit for auto-routing. Values: `"fast"` | `"primary"` | `"heavy"`.

Response includes routing metadata:
```json
{
  "choices": [...],
  "_routing": {
    "tier": "heavy",
    "device": "macbook-pro",
    "latency_ms": 1420,
    "fallback": false
  }
}
```

### `POST /classify` — Fast tier only
```json
{"text": "Please fix the bug in the login module."}
```
→ `{"category": "request", "latency_ms": 270}`

### `POST /compress` — Primary tier only
```json
{"text": "...", "words": 20}
```
→ `{"compressed": "...", "latency_ms": 1570}`

### `GET /health`
```json
{"tiers": {"fast": "ok", "primary": "ok", "heavy": "offline"}}
```

### `GET /devices`
```json
{
  "mac-mini": {
    "ip": "100.75.223.113",
    "capabilities": ["e2b", "e4b", "gateway"],
    "status": "online",
    "last_seen": "2026-04-04T17:30:00Z"
  },
  "macbook-pro": {
    "ip": "100.x.x.x",
    "capabilities": ["26b-a4b"],
    "status": "offline",
    "last_seen": null
  }
}
```

### `GET /metrics`
→ Per-tier request counts and p50/p95 latency histograms

---

## Production Use Cases

### Tier 1 — Fast: E2B (0.27s, 126 tok/s, always on)
| Use Case | Integration |
|----------|------------|
| Slack/Discord message triage | Webhook → `POST /classify` |
| Email ticket routing | Per-email → classify to queue |
| Git pre-commit hook | Reject vague commit messages in <0.5s |
| Intent detection | Classify voice transcription before routing |
| Content moderation | Gate all UGC before storage |

### Tier 2 — Primary: E4B (1.57s, 32 tok/s, always on)
| Use Case | Integration |
|----------|------------|
| PR diff summarizer | GitHub webhook → auto-summary on every PR |
| Meeting transcript digest | Daily cron → compress transcripts |
| Vector DB chunking | Pre-ingest pipeline → chunk + summarize |
| Release notes drafts | CI trigger → changelog from git log |

### Tier 3 — Heavy: 26B-A4B (~50-70 tok/s on M4 Max, when online)
| Use Case | Integration |
|----------|------------|
| Deep code review | PR webhook → escalate to heavy tier |
| Technical documentation | Post-deploy → draft API docs |
| Long document analysis | Feed 256K context → structured output |
| Architecture trade-off analysis | On-demand via web UI |
| Complex SQL generation | When E4B is insufficient → escalate |

---

## Cost Analysis

### Hardware

| Device | Cost | Monthly Cloud Equivalent | Break-Even |
|--------|------|-------------------------|------------|
| Mac Mini M4 16GB | $599 | $360/mo (A100 spot 8h/day) | **50 days** |
| MacBook Pro M4 Max 36GB | ~$3,499 | $600/mo (A100 spot full-day) | ~6 months |
| GCP Cloud Run proxy | ~$0/mo | — | Immediate |
| **Total** | **~$4,100** | **~$960/mo cloud** | **~4.3 months** |

After break-even: **~$8/month electricity** for unlimited private inference across both devices.

### Funding Tiers (Apple for Education/Business pricing)

| Tier | Budget | What You Get | Key Unlock |
|------|--------|-------------|------------|
| 0 | **$599** | Mac Mini M4 16GB — E2B + E4B always-on | Private AI foundation |
| 1 | **~$4,100** | + MacBook Pro M4 Max 36GB — 26B heavy tier | **50-70 tok/s heavy, 256K context** |
| 2 | **~$5,800** | + Mac Mini M4 Pro 24GB (redundancy) | High availability, GPU 26B on Mini too |
| 3 | **~$12,500** | + Mac Studio M4 Ultra 192GB (replace MacBook) | 405B models, 100+ concurrent users |
| 4 | **~$25,000** | Multi-Ultra cluster + NAS | Model parallelism, A/B testing, fine-tuning |

---

## File Structure

```
gemma4-stack/
├── README.md
├── .gitignore
│
├── scripts/
│   ├── gateway.py               ← Multi-device FastAPI gateway
│   ├── local_llm.py             ← Python SDK for tier routing
│   ├── benchmark.py             ← Benchmark all active tiers
│   ├── health_check.sh          ← Check all services
│   │
│   ├── phase1_setup.sh          ← Mac Mini full bootstrap
│   ├── setup-macbook.sh         ← MacBook Pro M4 Max bootstrap
│   ├── setup_tailscale.sh       ← Tailscale mesh setup
│   │
│   ├── start_fast.sh            ← Start E2B (MLX :8082)
│   ├── start_primary.sh         ← Start E4B (MLX :8083)
│   ├── start_gateway.sh         ← Start gateway (:8080)
│   ├── start-macbook.sh         ← Start 26B on MacBook Pro (:8084)
│   │
│   ├── stop_fast.sh             ← Stop E2B
│   ├── stop_primary.sh          ← Stop E4B
│   └── stop-macbook.sh          ← Stop 26B on MacBook Pro
│
├── cloud/
│   ├── deploy.sh                ← GCP Cloud Run deploy script
│   ├── proxy/
│   │   ├── main.py              ← Cloud Run proxy (FastAPI)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── web/
│       └── index.html           ← Public web chat UI
│
├── notebooks/
│   └── gemma4_local_inference.ipynb
│
└── docs/
    └── IMPLEMENTATION_LOG.md
```

---

## iPhone App

The architecture supports two paths for iOS access:

### Path 1: Web App (works now)
The Cloud Run web UI is responsive and works on iOS Safari. Add to Home Screen for an app-like experience. No App Store submission required.

### Path 2: Native SwiftUI App (planned)
A native iOS app connecting to either:
- **Public URL** (Cloud Run proxy) — works from anywhere
- **Tailscale IP** (direct) — lower latency on private network

The app would show device status, let you pick tiers, stream responses, and store chat history locally. [SwiftLM's SwiftBuddy](https://github.com/SharpAI/SwiftLM) is a reference implementation that already handles model downloads and on-device inference — useful if the iPhone should also run the E2B model locally for offline use.

---

## Future Optimizations

| Optimization | Impact | Effort | Status |
|-------------|--------|--------|--------|
| **TurboQuant KV cache** | 4.6x cache compression → 256K context on M4 Max | Medium | [MLX impl available](https://github.com/rachittshah/mlx-turboquant) |
| **SwiftLM backend** | Eliminate Python GIL, native Metal, built-in TurboQuant | High | [Exists, needs eval](https://github.com/SharpAI/SwiftLM) |
| **8-bit models on M4 Max 48GB+** | Higher quality (Q8 vs Q4) at same throughput | Low | Model exists: `mlx-community/gemma-4-26b-a4b-mxfp8` |
| **Speculative decoding** | Use E2B as draft model for 26B verification | Medium | MLX supports this |
| **LaunchAgents** | Auto-start tiers on boot (Mac Mini) | Low | Plist files ready |
| **Cloudflare Tunnel** | Alternative to GCP for public access | Low | Free tier available |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `gemma4 not supported` in mlx-lm | Install from GitHub: `pip install git+https://github.com/ml-explore/mlx-lm` |
| Heavy tier always "offline" | MacBook Pro must be awake + connected to Tailscale + `start-macbook.sh` running |
| Gateway doesn't see MacBook Pro | Set `HEAVY_URL=http://<tailscale-ip>:8084` before starting gateway |
| Tailscale daemon not running | Install official .pkg from pkgs.tailscale.com (Homebrew CLI alone is not enough) |
| Duplicate MLX server processes | `pkill -f "mlx_lm server"` then restart |
| Cloud Run deploy fails | Ensure `gcloud` is authenticated and billing is enabled on the project |
| Web UI shows both devices red | Gateway may be down; check `curl http://localhost:8080/health` on Mac Mini |

---

## License

Models: Apache 2.0 (Google Gemma 4)
Code: Apache 2.0

---

Sources for technical claims:
- [Gemma 4 model card & benchmarks](https://huggingface.co/blog/gemma4)
- [Google blog: Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [TurboQuant: Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [mlx-turboquant: MLX implementation](https://github.com/rachittshah/mlx-turboquant)
- [SwiftLM: Native Swift inference](https://github.com/SharpAI/SwiftLM)
- [MLX 26B-A4B model](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) (15.6 GB)
- [MLX 31B Dense model](https://huggingface.co/mlx-community/gemma-4-31b-it-4bit) (18.4 GB)
- [Gemma 4 vs Qwen 3.5 vs Llama 4 benchmarks](https://ai.rs/ai-developer/gemma-4-vs-qwen-3-5-vs-llama-4-compared)
