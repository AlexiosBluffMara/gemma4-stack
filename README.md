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

| Tier | Model | Engine | Speed | RAM Resident | Latency |
|------|-------|--------|-------|-------------|---------|
| Fast | E2B MLX 4-bit | MLX 0.31.2 | **126 tok/s** | 2.6 GB | **0.27s** classify |
| Primary | E4B MLX 4-bit | MLX 0.31.2 | **32 tok/s** | 4.3 GB | **1.57s** summarize |
| Heavy | 26B-A4B GGUF | llama.cpp CPU | **1.6 tok/s** | ~4 GB resident | 30-60s complex |

> **Fast and Primary tiers are 6-9× faster than the original plan's targets** thanks to MLX's native Metal GPU path on Apple Silicon. The heavy tier runs CPU-only on 16 GB due to the Metal memory ceiling. See [Architecture Deep Dive](#architecture-deep-dive) and [Hardware Notes](#hardware-notes).

---

## Architecture Deep Dive

### Why Apple Silicon Is Different: The Unified Memory Model

Every conventional AI inference setup involves a discrete GPU with its own VRAM, separated from system RAM by a PCIe bus. When a CUDA kernel needs model weights it copies them over that bus — a hard ceiling of ~64 GB/s on PCIe 4.0 x16. Apple M4's unified memory architecture eliminates this entirely:

```
Conventional GPU Server:               Apple M4 Mac Mini:
┌──────────────────────┐               ┌────────────────────────────────────┐
│ System RAM (DDR5)    │               │  Unified Memory (LPDDR5X 120 GB/s) │
│  128 GB  @ 50 GB/s   │               │                                    │
└──────────┬───────────┘               │  ┌──────────┐   ┌──────────────┐  │
           │ PCIe 4.0 x16              │  │ CPU Dies │   │  GPU (10-c.) │  │
           │ ~64 GB/s ceiling          │  │ (4P + 6E)│   │  Metal 3     │  │
           ▼                           │  └────┬─────┘   └──────┬───────┘  │
┌──────────────────────┐               │       │                 │          │
│ VRAM (GDDR6X)        │               │       └────────┬────────┘          │
│  24 GB  @ 1 TB/s     │               │                │                   │
└──────────────────────┘               │    Fabric bus (all cores share)    │
                                       └────────────────────────────────────┘
```

The M4's CPU, GPU, Neural Engine, and media engine all read from and write to the **same physical DRAM chips**. No copy. No bus bottleneck. Model weights loaded by the CPU are immediately accessible to the GPU at full memory bandwidth — **120 GB/s** vs PCIe's 64 GB/s ceiling. LLM inference is overwhelmingly memory-bandwidth-bound (not compute-bound), so this is the dominant hardware advantage.

---

### Tier 1 — Fast: MLX E2B (`mlx-community/gemma-4-e2b-it-4bit`)

**2.3B active parameters, 4-bit quantized, served on port 8082 by `mlx_lm.server`.**

```
Safetensors weights → MLX array (unified memory)
                                │
                      MLX lazy computation graph
                                │
                    Metal kernel (MSL) compilation
                                │
              GPU executes attention + FFN in-place
               (same memory pool, zero data copy)
                                │
                       Token sampled → output
```

MLX represents model weights as arrays in unified memory. When a generation call arrives, MLX builds a lazy computation graph and dispatches Metal kernels that read directly from those arrays — no staging buffer, no host-to-device transfer. 4-bit quantization packs two weights into one byte, halving the effective memory bandwidth requirement. The dequantization kernel runs on-GPU, so throughput is limited only by raw memory bandwidth, not instruction throughput.

**Why it's fast:** 2.3B active params × 4-bit ≈ 1.15 GB working set — trivially inside the 12.7 GB Metal budget with substantial headroom for KV cache growth.

---

### Tier 2 — Primary: MLX E4B (`mlx-community/gemma-4-e4b-it-4bit`)

Same execution path as E2B. 8.0B total parameters, 4.5B active per token, ~2.25 GB working set at 4-bit. Runs **simultaneously with E2B** — together they consume ~3.5 GB, comfortably under the Metal limit, leaving the heavy tier able to operate in CPU mode without evicting either.

---

### Tier 3 — Heavy: llama.cpp 26B-A4B + mmap

This is where hardware physics imposes itself. The 26B-A4B GGUF weighs ~16 GB. The Metal `recommendedMaxWorkingSetSize` on a 16 GB M4 is **12.7 GB** (Apple reserves ~20% for OS, display, and other processes). GPU acceleration cannot fit the model. The solution is `mmap` with CPU inference:

```
SSD (NVMe ~7.5 GB/s read)
        │
        │  OS page cache (kernel-managed)
        ▼
Virtual address space  ←── llama-server process
        │                   "sees" all 16 GB via mmap
        │
        │  Physical RAM — only hot pages reside here:
        ▼
  Active expert weights   ≈ 4.0 GB resident
  KV cache                ≈ 0.5 GB
  Activations/overhead    ≈ 0.3 GB
  ─────────────────────────────────
  Total physical RAM use  ≈ 4.8 GB
  Remaining for OS+tiers  ≈ 11 GB
```

`--mmap` instructs llama-server to use `mmap(2)` instead of `malloc` + `fread`. The OS maps the GGUF file's virtual pages into the process address space. When llama-server accesses an expert weight block not in physical RAM, the CPU takes a **page fault** — execution pauses, the OS reads the 4 KB page from SSD at ~7.5 GB/s, and resumes. Recently-used pages stay cached automatically.

**The MoE architecture is what makes this viable.** A dense 26B model needs all 26B parameters resident per forward pass. This model's Mixture-of-Experts design activates only **4 of N expert sub-networks per token** — the hot working set is ~4B active params (~2 GB at Q4), not 26B. Inactive expert blocks are cold pages that live on SSD until called.

---

### The Auto-Routing Gateway

```
Incoming request
        │
        ▼
  classify_request()     ← keyword scan, no model call
        │
        ├─ "hello", "thanks", "what is", greeting → Tier 1 Fast
        ├─ "summarize", "compress", "tldr", "digest" → Tier 2 Primary
        └─ all other (complex, long, code, analysis) → Tier 1 Fast (default)
                │                                      (escalate to Heavy if forced)
                ▼
        POST to selected tier's /v1/chat/completions
                │
                ▼
        Unified response + _routing metadata:
        {"tier": "fast", "latency_ms": 270}
```

The gateway also handles the Ollama/llama.cpp health check difference: Ollama exposes `/api/tags` (not `/health`). The gateway tries `/health` first and falls back to `/api/tags`, which is why both backends report `"ok"` through the same `/health` endpoint.

---

## Advantages

### 1. Privacy by Design — Structurally Air-Gapped
Every token stays on-device. For use cases involving proprietary code, legal documents, medical records, or personal data, local inference is non-negotiable. You cannot accidentally expose sensitive context to a third-party API. GDPR, HIPAA, attorney-client privilege — all satisfied structurally rather than contractually.

### 2. True Zero Marginal Cost After Break-Even
At $1.50/hr for an A100 spot instance running 8 hrs/day: $360/month. The $599 Mac Mini breaks even in ~50 days. After that, inference is effectively free — roughly $5-8/month in electricity. At 200 requests/day × 500 tokens each, you'd pay $60-120/month on GPT-4o-mini. On this stack: $0 ongoing.

### 3. Apple Silicon Memory Bandwidth Advantage
120 GB/s unified memory bandwidth vs ~50 GB/s for typical server DDR5. For small models (E2B/E4B), you're getting bandwidth that rivals mid-range GPU cards at a fraction of the cost and power draw. This is why 126 tok/s on E2B is achievable on a $599 machine.

### 4. Native Multimodal on All Tiers
Gemma 4 supports text + vision + audio (E2B/E4B) and text + vision (26B). Most $599 self-hosted setups give you text-only. Image and audio understanding on a machine this size is genuinely unusual.

### 5. OpenAI-Compatible API Surface
Any tool built against OpenAI's API — LangChain, LlamaIndex, Continue.dev, Cursor, Aider, Open WebUI, n8n — works against this stack with one base URL change. Zero migration cost.

### 6. Tailscale Mesh — Zero Infrastructure Overhead
No port forwarding, no dynamic DNS, no cloud proxy, no VPN server to maintain. Anyone you add to your Tailscale network hits `http://100.75.223.113:8080` from anywhere in the world. WireGuard-based, cryptographically authenticated, zero open ports on your router.

### 7. MoE Efficiency Is Unusually Well-Matched to Constrained Memory
A dense 26B model on 16 GB would not run at all. The MoE architecture's small active-parameter footprint is what makes mmap viable — 26B-class reasoning ability on a 4B-class RAM budget per token.

---

## Disadvantages

### 1. The 16 GB Hard Ceiling — The Dominant Constraint
The `recommendedMaxWorkingSetSize` of 12.7 GB means GPU acceleration stops for models above ~10 GB at 4-bit. The heavy tier runs at **1.6 tok/s** instead of a potential 15-25 tok/s on 24 GB. This is a 10-15× performance gap that no software change can close — it is a physics constraint imposed by the memory subsystem. The heavy tier is viable for background/async tasks, not interactive streaming.

### 2. Effective Single-Tenancy
The fast and primary tiers handle modest concurrency (MLX queues requests). The heavy tier cannot: `--parallel 1` with CPU inference means a second request queues and waits several minutes while the first completes. This setup serves one person well, a small team tolerably for non-interactive workloads, and real concurrent multi-user demand poorly.

### 3. Docker ≠ MLX
MLX requires the Metal framework, which requires native macOS on Apple Silicon. Docker containers use Ollama (llama.cpp backend) instead. You lose 3-8× inference speed in containers. The Docker path is for cross-platform portability, not performance.

### 4. Sustained CPU Inference Thermals
When the heavy tier runs CPU-only inference, all 10 cores peg at 100%. On the Mac Mini's passive cooling, sustained inference beyond ~10 minutes triggers thermal throttling, dropping tok/s from ~1.59 to ~0.9. The machine stays functional; inference degrades nonlinearly.

### 5. SSD Wear from mmap Paging
Heavy mmap usage reads weight pages from SSD on cache misses. At ~400 KB read per token × 50,000 tokens/day → ~20 GB/day of SSD reads → the M4's 600 TBW endurance would take ~82 years to exhaust at this rate. Not a real concern in practice, but it means the SSD is active during inference, which can marginally slow other I/O.

### 6. No Fine-Tuning
This stack is inference-only. Training even a small LoRA adapter on E2B at 4-bit requires gradient buffers, optimizer state, and activation storage — easily 3-4× inference memory. You'd hit OOM on 16 GB for anything beyond a toy experiment.

### 7. No Redundancy or Failover
Single machine, single point of failure. No load balancing, no hot standby. If the Mac Mini goes down, all tiers go down. Acceptable for personal use; not acceptable for any production SLA.

---

## Cost Effectiveness

### Hardware Break-Even vs Cloud

| Setup | Cost | 26B Speed | Break-Even vs A100 Spot |
|-------|------|-----------|------------------------|
| **Mac Mini M4 16GB** | **$599** | 1.6 tok/s CPU | **~50 days** |
| Mac Mini M4 Pro 24GB | $1,399 | ~15-20 tok/s GPU | ~116 days |
| Mac Studio M4 Max 96GB | $3,999 | ~50-70 tok/s GPU | ~333 days |
| Mac Studio M4 Ultra 192GB | $9,999 | ~80-110 tok/s GPU | ~833 days |
| Cloud A100 (spot, $1.50/hr) | ongoing | 80-120 tok/s | never |
| Cloud H100 (spot, $2.50/hr) | ongoing | 150-250 tok/s | never |

`Break-even = hardware cost ÷ ($1.50/hr × 8 hrs/day × 30 days/month)`
After break-even: **~$5-8/month electricity for unlimited private inference.**

---

## Upgrade Path & Funding Tiers

> All prices reflect **Apple for Education** (~10-15% off retail) or **Apple Business Program** estimates. Retail prices in parentheses. Apple Financial Services leasing options noted where relevant.

---

### 🔵 Tier 0 — What This Repo Runs On: ~$599
**Mac Mini M4 | 16 GB | 10-core CPU | 10-core GPU**

| Capability | Status |
|---|---|
| E2B fast tier (GPU, MLX) | ✅ 126 tok/s |
| E4B primary tier (GPU, MLX) | ✅ 32 tok/s |
| 26B-A4B heavy tier | ⚠️ 1.6 tok/s CPU-only |
| All tiers simultaneously | ✅ (heavy runs CPU alongside GPU tiers) |
| Concurrent users | ~1 effectively |
| Largest GPU model | ~8B |
| Fine-tuning | ❌ |
| Vision | ✅ |
| Audio | ✅ (E2B/E4B only) |

**Good for:** Personal productivity, private document analysis, solo developer copilot, triage/routing pipelines, learning and experimentation.

---

### 🟢 Tier 1 — Critical Upgrade: ~$1,200–1,700
**Mac Mini M4 Pro | 24 GB | 12-core CPU | 20-core GPU**
*(Retail ~$1,399 → Edu ~$1,219)*

**The most important upgrade threshold in this entire stack.**

The Metal `recommendedMaxWorkingSetSize` on 24 GB is **~19 GB**. The 16 GB GGUF fits with 3 GB to spare for KV cache and activations. `--gpu-layers 99` now works. This is the boundary where the heavy tier goes from "background task" to "interactive use."

| Metric | Tier 0 (16 GB) | Tier 1 (24 GB) | Delta |
|---|---|---|---|
| 26B generation speed | 1.6 tok/s (CPU) | ~15-20 tok/s (GPU) | **+10-12×** |
| All tiers GPU-accelerated simultaneously | ❌ | ✅ | New capability |
| Concurrent users | ~1 | ~3-5 | +3-4 |
| Largest GPU model | ~8B | ~13B | +5B |

**What becomes possible:**
- Heavy tier is now viable for interactive streaming (~15 tok/s renders in real time)
- All three tiers resident in Metal simultaneously, no stopping/starting
- Can serve a small team of 3-5 concurrent users
- 70B models at extreme quantization (Q2_K) become marginal

**Incremental cost over Tier 0:** ~$620 edu. The single highest return-on-investment upgrade available in this architecture.

---

### 🟡 Tier 2 — Serious Capability: ~$3,500–5,500

Three options at this price point depending on use case:

| Machine | RAM | GPU Cores | Bandwidth | Edu Price | Best For |
|---|---|---|---|---|---|
| Mac Mini M4 Pro 48 GB | 48 GB | 20-core | 120 GB/s | ~$1,659 | Cost-per-GB efficiency |
| Mac Studio M4 Max 36 GB | 36 GB | 40-core | 410 GB/s | ~$1,749 | Speed + headroom |
| Mac Studio M4 Max 96 GB | 96 GB | 40-core | 410 GB/s | ~$3,499 | Large models + concurrency |

**The M4 Max's memory bandwidth jump: 120 GB/s → 410 GB/s (3.4×).** Since LLM generation is memory-bandwidth-bound, this translates almost linearly into ~3× generation speed for the same model at the same quantization — without any software changes.

| Metric | Tier 1 (24 GB) | Tier 2 (96 GB M4 Max) | Delta |
|---|---|---|---|
| 26B tok/s | 15-20 | 50-70 | **+3-4×** |
| 70B models | ❌ | ✅ ~20-30 tok/s | New capability |
| Concurrent users | ~3-5 | ~10-20 | +15 |
| LoRA fine-tuning (E2B/E4B) | ❌ | ✅ | New capability |
| 128K context window | marginal | comfortable | Qualitative |

**What becomes possible:**
- 70B models (Llama 3.3 70B, Qwen 2.5 72B) at GPU speed
- LoRA fine-tuning on the smaller tiers — adapt models to your domain
- True multi-user concurrent inference for a team of 10-20
- Batch document processing (legal review, codebase analysis) at practical throughput
- Speculative decoding: run a small draft model + large verifier model simultaneously

---

### 🟠 Tier 3 — Production Grade: ~$8,500–10,000
**Mac Studio M4 Ultra | 192 GB | 32-core CPU | 80-core GPU | 800 GB/s**
*(Retail ~$9,999 → Edu ~$8,499, or Apple Financial Services leasing ~$350/mo)*

This is the inflection point where the constraint shifts from hardware memory to application design.

| Metric | Tier 2 (96 GB) | Tier 3 (192 GB Ultra) | Delta |
|---|---|---|---|
| 26B tok/s | 50-70 | 80-110 | +60% |
| 70B tok/s | 20-30 | 55-80 | **+2.5×** |
| 405B class (Q4 ~200 GB) | ❌ | ✅ ~8-12 tok/s | New capability |
| Concurrent users | 10-20 | 30-60 | +40 |
| Full 70B LoRA fine-tuning | ❌ | ✅ | New capability |
| All Gemma 4 models simultaneously | ❌ | ✅ | New capability |

**What becomes possible:**
- Full Gemma 4 family (E2B + E4B + 26B + 27B dense) all resident simultaneously — no eviction, no tier conflicts
- Llama 3.1 405B at Q4 (~200 GB) barely fits — genuinely frontier-scale open-source inference on a single machine
- Full LoRA fine-tuning on 70B class models for domain adaptation
- Multimodal batch pipelines: process hundreds of images or audio files per hour
- Production SLA for a team of 30-60 concurrent users
- Speculative decoding at full scale

> **Note:** The M4 Ultra is two M4 Max dies connected via Apple's die-to-die interconnect at 32 GB/s. The OS presents a single 192 GB flat address space. MLX and llama.cpp both see one pool — no distributed programming required.

---

### 🔴 Tier 4 — Research Cluster: ~$20,000–40,000
**2-4× Mac Studio M4 Ultra + 10 GbE switch + NAS**
*(2× Ultra ~$17,000 edu + switch/NAS ~$3,000-8,000)*

At this tier, the architecture changes from a single machine to a small cluster. Orchestration becomes the design problem:

```
                    ┌─────────────────────────────┐
                    │   10 GbE / 25 GbE Switch     │
                    └──┬───────┬───────┬───────────┘
                       │       │       │
              ┌────────┴──┐ ┌──┴────┐ ┌┴──────────┐
              │ Ultra #1   │ │Ultra#2│ │ NAS (RAID)│
              │ 192 GB     │ │192 GB │ │  ~50 TB   │
              │ Tiers 1-2  │ │Tier 3 │ │ Model Zoo │
              └────────────┘ └───────┘ └───────────┘
                       │
              ┌────────┴───────────┐
              │  Load balancer     │
              │  (gateway.py ×2)   │
              └────────────────────┘
```

**What becomes possible:**
- **Model parallelism:** Run a single 400B+ model split across two machines using llama.cpp's RPC backend or MLX-LM distributed mode
- **A/B testing infrastructure:** Route 50% of traffic to each machine for model comparison
- **Simultaneous fine-tuning + production inference:** Dedicate one machine to training while the other serves traffic
- **High availability failover:** If one machine goes down, the gateway reroutes — no downtime
- **NAS model zoo:** Keep all model variants on shared storage, hot-swap without re-downloading
- **100+ concurrent users** with proper load balancing

---

### 🟣 Tier 5 — AI Lab: ~$60,000–100,000
**4-6× Mac Pro M4 Ultra + ProRes infrastructure + high-speed networking**
*(Mac Pro M4 Ultra ~$12,000-16,000 edu each)*

At this scale you're building a genuine **internal AI research lab**:

- Full pre-training runs on small models (1-3B) — not just fine-tuning
- Custom tokenizer training and evaluation harness infrastructure running continuously
- Synthetic data generation pipeline (use cheap models to generate training data for expensive ones)
- Multi-modal training: vision encoder alignment, audio integration
- Dedicated embedding and re-ranking infrastructure for enterprise-scale RAG
- CI/CD for model versions: train → eval → deploy without human intervention
- Integration with Final Cut Pro, Logic Pro, and Core ML for creative/multimodal research

The Apple ecosystem compounds here: **Final Cut Pro, Logic Pro, Core ML, Create ML** all run natively on the same hardware. Video understanding, audio transcription, and multimodal AI pipelines share the same machines — no GPU time-slicing between creative and AI workloads.

---

### Upgrade Summary

| Tier | Budget | Hardware | Key Unlock | Heavy Tier Speed | Concurrent Users |
|------|--------|----------|------------|-----------------|-----------------|
| 0 | $599 | Mac Mini M4 16GB | Foundation, private AI | 1.6 tok/s (CPU) | ~1 |
| 1 | ~$1,700 | Mac Mini M4 Pro 24GB | **26B goes GPU — 10× speedup** | 15-20 tok/s | 3-5 |
| 2 | ~$3,500-5,500 | Mac Studio M4 Max 96GB | **70B models + LoRA fine-tuning** | 50-70 tok/s | 10-20 |
| 3 | ~$8,500 | Mac Studio M4 Ultra 192GB | **405B class + true production** | 80-110 tok/s | 30-60 |
| 4 | ~$20,000-40,000 | 2-4× Ultra + NAS | **Cluster, failover, model parallel** | 160-220 tok/s | 100+ |
| 5 | ~$60,000-100,000 | Mac Pro cluster | **Pre-training, full research lab** | training-scale | org |

> **The single highest-leverage upgrade is Tier 0 → Tier 1 (16 GB → 24 GB).** The 10-15× improvement in heavy tier throughput is the largest performance-per-dollar gain available anywhere in this stack. Everything above it is incremental. The 24 GB threshold is where the model physically fits in Metal's working set — a qualitative architectural shift, not just a speed improvement.

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

# 3. Start always-on tiers (models auto-download on first start, ~17 GB)
cd docker
docker compose up -d fast primary gateway

# 4. Watch model download progress
docker compose logs -f fast

# 5. Verify (wait ~5-15 minutes for model downloads on first run)
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

# Native macOS — start heavy tier
bash scripts/start_heavy.sh

# Docker — mount models dir and start
docker compose --profile heavy up -d heavy
```

---

## Tailscale Mesh Access

Share the stack with any device on your Tailscale network:

### macOS Setup
```bash
# Install official Tailscale (Homebrew CLI alone is not enough — needs the .pkg)
# Download: https://pkgs.tailscale.com/stable/#macos
# Install .pkg → approve Network Extension in System Settings

bash scripts/setup_tailscale.sh
# Prints all access URLs once connected
```

### Use from Any Device
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<your-tailscale-ip>:8080",
    api_key="not-required"
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Summarize this PR diff: ..."}]
)
print(response.choices[0].message.content)
print(response._routing)  # {"tier": "primary", "latency_ms": 1570}
```

---

## API Reference

All endpoints on the gateway at `:8080`.

### `POST /v1/chat/completions` — OpenAI-compatible
```json
{
  "messages": [{"role": "user", "content": "your message"}],
  "max_tokens": 512,
  "tier": "fast"
}
```
> `"tier"` is optional — omit to auto-route. Values: `"fast"` | `"primary"` | `"heavy"`

Response includes routing metadata:
```json
{
  "choices": [...],
  "_routing": {"tier": "primary", "latency_ms": 1640}
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
→ `{"tiers": {"fast": "ok", "primary": "ok", "heavy": "offline"}}`

### `GET /metrics`
→ Per-tier request counts and p50/p95 latency histograms

---

## Production Use Cases

### Tier 1 — Fast (0.27s, 126 tok/s) — Always On
| Use Case | Integration |
|----------|------------|
| Slack/Discord message triage | Webhook → `POST /classify` |
| Email ticket routing | Per-email → classify to queue |
| Git pre-commit hook | Reject vague commit messages in <0.5s |
| Intent detection | Classify voice transcription before routing |
| Content moderation | Gate all submissions before storage |

### Tier 2 — Primary (1.57s, 32 tok/s) — Always On
| Use Case | Integration |
|----------|------------|
| PR diff summarizer | GitHub webhook → auto-summary on every PR |
| Meeting transcript digest | Daily cron → compress transcripts |
| Vector DB chunking | Pre-ingest pipeline → chunk + summarize |
| Standup aggregator | Team async updates → single digest |
| Release notes drafts | CI trigger → changelog from git log |

### Tier 3 — Heavy (30-60s, 1.6 tok/s CPU) — On-Demand
| Use Case | Integration |
|----------|------------|
| Deep code review | Manual trigger or PR open webhook |
| Technical documentation | Post-deploy → draft API docs |
| Incident postmortem | Feed structured data → full postmortem |
| Architecture analysis | On request → trade-off analysis |
| Complex SQL generation | Escalate when Tier 2 is insufficient |

---

## Hardware Notes

### Why 16 GB Is Enough for Tiers 1 & 2
MLX runs E2B and E4B natively on the M4 GPU with 4-bit quantization, using only 2.6 GB and 4.3 GB respectively. Total footprint for both tiers + macOS ≈ 10 GB, leaving 6 GB headroom.

### Why 16 GB Limits the 26B Heavy Tier
The 26B-A4B GGUF is 16 GB on disk. When llama.cpp tries to offload all layers to Metal, the combined weight buffers + KV cache + activations exceed the M4's `recommendedMaxWorkingSetSize` of 12.7 GB. Result: `kIOGPUCommandBufferCallbackErrorOutOfMemory`.

**Solution used:** `--gpu-layers 0` — CPU-only via Apple Accelerate/BLAS. Reliable at ~1.6 tok/s.

**For GPU-accelerated 26B:** Requires 24 GB+ unified memory (Mac Mini M4 Pro ≥$1,399).

### mmap Explained
`mmap` maps a file into the process's virtual address space without reading it into RAM upfront. The OS loads only the pages actively accessed. For a MoE model:
- Total weights: 26B (16 GB on disk)
- Active per token: 4B (only selected experts fire)
- RAM resident: ~4-5 GB at any time
- Rest: on SSD, fetched on demand at ~7.5 GB/s when page-faulted

**Advantages:** No OOM crashes; zero-copy load (maps in seconds); OS handles hot/cold pages automatically; graceful performance degradation instead of cliff-edge failure.

**Drawbacks:** SSD bandwidth becomes the throughput floor on cache misses; latency variance per token (2-10× depending on what's cached); marginal SSD wear; can cause system-wide I/O pressure under sustained load.

---

## Architecture: Native vs Docker

```
Native macOS (fast path):          Docker (portable path):
────────────────────────           ───────────────────────
E2B: mlx_lm server :8082           E2B: ollama :8082 (llama.cpp backend)
E4B: mlx_lm server :8083           E4B: ollama :8083 (llama.cpp backend)
26B: llama-server  :8081           26B: llama.cpp:server :8081
GW:  uvicorn/fastapi :8080         GW:  uvicorn/fastapi :8080

LaunchAgents auto-start E2B,       docker compose manages lifecycle
E4B, gateway on login.             Heavy tier uses --profile heavy
Metal GPU acceleration             CPU-only (MLX = macOS native only)
126 tok/s E2B                      ~18-25 tok/s E2B
```

---

## File Structure

```
gemma4-stack/
├── README.md
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
| Metal `Insufficient Memory` on 26B | Use `start_heavy.sh cpu` — CPU-only mode bypasses Metal |
| Tailscale daemon not running | Install official .pkg from pkgs.tailscale.com (Homebrew CLI alone is not enough) |
| Duplicate MLX server processes | `pkill -f "mlx_lm server"` then restart |
| llama-server `--n-parallel` error | Use `--parallel` (correct long-form flag) |
| Docker containers unhealthy | `curl` is not in the Ollama image — healthcheck uses `ollama list` in this repo |
| Docker models not downloading | `docker compose logs -f fast` — first pull is ~3.5 GB, takes 5-15 min |
| Heavy tier hangs indefinitely | GPU mode with 16 GB RAM — use `--gpu-layers 0` (already set in scripts) |

---

## Implementation Notes

See [`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md) for the full plan-vs-reality log including:
- Why MLX was chosen over Ollama for native tiers (5-8× faster, 55-63% less RAM)
- Why the 26B tier uses llama.cpp instead of MLX (Metal OOM on 16 GB)
- Why mlx-lm needed GitHub main instead of PyPI (Gemma 4 architecture not in 0.31.1)
- Why Tailscale requires the official .pkg on macOS (Network Extension)
- Why Docker healthchecks use `ollama list` instead of `curl` (curl not in image)
- Tailscale live deployment: `100.75.223.113`, confirmed working 2026-04-04

---

## License

Models: Apache 2.0 (Google Gemma 4, Unsloth quantizations)
Code: Apache 2.0
