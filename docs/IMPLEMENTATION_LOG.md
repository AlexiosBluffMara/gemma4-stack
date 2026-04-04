# Gemma 4 Local Inference Stack — Implementation Log

**Machine:** Mac Mini M4 (Mac16,10) — MU9D3LL/A
**Date Started:** 2026-04-04
**Hardware:** Apple M4 (10-core CPU), 16 GB Unified Memory, 228 GB SSD (176 GB free)
**OS:** macOS 26.3.1 (Tahoe) Build 25D2128

---

## Pre-Implementation System Audit

### What the Plan Assumed
| Component | Plan Spec | Notes |
|-----------|-----------|-------|
| Chip | Apple M4 | — |
| RAM | 16 GB | — |
| SSD | 256 GB | — |
| CPU Cores | 8 | Referenced `--threads 8` |
| Python | 3.11+ | — |
| macOS | Not specified | — |

### What We Actually Found
| Component | Actual | Delta |
|-----------|--------|-------|
| Chip | Apple M4 | Exact match |
| RAM | 16 GB | Exact match |
| SSD | 228 GB formatted, 176 GB free | 176 GB free >> 50 GB needed |
| CPU Cores | 10 | +2 over plan (will use `--threads 10`) |
| Python | 3.9.6 (system only) | Needs 3.12 via Homebrew |
| macOS | 26.3.1 Tahoe | Newer than plan assumed |
| Homebrew | Not installed | Fresh install needed |
| Ollama | Not installed | Fresh install needed |
| llama.cpp | Not installed | Fresh install needed |
| Tailscale | Not installed | Not in original plan; adding for mesh |

### Key Deviations from Plan
1. **10 cores, not 8:** We'll use `--threads 10` instead of `--threads 8` for llama-server. This gives ~25% more parallelism for the heavy tier.
2. **Python 3.9.6 too old:** System Python is 3.9.6. Plan requires 3.11+. Installing Python 3.12 via Homebrew.
3. **Tailscale added:** Original plan had no network sharing. We're adding Tailscale for mesh access.
4. **Interactive sudo required:** Homebrew installer can't run non-interactively. Created `phase1_setup.sh` for user to run in Terminal.

---

## Phase 1: Environment Preparation

### Planned Commands vs Actual

| Step | Plan | Actual | Reason for Change |
|------|------|--------|-------------------|
| Homebrew install | Direct curl pipe | Wrapped in `phase1_setup.sh` | Needs interactive sudo |
| brew install | `ollama llama.cpp cmake git python3` | `ollama llama.cpp cmake git python@3.12` | Explicit Python 3.12, system python3 is 3.9.6 |
| pip install | `pip3 install ...` | `python3.12 -m pip install ...` | Target correct Python version |
| Env vars | Separate echo commands | Single heredoc block with idempotency check | Cleaner, won't duplicate on re-run |

### Status: WAITING — User must run `~/ai-scripts/phase1_setup.sh` in Terminal

---

## Cost-Effectiveness Analysis

### Hardware Cost
| Setup | Cost | Performance (tok/s on 26B) | $/tok/s |
|-------|------|---------------------------|---------|
| **Mac Mini M4 16GB** | **$599** | **8-17** | **$35-75** |
| Mac Mini M4 Pro 24GB | $1,399 | 15-25 | $56-93 |
| Mac Studio M4 Max 64GB | $1,999 | 30-50 | $40-67 |
| Mac Studio M4 Ultra 192GB | $6,999 | 60-100 | $70-117 |
| MacBook Pro M4 Max 36GB | $3,499 | 25-40 | $87-140 |
| Cloud GPU (A100 spot) | ~$1.50/hr ongoing | 80-120 | Ongoing cost |
| Cloud GPU (H100 spot) | ~$2.50/hr ongoing | 150-250 | Ongoing cost |

### Break-Even Analysis (vs Cloud)
- Cloud A100 spot at $1.50/hr for 8hrs/day = $360/month
- Mac Mini M4 at $599 one-time = **breaks even in ~50 days**
- After break-even: effectively free inference forever (minus ~$5/mo electricity)

### The 16GB Constraint — mmap to the Rescue
The 26B-A4B model weights are ~16-18 GB in Q4 quantization. With only 16 GB of unified memory, loading the entire model into RAM is impossible. This is where **mmap (memory-mapped files)** becomes the critical enabler.

#### How mmap Works for LLM Inference
1. The OS maps the GGUF file directly into virtual address space
2. Only the actively-needed pages are loaded into physical RAM
3. Unused pages are evicted back to SSD transparently
4. The M4's SSD (~7.5 GB/s read) acts as overflow memory

#### Advantages
- **No OOM crashes:** The model "fits" in virtual memory even if physical RAM is insufficient
- **Zero-copy loading:** Model loads in seconds (mapping, not copying)
- **Automatic paging:** OS handles hot/cold page management — no manual memory tuning
- **Graceful degradation:** Performance scales smoothly with available RAM rather than cliff-edge failures

#### Drawbacks
- **SSD bandwidth bottleneck:** When tokens need weights from evicted pages, inference stalls while SSD fetches them. This is the primary source of variance in tok/s.
- **SSD wear:** Continuous paging reads from SSD. NVMe endurance is high (~600 TBW for M4's SSD) but it's non-zero wear.
- **Unpredictable latency:** Individual token generation time varies 2-10x depending on which expert weights are resident in RAM vs paged out. MoE models like 26B-A4B mitigate this — only 4B params active per token, so the working set is much smaller than 26B.
- **System-wide impact:** When the heavy model is paging heavily, other apps may experience slowdowns as the OS juggles pages.

#### Why MoE + mmap is the Sweet Spot for 16GB
The Gemma 4 26B-A4B is a Mixture-of-Experts model. "26B" total params, but only "A4B" (4 billion) are active per forward pass. This means:
- The active working set is ~4 GB, not 26 GB
- Only the router + active experts need to be in RAM at any given time
- Inactive expert weights can live on SSD without penalty
- This is why 8-17 tok/s is achievable on 16 GB — the effective memory pressure is far lower than a dense 26B model

---

## Model Verification (Pre-Download)

All models verified available on Hugging Face as of 2026-04-04:

| Model | Repo | Total Params | Effective Params | Downloads | License | Modalities |
|-------|------|-------------|-----------------|-----------|---------|------------|
| E2B | google/gemma-4-E2B-it | 5.1B | 2.3B | 90.2K | Apache 2.0 | Text, Vision, Audio |
| E4B | google/gemma-4-E4B-it | 8.0B | 4.5B | 108.3K | Apache 2.0 | Text, Vision, Audio |
| 26B-A4B | unsloth/gemma-4-26B-A4B-it-GGUF | 26B | 4B | 301.3K | Apache 2.0 | Text, Vision |

**Note:** E2B and E4B are `any-to-any` models (vision + audio). The 26B GGUF is `image-text-to-text` (vision only, no audio). This means audio processing must route through Tier 1 or 2, never Tier 3.

---

## Tailscale Mesh Network Extension (Phase 7 — Added)

### Purpose
Expose the inference stack to other devices on a Tailscale mesh network, enabling:
- Team members to query the models from any device
- Integration with remote automation pipelines
- Secure access without port forwarding or public exposure

### Architecture
```
[Remote Device] --Tailscale--> [Mac Mini M4 100.x.x.x]
                                  |
                                  +-- :11434 Ollama (E2B/E4B)
                                  +-- :8081 llama-server (26B)
                                  +-- :8080 API Gateway (router)
```
