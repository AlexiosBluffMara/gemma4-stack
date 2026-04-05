# Gemma 4 Stack — Architecture Reference

**Last updated:** 2026-04-04
**Stack version:** 2.0.0
**Python:** 3.14.3 (gateway-venv) / 3.14 (Homebrew)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Map](#component-map)
3. [Gateway Deep Dive](#gateway-deep-dive)
4. [Media Processing Pipeline](#media-processing-pipeline)
5. [Cloud Proxy Architecture](#cloud-proxy-architecture)
6. [Network Topology](#network-topology)
7. [Security Model](#security-model)
8. [Reliability Patterns](#reliability-patterns)
9. [Performance Characteristics](#performance-characteristics)
10. [Data Flow Diagrams](#data-flow-diagrams)

---

## System Overview

The Gemma 4 inference network is a three-tier private AI serving system built on Apple Silicon. It exposes an OpenAI-compatible HTTP API both locally (via Tailscale mesh) and publicly (via Google Cloud Run). All model inference happens on-device using MLX — Apple's machine learning framework optimised for Apple Silicon's unified memory architecture.

### Design Principles

1. **On-device inference** — model weights never leave the hardware; no data is sent to cloud AI services
2. **Tier-aware routing** — requests are classified and routed to the smallest capable model, minimising latency and memory pressure
3. **Graceful degradation** — circuit breakers and fallback logic ensure the system remains functional when devices go offline
4. **Streaming-first** — all media uploads are streamed to disk; no large payloads are buffered in process memory
5. **Zero-copy where possible** — the Cloud Run proxy streams request bodies directly to the gateway without buffering

---

## Component Map

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  CLIENTS                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐     │
│  │ Browser  │  │ iPhone   │  │ API CLI  │  │ Python SDK (local_llm.py)│     │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────┬───────────────┘     │
└───────┼─────────────┼─────────────┼─────────────────────┼────────────────────┘
        │ HTTPS       │ HTTPS       │ HTTPS               │ HTTP (local)
        ▼             ▼             ▼                     ▼
┌──────────────────────────────────────┐   ┌─────────────────────────────────┐
│  GCP Cloud Run (us-central1)         │   │  Mac Mini :8080 (Tailscale LAN) │
│  ┌────────────────────────────────┐  │   │  FastAPI Gateway                │
│  │  gemma4-proxy (FastAPI)        │  │   └─────────────────────────────────┘
│  │  - Firebase JWT validation     │  │
│  │  - Rate limiting (per-IP)      │  │
│  │  - Circuit breaker             │  │
│  │  - Streaming media proxy       │  │
│  └──────────────┬─────────────────┘  │
│                 │ HTTP via SOCKS5     │
│  ┌──────────────▼─────────────────┐  │
│  │  Tailscale sidecar             │  │
│  │  (WireGuard userspace)         │  │
│  └──────────────┬─────────────────┘  │
└─────────────────┼────────────────────┘
                  │ WireGuard tunnel (encrypted)
                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Mac Mini M4 16GB (Tailscale IP: 100.75.223.113)                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Gateway (scripts/gateway.py)  :8080                                │    │
│  │  - Request classification (fast tier)                               │    │
│  │  - Tier routing + fallback                                          │    │
│  │  - Circuit breaker per tier                                         │    │
│  │  - Media processing (media.py)                                      │    │
│  │  - Metrics collection                                               │    │
│  │  - Background health checker (30s interval)                        │    │
│  └──────────┬──────────────────┬────────────────────┬─────────────────┘    │
│             │                  │                    │                       │
│             ▼                  ▼                    ▼                       │
│  ┌──────────────┐  ┌─────────────────┐  ┌────────────────────┐            │
│  │  E2B MLX     │  │  E4B MLX        │  │  26B-A4B MLX       │            │
│  │  :8082       │  │  :8083          │  │  :8084             │            │
│  │  126 tok/s   │  │  32 tok/s       │  │  (MacBook Pro)     │            │
│  │  2.6 GB RAM  │  │  4.3 GB RAM     │  │  50-70 tok/s       │            │
│  │  Fast tier   │  │  Primary tier   │  │  Heavy tier        │            │
│  └──────────────┘  └─────────────────┘  └──────────┬─────────┘            │
│                                                     │ Tailscale            │
└─────────────────────────────────────────────────────┼──────────────────────┘
                                                      │
                                          ┌───────────▼──────────────┐
                                          │  MacBook Pro M4 Max      │
                                          │  (when online)           │
                                          │  26B-A4B MLX :8084       │
                                          └──────────────────────────┘
```

---

## Gateway Deep Dive

The gateway (`scripts/gateway.py`) is a FastAPI application running under uvicorn with Python 3.14.3 inside an isolated virtual environment (`gateway-venv/`).

### Request Lifecycle

```
1. HTTP request arrives at :8080
2. Request ID assigned (UUID, added to all log entries)
3. Rate limiter check (sliding window, per client IP)
4. Route matching → appropriate handler
5. Handler executes (see below per endpoint)
6. Response returned with _routing metadata appended
7. Metrics recorded (latency, tier, success/failure)
```

### `/v1/chat/completions` Handler

```
parse body → extract messages, tier hint, max_tokens
     │
     ▼
classify_request(last_user_message)
     │  [calls E2B via httpx, ~270ms]
     │
     ├── If tier explicitly specified → use that tier
     │
     └── Route by classification:
              greeting/fyi    → fast
              question/request → primary
              idea/complex     → heavy (fallback to primary if offline)
     │
     ▼
_select_tier_with_fallback(chosen_tier)
     │  [checks circuit breaker state]
     │  [if heavy OPEN and no explicit tier → fallback to primary]
     │
     ▼
_forward(client, url, model, messages, max_tokens, temperature)
     │  [httpx async POST to MLX server]
     │  [30s timeout text, 300s timeout media]
     │
     ▼
append _routing metadata to response
record latency in metrics histogram
return JSONResponse
```

### `/v1/media/upload` Handler

```
POST multipart/form-data arrives
     │
     ▼
rate_limiter.check_media(client_ip)  ← 20 req/min
     │
     ▼
stream upload to tempfile.mkstemp()  ← 4 MB chunks, no RAM buffering
     │  monitor total_bytes, reject if > 10 GB
     │
     ▼
tier_selection  ← BEFORE media processing (frame budget depends on tier)
     │
     ▼
media_processor.process_upload(path=tmp, filename, mime, tier, file_size)
     │  [reads 16-byte header for magic byte detection]
     │
     ├── image → read full bytes → _process_image() → ProcessedMedia
     │
     ├── audio → read full bytes → _process_audio() → ProcessedMedia
     │
     └── video → process_video_chunked(path, ...) →
                      │
                      ├── ≤60s → _process_video_from_path() → ProcessedMedia
                      │
                      └── >60s → extract frames per chunk → ChunkedVideoMedia
     │
     ├── isinstance(result, ChunkedVideoMedia) →
     │        _process_chunked_video():
     │             for each chunk: _forward(prompt + frames)
     │             aggregation: _forward(all_segment_analyses + original_prompt)
     │             return final_response
     │
     └── isinstance(result, ProcessedMedia) →
              _forward(prompt + content_parts)
              return direct response
     │
     finally:
          os.remove(tmp)           ← upload temp file
          os.remove(audio_wav)     ← result.temp_files
```

### Background Health Checker

A separate asyncio task runs in the gateway process and checks all tier endpoints every 30 seconds:

```python
# Simplified pseudo-code
async def _health_loop():
    while True:
        for tier, (url, model) in TIER_MAP.items():
            try:
                r = await client.get(url.replace("/v1/chat/completions", "/health"))
                _tier_status[tier] = r.status_code == 200
            except Exception:
                _tier_status[tier] = False
        await asyncio.sleep(30)
```

This is what enables the heavy tier to appear online automatically when the MacBook Pro starts its MLX server — no manual intervention needed.

### Metrics Collection

The gateway maintains an in-memory sliding histogram (last 1000 requests per tier):

```python
@dataclass
class TierMetrics:
    request_count: int = 0
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_request(self, latency_s: float):
        self.request_count += 1
        self.latencies.append(latency_s)

    def percentile(self, p: float) -> float:
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * p / 100)
        return sorted_l[idx] if sorted_l else 0.0
```

`GET /metrics` returns p50/p95/p99 computed from these histograms.

---

## Media Processing Pipeline

### Type Detection

Type detection always uses **magic bytes** from the first 16 bytes of the file, never the declared MIME type or filename. This prevents type confusion (e.g., renaming a shell script to `.jpg`).

```python
# Key magic byte signatures
b"\xff\xd8\xff"      → image/jpeg
b"\x89PNG\r\n\x1a\n" → image/png
b"RIFF....WEBP"      → image/webp
b"RIFF....WAVE"      → audio/wav
b"\x1a\x45\xdf\xa3"  → video/webm
data[4:8] == b"ftyp" → video/mp4 or video/quicktime (check bytes 8-12)
b"caff"              → audio/caf (iPhone native)
```

### Image Processing (`_process_image`)

```
Raw bytes (≤20 MB)
     │
Pillow Image.open()
     │
Convert to RGB (handles RGBA, palette, CMYK, L)
     │
Resize to fit within 1280×1280 (maintain aspect ratio, LANCZOS)
     │
Strip EXIF (security: GPS, device fingerprint, embedded thumbnails)
     → Create new Image, copy pixel data only
     │
Encode as JPEG (quality=85, optimize=True)
     │
base64-encode → content_part {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
     │
return ProcessedMedia
```

### Audio Processing (`_process_audio`)

```
Raw bytes (≤50 MB)
     │
Write to temp file
     │
ffprobe → duration
     │
if duration > 30s → will truncate (Gemma 4 audio limit: 750 tokens × 40ms = 30s)
     │
ffmpeg -ar 16000 -ac 1 -sample_fmt s16 -f wav [-t 30] → output.wav
     │
Write WAV to persistent temp file (gemma4_audio_<uuid>.wav)
     │  [persistent because mlx_vlm.load_audio() needs a file path]
     │
return ProcessedMedia(temp_files=[wav_path])
     │
gateway finally block: os.remove(wav_path)
```

### Video Processing — Short Path (`_process_video_from_path`)

```
File already on disk (path passed in)
     │
ffprobe → {duration_s, width, height, fps}
     │
if duration ≤ max_frames seconds → interval = 1.0 fps
else → interval = duration / max_frames  (evenly spread)
     │
for i in range(frame_count):
    timestamp = i * interval
    ffmpeg -ss timestamp -i path -vframes 1 -vf scale='min(768,iw)...' -q:v 3 → frame_N.jpg
     │
for each frame:
    Pillow: open, convert to RGB, resize ≤768px, JPEG re-encode (quality=80)
    base64-encode → content_part
     │
return ProcessedMedia(content_parts=[text_header, frame_0, frame_1, ...])
```

### Video Processing — Long Path (`process_video_chunked`)

```
File already on disk (path passed in, size known)
     │
_get_video_info(path) → {duration_s, width, height, fps}
     │
_compute_chunk_plan(duration_s, tier):
     │  chunk_dur = {fast: 30s, primary: 60s, heavy: 120s}
     │  n_frames  = {fast: 4,   primary: 8,   heavy: 16 }
     │  windows   = [(0, 60), (59, 120), (119, 180), ...]  ← 1s overlap
     │  if len(windows) > 30: subsample to 30 windows
     │
TemporaryDirectory(prefix="gemma4_chunks_")
     │
for each (start_s, end_s) window:
    _extract_frames_for_window(path, start_s, end_s, n_frames, tmpdir, i)
         │
         └── asyncio.gather(*[_extract_one_frame(...) for each timestamp])
                  │  [asyncio.Semaphore(4) limits concurrent ffmpeg processes]
                  │
              for each success:
                  Pillow: open, resize ≤768px, JPEG re-encode
                  base64-encode → b64_frame
         │
    append VideoChunk(index, total, start_s, end_s, frames, timestamps)
     │
return ChunkedVideoMedia(chunks, duration_s, tier, chunk_duration_s, frames_per_chunk)
```

### Multi-Pass Aggregation (gateway)

```python
async def _process_chunked_video(chunked, prompt, client, url, model, request_id, tier):
    chunk_analyses = []
    t0 = time.monotonic()

    for chunk in chunked.chunks:
        segment_prompt = (
            f"[Segment {chunk.index+1}/{chunk.total} "
            f"({chunk.start_s:.0f}s–{chunk.end_s:.0f}s of {chunked.duration_s:.0f}s)]\n"
            f"{prompt}\n"
            "Focus only on this time window. Be concise."
        )
        content = [{"type": "text", "text": segment_prompt}]
        for b64 in chunk.frames:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

        result = await _forward(client, url, model,
                                messages=[{"role": "user", "content": content}],
                                max_tokens=512, temperature=0.0)
        chunk_analyses.append(result["choices"][0]["message"]["content"])

    # Final aggregation pass
    agg_prompt = (
        f"Original request: {prompt}\n\n"
        + "\n\n".join(f"[{i+1}/{N}] {a}" for i, a in enumerate(chunk_analyses))
        + "\n\nProvide a unified analysis covering the entire video."
    )
    final = await _forward(client, url, model,
                           messages=[{"role": "user", "content": agg_prompt}],
                           max_tokens=2048, temperature=0.0)

    latency = time.monotonic() - t0
    return final, chunk_analyses, latency
```

---

## Cloud Proxy Architecture

### Request Flow (text)

```
Client → Cloud Run proxy
              │
              1. Middleware: request_id = uuid4()
              2. Middleware: validate Content-Type
              3. _check_auth(request) → verify Firebase JWT
              4. _check_rate_limit(ip, "text", 30/min)
              5. await circuit_breaker.allow_request()
              6. body = await request.body()   ← buffered (text is small)
              7. _proxy_upstream(POST, GATEWAY_URL/v1/..., body, headers)
                      │  retry up to 2× on 503/504/ConnectError
                      │  exponential backoff: 0.5s, 1.0s
                      │
              8. forward response to client
              9. circuit_breaker.record_success() or record_failure()
```

### Request Flow (media streaming)

```
Client → Cloud Run proxy
              │
              1-4. Same as text (auth, rate limit, circuit breaker check)
              │
              5. content_length = request.headers.get("content-length")
                 if content_length > 2 GB → return 413 immediately
              │
              6. async with _media_client.stream(
                      "POST", GATEWAY_URL/v1/media/upload,
                      content=request.stream(),   ← async generator, zero buffering
                      headers=forward_headers,
                 ) as upstream:
                      response_body = await upstream.aread()
              │
              7. return Response(response_body, status=upstream.status_code)
              │  [NO GCS background upload for streamed requests]
```

**Key difference from text:** Media bodies are **never loaded into process memory** in the proxy. The `request.stream()` async generator is piped directly into the httpx streaming request. The httpx `_media_client` has `write=3600s` timeout to support sustained multi-GB uploads.

### Circuit Breaker State Machine

```
Initial state: CLOSED (healthy)

CLOSED:
  - All requests allowed
  - failure_count tracked per request
  - On failure_count >= THRESHOLD (5):
      → transition to OPEN

OPEN:
  - All requests immediately rejected with 503
  - After RECOVERY_TIMEOUT (60s):
      → transition to HALF_OPEN

HALF_OPEN:
  - One probe request allowed
  - If probe succeeds:
      → transition to CLOSED, reset failure_count
  - If probe fails:
      → transition back to OPEN, reset timeout
```

---

## Network Topology

### Tailscale Mesh

```
tailnet: miniapple.ts.net (example)

Nodes:
  mac-mini         100.75.223.113   (always online)
  macbook-pro      100.x.x.x        (online when awake)
  gemma4-cloudrun  100.x.x.x        (Cloud Run sidecar)
  iphone           100.x.x.x        (when on Tailscale)

All traffic: WireGuard (UDP, encrypted, authenticated)
No port forwarding, no firewall rules, no public IPs exposed
```

### Port Map

| Host | Port | Service | Protocol |
|------|------|---------|---------|
| mac-mini | 8080 | FastAPI gateway | HTTP |
| mac-mini | 8082 | E2B MLX server | HTTP (OpenAI compat) |
| mac-mini | 8083 | E4B MLX server | HTTP (OpenAI compat) |
| macbook-pro | 8084 | 26B MLX server | HTTP (OpenAI compat) |
| cloud-run | 443 | gemma4-proxy | HTTPS |
| cloud-run sidecar | 1055 | Tailscale SOCKS5 | SOCKS5 |
| cloud-run sidecar | 9002 | Tailscale health | HTTP |

### Firewall / Network Extension

The Mac Mini gateway binds to `0.0.0.0:8080`. Access is controlled at the network layer:
- **LAN (home network):** Directly accessible by IP
- **Tailscale mesh:** Accessible from any Tailscale node
- **Public internet:** NOT directly accessible (no port forwarding)
- **Cloud Run:** Only reachable via Tailscale mesh

---

## Security Model

### Threat Model

| Threat | Mitigation |
|--------|-----------|
| Unauthenticated public API access | Firebase JWT validation on Cloud Run proxy |
| SQL injection / command injection | No SQL; subprocess called with list args, never `shell=True` |
| Path traversal via filename | Filename sanitized: `re.sub(r"[^a-zA-Z0-9._-]", "_", filename)` |
| MIME type spoofing | Magic-byte detection; declared MIME used only as fallback |
| Oversized upload DoS | Size checked during streaming (not after buffering); 10 GB hard limit |
| Prompt injection via filenames | Filename sanitized before any model context insertion |
| Tailscale auth key exposure | Stored in Google Secret Manager, not in code or environment |
| HF token exposure | Stored in LaunchAgent plist (local machine only, not in git) |
| Model exfiltration | Models live only on local hardware; no cloud inference |

### What Is NOT in the Security Model

- **Rate limiting bypass via IP spoofing:** Rate limits use `X-Forwarded-For` headers which can be spoofed. This is a trade-off for simplicity; not production-hardened for adversarial clients.
- **Tailscale node compromise:** If a Tailscale node is compromised, it can access the gateway directly. This is inherent to the mesh model.
- **Local file system access:** The gateway process can read/write the local file system. Restrict with macOS Sandbox if needed.

---

## Reliability Patterns

### Tier Fallback Chain

```
heavy requested → heavy circuit CLOSED? → use heavy
                                     └─ OPEN? → use primary (with fallback note)

primary requested → primary circuit CLOSED? → use primary
                                         └─ OPEN? → use fast (or return error)

fast requested → fast circuit CLOSED? → use fast
                                   └─ OPEN? → return 503 (no fallback below fast)
```

### Temp File Cleanup

Every request that touches the file system has a `finally` block that cleans up:

```python
upload_tmp_path = None
result = None
try:
    tmp_fd, upload_tmp_path = tempfile.mkstemp(...)
    # ... stream upload, process, forward ...
except Exception as e:
    log.error(...)
    return JSONResponse({"error": str(e)}, status_code=503)
finally:
    if upload_tmp_path:
        try: os.remove(upload_tmp_path)
        except OSError: pass
    for tf in getattr(result, "temp_files", []):
        try: os.remove(tf)
        except OSError: pass
```

This ensures temp files are always cleaned up, even if:
- The model call times out
- The client disconnects mid-upload
- An unexpected exception is raised in media processing

### FFmpeg Concurrency Limit

Frame extraction uses an `asyncio.Semaphore(4)` to prevent spawning more than 4 concurrent ffmpeg processes:

```python
_EXTRACT_SEM: Optional[asyncio.Semaphore] = None

def _get_extract_sem() -> asyncio.Semaphore:
    global _EXTRACT_SEM
    if _EXTRACT_SEM is None:
        _EXTRACT_SEM = asyncio.Semaphore(4)
    return _EXTRACT_SEM

async def _extract_one_frame(...):
    async with _get_extract_sem():
        # ffmpeg subprocess here
```

The semaphore is lazily initialized inside an async context to avoid event loop issues at import time.

---

## Performance Characteristics

### Measured Latencies (Mac Mini M4 16GB, April 2026)

| Operation | p50 | p95 | p99 | Notes |
|-----------|-----|-----|-----|-------|
| Text classify (fast, E2B) | 271ms | 390ms | 512ms | ~10-token response |
| Text summarize (primary, E4B) | 1,572ms | 2,100ms | 3,200ms | 20-word output |
| Gateway overhead (httpx roundtrip) | <5ms | <10ms | <20ms | Async, connection pooled |
| Image processing (1 MP JPEG) | 15ms | 30ms | 50ms | Pillow resize + JPEG encode |
| Audio processing (10s WAV) | 200ms | 350ms | 600ms | ffmpeg convert + read |
| Video frame extract (per frame) | 80ms | 150ms | 300ms | ffmpeg seek + extract |
| Video chunked (90s, 2 chunks) | 8,400ms | 12,000ms | 18,000ms | 2× model call + aggregation |

### Memory Footprint (Mac Mini M4 16GB)

| Component | RAM Usage |
|-----------|----------|
| E2B model (4-bit MLX) | 2.6 GB |
| E4B model (4-bit MLX) | 4.3 GB |
| Gateway process | ~80 MB |
| macOS + system | ~4 GB |
| Available headroom | ~5 GB |

The Mac Mini runs E2B + E4B + gateway comfortably within 16 GB. The 26B model (15.6 GB) runs on the MacBook Pro which has ≥36 GB.

### Throughput Limits

| Bottleneck | Limit | Notes |
|-----------|-------|-------|
| Rate limiter (text) | 30 req/min per IP | Sliding window |
| Rate limiter (media) | 20 req/min per IP | Sliding window |
| E2B generation | 126 tok/s | Single-user; degrades with concurrency |
| E4B generation | 32 tok/s | Single-user; MLX serialises GPU work |
| FFmpeg concurrency | 4 simultaneous | asyncio.Semaphore(4) |
| Cloud Run instances | 3 max | service.yaml maxScale: 3 |

MLX servers are single-threaded for inference (GPU work is serialised). The gateway is async and can handle many concurrent requests, but concurrent inference requests queue at the MLX server level.

---

## Data Flow Diagrams

### Simple Text Request

```
Client                Gateway              MLX Server (E4B)
  │                     │                        │
  │ POST /v1/chat/...   │                        │
  │────────────────────►│                        │
  │                     │                        │
  │                     │ classify (E2B, fast)   │
  │                     │◄──────────────────────►│  [classify returns "request"]
  │                     │                        │
  │                     │ POST /v1/chat/...       │
  │                     │───────────────────────►│  [forward to E4B]
  │                     │                        │
  │                     │   response + usage      │
  │                     │◄───────────────────────│
  │                     │                        │
  │  response + _routing│                        │
  │◄────────────────────│                        │
```

### Large Video Upload

```
Client              Gateway           media.py         MLX Server (primary)
  │                   │                   │                    │
  │ POST /v1/media/upload (2 GB video)   │                    │
  │──────────────────►│                   │                    │
  │                   │                   │                    │
  │  [4 MB chunks]    │ stream to         │                    │
  │──────────────────►│ tempfile          │                    │
  │  [continues...]   │                   │                    │
  │──────────────────►│                   │                    │
  │                   │                   │                    │
  │                   │ process_upload(   │                    │
  │                   │  path=tmp,        │                    │
  │                   │  tier="primary")  │                    │
  │                   │──────────────────►│                    │
  │                   │                   │                    │
  │                   │                   │ ffprobe → 4 chunks │
  │                   │                   │ ffmpeg frame extract│
  │                   │                   │ ChunkedVideoMedia   │
  │                   │                   │◄────────────────────│
  │                   │◄──────────────────│                    │
  │                   │                   │                    │
  │                   │   [chunk 1 frames]│                    │
  │                   │───────────────────────────────────────►│
  │                   │◄───────────────────────────────────────│
  │                   │   [chunk 2 frames]│                    │
  │                   │───────────────────────────────────────►│
  │                   │◄───────────────────────────────────────│
  │                   │   [... chunks 3, 4]                    │
  │                   │   [aggregation pass]                   │
  │                   │───────────────────────────────────────►│
  │                   │◄───────────────────────────────────────│
  │                   │                   │                    │
  │  response +       │                   │                    │
  │  _media metadata  │                   │                    │
  │◄──────────────────│                   │                    │
  │                   │ finally: rm tmp   │                    │
```

### Cloud Run to Gateway (Streaming)

```
Browser              Cloud Run Proxy          Mac Mini Gateway
    │                      │                         │
    │ HTTPS POST (1 GB)     │                         │
    │─────────────────────►│                         │
    │                      │                         │
    │  [body stream chunk] │                         │
    │─────────────────────►│ httpx.stream(           │
    │  [body stream chunk] │  content=request.stream │
    │─────────────────────►│ )                       │
    │  [body stream chunk] │─────────────────────────►│
    │─────────────────────►│  [forwarded chunks]      │
    │  [...]               │─────────────────────────►│
    │─────────────────────►│  [forwarded chunks]      │
    │  [complete]          │─────────────────────────►│
    │                      │                         │
    │                      │         response body   │
    │                      │◄─────────────────────────│
    │   HTTPS response     │                         │
    │◄─────────────────────│                         │
```

**Zero-copy guarantee:** The Cloud Run proxy never calls `await request.body()` for media uploads. The `request.stream()` async generator yields bytes as they arrive from the client; these are immediately fed into the httpx outgoing stream. Peak RAM usage in the proxy for a 1 GB video upload is approximately the httpx HTTP/1.1 buffer size (~64 KB), not 1 GB.
