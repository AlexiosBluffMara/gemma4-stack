# Gemma 4 Stack — Complete API Reference

**Base URLs:**
- Local gateway: `http://localhost:8080`
- Tailscale: `http://100.75.223.113:8080`
- Cloud Run proxy: `https://gemma4-proxy-yw44gr4drq-uc.a.run.app`

**Authentication:**
- Local / Tailscale: No authentication required
- Cloud Run: Firebase Google Sign-in JWT (`Authorization: Bearer <id_token>`)

**Content-Type:** All POST endpoints expect `application/json` or `multipart/form-data` (media upload only).

---

## Inference Endpoints

### `POST /v1/chat/completions`

OpenAI-compatible chat completions with automatic tier routing.

**Request body:**

```json
{
  "messages": [
    {"role": "system", "content": "You are a concise assistant."},
    {"role": "user",   "content": "Summarise the Gemma 4 model family."}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "tier": "primary"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array | Yes | Array of `{role, content}` objects. Roles: `system`, `user`, `assistant` |
| `max_tokens` | integer | No | Maximum tokens to generate (default: 512) |
| `temperature` | float | No | Sampling temperature 0.0–2.0 (default: 0.7). `0.0` = deterministic |
| `tier` | string | No | Force a specific tier: `"fast"`, `"primary"`, `"heavy"`. Omit for auto-routing |

**Response:**

```json
{
  "model": "mlx-community/gemma-4-e4b-it-4bit",
  "choices": [
    {
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Gemma 4 is Google's open model family...",
        "tool_calls": []
      }
    }
  ],
  "usage": {
    "input_tokens": 28,
    "output_tokens": 94,
    "total_tokens": 122,
    "prompt_tps": 302.5,
    "generation_tps": 32.1,
    "peak_memory": 4.3
  },
  "_routing": {
    "tier": "primary",
    "model": "mlx-community/gemma-4-e4b-it-4bit",
    "latency_ms": 2940,
    "strategy": "direct",
    "fallback": null
  }
}
```

**`_routing` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `tier` | string | Tier actually used (`fast`/`primary`/`heavy`) |
| `model` | string | Full model identifier |
| `latency_ms` | integer | End-to-end latency in milliseconds |
| `strategy` | string | `"direct"` for all text requests |
| `fallback` | string\|null | `"heavy→primary"` if heavy was requested but offline, else `null` |

**Error responses:**

| Status | Meaning |
|--------|---------|
| 400 | Malformed request body |
| 429 | Rate limit exceeded (30 req/min for text) |
| 503 | All tiers offline (circuit breakers open) |

**cURL example:**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 10,
    "tier": "fast"
  }'
```

**Python example:**

```python
import httpx

r = httpx.post("http://localhost:8080/v1/chat/completions", json={
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
})
print(r.json()["choices"][0]["message"]["content"])
```

**OpenAI SDK compatibility:**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",  # gateway ignores this field
)

response = client.chat.completions.create(
    model="gemma-4",       # any string; gateway ignores this, uses tier routing
    messages=[{"role": "user", "content": "Explain MLX"}],
    max_tokens=200,
    extra_body={"tier": "heavy"},  # optional tier override
)
print(response.choices[0].message.content)
```

---

### `POST /classify`

Fast-tier message classification. Always uses E2B. Returns in ~270ms.

**Request:**

```json
{"text": "Can you fix the authentication bug in production?"}
```

**Response:**

```json
{"category": "request", "latency_ms": 271}
```

**Categories:**

| Category | Examples |
|----------|---------|
| `question` | "How does X work?", "What is the status of Y?" |
| `request` | "Please fix Z", "Can you implement W?", "Update the docs" |
| `idea` | "We should try X", "What if we Y?", "Idea: use Z" |
| `greeting` | "Hello", "Hey", "How are you?", "Thanks" |
| `fyi` | "Heads up: X happened", "FYI the deploy is done", "Note: server is down" |

**cURL example:**

```bash
curl -X POST http://localhost:8080/classify \
  -H 'Content-Type: application/json' \
  -d '{"text": "Deploy the new version to staging please"}'
# → {"category": "request", "latency_ms": 268}
```

---

### `POST /compress`

Primary-tier text compression. Always uses E4B. Compresses to approximately N words.

**Request:**

```json
{
  "text": "The quarterly earnings report showed strong growth across all business segments with total revenue increasing by 23% year-over-year to $4.2 billion, driven primarily by cloud services which grew 41%...",
  "words": 20
}
```

| Field | Type | Required | Default |
|-------|------|----------|---------|
| `text` | string | Yes | — |
| `words` | integer | No | `30` |

**Response:**

```json
{"compressed": "Revenue up 23% to $4.2B; cloud grew 41%.", "latency_ms": 1572}
```

**cURL example:**

```bash
curl -X POST http://localhost:8080/compress \
  -H 'Content-Type: application/json' \
  -d '{"text": "...long text...", "words": 15}'
```

---

## Media Endpoints

### `POST /v1/media/upload`

Upload any image, audio, or video file for multimodal analysis. The gateway processes the file server-side (validates, converts, extracts frames) before forwarding to the model.

**Request (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | binary | Yes | The media file |
| `prompt` | string | No | Text prompt (default: `"Describe this in detail."`) |
| `tier` | string | No | `fast` / `primary` / `heavy` (default: auto-select) |

**Supported formats and limits:**

| Category | Formats | Max Size | Notes |
|----------|---------|----------|-------|
| Image | JPEG, PNG, WebP, GIF, BMP | 20 MB | Resized to ≤1280px, EXIF stripped |
| Audio | WAV, MP3, M4A, AAC, FLAC, OGG, CAF | 50 MB | Converted to 16 kHz mono WAV, max 30s |
| Video | MP4, MOV, WebM, MKV, AVI, 3GP | 10 GB (direct) / 2 GB (via Cloud Run) | Frame extraction, dynamic chunking |

**cURL examples:**

```bash
# Image
curl -X POST http://localhost:8080/v1/media/upload \
  -F "file=@screenshot.png;type=image/png" \
  -F "prompt=What does this UI show?"

# Audio
curl -X POST http://localhost:8080/v1/media/upload \
  -F "file=@meeting.wav;type=audio/wav" \
  -F "prompt=Transcribe and summarise this meeting."

# Short video (≤60s, single-pass)
curl -X POST http://localhost:8080/v1/media/upload \
  -F "file=@demo.mp4;type=video/mp4" \
  -F "prompt=Describe what happens in this video." \
  -F "tier=primary"

# Long video (>60s, chunked — upload directly for files >2 GB)
curl -X POST http://100.75.223.113:8080/v1/media/upload \
  -F "file=@lecture.mp4;type=video/mp4" \
  -F "prompt=What topics are covered in this lecture?" \
  -F "tier=heavy"
```

**Response — image or short video (direct):**

```json
{
  "model": "mlx-community/gemma-4-e4b-it-4bit",
  "choices": [{"message": {"role": "assistant", "content": "The image shows..."}}],
  "usage": {"input_tokens": 310, "output_tokens": 45, "generation_tps": 31.2},
  "_routing": {
    "tier": "primary",
    "model": "mlx-community/gemma-4-e4b-it-4bit",
    "latency_ms": 5200,
    "strategy": "direct"
  },
  "_media": {
    "category": "image",
    "original_name": "screenshot.png",
    "original_size": 524288,
    "processed_size": 87043,
    "frame_count": 0,
    "duration_s": 0,
    "warnings": ["Resized from 2560x1440 to 1280x720"]
  }
}
```

**Response — long video (chunked):**

```json
{
  "model": "mlx-community/gemma-4-e4b-it-4bit",
  "choices": [{"message": {"role": "assistant", "content": "Across all segments, the video..."}}],
  "usage": {"input_tokens": 2840, "output_tokens": 312, "generation_tps": 29.1},
  "_routing": {
    "tier": "primary",
    "model": "mlx-community/gemma-4-e4b-it-4bit",
    "latency_ms": 22400,
    "strategy": "chunked",
    "chunks": 4
  },
  "_media": {
    "category": "video",
    "original_name": "lecture.mp4",
    "original_size": 2684354560,
    "duration_s": 240.0,
    "strategy": "chunked",
    "chunk_count": 4,
    "chunk_duration_s": 60.0,
    "frames_per_chunk": 8,
    "total_frames": 32,
    "warnings": [
      "Long video (240s) split into 4 segment(s) of ~60s each, 8 frames/segment (primary tier)"
    ]
  }
}
```

**`_media` fields:**

| Field | Applies To | Description |
|-------|-----------|-------------|
| `category` | all | `"image"`, `"audio"`, `"video"` |
| `original_name` | all | Client-provided filename |
| `original_size` | all | Bytes as received |
| `processed_size` | image/audio/short video | Bytes after conversion |
| `frame_count` | video (direct) | Number of frames extracted |
| `duration_s` | audio/video | Duration in seconds |
| `strategy` | video | `"direct"` or `"chunked"` |
| `chunk_count` | chunked video | Number of temporal segments |
| `chunk_duration_s` | chunked video | Seconds per segment |
| `frames_per_chunk` | chunked video | Frames extracted per segment |
| `total_frames` | chunked video | `chunk_count × frames_per_chunk` |
| `warnings` | all | Non-fatal processing notes (resize, truncation, etc.) |

**Error responses:**

| Status | Code | Meaning |
|--------|------|---------|
| 400 | `no_media` | Empty file |
| 400 | `invalid_type` | Unsupported file format |
| 413 | `too_large` | File exceeds category limit |
| 422 | `processing_failed` | ffmpeg/Pillow processing error |
| 429 | — | Rate limit exceeded (20 req/min for media) |

---

### `POST /v1/media/analyze`

Analyze media by URL rather than file upload.

**Request:**

```json
{
  "url": "https://example.com/photo.jpg",
  "prompt": "Describe what you see in detail.",
  "tier": "primary"
}
```

**Response:** Same structure as `/v1/media/upload`.

---

## Status & Monitoring Endpoints

### `GET /health`

Returns detailed tier status including circuit breaker state.

**Response:**

```json
{
  "version": "2.0.0",
  "uptime_s": 86403,
  "tiers": {
    "fast": {
      "status": "ok",
      "url": "http://localhost:8082",
      "circuit_breaker": {
        "state": "closed",
        "failure_count": 0,
        "last_error": null,
        "failure_threshold": 5,
        "recovery_timeout_s": 60.0
      }
    },
    "primary": {
      "status": "ok",
      "url": "http://localhost:8083",
      "circuit_breaker": {
        "state": "closed",
        "failure_count": 0,
        "last_error": null,
        "failure_threshold": 5,
        "recovery_timeout_s": 60.0
      }
    },
    "heavy": {
      "status": "offline",
      "url": "http://100.x.x.x:8084",
      "circuit_breaker": {
        "state": "open",
        "failure_count": 5,
        "last_error": "ConnectError: connection refused",
        "failure_threshold": 5,
        "recovery_timeout_s": 60.0
      }
    }
  }
}
```

---

### `GET /metrics`

Per-tier request counts and latency percentiles since last gateway restart.

**Response:**

```json
{
  "fast": {
    "requests": 1247,
    "errors": 3,
    "p50_ms": 271,
    "p95_ms": 392,
    "p99_ms": 517
  },
  "primary": {
    "requests": 438,
    "errors": 1,
    "p50_ms": 1571,
    "p95_ms": 2108,
    "p99_ms": 3241
  },
  "heavy": {
    "requests": 12,
    "errors": 0,
    "p50_ms": 3102,
    "p95_ms": 5382,
    "p99_ms": 8193
  }
}
```

---

### `GET /devices`

Registered device registry with capability and online status.

**Response:**

```json
{
  "mac-mini": {
    "ip": "100.75.223.113",
    "capabilities": ["fast", "primary", "gateway"],
    "status": "online",
    "last_seen": "2026-04-04T17:30:00.000Z"
  },
  "macbook-pro": {
    "ip": "100.x.x.x",
    "capabilities": ["heavy"],
    "status": "offline",
    "last_seen": null
  }
}
```

---

### `GET /healthz`

Kubernetes/Cloud Run liveness probe. Always returns `200 OK` if the process is running.

**Response:** `200 OK`, body: `{"status": "alive"}`

**Note:** Cloud Run's internal load balancer intercepts `/healthz` for its own health routing. For external readiness checks, use `/_ready`.

---

### `GET /_ready`

Readiness probe. Returns `503` when no tier's circuit breaker is CLOSED (system cannot serve any requests).

**Response — ready:**

```json
{"status": "ready", "circuit": {"state": "closed", "failure_count": 0}}
```

**Response — not ready:**

```json
{"status": "not_ready", "reason": "circuit_breaker_open"}
```

---

## Rate Limits

All limits apply per source IP address using a sliding window.

| Endpoint Group | Limit | Scope |
|---------------|-------|-------|
| `/v1/chat/completions`, `/classify`, `/compress` | 30 requests / minute | Per IP |
| `/v1/media/upload`, `/v1/media/analyze` | 20 requests / minute | Per IP |
| `/health`, `/_ready`, `/healthz` | 120 requests / minute | Per IP |

**Rate limit response:**

```json
{
  "error": "Rate limit exceeded (30 req/min)",
  "request_id": "req_abc123"
}
```
HTTP status: `429 Too Many Requests`

**Cloud Run proxy limits:** Same as above, enforced independently before forwarding.

---

## Response Metadata

Every inference response includes `_routing` metadata appended by the gateway. This is informational and not part of the OpenAI API specification.

```json
"_routing": {
  "tier": "primary",
  "model": "mlx-community/gemma-4-e4b-it-4bit",
  "latency_ms": 1572,
  "strategy": "direct",
  "fallback": null
}
```

**Strategy values:**

| Value | Meaning |
|-------|---------|
| `"direct"` | Text request or short video (single model call) |
| `"chunked"` | Long video split into temporal segments + aggregation |

**Fallback values:**

| Value | Meaning |
|-------|---------|
| `null` | No fallback; tier used as requested/routed |
| `"heavy→primary"` | Heavy tier offline; fell back to primary |
| `"primary→fast"` | Primary tier offline; fell back to fast |

---

## Error Reference

### Gateway Errors

| HTTP Status | Condition |
|-------------|-----------|
| `400 Bad Request` | Missing required fields, empty body, unsupported file type |
| `413 Request Entity Too Large` | File exceeds size limit (10 GB video, 20 MB image, 50 MB audio) |
| `422 Unprocessable Entity` | File passed type check but failed processing (corrupted video, etc.) |
| `429 Too Many Requests` | Rate limit exceeded |
| `503 Service Unavailable` | All tier circuit breakers open; no model available |

### Cloud Run Proxy Errors

| HTTP Status | Condition |
|-------------|-----------|
| `401 Unauthorized` | Missing or invalid Firebase ID token |
| `413 Request Entity Too Large` | Media file >2 GB (must upload directly to gateway) |
| `429 Too Many Requests` | Proxy-level rate limit exceeded |
| `502 Bad Gateway` | Gateway connect error |
| `503 Service Unavailable` | Proxy circuit breaker open (gateway appears down) |
| `504 Gateway Timeout` | Gateway took too long to respond |

---

## Python SDK Reference

`scripts/local_llm.py` — synchronous SDK for direct tier access:

```python
from local_llm import classify, compress, heavy_query, route

# classify(text: str) -> str
# Calls fast tier. Returns category string.
category = classify("Fix the deploy bug please")
# → "request"

# compress(text: str, words: int = 30) -> str
# Calls primary tier. Returns compressed text.
summary = compress("Long meeting transcript...", words=20)
# → "Discussed Q3 goals, assigned tasks, next meeting Friday."

# heavy_query(prompt: str, max_tokens: int = 1024) -> str
# Calls heavy tier. Raises if MacBook Pro offline.
analysis = heavy_query("Review this diff for security issues:\n" + diff_text)

# route(text: str) -> tuple[str, str]
# Classify then route automatically. Returns (tier_used, response).
tier, response = route("Hello, can you help me?")
# → ("fast", "Of course! What do you need help with?")
```

**Note:** The SDK calls tier endpoints directly (`:8082`, `:8083`, `:8084`), bypassing the gateway's routing logic. Use the gateway's `/v1/chat/completions` with `tier=` for the same effect with full metadata in the response.
