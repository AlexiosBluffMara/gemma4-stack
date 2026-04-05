"""
Gemma 4 API Gateway — Production-grade multi-device FastAPI router for MLX inference network.

Devices:
  Mac Mini M4 (always on):  fast (E2B :8082), primary (E4B :8083)
  MacBook Pro M4 Max (intermittent via Tailscale): heavy (26B-A4B :8084)

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible, auto-routes across tiers
  POST /v1/media/analyze     — Convenience wrapper for multimodal analysis
  POST /v1/media/upload      — Direct file upload for multimodal inference
  POST /classify             — Fast tier classification only
  POST /compress             — Primary tier compression only
  GET  /health               — Detailed status of all tiers with circuit breaker state
  GET  /healthz              — Kubernetes liveness probe (always 200)
  GET  /_ready               — Readiness probe (503 if no tiers healthy)
  GET  /metrics              — Per-tier request counts, latencies, errors
  GET  /devices              — Registered devices, IPs, online status, capabilities
  GET  /                     — Web UI

Run:
  uvicorn gateway:app --host 0.0.0.0 --port 8080
"""

import asyncio
import json
import logging
import os
import signal
import statistics
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

import media as media_processor

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

__version__ = "2.0.0"

# ---------------------------------------------------------------------------
# Structured JSON Logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields attached to the record
        for key in ("request_id", "tier", "latency_ms", "status_code", "client_ip", "method", "path"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry, default=str)


def _setup_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger = logging.getLogger("gateway")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


log = _setup_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAST_URL = os.getenv("FAST_URL", "http://localhost:8082")
PRIMARY_URL = os.getenv("PRIMARY_URL", "http://localhost:8083")
HEAVY_URL = os.getenv("HEAVY_URL", "")        # empty = disabled
HEAVY_API_KEY = os.getenv("HEAVY_API_KEY", "")  # shared secret for Cloud Run heavy tier
DEVICE_NAME = os.getenv("DEVICE_NAME", "mac-mini")

MODEL_FAST = "mlx-community/gemma-4-e2b-it-4bit"
MODEL_PRIMARY = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_HEAVY = "mlx-community/gemma-4-26b-a4b-it-4bit"

# Tier map — heavy tier only registered if HEAVY_URL is configured
TIER_MAP: dict[str, tuple[str, str]] = {
    "fast": (FAST_URL, MODEL_FAST),
    "primary": (PRIMARY_URL, MODEL_PRIMARY),
}
if HEAVY_URL:
    TIER_MAP["heavy"] = (HEAVY_URL, MODEL_HEAVY)

HEALTH_CHECK_INTERVAL = 30  # seconds

# Circuit breaker configuration
CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "5"))
CB_RECOVERY_TIMEOUT = float(os.getenv("CB_RECOVERY_TIMEOUT", "60.0"))  # seconds
CB_HALF_OPEN_MAX = 1  # number of probe requests in half-open state

# Rate limiting
RATE_LIMIT_TEXT_PER_MIN = int(os.getenv("RATE_LIMIT_TEXT", "60"))
RATE_LIMIT_MEDIA_PER_MIN = int(os.getenv("RATE_LIMIT_MEDIA", "20"))

# Retry configuration
MAX_RETRIES = 2
RETRY_DELAYS = [0.5, 1.0]

# httpx timeout configuration
# Standard timeout for fast/primary tiers (local MLX servers)
HTTPX_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=10.0)
# Extended timeout for heavy tier — Cloud Run GPU cold start can take ~110 seconds
# (model download from GCS + GPU load), plus inference up to 600s.
HTTPX_TIMEOUT_HEAVY = httpx.Timeout(connect=120.0, read=600.0, write=10.0, pool=10.0)
HTTPX_LIMITS = httpx.Limits(
    max_connections=50,
    max_keepalive_connections=20,
    keepalive_expiry=30,
)

# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-tier circuit breaker: CLOSED -> OPEN -> HALF_OPEN -> CLOSED."""

    def __init__(self, name: str, failure_threshold: int = CB_FAILURE_THRESHOLD,
                 recovery_timeout: float = CB_RECOVERY_TIMEOUT):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0.0
        self._last_error: Optional[str] = None
        self._half_open_in_flight = 0

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_in_flight = 0
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    @property
    def last_error(self) -> Optional[str]:
        with self._lock:
            return self._last_error

    def allow_request(self) -> bool:
        """Return True if a request is allowed through the breaker."""
        current = self.state
        if current == CircuitState.CLOSED:
            return True
        if current == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_in_flight < CB_HALF_OPEN_MAX:
                    self._half_open_in_flight += 1
                    return True
            return False
        # OPEN
        return False

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._half_open_in_flight = 0

    def record_failure(self, error: str) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            self._last_error = error
            if self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed, go back to open
                self._state = CircuitState.OPEN

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "last_error": self.last_error,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_s": self.recovery_timeout,
        }


_circuit_breakers: dict[str, CircuitBreaker] = {
    name: CircuitBreaker(name) for name in TIER_MAP
}

# ---------------------------------------------------------------------------
# Device registry
# ---------------------------------------------------------------------------

_devices: dict[str, dict] = {
    DEVICE_NAME: {
        "ip": "localhost",
        "online": True,
        "last_seen": datetime.now(timezone.utc).isoformat(),
        "capabilities": ["fast", "primary"],
        "tiers": {
            "fast": {"url": FAST_URL, "model": MODEL_FAST},
            "primary": {"url": PRIMARY_URL, "model": MODEL_PRIMARY},
        },
    },
}

if HEAVY_URL:
    _devices["macbook-pro"] = {
        "ip": HEAVY_URL.replace("http://", "").split(":")[0],
        "online": False,
        "last_seen": None,
        "capabilities": ["heavy"],
        "tiers": {
            "heavy": {"url": HEAVY_URL, "model": MODEL_HEAVY},
        },
    }

# ---------------------------------------------------------------------------
# Tier health status (updated by background task)
# ---------------------------------------------------------------------------

_tier_status: dict[str, bool] = {t: False for t in TIER_MAP}

# ---------------------------------------------------------------------------
# Startup state
# ---------------------------------------------------------------------------

_startup_ready = False
_start_time = time.monotonic()

# ---------------------------------------------------------------------------
# Metrics (thread-safe)
# ---------------------------------------------------------------------------


class TierMetrics:
    """Thread-safe per-tier metrics collector."""

    def __init__(self):
        self._lock = Lock()
        self.request_count: int = 0
        self.error_count: int = 0
        self.last_error: Optional[str] = None
        self.latencies: list[float] = []
        # Histogram buckets (seconds): <0.1, <0.5, <1, <2, <5, <10, <30, <60, >=60
        self.histogram_buckets: dict[str, int] = {
            "<0.1s": 0, "<0.5s": 0, "<1s": 0, "<2s": 0,
            "<5s": 0, "<10s": 0, "<30s": 0, "<60s": 0, ">=60s": 0,
        }

    def record_request(self, latency: float) -> None:
        with self._lock:
            self.request_count += 1
            self.latencies.append(latency)
            if len(self.latencies) > 2000:
                self.latencies = self.latencies[-1000:]
            # Histogram
            if latency < 0.1:
                self.histogram_buckets["<0.1s"] += 1
            elif latency < 0.5:
                self.histogram_buckets["<0.5s"] += 1
            elif latency < 1.0:
                self.histogram_buckets["<1s"] += 1
            elif latency < 2.0:
                self.histogram_buckets["<2s"] += 1
            elif latency < 5.0:
                self.histogram_buckets["<5s"] += 1
            elif latency < 10.0:
                self.histogram_buckets["<10s"] += 1
            elif latency < 30.0:
                self.histogram_buckets["<30s"] += 1
            elif latency < 60.0:
                self.histogram_buckets["<60s"] += 1
            else:
                self.histogram_buckets[">=60s"] += 1

    def record_error(self, error: str) -> None:
        with self._lock:
            self.error_count += 1
            self.last_error = error

    def snapshot(self) -> dict:
        with self._lock:
            lats = list(self.latencies)
            return {
                "requests": self.request_count,
                "errors": self.error_count,
                "last_error": self.last_error,
                "p50_ms": round(statistics.median(lats) * 1000) if lats else None,
                "p90_ms": (
                    round(sorted(lats)[int(len(lats) * 0.90)] * 1000)
                    if len(lats) >= 10 else None
                ),
                "p95_ms": (
                    round(sorted(lats)[int(len(lats) * 0.95)] * 1000)
                    if len(lats) >= 20 else None
                ),
                "p99_ms": (
                    round(sorted(lats)[int(len(lats) * 0.99)] * 1000)
                    if len(lats) >= 100 else None
                ),
                "histogram": dict(self.histogram_buckets),
            }


_metrics: dict[str, TierMetrics] = {t: TierMetrics() for t in TIER_MAP}
_global_error_count = 0
_global_error_lock = Lock()

# ---------------------------------------------------------------------------
# Rate Limiter (in-memory, per-IP)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window per-IP rate limiter."""

    def __init__(self):
        self._lock = Lock()
        # ip -> list of timestamps
        self._text_windows: dict[str, list[float]] = defaultdict(list)
        self._media_windows: dict[str, list[float]] = defaultdict(list)

    def _prune(self, timestamps: list[float], now: float) -> list[float]:
        cutoff = now - 60.0
        # Binary search would be faster, but for typical counts list comp is fine
        return [t for t in timestamps if t > cutoff]

    def check_text(self, ip: str) -> bool:
        """Return True if request is allowed."""
        now = time.monotonic()
        with self._lock:
            self._text_windows[ip] = self._prune(self._text_windows[ip], now)
            if len(self._text_windows[ip]) >= RATE_LIMIT_TEXT_PER_MIN:
                return False
            self._text_windows[ip].append(now)
            return True

    def check_media(self, ip: str) -> bool:
        """Return True if request is allowed."""
        now = time.monotonic()
        with self._lock:
            self._media_windows[ip] = self._prune(self._media_windows[ip], now)
            if len(self._media_windows[ip]) >= RATE_LIMIT_MEDIA_PER_MIN:
                return False
            self._media_windows[ip].append(now)
            return True


_rate_limiter = RateLimiter()

# ---------------------------------------------------------------------------
# Shutdown management
# ---------------------------------------------------------------------------

_shutting_down = False
_in_flight = 0
_in_flight_lock = Lock()
_shutdown_event = asyncio.Event() if hasattr(asyncio, "Event") else None


def _increment_in_flight():
    global _in_flight
    with _in_flight_lock:
        _in_flight += 1


def _decrement_in_flight():
    global _in_flight
    with _in_flight_lock:
        _in_flight -= 1


# ---------------------------------------------------------------------------
# Middleware: Request ID, Security Headers, Rate Limiting
# ---------------------------------------------------------------------------


class GatewayMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        global _shutting_down

        # Generate request correlation ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        # Reject during shutdown
        if _shutting_down:
            return JSONResponse(
                {"error": "Server is shutting down", "request_id": request_id},
                status_code=503,
                headers={"X-Request-ID": request_id},
            )

        _increment_in_flight()
        t0 = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            _decrement_in_flight()
            raise
        latency_ms = round((time.monotonic() - t0) * 1000)
        _decrement_in_flight()

        # Attach correlation and security headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains"

        # Structured access log
        log.info(
            "%s %s %s %dms",
            request.method, request.url.path, response.status_code, latency_ms,
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": str(request.url.path),
                "status_code": response.status_code,
                "latency_ms": latency_ms,
                "client_ip": _get_client_ip(request),
            },
        )
        return response


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


# ---------------------------------------------------------------------------
# Health check background task
# ---------------------------------------------------------------------------

async def _check_tier(client: httpx.AsyncClient, name: str, url: str) -> bool:
    cb = _circuit_breakers.get(name)
    try:
        r = await client.get(f"{url}/health", timeout=5.0)
        ok = r.status_code == 200
        if ok and cb:
            cb.record_success()
        return ok
    except Exception as exc:
        if cb:
            cb.record_failure(str(exc))
        return False


async def _health_loop(app: FastAPI) -> None:
    """Periodically health-check all tiers and update device/tier status."""
    global _startup_ready
    client = app.state.client
    first_check_done = False

    while True:
        # Check local tiers
        for tier_name in ("fast", "primary"):
            if tier_name in TIER_MAP:
                url, _ = TIER_MAP[tier_name]
                _tier_status[tier_name] = await _check_tier(client, tier_name, url)

        # Update local device status
        local_dev = _devices[DEVICE_NAME]
        local_dev["online"] = any(
            _tier_status.get(t, False) for t in local_dev["capabilities"]
        )
        if local_dev["online"]:
            local_dev["last_seen"] = datetime.now(timezone.utc).isoformat()

        # Check remote heavy tier (only if configured)
        if HEAVY_URL and "heavy" in TIER_MAP:
            heavy_online = await _check_tier(client, "heavy", HEAVY_URL)
            _tier_status["heavy"] = heavy_online
            if "macbook-pro" in _devices:
                remote_dev = _devices["macbook-pro"]
                remote_dev["online"] = heavy_online
                if heavy_online:
                    remote_dev["last_seen"] = datetime.now(timezone.utc).isoformat()

        if not first_check_done:
            first_check_done = True
            _startup_ready = True
            healthy = [t for t, ok in _tier_status.items() if ok]
            log.info(
                "Initial health check complete. Healthy tiers: %s",
                healthy or "none",
            )

        await asyncio.sleep(HEALTH_CHECK_INTERVAL)


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------

def _has_media(messages: list) -> bool:
    """Check if any message contains multimodal content (images or audio)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in (
                    "image_url",
                    "input_audio",
                ):
                    return True
    return False


# ---------------------------------------------------------------------------
# Forwarding helpers with retry and circuit breaker
# ---------------------------------------------------------------------------

def _is_retryable(exc: Exception) -> bool:
    """Determine if an error warrants a retry."""
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


async def _forward(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.0,
    request_id: str = "",
    tier: str = "",
    extra_headers: Optional[dict] = None,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Use extended timeout for heavy tier (Cloud Run GPU cold start + long inference)
    timeout = HTTPX_TIMEOUT_HEAVY if tier == "heavy" else HTTPX_TIMEOUT

    # Build headers: add API key auth for Cloud Run heavy tier
    headers: dict = {}
    if extra_headers:
        headers.update(extra_headers)
    if tier == "heavy" and HEAVY_API_KEY:
        headers["Authorization"] = f"Bearer {HEAVY_API_KEY}"

    cb = _circuit_breakers.get(tier)
    last_exc: Optional[Exception] = None

    for attempt in range(1 + MAX_RETRIES):
        try:
            r = await client.post(
                f"{url}/v1/chat/completions",
                json=payload,
                headers=headers or None,
                timeout=timeout,
            )
            r.raise_for_status()
            if cb:
                cb.record_success()
            return r.json()

        except httpx.ConnectTimeout as exc:
            last_exc = exc
            msg = f"Connect timeout to {tier} tier at {url}"
            log.warning(msg, extra={"request_id": request_id, "tier": tier})
            if cb:
                cb.record_failure(msg)
            if _metrics.get(tier):
                _metrics[tier].record_error(msg)

        except httpx.ReadTimeout as exc:
            last_exc = exc
            msg = f"Read timeout from {tier} tier (inference took too long)"
            log.warning(msg, extra={"request_id": request_id, "tier": tier})
            if cb:
                cb.record_failure(msg)
            if _metrics.get(tier):
                _metrics[tier].record_error(msg)
            # Read timeouts are not retryable (inference is slow, retrying won't help)
            raise

        except httpx.WriteTimeout as exc:
            last_exc = exc
            msg = f"Write timeout sending request to {tier} tier"
            log.warning(msg, extra={"request_id": request_id, "tier": tier})
            if cb:
                cb.record_failure(msg)
            if _metrics.get(tier):
                _metrics[tier].record_error(msg)

        except httpx.PoolTimeout as exc:
            last_exc = exc
            msg = f"Connection pool exhausted for {tier} tier"
            log.warning(msg, extra={"request_id": request_id, "tier": tier})
            if cb:
                cb.record_failure(msg)
            if _metrics.get(tier):
                _metrics[tier].record_error(msg)

        except httpx.ConnectError as exc:
            last_exc = exc
            msg = f"Connection refused by {tier} tier at {url}"
            log.warning(msg, extra={"request_id": request_id, "tier": tier})
            if cb:
                cb.record_failure(msg)
            if _metrics.get(tier):
                _metrics[tier].record_error(msg)

        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status = exc.response.status_code
            msg = f"{tier} tier returned HTTP {status}"
            log.warning(msg, extra={"request_id": request_id, "tier": tier, "status_code": status})
            if cb:
                cb.record_failure(msg)
            if _metrics.get(tier):
                _metrics[tier].record_error(msg)
            # Don't retry 4xx errors
            if status < 500:
                raise

        except Exception as exc:
            last_exc = exc
            msg = f"Unexpected error from {tier} tier: {exc}"
            log.error(msg, extra={"request_id": request_id, "tier": tier})
            if cb:
                cb.record_failure(msg)
            if _metrics.get(tier):
                _metrics[tier].record_error(msg)
            raise

        # Retry with exponential backoff
        if attempt < MAX_RETRIES:
            delay = RETRY_DELAYS[attempt]
            log.info(
                "Retrying %s tier (attempt %d/%d) after %.1fs",
                tier, attempt + 1, MAX_RETRIES, delay,
                extra={"request_id": request_id, "tier": tier},
            )
            await asyncio.sleep(delay)

    # All retries exhausted
    raise last_exc  # type: ignore[misc]


def _extract_text(messages: list) -> str:
    """Extract the last user message text, handling both string and multimodal formats."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return " ".join(text_parts)
    return ""


async def _classify(client: httpx.AsyncClient, text: str, request_id: str = "") -> str:
    data = await _forward(
        client,
        FAST_URL,
        MODEL_FAST,
        [
            {
                "role": "user",
                "content": (
                    "Classify into one word "
                    "(greeting/fyi/summarize/compress/question/request/code/analysis):\n"
                    f"{text}"
                ),
            }
        ],
        max_tokens=8,
        temperature=0.0,
        request_id=request_id,
        tier="fast",
    )
    return data["choices"][0]["message"]["content"].strip().lower()


def _select_tier_with_fallback(preferred: str, request_id: str = "") -> tuple[str, Optional[str]]:
    """Select a tier, falling back through the chain if circuit is open or tier is down.

    Returns (tier_name, fallback_note_or_none).
    Fallback order: heavy -> primary -> fast.
    """
    fallback_chain = {
        "heavy": ["primary", "fast"],
        "primary": ["fast"],
        "fast": ["primary"],
    }
    fallback_note = None

    # Try preferred tier
    if preferred in TIER_MAP:
        cb = _circuit_breakers.get(preferred)
        if cb and cb.allow_request() and _tier_status.get(preferred, False):
            return preferred, None
        elif cb and not cb.allow_request():
            fallback_note = (
                f"{preferred.capitalize()} tier circuit breaker is {cb.state.value}. "
            )
            log.info(
                "Circuit breaker %s for tier %s, falling back",
                cb.state.value, preferred,
                extra={"request_id": request_id, "tier": preferred},
            )
        elif not _tier_status.get(preferred, False):
            fallback_note = f"{preferred.capitalize()} tier is offline. "

    # Try fallbacks
    for fallback in fallback_chain.get(preferred, []):
        if fallback in TIER_MAP:
            cb = _circuit_breakers.get(fallback)
            if cb and cb.allow_request() and _tier_status.get(fallback, False):
                note = (fallback_note or "") + f"Using {fallback} tier instead."
                log.info(
                    "Falling back from %s to %s", preferred, fallback,
                    extra={"request_id": request_id, "tier": fallback},
                )
                return fallback, note

    # Nothing healthy — return preferred anyway (will likely fail and trip circuit breaker)
    # This allows the request to flow through and generate a proper error
    if preferred in TIER_MAP:
        return preferred, (fallback_note or "") + "All tiers degraded."
    # Last resort: first available tier
    first = next(iter(TIER_MAP))
    return first, "All tiers degraded, using first available."


# ---------------------------------------------------------------------------
# Chunked video processing
# ---------------------------------------------------------------------------

async def _process_chunked_video(
    chunked: "media_processor.ChunkedVideoMedia",
    prompt: str,
    client: httpx.AsyncClient,
    url: str,
    model: str,
    request_id: str,
    tier: str,
) -> tuple:
    """
    Run sequential per-segment model calls on a ChunkedVideoMedia, then
    aggregate with a final summarisation pass.

    Returns:
        (final_response: dict, chunk_analyses: list[str], total_latency_s: float)
    """
    chunks = chunked.chunks
    n_chunks = len(chunks)
    duration = chunked.duration_s
    chunk_analyses: list = []
    total_latency = 0.0

    log.info(
        "Chunked video: %d segment(s) × %d frames, %.0fs total",
        n_chunks, chunked.frames_per_chunk, duration,
        extra={"request_id": request_id},
    )

    # ── Per-segment analysis ──────────────────────────────────────────────────
    for chunk in chunks:
        seg_label = (
            f"Segment {chunk.index + 1}/{n_chunks} "
            f"({chunk.start_s:.0f}s–{chunk.end_s:.0f}s of {duration:.0f}s)"
        )

        # Build multimodal content for this segment
        seg_content = [
            {
                "type": "text",
                "text": (
                    f"[{seg_label}]\n"
                    f"{prompt}\n"
                    f"Focus only on this time window. Be concise — "
                    f"all segments will be combined into one final answer."
                ),
            }
        ]
        for b64 in chunk.frames:
            seg_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        t0 = time.monotonic()
        try:
            resp = await _forward(
                client, url, model,
                [{"role": "user", "content": seg_content}],
                max_tokens=512,
                temperature=0.0,
                request_id=f"{request_id}_s{chunk.index}",
                tier=tier,
            )
            seg_latency = time.monotonic() - t0
            total_latency += seg_latency
            analysis = (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            chunk_analyses.append(analysis)
            log.info(
                "Segment %d/%d: %.1fs, %d chars",
                chunk.index + 1, n_chunks, seg_latency, len(analysis),
                extra={"request_id": request_id},
            )
        except Exception as exc:
            log.warning(
                "Segment %d/%d failed (%s) — continuing",
                chunk.index + 1, n_chunks, exc,
                extra={"request_id": request_id},
            )
            chunk_analyses.append(f"[segment {chunk.index + 1} analysis unavailable: {exc}]")

    # ── Aggregation pass ──────────────────────────────────────────────────────
    if len(chunk_analyses) <= 1:
        # Single chunk — return the segment analysis directly (no aggregation needed)
        content = chunk_analyses[0] if chunk_analyses else "No analysis available."
        return (
            {
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "model": model,
                "usage": {},
            },
            chunk_analyses,
            total_latency,
        )

    # Build aggregation prompt with all segment summaries
    seg_lines = "\n".join(
        f"  • Segment {i + 1}/{n_chunks} "
        f"({chunks[i].start_s:.0f}s–{chunks[i].end_s:.0f}s): {analysis}"
        for i, analysis in enumerate(chunk_analyses)
    )
    agg_text = (
        f"The following are concise analyses of {n_chunks} sequential segments "
        f"from a {duration:.0f}s video "
        f"(~{chunked.chunk_duration_s:.0f}s per segment, {chunked.frames_per_chunk} frames each):\n\n"
        f"{seg_lines}\n\n"
        f'User\'s original question: "{prompt}"\n\n'
        f"Using all segment analyses above, provide a single comprehensive answer "
        f"as if you had watched the entire video from start to finish."
    )

    t0 = time.monotonic()
    try:
        final_resp = await _forward(
            client, url, model,
            [{"role": "user", "content": [{"type": "text", "text": agg_text}]}],
            max_tokens=2048,
            temperature=0.0,
            request_id=f"{request_id}_agg",
            tier=tier,
        )
        total_latency += time.monotonic() - t0
    except Exception as exc:
        log.error(
            "Aggregation pass failed: %s — falling back to concatenated analyses",
            exc, extra={"request_id": request_id},
        )
        # Fallback: stitch segment analyses together
        combined = "\n\n".join(
            f"[{chunks[i].start_s:.0f}s–{chunks[i].end_s:.0f}s] {a}"
            for i, a in enumerate(chunk_analyses)
        )
        final_resp = {
            "choices": [{"message": {"content": combined}, "finish_reason": "stop"}],
            "model": model,
            "usage": {},
            "_aggregation_fallback": True,
        }

    return final_resp, chunk_analyses, total_latency


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _shutting_down, _shutdown_event

    # Initialize the shutdown event in the running loop
    _shutdown_event = asyncio.Event()

    app.state.client = httpx.AsyncClient(
        timeout=HTTPX_TIMEOUT,
        limits=HTTPX_LIMITS,
    )
    health_task = asyncio.create_task(_health_loop(app))

    # Register signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()

    def _signal_handler(sig):
        global _shutting_down
        log.info("Received signal %s, initiating graceful shutdown", sig.name)
        _shutting_down = True
        _shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _signal_handler, sig)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    log.info("Gateway v%s starting up", __version__)
    yield

    # Graceful shutdown: drain in-flight requests
    log.info("Shutting down, draining in-flight requests...")
    _shutting_down = True
    drain_start = time.monotonic()
    while _in_flight > 0 and (time.monotonic() - drain_start) < 30.0:
        await asyncio.sleep(0.1)
    if _in_flight > 0:
        log.warning("Shutdown timeout, %d requests still in flight", _in_flight)
    else:
        log.info("All in-flight requests drained")

    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass
    await app.state.client.aclose()
    log.info("Gateway shutdown complete")


app = FastAPI(title="Gemma 4 Gateway", version=__version__, lifespan=lifespan)

# CORS — allow all origins for the cloud-hosted web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware (added after CORS so CORS runs first)
app.add_middleware(GatewayMiddleware)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    tier: Optional[str] = None  # force a tier: "fast" | "primary" | "heavy"

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v):
        if not v:
            raise ValueError("messages array must not be empty")
        return v

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_range(cls, v):
        if v is not None and (v < 1 or v > 4096):
            raise ValueError("max_tokens must be between 1 and 4096")
        return v

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v):
        if v is not None and (v < 0.0 or v > 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class ClassifyRequest(BaseModel):
    text: str


class CompressRequest(BaseModel):
    text: str
    words: Optional[int] = 30


class MediaAnalyzeRequest(BaseModel):
    media_url: str  # base64 data URL, e.g. "data:image/jpeg;base64,..."
    media_type: str = "image"  # "image", "audio", or "video"
    prompt: str = "Describe this in detail"
    tier: Optional[str] = None  # optional tier override


# ---------------------------------------------------------------------------
# Validation error handler — return 422 with clear messages
# ---------------------------------------------------------------------------

from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    errors = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error["loc"] if loc != "body")
        errors.append({"field": field, "message": error["msg"]})
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "details": errors,
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz():
    """Kubernetes liveness probe. Always returns 200 if the process is alive."""
    return JSONResponse({"status": "alive"})


@app.get("/_ready")
async def readiness():
    """Readiness probe. Returns 503 until at least one tier is healthy."""
    if not _startup_ready:
        return JSONResponse(
            {"status": "starting", "message": "Initial health check not yet completed"},
            status_code=503,
        )
    healthy_tiers = [t for t, ok in _tier_status.items() if ok]
    if not healthy_tiers:
        return JSONResponse(
            {"status": "not_ready", "healthy_tiers": [], "message": "No healthy tiers available"},
            status_code=503,
        )
    return JSONResponse({"status": "ready", "healthy_tiers": healthy_tiers})


@app.get("/health")
async def health(request: Request):
    client = request.app.state.client
    request_id = getattr(request.state, "request_id", "unknown")
    status = {}

    # Live-check all registered tiers
    for name, (url, _) in TIER_MAP.items():
        ok = await _check_tier(client, name, url)
        _tier_status[name] = ok
        cb = _circuit_breakers[name]
        status[name] = {
            "status": "ok" if ok else "offline",
            "circuit_breaker": cb.to_dict(),
        }

    uptime_s = round(time.monotonic() - _start_time)
    result = {
        "version": __version__,
        "uptime_s": uptime_s,
        "tiers": status,
        "request_id": request_id,
    }
    if HEAVY_URL:
        result["heavy_device"] = {
            "url": HEAVY_URL,
            "online": _tier_status.get("heavy", False),
        }
    return result


@app.get("/metrics")
async def metrics():
    out = {}
    for tier in TIER_MAP:
        m = _metrics.get(tier)
        if m:
            out[tier] = m.snapshot()
        else:
            out[tier] = {"requests": 0, "errors": 0}
    with _global_error_lock:
        out["global_errors"] = _global_error_count
    return out


@app.get("/devices")
async def devices():
    return {"devices": _devices}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    global _global_error_count
    client = request.app.state.client
    request_id = getattr(request.state, "request_id", "unknown")
    client_ip = _get_client_ip(request)
    t0 = time.monotonic()
    fallback_note = None

    # Startup readiness gate
    if not _startup_ready:
        return JSONResponse(
            {"error": "Gateway is starting up, not yet ready", "request_id": request_id},
            status_code=503,
        )

    # Rate limiting
    is_media = _has_media(req.messages)
    if is_media:
        if not _rate_limiter.check_media(client_ip):
            log.warning(
                "Media rate limit exceeded for %s", client_ip,
                extra={"request_id": request_id, "client_ip": client_ip},
            )
            return JSONResponse(
                {"error": "Rate limit exceeded (media: 20 req/min)", "request_id": request_id},
                status_code=429,
            )
    else:
        if not _rate_limiter.check_text(client_ip):
            log.warning(
                "Text rate limit exceeded for %s", client_ip,
                extra={"request_id": request_id, "client_ip": client_ip},
            )
            return JSONResponse(
                {"error": "Rate limit exceeded (text: 60 req/min)", "request_id": request_id},
                status_code=429,
            )

    try:
        # Determine tier
        if req.tier and req.tier in TIER_MAP:
            tier = req.tier
        elif is_media:
            if "heavy" in TIER_MAP and _tier_status.get("heavy"):
                tier = "heavy"
            else:
                tier = "primary"
        else:
            # Auto-route based on fast classification of last user message
            last_user = _extract_text(req.messages)
            try:
                category = await _classify(client, last_user, request_id)
            except Exception:
                # If classification fails, default to primary
                category = "question"
                log.warning(
                    "Classification failed, defaulting to primary",
                    extra={"request_id": request_id},
                )
            if category in ("greeting", "fyi"):
                tier = "fast"
            elif category in ("summarize", "compress"):
                tier = "primary"
            elif "heavy" in TIER_MAP and _tier_status.get("heavy"):
                tier = "heavy"
            else:
                tier = "primary"

        # Apply circuit breaker and fallback
        tier, fallback_note = _select_tier_with_fallback(tier, request_id)
        url, model = TIER_MAP[tier]

        log.info(
            "Routing to %s tier", tier,
            extra={"request_id": request_id, "tier": tier},
        )

        result = await _forward(
            client, url, model, req.messages, req.max_tokens, req.temperature,
            request_id=request_id, tier=tier,
        )
        latency = time.monotonic() - t0
        _metrics[tier].record_request(latency)

        # Inject routing metadata
        result["_routing"] = {
            "tier": tier,
            "model": model,
            "latency_ms": round(latency * 1000),
        }
        if fallback_note:
            result["_routing"]["fallback"] = fallback_note

        return JSONResponse(result)

    except Exception as e:
        with _global_error_lock:
            _global_error_count += 1
        latency = time.monotonic() - t0
        log.error(
            "Request failed after %.1fs: %s", latency, str(e),
            extra={"request_id": request_id},
        )

        # Check if any tier is healthy for the error response
        any_healthy = any(_tier_status.get(t, False) for t in TIER_MAP)
        status_code = 503
        detail = {
            "error": str(e),
            "request_id": request_id,
            "degraded": not any_healthy,
        }
        if not any_healthy:
            detail["message"] = "All inference tiers are currently unavailable"

        return JSONResponse(detail, status_code=status_code)


@app.post("/v1/media/analyze")
async def media_analyze(req: MediaAnalyzeRequest, request: Request):
    """Convenience endpoint for multimodal analysis.

    Constructs the proper OpenAI multimodal message format from a simple
    media_url + prompt payload, then forwards to the appropriate tier.
    """
    # Build the multimodal content parts
    content_parts = [
        {"type": "text", "text": req.prompt},
    ]

    if req.media_type == "image":
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": req.media_url},
        })
    elif req.media_type == "audio":
        content_parts.append({
            "type": "input_audio",
            "input_audio": {"data": req.media_url, "format": "wav"},
        })
    elif req.media_type == "video":
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": req.media_url},
        })
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported media_type: {req.media_type}. Use 'image', 'audio', or 'video'.",
        )

    messages = [{"role": "user", "content": content_parts}]

    chat_req = ChatRequest(
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        tier=req.tier,
    )
    return await chat(chat_req, request)


@app.post("/v1/media/upload")
async def media_upload(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form("Describe this in detail."),
    tier: Optional[str] = Form(None),
):
    """Process uploaded media (image/audio/video) and run inference.

    Accepts multipart/form-data with:
      - file: the media file (image, audio, or video)
      - prompt: text prompt to accompany the media (default: "Describe this in detail.")
      - tier: optional tier override ("fast", "primary", "heavy")

    The server validates, converts, and compresses the file to Gemma 4's native
    formats before forwarding to the model. No client-side conversion needed.
    """
    global _global_error_count
    request_id = getattr(request.state, "request_id", "unknown")
    client_ip = _get_client_ip(request)

    # Rate limit (media)
    if not _rate_limiter.check_media(client_ip):
        return JSONResponse(
            {"error": "Rate limit exceeded (media: 20 req/min)", "request_id": request_id},
            status_code=429,
        )

    # ── Stream upload to disk — avoids loading large files into RAM ────────────
    # Videos up to 10 GB are supported; images/audio are still small in practice.
    upload_tmp_path: Optional[str] = None
    result = None
    try:
        tmp_fd, upload_tmp_path = tempfile.mkstemp(prefix="gemma4_upload_")
        total_bytes = 0
        max_total = media_processor.MAX_VIDEO_BYTES
        with os.fdopen(tmp_fd, "wb") as fout:
            while True:
                chunk = await file.read(media_processor.UPLOAD_STREAM_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_total:
                    return JSONResponse(
                        {
                            "error": f"File too large (>{max_total // 1024 // 1024 // 1024} GB)",
                            "code": "too_large",
                        },
                        status_code=413,
                    )
                fout.write(chunk)

        if total_bytes == 0:
            return JSONResponse({"error": "Empty file", "code": "no_media"}, status_code=400)

        # ── Tier selection BEFORE media processing (frame budget depends on tier) ──
        chosen_tier_pre = tier if (tier and tier in TIER_MAP) else None
        if not chosen_tier_pre:
            if "heavy" in TIER_MAP and _tier_status.get("heavy"):
                chosen_tier_pre = "heavy"
            else:
                chosen_tier_pre = "primary"

        # Apply circuit-breaker fallback
        chosen_tier, fallback_note = _select_tier_with_fallback(chosen_tier_pre, request_id)
        url, model = TIER_MAP[chosen_tier]

        # Process the media server-side (tier-aware: controls frame budget for video)
        result = await media_processor.process_upload(
            path=upload_tmp_path,
            filename=file.filename or "unknown",
            declared_mime=file.content_type,
            tier=chosen_tier,
            file_size=total_bytes,
        )

        if isinstance(result, media_processor.MediaError):
            status = (
                400 if result.code in ("invalid_type", "no_media")
                else 413 if result.code == "too_large"
                else 422
            )
            return JSONResponse({"error": result.error, "code": result.code}, status_code=status)

        client = request.app.state.client
        t0 = time.monotonic()

        log.info(
            "Media upload: tier=%s strategy=%s file=%s",
            chosen_tier,
            getattr(result, "strategy", "direct"),
            file.filename,
            extra={"request_id": request_id},
        )

        # ── CHUNKED VIDEO: multiple model calls + aggregation ─────────────────
        if isinstance(result, media_processor.ChunkedVideoMedia):
            forward_result, chunk_analyses, latency = await _process_chunked_video(
                chunked=result,
                prompt=prompt,
                client=client,
                url=url,
                model=model,
                request_id=request_id,
                tier=chosen_tier,
            )
            _metrics[chosen_tier].record_request(latency)
            forward_result["_routing"] = {
                "tier": chosen_tier,
                "model": model,
                "latency_ms": round(latency * 1000),
                "strategy": "chunked",
                "chunks": len(result.chunks),
            }
            if fallback_note:
                forward_result["_routing"]["fallback"] = fallback_note
            forward_result["_media"] = {
                "category": "video",
                "original_name": result.original_name,
                "original_size": result.original_size,
                "duration_s": result.duration_s,
                "strategy": result.strategy,
                "chunk_count": len(result.chunks),
                "chunk_duration_s": result.chunk_duration_s,
                "frames_per_chunk": result.frames_per_chunk,
                "total_frames": result.total_frames,
                "warnings": result.warnings,
            }
            return JSONResponse(forward_result)

        # ── SINGLE-PASS (short video / image / audio) ─────────────────────────
        content_parts = []
        if prompt:
            content_parts.append({"type": "text", "text": prompt})
        content_parts.extend(result.content_parts)
        messages = [{"role": "user", "content": content_parts}]

        forward_result = await _forward(
            client, url, model, messages,
            max_tokens=1024, temperature=0.0,
            request_id=request_id, tier=chosen_tier,
        )
        latency = time.monotonic() - t0
        _metrics[chosen_tier].record_request(latency)

        forward_result["_routing"] = {
            "tier": chosen_tier,
            "model": model,
            "latency_ms": round(latency * 1000),
            "strategy": "direct",
        }
        if fallback_note:
            forward_result["_routing"]["fallback"] = fallback_note
        forward_result["_media"] = {
            "category": result.category,
            "original_name": result.original_name,
            "original_size": result.original_size,
            "processed_size": result.processed_size,
            "frame_count": result.frame_count,
            "duration_s": result.duration_s,
            "warnings": result.warnings,
        }
        return JSONResponse(forward_result)

    except Exception as e:
        with _global_error_lock:
            _global_error_count += 1
        log.error(
            "Media upload request failed: %s", str(e),
            extra={"request_id": request_id},
        )
        return JSONResponse(
            {"error": str(e), "request_id": request_id, "degraded": True},
            status_code=503,
        )
    finally:
        # Clean up: streamed upload temp file + any result temp files (audio WAV, etc.)
        if upload_tmp_path:
            try:
                os.remove(upload_tmp_path)
            except OSError:
                pass
        for tf in getattr(result, "temp_files", []):
            try:
                os.remove(tf)
            except OSError:
                pass


@app.post("/classify")
async def classify_endpoint(req: ClassifyRequest, request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    client = request.app.state.client
    t0 = time.monotonic()
    category = await _classify(client, req.text, request_id)
    latency = time.monotonic() - t0
    _metrics["fast"].record_request(latency)
    return {"category": category, "latency_ms": round(latency * 1000)}


@app.post("/compress")
async def compress_endpoint(req: CompressRequest, request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    client = request.app.state.client
    t0 = time.monotonic()
    data = await _forward(
        client,
        PRIMARY_URL,
        MODEL_PRIMARY,
        [
            {
                "role": "user",
                "content": (
                    f"Compress to {req.words} words, preserve key meaning:\n\n"
                    f"{req.text}"
                ),
            }
        ],
        max_tokens=128,
        request_id=request_id,
        tier="primary",
    )
    latency = time.monotonic() - t0
    _metrics["primary"].record_request(latency)
    content = data["choices"][0]["message"]["content"].strip()
    return {"compressed": content, "latency_ms": round(latency * 1000)}


# ---------------------------------------------------------------------------
# Web UI — serve the PWA from the gateway so everything is on one port
# ---------------------------------------------------------------------------

_WEB_DIR = Path(__file__).resolve().parent.parent / "cloud" / "web"


@app.get("/")
async def serve_ui():
    index = _WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html")
    return JSONResponse({"error": "Web UI not found"}, status_code=404)
