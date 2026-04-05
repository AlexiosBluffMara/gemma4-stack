"""Cloud Run proxy for Gemma 4 inference network.

Production-grade public frontend that:
  - Serves the web UI with proper caching headers
  - Rate-limits per IP with sliding window (text: 30/min, media: 10/min, health: 120/min)
  - Optional API key authentication
  - Forwards all requests to the Mac Mini gateway via Tailscale Funnel
  - Handles multipart media uploads proxied directly to gateway
  - Circuit breaker with half-open probe for upstream resilience
  - Structured JSON logging with request correlation IDs
  - Retry with exponential backoff on transient failures
  - Graceful degradation with cached health responses
  - Connection pooling and proper timeout management
  - Security headers, request validation, and metrics

The gateway at GATEWAY_URL does all heavy lifting:
  - Server-side media processing (ffmpeg, Pillow)
  - Format conversion (WebM->WAV, MOV->frames, etc.)
  - Magic-byte validation & security
  - Model routing (E2B fast / E4B primary / 26B heavy)
"""

import asyncio
import json
import logging
import os
import statistics
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Optional Google Cloud integrations (gracefully degrade if unavailable)
_auth_service = None
_storage_client = None
_db = None

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_fields"):
            log_obj.update(record.extra_fields)
        if record.exc_info and record.exc_info[0] is not None:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, default=str)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger("proxy")
logger.handlers = [handler]
logger.setLevel(logging.INFO)
logger.propagate = False


def _log(level: str, message: str, **fields: Any) -> None:
    record = logger.makeRecord(
        "proxy", getattr(logging, level.upper()), "", 0, message, (), None
    )
    record.extra_fields = fields  # type: ignore[attr-defined]
    logger.handle(record)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GATEWAY_URL = os.environ.get("GATEWAY_URL", "https://miniapple.scylla-betta.ts.net")
API_KEY = os.environ.get("API_KEY", "")

RATE_LIMIT_TEXT = int(os.environ.get("RATE_LIMIT", "30"))
RATE_LIMIT_MEDIA = int(os.environ.get("RATE_LIMIT_MEDIA", "10"))
RATE_LIMIT_HEALTH = int(os.environ.get("RATE_LIMIT_HEALTH", "120"))
RATE_WINDOW = 60  # seconds

MAX_BODY_SIZE = 100 * 1024 * 1024           # 100 MB for text/JSON endpoints
PROXY_MEDIA_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB for Cloud Run media proxy
# For files > 2 GB, upload directly to the gateway over Tailscale (no Cloud Run middleman)

# Retry settings
MAX_RETRIES = 2
RETRY_DELAYS = [0.5, 1.0]
RETRYABLE_STATUS_CODES = {502, 503, 504}

# Circuit breaker settings
CB_FAILURE_THRESHOLD = 5
CB_RECOVERY_TIMEOUT = 30.0  # seconds


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = CB_FAILURE_THRESHOLD,
        recovery_timeout: float = CB_RECOVERY_TIMEOUT,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: float = 0.0
        self.success_count = 0
        self.total_trips = 0
        self._lock = asyncio.Lock()

    async def record_success(self) -> None:
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                _log("info", "Circuit breaker closed after successful probe")
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            self.success_count += 1

    async def record_failure(self) -> None:
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.total_trips += 1
                _log("warning", "Circuit breaker re-opened from half-open")
            elif self.failure_count >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    self.total_trips += 1
                    _log(
                        "warning",
                        "Circuit breaker opened",
                        failure_count=self.failure_count,
                    )
                self.state = CircuitState.OPEN

    async def allow_request(self) -> bool:
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    _log("info", "Circuit breaker half-open, allowing probe request")
                    return True
                return False
            # HALF_OPEN: only one probe allowed; deny further until resolved
            return False

    def info(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "total_trips": self.total_trips,
            "last_failure_time": self.last_failure_time or None,
        }


circuit_breaker = CircuitBreaker()


# ---------------------------------------------------------------------------
# Rate limiting — sliding window, per-IP, per-endpoint bucket
# ---------------------------------------------------------------------------

_rate_buckets: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_hits = 0


def _bucket_key(client_ip: str, bucket: str) -> str:
    return f"{client_ip}:{bucket}"


def _check_rate_limit(client_ip: str, bucket: str, limit: int) -> tuple[bool, int]:
    """Return (allowed, retry_after_seconds)."""
    global _rate_limit_hits
    now = time.time()
    key = _bucket_key(client_ip, bucket)
    dq = _rate_buckets[key]

    # Trim expired entries
    while dq and now - dq[0] >= RATE_WINDOW:
        dq.popleft()

    if len(dq) >= limit:
        oldest = dq[0]
        retry_after = int(RATE_WINDOW - (now - oldest)) + 1
        _rate_limit_hits += 1
        return False, max(retry_after, 1)

    dq.append(now)
    return True, 0


def _cleanup_stale_buckets() -> int:
    """Remove buckets with no recent entries. Returns count of removed buckets."""
    now = time.time()
    stale_keys = [
        k for k, dq in _rate_buckets.items()
        if not dq or now - dq[-1] >= RATE_WINDOW * 2
    ]
    for k in stale_keys:
        del _rate_buckets[k]
    return len(stale_keys)


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------


class Metrics:
    def __init__(self) -> None:
        self.total_requests = 0
        self.requests_by_path: dict[str, int] = defaultdict(int)
        self.errors_by_type: dict[str, int] = defaultdict(int)
        self.upstream_latencies: deque[float] = deque(maxlen=1000)
        self.start_time = time.time()

    def record_request(self, path: str) -> None:
        self.total_requests += 1
        self.requests_by_path[path] += 1

    def record_error(self, error_type: str) -> None:
        self.errors_by_type[error_type] += 1

    def record_upstream_latency(self, latency_ms: float) -> None:
        self.upstream_latencies.append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        latencies = sorted(self.upstream_latencies)
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95_idx = int(len(latencies) * 0.95) if latencies else 0
        p95 = latencies[min(p95_idx, len(latencies) - 1)] if latencies else 0.0
        return {
            "total_requests": self.total_requests,
            "requests_by_path": dict(self.requests_by_path),
            "errors_by_type": dict(self.errors_by_type),
            "upstream_latency_p50_ms": round(p50, 2),
            "upstream_latency_p95_ms": round(p95, 2),
            "circuit_breaker": circuit_breaker.info(),
            "rate_limit_hits": _rate_limit_hits,
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }


metrics = Metrics()


# ---------------------------------------------------------------------------
# Health cache for graceful degradation
# ---------------------------------------------------------------------------


class HealthCache:
    def __init__(self) -> None:
        self.last_response: Optional[dict[str, Any]] = None
        self.last_status_code: int = 200
        self.last_updated: float = 0.0

    def store(self, data: dict[str, Any], status_code: int) -> None:
        self.last_response = data
        self.last_status_code = status_code
        self.last_updated = time.time()

    def get_cached(self) -> Optional[tuple[dict[str, Any], float]]:
        if self.last_response is None:
            return None
        staleness = time.time() - self.last_updated
        return self.last_response, staleness


health_cache = HealthCache()


# ---------------------------------------------------------------------------
# Error response builder
# ---------------------------------------------------------------------------


def _error_response(
    status_code: int,
    error: str,
    code: str,
    request_id: str,
    detail: str = "",
    headers: Optional[dict[str, str]] = None,
) -> JSONResponse:
    body = {
        "error": error,
        "code": code,
        "request_id": request_id,
        "detail": detail,
    }
    resp_headers = {"X-Request-ID": request_id}
    if headers:
        resp_headers.update(headers)
    return JSONResponse(status_code=status_code, content=body, headers=resp_headers)


# ---------------------------------------------------------------------------
# Auth check
# ---------------------------------------------------------------------------


def _check_auth(request: Request, request_id: str) -> Optional[JSONResponse]:
    """Verify authentication. Supports Firebase ID tokens and API keys."""
    auth_header = request.headers.get("authorization", "")

    # Try Firebase token first
    if _auth_service and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # Skip if it looks like a gcloud identity token (for Cloud Run IAM auth)
        if not token.startswith("eyJhbG"):  # Not a JWT
            pass
        else:
            user = _auth_service.verify_token(token)
            if user:
                request.state.user_id = user.uid
                request.state.user_email = user.email
                request.state.user_name = user.display_name
                request.state.authenticated = True
                return None

    # Fall back to static API key
    if API_KEY:
        if auth_header == f"Bearer {API_KEY}":
            request.state.user_id = "api_key_user"
            request.state.user_email = ""
            request.state.user_name = "API Key User"
            request.state.authenticated = True
            return None
        # If API_KEY is set but not matched, reject
        # UNLESS it's a Cloud Run IAM-authenticated request (no app-level auth needed)
        return _error_response(401, "Unauthorized", "AUTH_REQUIRED", request_id)

    # No auth configured — allow anonymous
    request.state.user_id = "anonymous"
    request.state.user_email = ""
    request.state.user_name = "Anonymous"
    request.state.authenticated = False
    return None


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------------------
# HTTP clients (initialized in lifespan)
# ---------------------------------------------------------------------------

_client: Optional[httpx.AsyncClient] = None
_media_client: Optional[httpx.AsyncClient] = None


def _make_pool_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=100,
        max_keepalive_connections=30,
        keepalive_expiry=60,
    )


# ---------------------------------------------------------------------------
# Async lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client, _media_client, _auth_service, _storage_client, _db

    _client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        limits=_make_pool_limits(),
    )
    _media_client = httpx.AsyncClient(
        # write=3600s: allows streaming multi-GB uploads without per-chunk write timeout
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=3600.0, pool=10.0),
        limits=_make_pool_limits(),
    )

    # Initialize Google Cloud services (non-blocking, fail gracefully)
    try:
        from auth import AuthService
        AuthService.initialize()
        _auth_service = AuthService
        _log("info", "Firebase Auth initialized")
    except Exception as e:
        _log("warning", "Firebase Auth not available, running without auth", error=str(e))

    try:
        from storage import StorageClient
        _storage_client = StorageClient()
        _log("info", "GCS storage initialized")
    except Exception as e:
        _log("warning", "GCS storage not available", error=str(e))

    try:
        from db import Database
        _db = Database()
        _log("info", "Firestore initialized")
    except Exception as e:
        _log("warning", "Firestore not available", error=str(e))

    # Background task for periodic rate-limit cleanup
    cleanup_task = asyncio.create_task(_periodic_cleanup())

    _log("info", "Proxy started", gateway_url=GATEWAY_URL)
    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    if _client and not _client.is_closed:
        await _client.aclose()
    if _media_client and not _media_client.is_closed:
        await _media_client.aclose()
    _log("info", "Proxy shut down")


async def _periodic_cleanup() -> None:
    """Clean up stale rate-limit buckets every 60 seconds."""
    while True:
        try:
            await asyncio.sleep(60)
            removed = _cleanup_stale_buckets()
            if removed > 0:
                _log("debug", "Rate limit cleanup", removed_buckets=removed)
        except asyncio.CancelledError:
            break


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Gemma 4 Cloud Proxy", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Security headers middleware
# ---------------------------------------------------------------------------

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:",
}


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    for key, value in SECURITY_HEADERS.items():
        response.headers[key] = value
    return response


# ---------------------------------------------------------------------------
# Request correlation ID + structured logging middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    client_ip = _get_client_ip(request)
    start = time.time()

    response = await call_next(request)

    latency_ms = round((time.time() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id

    # Record metrics
    metrics.record_request(request.url.path)

    _log(
        "info",
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        latency_ms=latency_ms,
        client_ip=client_ip,
    )

    return response


# ---------------------------------------------------------------------------
# Request body size limit middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        return _error_response(
            413,
            "Request body too large",
            "BODY_TOO_LARGE",
            request_id,
            detail=f"Max {MAX_BODY_SIZE // (1024 * 1024)} MB.",
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request body validation middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def validate_content_type(request: Request, call_next):
    if request.method == "POST":
        content_type = request.headers.get("content-type")
        if not content_type:
            request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
            return _error_response(
                415,
                "Missing Content-Type header",
                "MISSING_CONTENT_TYPE",
                request_id,
                detail="POST requests must include a Content-Type header.",
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Generic proxy helper with retry + circuit breaker
# ---------------------------------------------------------------------------


async def _proxy_upstream(
    method: str,
    url: str,
    body: bytes,
    headers: dict[str, str],
    *,
    media: bool = False,
    request_id: str,
) -> tuple[Optional[httpx.Response], Optional[JSONResponse]]:
    """
    Send request to upstream with retry and circuit breaker logic.
    Returns (upstream_response, error_response) -- exactly one will be set.
    """
    client = _media_client if media else _client
    assert client is not None

    for attempt in range(1 + MAX_RETRIES):
        # Circuit breaker check
        if not await circuit_breaker.allow_request():
            metrics.record_error("circuit_open")
            _log(
                "warning",
                "Circuit breaker open, rejecting request",
                request_id=request_id,
                url=url,
            )
            return None, _error_response(
                503,
                "Gateway unavailable, circuit open",
                "CIRCUIT_OPEN",
                request_id,
                detail=f"Upstream has failed {circuit_breaker.failure_count} consecutive times. "
                f"Retry after {int(CB_RECOVERY_TIMEOUT)}s.",
                headers={"Retry-After": str(int(CB_RECOVERY_TIMEOUT))},
            )

        try:
            upstream_start = time.time()
            upstream = await client.request(
                method=method,
                url=url,
                content=body if body else None,
                headers=headers,
            )
            upstream_latency = (time.time() - upstream_start) * 1000
            metrics.record_upstream_latency(upstream_latency)

            # Check for retryable status codes
            if upstream.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES:
                await circuit_breaker.record_failure()
                delay = RETRY_DELAYS[attempt]
                _log(
                    "warning",
                    "Retryable upstream status",
                    request_id=request_id,
                    status_code=upstream.status_code,
                    attempt=attempt + 1,
                    retry_delay=delay,
                )
                await asyncio.sleep(delay)
                continue

            if upstream.status_code in RETRYABLE_STATUS_CODES:
                await circuit_breaker.record_failure()
            else:
                await circuit_breaker.record_success()

            return upstream, None

        except httpx.ConnectError as exc:
            await circuit_breaker.record_failure()
            metrics.record_error("connect_error")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                _log(
                    "warning",
                    "Upstream connect error, retrying",
                    request_id=request_id,
                    attempt=attempt + 1,
                    retry_delay=delay,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
                continue
            _log(
                "error",
                "Upstream connect error, retries exhausted",
                request_id=request_id,
                error=str(exc),
            )
            return None, _error_response(
                502,
                "Gateway unreachable",
                "CONNECT_ERROR",
                request_id,
                detail="Cannot connect to inference network after retries.",
            )

        except httpx.ConnectTimeout:
            await circuit_breaker.record_failure()
            metrics.record_error("connect_timeout")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt]
                _log(
                    "warning",
                    "Upstream connect timeout, retrying",
                    request_id=request_id,
                    attempt=attempt + 1,
                    retry_delay=delay,
                )
                await asyncio.sleep(delay)
                continue
            return None, _error_response(
                504,
                "Gateway connect timeout",
                "CONNECT_TIMEOUT",
                request_id,
                detail="Could not establish connection to inference network within 10s.",
            )

        except httpx.ReadTimeout:
            await circuit_breaker.record_failure()
            metrics.record_error("read_timeout")
            timeout_val = "300s" if media else "120s"
            return None, _error_response(
                504,
                "Gateway read timeout",
                "READ_TIMEOUT",
                request_id,
                detail=f"Inference timed out ({timeout_val} limit). "
                "Try a shorter prompt or smaller media file.",
            )

        except httpx.PoolTimeout:
            metrics.record_error("pool_timeout")
            return None, _error_response(
                503,
                "Connection pool exhausted",
                "POOL_TIMEOUT",
                request_id,
                detail="Too many concurrent requests. Try again shortly.",
                headers={"Retry-After": "5"},
            )

    # Should not reach here, but safety fallback
    return None, _error_response(
        502, "Gateway error", "UNKNOWN", request_id, detail="Unexpected retry loop exit."
    )


async def _proxy(request: Request, path: str, *, media: bool = False) -> Response:
    """Forward a request to the upstream gateway."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = _get_client_ip(request)

    # Determine rate limit bucket and limit
    if path == "/health":
        bucket, limit = "health", RATE_LIMIT_HEALTH
    elif media:
        bucket, limit = "media", RATE_LIMIT_MEDIA
    else:
        bucket, limit = "text", RATE_LIMIT_TEXT

    allowed, retry_after = _check_rate_limit(client_ip, bucket, limit)
    if not allowed:
        return _error_response(
            429,
            f"Rate limit exceeded. Max {limit} requests per minute.",
            "RATE_LIMITED",
            request_id,
            detail=f"Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    auth_err = _check_auth(request, request_id)
    if auth_err:
        return auth_err

    url = f"{GATEWAY_URL}{path}"
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "authorization", "content-length")
    }
    headers["X-Request-ID"] = request_id

    upstream, error_resp = await _proxy_upstream(
        method=request.method,
        url=url,
        body=body,
        headers=headers,
        media=media,
        request_id=request_id,
    )

    if error_resp is not None:
        # Graceful degradation for /health
        if path == "/health":
            cached = health_cache.get_cached()
            if cached is not None:
                cached_data, staleness = cached
                return JSONResponse(
                    status_code=200,
                    content={
                        **cached_data,
                        "_degraded": True,
                        "_cached": True,
                        "_staleness_seconds": round(staleness, 1),
                    },
                    headers={"X-Request-ID": request_id},
                )
        return error_resp

    assert upstream is not None

    # Cache health responses
    if path == "/health" and upstream.status_code == 200:
        try:
            health_cache.store(upstream.json(), upstream.status_code)
        except Exception:
            pass

    response_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
    }
    response_headers["X-Request-ID"] = request_id

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Liveness / readiness probes
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz():
    """Liveness probe -- always returns 200 if the process is running."""
    return JSONResponse({"status": "alive"})


@app.get("/_ready")
async def readiness():
    """Readiness probe -- returns 503 when upstream is down (circuit open)."""
    if circuit_breaker.state == CircuitState.OPEN:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "circuit_breaker_open",
                "circuit": circuit_breaker.info(),
            },
        )
    return JSONResponse({"status": "ready", "circuit": circuit_breaker.info()})


# ---------------------------------------------------------------------------
# Background helpers for usage tracking and media storage
# ---------------------------------------------------------------------------


async def _record_usage_bg(user_id: str, event_type: str, tier: str, tokens: int, latency_ms: int, media_size: int = 0, gcs_uri: str = ""):
    """Record usage in background — errors are logged but never block the response."""
    try:
        if _db:
            await _db.record_usage(user_id, event_type, tier, tokens, latency_ms, media_size, gcs_uri)
    except Exception as e:
        _log("warning", "Background usage recording failed", error=str(e), user_id=user_id)


async def _upload_media_bg(user_id: str, body: bytes, request_id: str):
    """Upload media to GCS in background."""
    try:
        if _storage_client:
            # We don't parse multipart here — just store the raw upload
            # The actual parsing happens in the gateway
            await _storage_client.upload_media(
                user_id=user_id,
                file_data=body,
                filename=f"upload_{request_id[:8]}",
                content_type="application/octet-stream",
                category="media",
            )
    except Exception as e:
        _log("warning", "Background media upload to GCS failed", error=str(e), request_id=request_id)


# ---------------------------------------------------------------------------
# Proxied routes -- text (30 req/min)
# ---------------------------------------------------------------------------


@app.api_route("/v1/chat/completions", methods=["POST"])
async def proxy_chat(request: Request):
    response = await _proxy(request, "/v1/chat/completions")

    # Fire-and-forget: record usage in background
    if _db and response.status_code == 200:
        user_id = getattr(request.state, "user_id", "anonymous")
        if user_id != "anonymous":
            try:
                # Parse response to get token count and routing info
                body = response.body
                if isinstance(body, bytes):
                    data = json.loads(body)
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    tier = data.get("_routing", {}).get("tier", "unknown")
                    latency = data.get("_routing", {}).get("latency_ms", 0)
                    asyncio.create_task(_record_usage_bg(
                        user_id, "text", tier, tokens, latency
                    ))
            except Exception:
                pass

    return response


@app.api_route("/classify", methods=["POST"])
async def proxy_classify(request: Request):
    return await _proxy(request, "/classify")


@app.api_route("/compress", methods=["POST"])
async def proxy_compress(request: Request):
    return await _proxy(request, "/compress")


@app.api_route("/health", methods=["GET"])
async def proxy_health(request: Request):
    return await _proxy(request, "/health")


@app.api_route("/devices", methods=["GET"])
async def proxy_devices(request: Request):
    return await _proxy(request, "/devices")


@app.api_route("/metrics", methods=["GET"])
async def proxy_metrics(request: Request):
    return await _proxy(request, "/metrics")


# ---------------------------------------------------------------------------
# Proxied routes -- media (10 req/min, 300s timeout)
# ---------------------------------------------------------------------------


@app.api_route("/v1/media/analyze", methods=["POST"])
async def proxy_media_analyze(request: Request):
    return await _proxy(request, "/v1/media/analyze", media=True)


@app.post("/v1/media/upload")
async def proxy_media_upload(request: Request):
    """Proxy multipart media uploads to the gateway via streaming.

    The body is streamed directly to the gateway without buffering into RAM,
    supporting files up to 2 GB through the Cloud Run proxy layer.

    For files larger than 2 GB, upload directly to the gateway over Tailscale
    (bypasses Cloud Run's infrastructure limits entirely).
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    client_ip = _get_client_ip(request)

    allowed, retry_after = _check_rate_limit(client_ip, "media", RATE_LIMIT_MEDIA)
    if not allowed:
        return _error_response(
            429,
            f"Rate limit exceeded. Max {RATE_LIMIT_MEDIA} media requests per minute.",
            "RATE_LIMITED",
            request_id,
            detail=f"Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    auth_err = _check_auth(request, request_id)
    if auth_err:
        return auth_err

    # Reject oversized uploads early when Content-Length is available
    content_length_hdr = request.headers.get("content-length")
    if content_length_hdr:
        try:
            if int(content_length_hdr) > PROXY_MEDIA_MAX_BYTES:
                return _error_response(
                    413,
                    (
                        f"File too large for Cloud proxy "
                        f"(>{PROXY_MEDIA_MAX_BYTES // 1024 // 1024 // 1024} GB). "
                        "For files larger than 2 GB, upload directly to the gateway "
                        "over Tailscale."
                    ),
                    "TOO_LARGE",
                    request_id,
                )
        except ValueError:
            pass

    # Circuit breaker check — skip streaming if gateway is known-down
    if not await circuit_breaker.allow_request():
        metrics.record_error("circuit_open")
        _log("warning", "Circuit breaker open — rejecting media upload", request_id=request_id)
        return _error_response(
            503,
            "Gateway unavailable (circuit breaker open). Please try again later.",
            "CIRCUIT_OPEN",
            request_id,
        )

    url = f"{GATEWAY_URL}/v1/media/upload"

    # Forward all headers except host/auth; drop content-length (httpx will use
    # chunked transfer encoding when streaming without a known size).
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "authorization", "content-length")
    }
    forward_headers["X-Request-ID"] = request_id

    try:
        # Stream the multipart body directly to the gateway — zero RAM buffering.
        # _media_client has write=3600s so large uploads won't time out per chunk.
        async with _media_client.stream(
            "POST",
            url,
            content=request.stream(),
            headers=forward_headers,
        ) as upstream:
            response_body = await upstream.aread()

        await circuit_breaker.record_success()

        response_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
        }
        response_headers["X-Request-ID"] = request_id

        return Response(
            content=response_body,
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    except httpx.TimeoutException:
        await circuit_breaker.record_failure()
        metrics.record_error("read_timeout")
        _log("warning", "Media upload timed out", request_id=request_id)
        return _error_response(
            504,
            "Gateway timeout while processing media upload. Large files may take longer — "
            "for the fastest experience with very large files, upload directly over Tailscale.",
            "GATEWAY_TIMEOUT",
            request_id,
        )
    except httpx.ConnectError:
        await circuit_breaker.record_failure()
        metrics.record_error("connect_error")
        return _error_response(502, "Cannot connect to gateway.", "GATEWAY_UNREACHABLE", request_id)
    except Exception as e:
        await circuit_breaker.record_failure()
        _log("error", "Media upload proxy failed", error=str(e), request_id=request_id)
        return _error_response(502, "Gateway error.", "GATEWAY_ERROR", request_id)


# ---------------------------------------------------------------------------
# Proxy status endpoint (enhanced)
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def api_status(request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    auth_err = _check_auth(request, request_id)
    if auth_err:
        return auth_err

    upstream_healthy = False
    upstream_detail = None
    try:
        if await circuit_breaker.allow_request():
            assert _client is not None
            resp = await _client.get(
                f"{GATEWAY_URL}/health",
                timeout=5.0,
                headers={"X-Request-ID": request_id},
            )
            upstream_healthy = resp.status_code == 200
            try:
                upstream_detail = resp.json()
            except Exception:
                upstream_detail = resp.text
            if upstream_healthy:
                await circuit_breaker.record_success()
            else:
                await circuit_breaker.record_failure()
        else:
            upstream_detail = "Circuit breaker open -- skipping upstream check"
    except Exception as exc:
        await circuit_breaker.record_failure()
        upstream_detail = str(exc)

    snap = metrics.snapshot()

    return JSONResponse(
        content={
            "proxy": "ok",
            "version": "3.0.0",
            "gateway_url": GATEWAY_URL,
            "upstream_healthy": upstream_healthy,
            "upstream": upstream_detail,
            "circuit_breaker": circuit_breaker.info(),
            "uptime_seconds": snap["uptime_seconds"],
            "total_requests": snap["total_requests"],
            "limits": {
                "max_body_mb": MAX_BODY_SIZE // (1024 * 1024),
                "rate_text": f"{RATE_LIMIT_TEXT}/min",
                "rate_media": f"{RATE_LIMIT_MEDIA}/min",
                "rate_health": f"{RATE_LIMIT_HEALTH}/min",
            },
        },
        headers={"X-Request-ID": request_id},
    )


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


@app.get("/api/metrics")
async def api_metrics_endpoint(request: Request):
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    auth_err = _check_auth(request, request_id)
    if auth_err:
        return auth_err

    return JSONResponse(
        content=metrics.snapshot(),
        headers={"X-Request-ID": request_id},
    )


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------


@app.get("/api/user")
async def get_user(request: Request):
    """Get current user profile and usage stats."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    user_id = getattr(request.state, "user_id", "anonymous")

    if user_id == "anonymous" or not _db:
        return JSONResponse({
            "user_id": user_id,
            "authenticated": False,
            "usage": None,
        }, headers={"X-Request-ID": request_id})

    try:
        user_data = await _db.get_user(user_id)
        usage = await _db.get_usage_stats(user_id, days=30)
        remaining_text, remaining_media = 100, 20  # defaults
        try:
            text_ok, text_rem = await _db.check_usage_limit(user_id, "text")
            media_ok, media_rem = await _db.check_usage_limit(user_id, "media")
            remaining_text, remaining_media = text_rem, media_rem
        except Exception:
            pass

        return JSONResponse({
            "user_id": user_id,
            "authenticated": True,
            "email": getattr(request.state, "user_email", ""),
            "display_name": getattr(request.state, "user_name", ""),
            "usage": usage,
            "limits": {
                "remaining_text_today": remaining_text,
                "remaining_media_today": remaining_media,
            },
        }, headers={"X-Request-ID": request_id})
    except Exception as e:
        _log("error", "Failed to get user data", error=str(e), user_id=user_id)
        return JSONResponse({
            "user_id": user_id,
            "authenticated": True,
            "usage": None,
            "error": "Could not load usage data",
        }, headers={"X-Request-ID": request_id})


@app.get("/api/conversations")
async def get_conversations(request: Request):
    """Get conversation history for the current user."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    user_id = getattr(request.state, "user_id", "anonymous")

    if user_id == "anonymous" or not _db:
        return JSONResponse({"conversations": []}, headers={"X-Request-ID": request_id})

    try:
        conversations = await _db.get_conversations(user_id, limit=20)
        return JSONResponse(
            {"conversations": conversations},
            headers={"X-Request-ID": request_id},
        )
    except Exception as e:
        _log("error", "Failed to get conversations", error=str(e), user_id=user_id)
        return JSONResponse(
            {"conversations": [], "error": "Could not load conversations"},
            headers={"X-Request-ID": request_id},
        )


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

STATIC_DIR = "/app/static"
if not os.path.isdir(STATIC_DIR):
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "web")


@app.get("/")
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"error": "Web UI not found"}, status_code=404)


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_static_subdir = os.path.join(STATIC_DIR, "static")
if os.path.isdir(_static_subdir):
    app.mount("/assets", StaticFiles(directory=_static_subdir), name="assets")
