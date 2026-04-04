"""Cloud Run proxy for Gemma 4 inference network.

Proxies requests to the Mac Mini gateway over Tailscale and serves the web UI.
Supports multimodal (vision/audio) uploads with increased body size limits.
"""

import base64
import mimetypes
import os
import time
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://100.75.223.113:8080")
API_KEY = os.environ.get("API_KEY", "")
RATE_LIMIT_TEXT = int(os.environ.get("RATE_LIMIT", "30"))
RATE_LIMIT_MEDIA = int(os.environ.get("RATE_LIMIT_MEDIA", "10"))
RATE_WINDOW = 60  # seconds

# Maximum request body size: 50 MB (for base64-encoded images/audio)
MAX_BODY_SIZE = 50 * 1024 * 1024

app = FastAPI(title="Gemma 4 Cloud Proxy", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request body size limit middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Reject requests larger than MAX_BODY_SIZE."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={
                "error": "Request body too large",
                "detail": f"Maximum allowed size is {MAX_BODY_SIZE // (1024 * 1024)} MB.",
            },
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory, per-IP, per-bucket)
# ---------------------------------------------------------------------------

_rate_counters: dict[str, list[float]] = defaultdict(list)


def _get_client_ip(request: Request) -> str:
    return request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")


def _check_rate_limit(client_ip: str, limit: int = RATE_LIMIT_TEXT) -> bool:
    """Return True if the request is allowed."""
    now = time.time()
    timestamps = _rate_counters[client_ip]
    # Purge entries older than the window
    _rate_counters[client_ip] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(_rate_counters[client_ip]) >= limit:
        return False
    _rate_counters[client_ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Auth check
# ---------------------------------------------------------------------------


def _check_auth(request: Request) -> Optional[JSONResponse]:
    """Return an error response if auth is required and missing/invalid."""
    if not API_KEY:
        return None
    auth = request.headers.get("authorization", "")
    if auth == f"Bearer {API_KEY}":
        return None
    return JSONResponse(status_code=401, content={"error": "Unauthorized"})


# ---------------------------------------------------------------------------
# Shared HTTP clients (text vs media timeouts)
# ---------------------------------------------------------------------------

_client: Optional[httpx.AsyncClient] = None
_media_client: Optional[httpx.AsyncClient] = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


async def _get_media_client() -> httpx.AsyncClient:
    """Client with extended timeout for multimodal inference."""
    global _media_client
    if _media_client is None or _media_client.is_closed:
        _media_client = httpx.AsyncClient(timeout=300.0)
    return _media_client


async def _proxy(request: Request, path: str, *, media: bool = False) -> Response:
    """Forward a request to the upstream gateway.

    Args:
        request: The incoming request.
        path: The upstream path to forward to.
        media: If True, use the media client (300s timeout) and media rate limit.
    """
    client_ip = _get_client_ip(request)
    limit = RATE_LIMIT_MEDIA if media else RATE_LIMIT_TEXT

    if not _check_rate_limit(client_ip, limit=limit):
        return JSONResponse(
            status_code=429,
            content={"error": f"Rate limit exceeded. Max {limit} requests per minute."},
        )

    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    url = f"{GATEWAY_URL}{path}"
    body = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "authorization", "content-length")
    }

    client = await (_get_media_client() if media else _get_client())
    try:
        upstream = await client.request(
            method=request.method,
            url=url,
            content=body if body else None,
            headers=headers,
        )
    except httpx.ConnectError:
        return JSONResponse(status_code=502, content={"error": "Gateway unreachable", "detail": "Cannot connect to Mac Mini gateway via Tailscale."})
    except httpx.ReadTimeout:
        return JSONResponse(status_code=504, content={"error": "Gateway timeout", "detail": "Upstream did not respond in time."})

    response_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Proxied routes — text (30 req/min)
# ---------------------------------------------------------------------------

@app.api_route("/v1/chat/completions", methods=["POST"])
async def proxy_chat(request: Request):
    return await _proxy(request, "/v1/chat/completions")


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
# Proxied routes — media (10 req/min, 300s timeout)
# ---------------------------------------------------------------------------

@app.api_route("/v1/media/analyze", methods=["POST"])
async def proxy_media_analyze(request: Request):
    """Proxy multimodal analysis requests to the gateway."""
    return await _proxy(request, "/v1/media/analyze", media=True)


@app.api_route("/v1/media/upload", methods=["POST"])
async def media_upload(
    request: Request,
    file: UploadFile = File(..., description="Image or audio file to analyze"),
    prompt: str = Form(..., description="Text prompt to accompany the media"),
    tier: Optional[str] = Form(None, description="Inference tier: e2b, e4b, or 26b"),
):
    """Accept a multipart file upload, convert to base64, and forward as an
    OpenAI-compatible multimodal chat completion to the gateway.

    This is a convenience endpoint for mobile clients and curl users who
    prefer multipart form uploads over constructing base64 JSON payloads.
    """
    client_ip = _get_client_ip(request)

    if not _check_rate_limit(client_ip, limit=RATE_LIMIT_MEDIA):
        return JSONResponse(
            status_code=429,
            content={"error": f"Rate limit exceeded. Max {RATE_LIMIT_MEDIA} media requests per minute."},
        )

    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    # Read and encode the uploaded file
    file_bytes = await file.read()
    if len(file_bytes) > MAX_BODY_SIZE:
        return JSONResponse(
            status_code=413,
            content={"error": "Uploaded file too large", "detail": f"Max {MAX_BODY_SIZE // (1024 * 1024)} MB."},
        )

    b64_data = base64.b64encode(file_bytes).decode("ascii")

    # Determine MIME type
    mime_type = file.content_type
    if not mime_type or mime_type == "application/octet-stream":
        guessed, _ = mimetypes.guess_type(file.filename or "file.bin")
        mime_type = guessed or "application/octet-stream"

    data_url = f"data:{mime_type};base64,{b64_data}"

    # Construct OpenAI multimodal message format
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
    }

    if tier:
        payload["tier"] = tier

    # Forward to the gateway's media/analyze endpoint
    import json

    body = json.dumps(payload).encode("utf-8")
    headers = {"content-type": "application/json"}

    client = await _get_media_client()
    try:
        upstream = await client.post(
            f"{GATEWAY_URL}/v1/media/analyze",
            content=body,
            headers=headers,
        )
    except httpx.ConnectError:
        return JSONResponse(status_code=502, content={"error": "Gateway unreachable", "detail": "Cannot connect to Mac Mini gateway via Tailscale."})
    except httpx.ReadTimeout:
        return JSONResponse(status_code=504, content={"error": "Gateway timeout", "detail": "Multimodal inference timed out (300s limit)."})

    response_headers = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
    }
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
        media_type=upstream.headers.get("content-type"),
    )


# ---------------------------------------------------------------------------
# Proxy status endpoint
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def api_status(request: Request):
    auth_err = _check_auth(request)
    if auth_err:
        return auth_err

    upstream_healthy = False
    upstream_detail = None
    client = await _get_client()
    try:
        resp = await client.get(f"{GATEWAY_URL}/health", timeout=5.0)
        upstream_healthy = resp.status_code == 200
        try:
            upstream_detail = resp.json()
        except Exception:
            upstream_detail = resp.text
    except Exception as exc:
        upstream_detail = str(exc)

    return {
        "proxy": "ok",
        "version": "1.1.0",
        "gateway_url": GATEWAY_URL,
        "upstream_healthy": upstream_healthy,
        "upstream": upstream_detail,
        "limits": {
            "max_body_mb": MAX_BODY_SIZE // (1024 * 1024),
            "rate_text": f"{RATE_LIMIT_TEXT}/min",
            "rate_media": f"{RATE_LIMIT_MEDIA}/min",
        },
    }


# ---------------------------------------------------------------------------
# Web UI
# ---------------------------------------------------------------------------

STATIC_DIR = "/app/static"
# Fallback for local development
if not os.path.isdir(STATIC_DIR):
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "web")


@app.get("/")
async def index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"error": "Web UI not found"}, status_code=404)


# Mount static files last so explicit routes take priority
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Also mount a static/ subdirectory if it exists (for app icons, assets, etc.)
_static_subdir = os.path.join(STATIC_DIR, "static")
if os.path.isdir(_static_subdir):
    app.mount("/assets", StaticFiles(directory=_static_subdir), name="assets")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def _shutdown():
    global _client, _media_client
    if _client and not _client.is_closed:
        await _client.aclose()
    if _media_client and not _media_client.is_closed:
        await _media_client.aclose()
