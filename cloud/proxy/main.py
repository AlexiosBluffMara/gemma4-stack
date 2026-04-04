"""Cloud Run proxy for Gemma 4 inference network.

Proxies requests to the Mac Mini gateway over Tailscale and serves the web UI.
"""

import os
import time
from collections import defaultdict
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://100.75.223.113:8080")
API_KEY = os.environ.get("API_KEY", "")
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "30"))
RATE_WINDOW = 60  # seconds

app = FastAPI(title="Gemma 4 Cloud Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate limiting (simple in-memory, per-IP)
# ---------------------------------------------------------------------------

_rate_counters: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if the request is allowed."""
    now = time.time()
    timestamps = _rate_counters[client_ip]
    # Purge entries older than the window
    _rate_counters[client_ip] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(_rate_counters[client_ip]) >= RATE_LIMIT:
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
# Shared HTTP client
# ---------------------------------------------------------------------------

_client: Optional[httpx.AsyncClient] = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


async def _proxy(request: Request, path: str) -> Response:
    """Forward a request to the upstream gateway."""
    client_ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "unknown")

    if not _check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded. Max 30 requests per minute."})

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

    client = await _get_client()
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
# Proxied routes
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
        "version": "1.0.0",
        "gateway_url": GATEWAY_URL,
        "upstream_healthy": upstream_healthy,
        "upstream": upstream_detail,
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


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("shutdown")
async def _shutdown():
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
