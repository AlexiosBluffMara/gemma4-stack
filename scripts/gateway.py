"""
Gemma 4 API Gateway — Multi-device FastAPI router for MLX inference network.

Devices:
  Mac Mini M4 (always on):  fast (E2B :8082), primary (E4B :8083)
  MacBook Pro M4 Max (intermittent via Tailscale): heavy (26B-A4B :8084)

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible, auto-routes across tiers
  POST /v1/media/analyze     — Convenience wrapper for multimodal analysis
  POST /classify             — Fast tier classification only
  POST /compress             — Primary tier compression only
  GET  /health               — Status of all three tiers
  GET  /metrics              — Per-tier request counts and p50/p95 latencies
  GET  /devices              — Registered devices, IPs, online status, capabilities

Run:
  uvicorn gateway:app --host 0.0.0.0 --port 8080
"""

import asyncio
import os
import time
import statistics
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAST_URL = os.getenv("FAST_URL", "http://localhost:8082")
PRIMARY_URL = os.getenv("PRIMARY_URL", "http://localhost:8083")
HEAVY_URL = os.getenv("HEAVY_URL", "http://localhost:8084")
DEVICE_NAME = os.getenv("DEVICE_NAME", "mac-mini")

MODEL_FAST = "mlx-community/gemma-4-e2b-it-4bit"
MODEL_PRIMARY = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_HEAVY = "mlx-community/gemma-4-26b-a4b-it-4bit"

TIER_MAP = {
    "fast": (FAST_URL, MODEL_FAST),
    "primary": (PRIMARY_URL, MODEL_PRIMARY),
    "heavy": (HEAVY_URL, MODEL_HEAVY),
}

HEALTH_CHECK_INTERVAL = 30  # seconds

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
    "macbook-pro": {
        "ip": HEAVY_URL.replace("http://", "").split(":")[0],
        "online": False,
        "last_seen": None,
        "capabilities": ["heavy"],
        "tiers": {
            "heavy": {"url": HEAVY_URL, "model": MODEL_HEAVY},
        },
    },
}

# ---------------------------------------------------------------------------
# Tier health status (updated by background task)
# ---------------------------------------------------------------------------

_tier_status: dict[str, bool] = {
    "fast": False,
    "primary": False,
    "heavy": False,
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_metrics: dict[str, dict] = {
    "fast": {"count": 0, "latencies": []},
    "primary": {"count": 0, "latencies": []},
    "heavy": {"count": 0, "latencies": []},
    "errors": 0,
}


def _record(tier: str, latency: float) -> None:
    _metrics[tier]["count"] += 1
    lats = _metrics[tier]["latencies"]
    lats.append(latency)
    if len(lats) > 1000:
        lats.pop(0)


# ---------------------------------------------------------------------------
# Health check background task
# ---------------------------------------------------------------------------

async def _check_tier(client: httpx.AsyncClient, name: str, url: str) -> bool:
    try:
        r = await client.get(f"{url}/health", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


async def _health_loop(app: FastAPI) -> None:
    """Periodically health-check all tiers and update device/tier status."""
    client = app.state.client
    while True:
        # Check local tiers
        for tier_name in ("fast", "primary"):
            url, _ = TIER_MAP[tier_name]
            _tier_status[tier_name] = await _check_tier(client, tier_name, url)

        # Update local device status
        local_dev = _devices[DEVICE_NAME]
        local_dev["online"] = any(
            _tier_status[t] for t in local_dev["capabilities"]
        )
        if local_dev["online"]:
            local_dev["last_seen"] = datetime.now(timezone.utc).isoformat()

        # Check remote heavy tier
        heavy_online = await _check_tier(client, "heavy", HEAVY_URL)
        _tier_status["heavy"] = heavy_online
        remote_dev = _devices["macbook-pro"]
        remote_dev["online"] = heavy_online
        if heavy_online:
            remote_dev["last_seen"] = datetime.now(timezone.utc).isoformat()

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
# Forwarding helpers
# ---------------------------------------------------------------------------

async def _forward(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    messages: list,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = await client.post(
        f"{url}/v1/chat/completions", json=payload, timeout=180.0
    )
    r.raise_for_status()
    return r.json()


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


async def _classify(client: httpx.AsyncClient, text: str) -> str:
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
    )
    return data["choices"][0]["message"]["content"].strip().lower()


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient()
    health_task = asyncio.create_task(_health_loop(app))
    yield
    health_task.cancel()
    try:
        await health_task
    except asyncio.CancelledError:
        pass
    await app.state.client.aclose()


app = FastAPI(title="Gemma 4 Gateway", lifespan=lifespan)

# CORS — allow all origins for the cloud-hosted web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: list
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    tier: Optional[str] = None  # force a tier: "fast" | "primary" | "heavy"


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
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health(request: Request):
    client = request.app.state.client
    status = {}
    for name, (url, _) in TIER_MAP.items():
        status[name] = "ok" if _tier_status.get(name) else "offline"
    # Also do a live check so /health is always fresh
    for name, (url, _) in TIER_MAP.items():
        ok = await _check_tier(client, name, url)
        _tier_status[name] = ok
        status[name] = "ok" if ok else "offline"
    return {
        "tiers": status,
        "heavy_device": {
            "url": HEAVY_URL,
            "online": _tier_status["heavy"],
        },
    }


@app.get("/metrics")
async def metrics():
    out = {}
    for tier in ("fast", "primary", "heavy"):
        data = _metrics[tier]
        lats = data["latencies"]
        out[tier] = {
            "requests": data["count"],
            "p50_ms": round(statistics.median(lats) * 1000) if lats else None,
            "p95_ms": (
                round(sorted(lats)[int(len(lats) * 0.95)] * 1000)
                if len(lats) >= 20
                else None
            ),
        }
    out["errors"] = _metrics["errors"]
    return out


@app.get("/devices")
async def devices():
    return {"devices": _devices}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    client = request.app.state.client
    t0 = time.time()
    fallback_note = None

    try:
        # Determine tier
        if req.tier and req.tier in TIER_MAP:
            tier = req.tier
        elif _has_media(req.messages):
            # Media requests: prefer heavy for best quality, fall back to primary
            # Never route media to fast tier (E2B can handle it but E4B is better)
            if _tier_status.get("heavy"):
                tier = "heavy"
            else:
                tier = "primary"
                if not _tier_status.get("primary"):
                    fallback_note = (
                        "Heavy tier offline, primary tier also offline. "
                        "Attempting primary anyway for media request."
                    )
        else:
            # Auto-route based on fast classification of last user message
            last_user = _extract_text(req.messages)
            category = await _classify(client, last_user)
            if category in ("greeting", "fyi"):
                tier = "fast"
            elif category in ("summarize", "compress"):
                tier = "primary"
            else:
                # complex / code / analysis / question / request -> heavy
                tier = "heavy"

        # Heavy tier fallback: if heavy is offline, use primary
        if tier == "heavy" and not _tier_status.get("heavy"):
            tier = "primary"
            fallback_note = fallback_note or (
                "Heavy tier (26B-A4B on MacBook Pro) is offline. "
                "Falling back to primary tier (E4B)."
            )

        url, model = TIER_MAP[tier]
        result = await _forward(
            client, url, model, req.messages, req.max_tokens, req.temperature
        )
        _record(tier, time.time() - t0)

        # Inject routing metadata
        result["_routing"] = {
            "tier": tier,
            "model": model,
            "latency_ms": round((time.time() - t0) * 1000),
        }
        if fallback_note:
            result["_routing"]["fallback"] = fallback_note

        return JSONResponse(result)

    except Exception as e:
        _metrics["errors"] += 1
        raise HTTPException(status_code=503, detail=str(e))


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
        # Video is treated as image_url (many backends accept video this way)
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

    # Build a ChatRequest and delegate to the chat endpoint
    chat_req = ChatRequest(
        messages=messages,
        max_tokens=1024,
        temperature=0.0,
        tier=req.tier,
    )
    return await chat(chat_req, request)


@app.post("/classify")
async def classify_endpoint(req: ClassifyRequest, request: Request):
    client = request.app.state.client
    t0 = time.time()
    category = await _classify(client, req.text)
    _record("fast", time.time() - t0)
    return {"category": category, "latency_ms": round((time.time() - t0) * 1000)}


@app.post("/compress")
async def compress_endpoint(req: CompressRequest, request: Request):
    client = request.app.state.client
    t0 = time.time()
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
    )
    _record("primary", time.time() - t0)
    content = data["choices"][0]["message"]["content"].strip()
    return {"compressed": content, "latency_ms": round((time.time() - t0) * 1000)}
