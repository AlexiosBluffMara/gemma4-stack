"""
Gemma 4 API Gateway — FastAPI router exposed over Tailscale.

Endpoints:
  POST /v1/chat/completions  — OpenAI-compatible, auto-routes across tiers
  POST /classify             — Fast tier classification only
  POST /compress             — Primary tier compression only
  GET  /health               — Status of all three tiers
  GET  /metrics              — Request counts and latency stats

Run:
  python3.12 ~/ai-scripts/gateway.py
  # or with uvicorn:
  uvicorn gateway:app --host 0.0.0.0 --port 8080
"""

import os
import time
import statistics
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Tier endpoints — configurable via env vars for Docker/Tailscale deployments
# Native macOS (MLX):  localhost:8082 / :8083 / :8081
# Docker (Ollama):     service names from docker-compose (fast:11434, etc.)
TIER_FAST    = os.getenv("FAST_URL",    "http://localhost:8082")
TIER_PRIMARY = os.getenv("PRIMARY_URL", "http://localhost:8083")
TIER_HEAVY   = os.getenv("HEAVY_URL",   "http://localhost:8081")

# Model names — MLX native mode uses full HF paths; Docker/Ollama mode uses short names
_docker_mode = os.getenv("DOCKER_MODE", "false").lower() == "true"
MODEL_FAST    = os.getenv("MODEL_FAST",    "gemma4:e2b"                       if _docker_mode else "mlx-community/gemma-4-e2b-it-4bit")
MODEL_PRIMARY = os.getenv("MODEL_PRIMARY", "gemma4:e4b"                       if _docker_mode else "mlx-community/gemma-4-e4b-it-4bit")
MODEL_HEAVY   = os.getenv("MODEL_HEAVY",   "gemma-4-26B")

TIER_MAP = {
    "fast":    (TIER_FAST,    MODEL_FAST),
    "primary": (TIER_PRIMARY, MODEL_PRIMARY),
    "heavy":   (TIER_HEAVY,   MODEL_HEAVY),
}

# Simple in-memory metrics
_metrics = {
    "fast":    {"count": 0, "latencies": []},
    "primary": {"count": 0, "latencies": []},
    "heavy":   {"count": 0, "latencies": []},
    "errors":  0,
}


def _record(tier, latency):
    _metrics[tier]["count"] += 1
    lats = _metrics[tier]["latencies"]
    lats.append(latency)
    if len(lats) > 1000:
        lats.pop(0)


async def _forward(client, url, model, messages, max_tokens=512, temperature=0.0):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = await client.post(f"{url}/v1/chat/completions", json=payload, timeout=180.0)
    r.raise_for_status()
    return r.json()


async def _classify(client, text):
    data = await _forward(
        client, TIER_FAST, MODEL_FAST,
        [{"role": "user", "content":
          f"Classify into one word (question/request/idea/greeting/fyi):\n{text}"}],
        max_tokens=8,
        temperature=0.0,
    )
    return data["choices"][0]["message"]["content"].strip().lower()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient()
    yield
    await app.state.client.aclose()


app = FastAPI(title="Gemma 4 Gateway", lifespan=lifespan)


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


@app.get("/health")
async def health(request: Request):
    client = request.app.state.client
    status = {}
    for name, (url, _) in TIER_MAP.items():
        # Try /health first (mlx_lm, llama.cpp), fall back to /api/tags (Ollama)
        for path in ["/health", "/api/tags"]:
            try:
                r = await client.get(f"{url}{path}", timeout=3.0)
                if r.status_code == 200:
                    status[name] = "ok"
                    break
            except Exception:
                pass
        else:
            status[name] = "offline"
    return {"tiers": status}


@app.get("/metrics")
async def metrics():
    out = {}
    for tier, data in _metrics.items():
        if tier == "errors":
            continue
        lats = data["latencies"]
        out[tier] = {
            "requests": data["count"],
            "p50_ms": round(statistics.median(lats) * 1000) if lats else None,
            "p95_ms": round(sorted(lats)[int(len(lats) * 0.95)] * 1000) if len(lats) >= 20 else None,
        }
    out["errors"] = _metrics["errors"]
    return out


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request):
    client = request.app.state.client
    t0 = time.time()

    try:
        # Determine tier
        if req.tier and req.tier in TIER_MAP:
            tier = req.tier
        else:
            # Auto-route based on fast classification of last user message
            last_user = next(
                (m["content"] for m in reversed(req.messages) if m["role"] == "user"),
                ""
            )
            category = await _classify(client, last_user)
            if category in ("greeting", "fyi"):
                tier = "fast"
            elif category in ("question", "request"):
                tier = "primary"
            else:
                tier = "heavy"

        url, model = TIER_MAP[tier]
        result = await _forward(client, url, model, req.messages, req.max_tokens, req.temperature)
        _record(tier, time.time() - t0)

        # Inject routing metadata into response
        result["_routing"] = {"tier": tier, "latency_ms": round((time.time() - t0) * 1000)}
        return JSONResponse(result)

    except Exception as e:
        _metrics["errors"] += 1
        raise HTTPException(status_code=503, detail=str(e))


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
        client, TIER_PRIMARY, MODEL_PRIMARY,
        [{"role": "user", "content": f"Compress to {req.words} words, preserve key meaning:\n\n{req.text}"}],
        max_tokens=128,
    )
    _record("primary", time.time() - t0)
    content = data["choices"][0]["message"]["content"].strip()
    return {"compressed": content, "latency_ms": round((time.time() - t0) * 1000)}
