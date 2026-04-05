"""
Gemma 4 26B Heavy Tier — Cloud Run GPU Service
===============================================
Runs on NVIDIA L4 (24 GB VRAM) with min-instances=0.

Startup sequence:
  1. Download model from GCS bucket (gemma4good-models) if not cached
     ~80 seconds from us-central1 at ~200 MB/s
  2. Load model into GPU via llama-cpp-python
     ~30 seconds on L4
  3. Start serving OpenAI-compatible /v1/chat/completions

Total cold start: ~110 seconds.
After that: ~50-70 tokens/second for the 26B-A4B model.

Environment variables (all from Secret Manager / Cloud Run env):
  PORT               Cloud Run injection (default: 8080)
  GCS_BUCKET         GCS bucket name (default: gemma4good-models)
  GCS_MODEL_BLOB     Object path inside bucket (default: gemma-4-26b-q4.gguf)
  HF_TOKEN           HuggingFace token for fallback download (optional)
  HEAVY_API_KEY      Shared secret sent by the gateway (required)
  N_GPU_LAYERS       GPU layers to offload, -1 = all (default: -1)
  CTX_SIZE           Context window size in tokens (default: 8192)
  N_PARALLEL         Simultaneous request slots (default: 4)
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("heavy")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("PORT", "8080"))
GCS_BUCKET = os.environ.get("GCS_BUCKET", "gemma4good-models")
GCS_MODEL_BLOB = os.environ.get("GCS_MODEL_BLOB", "gemma-4-26b-q4.gguf")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HEAVY_API_KEY = os.environ.get("HEAVY_API_KEY", "")
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "-1"))  # -1 = all on GPU
CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))
N_PARALLEL = int(os.environ.get("N_PARALLEL", "4"))
MODEL_PATH = Path("/tmp/gemma-4-26b.gguf")

# HuggingFace fallback (if GCS download fails)
# Override via env vars to avoid rebuilding the image when switching quants.
HF_REPO = os.environ.get("HF_REPO", "unsloth/gemma-4-26B-A4B-it-GGUF")
HF_FILE = os.environ.get("HF_FILE", "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------


def _download_from_gcs() -> bool:
    """Download model from GCS bucket. Returns True on success."""
    try:
        from google.cloud import storage
        log.info(f"Downloading model from gs://{GCS_BUCKET}/{GCS_MODEL_BLOB}")
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_BLOB)

        # Stream to temp file first, then atomic rename
        tmp = MODEL_PATH.with_suffix(".tmp")
        t0 = time.monotonic()
        blob.download_to_filename(str(tmp))
        tmp.rename(MODEL_PATH)
        elapsed = time.monotonic() - t0
        size_gb = MODEL_PATH.stat().st_size / 1e9
        log.info(f"GCS download complete: {size_gb:.1f} GB in {elapsed:.0f}s")
        return True
    except Exception as e:
        log.warning(f"GCS download failed: {e}")
        return False


def _download_from_hf() -> bool:
    """Download model from HuggingFace as fallback. Returns True on success."""
    try:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from huggingface_hub import hf_hub_download
        log.info(f"Downloading model from HuggingFace: {HF_REPO}/{HF_FILE}")
        t0 = time.monotonic()
        local = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILE,
            local_dir="/tmp/hf_dl",
            token=HF_TOKEN or None,
        )
        shutil.move(local, MODEL_PATH)
        elapsed = time.monotonic() - t0
        size_gb = MODEL_PATH.stat().st_size / 1e9
        log.info(f"HF download complete: {size_gb:.1f} GB in {elapsed:.0f}s")
        return True
    except Exception as e:
        log.error(f"HuggingFace download also failed: {e}")
        return False


def ensure_model():
    """Download model if not already on disk. Tries GCS first, HF as fallback."""
    if MODEL_PATH.exists():
        size_gb = MODEL_PATH.stat().st_size / 1e9
        log.info(f"Model already at {MODEL_PATH} ({size_gb:.1f} GB)")
        return

    log.info("Model not found — downloading...")
    if not _download_from_gcs():
        if not _download_from_hf():
            raise RuntimeError("All model download attempts failed — cannot start heavy tier")


# ---------------------------------------------------------------------------
# LLM state
# ---------------------------------------------------------------------------

_llm = None
_model_loaded = False


def load_model():
    """Load model into GPU using llama-cpp-python."""
    global _llm, _model_loaded
    from llama_cpp import Llama
    log.info(f"Loading model into GPU (n_gpu_layers={N_GPU_LAYERS}, ctx={CTX_SIZE}, parallel={N_PARALLEL})")
    t0 = time.monotonic()
    _llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=CTX_SIZE,
        n_gpu_layers=N_GPU_LAYERS,
        n_threads=4,
        n_batch=512,
        verbose=False,
    )
    elapsed = time.monotonic() - t0
    _model_loaded = True
    log.info(f"Model loaded in {elapsed:.1f}s — ready to serve")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run blocking download + model load in a thread (don't block the event loop)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, ensure_model)
    await loop.run_in_executor(None, load_model)
    log.info(f"Heavy tier ready on port {PORT}")
    yield
    # Cleanup (llama_cpp Llama has no explicit close; GC handles it)
    global _llm
    _llm = None


app = FastAPI(
    title="Gemma 4 Heavy Tier",
    description="26B-A4B inference via llama-cpp-python on NVIDIA L4",
    lifespan=lifespan,
)


def _check_api_key(request: Request):
    """Validate the shared API key from the gateway."""
    if not HEAVY_API_KEY:
        return  # No key configured — open access (dev mode)
    auth = request.headers.get("Authorization", "")
    if auth == f"Bearer {HEAVY_API_KEY}":
        return
    # Also accept X-Heavy-Key header
    if request.headers.get("X-Heavy-Key", "") == HEAVY_API_KEY:
        return
    raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = "gemma-4-26b-a4b"
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request):
    """OpenAI-compatible chat completions endpoint."""
    _check_api_key(request)

    if not _model_loaded or _llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet — cold start in progress")

    # Convert to llama_cpp format
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # Run inference in executor so we don't block the event loop
    loop = asyncio.get_event_loop()
    t0 = time.monotonic()

    response = await loop.run_in_executor(
        None,
        lambda: _llm.create_chat_completion(
            messages=messages,
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature if req.temperature is not None else 0.0,
            top_p=req.top_p or 0.95,
        ),
    )

    latency = time.monotonic() - t0
    output_tokens = response.get("usage", {}).get("completion_tokens", 0)
    tps = output_tokens / latency if latency > 0 else 0
    log.info(f"Inference: {output_tokens} tokens in {latency:.1f}s ({tps:.1f} tok/s)")

    # Ensure response includes model name
    response["model"] = "mlx-community/gemma-4-26b-a4b-it-4bit"
    return JSONResponse(response)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "mlx-community/gemma-4-26b-a4b-it-4bit",
            "object": "model",
            "owned_by": "google",
        }],
    }


# ---------------------------------------------------------------------------
# Health / readiness probes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {
        "status": "ok" if _model_loaded else "loading",
        "model_loaded": _model_loaded,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    }


@app.get("/healthz")
async def healthz():
    """Liveness probe — always 200 if process is running."""
    return {"status": "alive"}


@app.get("/_ready")
async def readiness():
    """Readiness probe — 503 until model is loaded."""
    if not _model_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "model_loading"},
        )
    return {"status": "ready"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="warning",
        workers=1,  # single worker — GPU is shared
    )
