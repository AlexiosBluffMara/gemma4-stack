"""
Shared fixtures for Gemma 4 stack end-to-end tests.

Fixtures:
  - gateway_client: httpx.AsyncClient pointed at localhost:8080 (the gateway)
  - cloud_client:   httpx.AsyncClient pointed at the Cloud Run proxy (requires CLOUD_PROXY_URL + CLOUD_API_KEY)
  - tiny_jpeg:      1x1 white pixel JPEG (bytes)
  - tiny_png:       1x1 white pixel PNG (bytes)
  - tiny_wav:       0.1s of silence at 16 kHz mono, 16-bit PCM WAV (bytes)
"""

import io
import os
import struct

import httpx
import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8080")
CLOUD_PROXY_URL = os.getenv("CLOUD_PROXY_URL", "")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")

# ---------------------------------------------------------------------------
# httpx clients
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def gateway_client():
    """Async HTTP client pointed at the local gateway."""
    async with httpx.AsyncClient(
        base_url=GATEWAY_URL, timeout=30.0
    ) as client:
        yield client


@pytest_asyncio.fixture
async def cloud_client():
    """Async HTTP client pointed at the Cloud Run proxy.

    Skips tests automatically when CLOUD_PROXY_URL is not set.
    """
    if not CLOUD_PROXY_URL:
        pytest.skip("CLOUD_PROXY_URL not set — skipping cloud proxy tests")
    headers = {}
    if CLOUD_API_KEY:
        headers["Authorization"] = f"Bearer {CLOUD_API_KEY}"
    async with httpx.AsyncClient(
        base_url=CLOUD_PROXY_URL, headers=headers, timeout=60.0
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Media fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_jpeg() -> bytes:
    """32x32 white pixel JPEG image (minimum size for MLX vision encoder)."""
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture
def tiny_png() -> bytes:
    """32x32 white pixel PNG image (minimum size for MLX vision encoder)."""
    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tiny_wav() -> bytes:
    """0.1 seconds of silence — 16 kHz mono 16-bit PCM WAV."""
    sample_rate = 16000
    num_channels = 1
    bits_per_sample = 16
    duration_s = 0.1
    num_samples = int(sample_rate * duration_s)
    bytes_per_sample = bits_per_sample // 8
    data_size = num_samples * num_channels * bytes_per_sample

    buf = io.BytesIO()
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))  # file size - 8
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 1))  # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * bytes_per_sample))  # byte rate
    buf.write(struct.pack("<H", num_channels * bytes_per_sample))  # block align
    buf.write(struct.pack("<H", bits_per_sample))
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)  # silence
    return buf.getvalue()


@pytest.fixture
def corrupted_jpeg() -> bytes:
    """Starts with JPEG magic bytes but is otherwise garbage."""
    return b"\xff\xd8\xff\xe0" + b"\xde\xad\xbe\xef" * 64


@pytest.fixture
def text_file_bytes() -> bytes:
    """Plain UTF-8 text — should be rejected by media endpoints."""
    return b"This is a plain text file, not an image."
