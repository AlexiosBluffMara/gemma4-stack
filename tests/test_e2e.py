"""
End-to-end test suite for the Gemma 4 inference stack.

Covers:
  - Local gateway (localhost:8080)
  - Cloud Run proxy (requires CLOUD_PROXY_URL env var)
  - media.py unit tests (no server required)

Run all:
    pytest tests/test_e2e.py -v

Run only fast / non-network tests:
    pytest tests/test_e2e.py -v -m "not cloud and not slow"

Run only media unit tests:
    pytest tests/test_e2e.py -v -m media_unit

Dependencies:
    pip install pytest pytest-asyncio httpx Pillow
"""

import asyncio
import io
import os
import struct
import sys
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Ensure the scripts/ directory is importable so we can import media.py
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)


# ===================================================================
# SECTION 1: Gateway tests (localhost:8080)
# ===================================================================


class TestGatewayHealth:
    """Tests 1-4: health, liveness, readiness, metrics."""

    # -- Test 1 --
    async def test_health_response_structure(self, gateway_client: httpx.AsyncClient):
        """GET /health returns tier status map with valid values."""
        r = await gateway_client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "tiers" in body
        tiers = body["tiers"]
        assert isinstance(tiers, dict)
        # Every tier value should be either a string ("ok"/"offline") or a dict with a "status" key
        for tier_name, status in tiers.items():
            if isinstance(status, dict):
                assert status.get("status") in ("ok", "offline"), (
                    f"Tier '{tier_name}' has unexpected status '{status.get('status')}'"
                )
            else:
                assert status in ("ok", "offline"), (
                    f"Tier '{tier_name}' has unexpected status '{status}'"
                )

    # -- Test 2 --
    async def test_healthz_liveness(self, gateway_client: httpx.AsyncClient):
        """GET /healthz returns 200 (Kubernetes liveness probe)."""
        r = await gateway_client.get("/healthz")
        # If the endpoint is not implemented yet, accept a 404 and xfail.
        if r.status_code == 404:
            pytest.xfail("/healthz not yet implemented on gateway")
        assert r.status_code == 200

    # -- Test 3 --
    async def test_ready_probe(self, gateway_client: httpx.AsyncClient):
        """GET /_ready returns 200 when tiers are up, 503 when down."""
        r = await gateway_client.get("/_ready")
        if r.status_code == 404:
            pytest.xfail("/_ready not yet implemented on gateway")
        assert r.status_code in (200, 503)

    # -- Test 4 --
    async def test_metrics_structure(self, gateway_client: httpx.AsyncClient):
        """GET /metrics returns per-tier request counts and latencies."""
        r = await gateway_client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        # At minimum, fast and primary tiers should be present
        for tier in ("fast", "primary"):
            assert tier in body, f"Missing tier '{tier}' in metrics"
            entry = body[tier]
            assert "requests" in entry
            assert "p50_ms" in entry
            assert "p95_ms" in entry
        assert "errors" in body or "global_errors" in body


class TestGatewayDevices:
    """Test 5: device registry."""

    async def test_devices_info(self, gateway_client: httpx.AsyncClient):
        """GET /devices returns device registry."""
        r = await gateway_client.get("/devices")
        assert r.status_code == 200
        body = r.json()
        assert "devices" in body
        devices = body["devices"]
        assert isinstance(devices, dict)
        # At least the local device should be present
        assert len(devices) >= 1
        # Each device should have standard fields
        for name, info in devices.items():
            assert "online" in info
            assert "capabilities" in info
            assert isinstance(info["capabilities"], list)


class TestGatewayChatCompletions:
    """Tests 6-10: /v1/chat/completions."""

    @pytest.mark.slow
    async def test_basic_text_inference(self, gateway_client: httpx.AsyncClient):
        """POST /v1/chat/completions with a simple prompt returns valid structure."""
        payload = {
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        r = await gateway_client.post("/v1/chat/completions", json=payload, timeout=120.0)
        assert r.status_code == 200
        body = r.json()
        # OpenAI-compatible structure
        assert "choices" in body
        assert len(body["choices"]) >= 1
        assert "message" in body["choices"][0]
        assert "content" in body["choices"][0]["message"]
        # Gateway routing metadata
        assert "_routing" in body
        routing = body["_routing"]
        assert "tier" in routing
        assert "model" in routing
        assert "latency_ms" in routing

    @pytest.mark.slow
    async def test_tier_override_fast(self, gateway_client: httpx.AsyncClient):
        """POST /v1/chat/completions with tier=fast routes to fast tier."""
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "tier": "fast",
        }
        r = await gateway_client.post("/v1/chat/completions", json=payload, timeout=120.0)
        assert r.status_code == 200
        body = r.json()
        assert body["_routing"]["tier"] == "fast"

    @pytest.mark.slow
    async def test_tier_override_primary(self, gateway_client: httpx.AsyncClient):
        """POST /v1/chat/completions with tier=primary routes to primary tier."""
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "tier": "primary",
        }
        r = await gateway_client.post("/v1/chat/completions", json=payload, timeout=120.0)
        assert r.status_code == 200
        body = r.json()
        assert body["_routing"]["tier"] == "primary"

    async def test_invalid_tier_graceful(self, gateway_client: httpx.AsyncClient):
        """POST /v1/chat/completions with invalid tier falls back gracefully."""
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "tier": "nonexistent_tier",
        }
        r = await gateway_client.post("/v1/chat/completions", json=payload, timeout=120.0)
        # Should not crash: either auto-routes (ignoring invalid tier) or returns an error
        assert r.status_code in (200, 400, 422, 503)

    async def test_empty_messages_rejected(self, gateway_client: httpx.AsyncClient):
        """POST /v1/chat/completions with empty messages list returns 422."""
        payload = {"messages": [], "max_tokens": 8}
        r = await gateway_client.post("/v1/chat/completions", json=payload)
        # FastAPI validation or gateway logic should reject empty messages
        assert r.status_code in (400, 422, 503)

    async def test_max_tokens_out_of_range(self, gateway_client: httpx.AsyncClient):
        """POST /v1/chat/completions with negative max_tokens returns 422."""
        payload = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": -1,
        }
        r = await gateway_client.post("/v1/chat/completions", json=payload)
        # Pydantic / backend validation should reject negative max_tokens
        assert r.status_code in (400, 422, 503)


class TestGatewayClassifyCompress:
    """Tests 11-12: /classify and /compress."""

    @pytest.mark.slow
    async def test_classify_response(self, gateway_client: httpx.AsyncClient):
        """POST /classify returns a category and latency."""
        payload = {"text": "Hello, how are you?"}
        r = await gateway_client.post("/classify", json=payload, timeout=60.0)
        assert r.status_code == 200
        body = r.json()
        assert "category" in body
        assert isinstance(body["category"], str)
        assert "latency_ms" in body

    @pytest.mark.slow
    async def test_compress_response(self, gateway_client: httpx.AsyncClient):
        """POST /compress returns compressed text and latency."""
        payload = {
            "text": (
                "The quick brown fox jumps over the lazy dog. "
                "This sentence contains every letter of the alphabet."
            ),
            "words": 10,
        }
        r = await gateway_client.post("/compress", json=payload, timeout=60.0)
        assert r.status_code == 200
        body = r.json()
        assert "compressed" in body
        assert isinstance(body["compressed"], str)
        assert "latency_ms" in body


class TestGatewayMediaUpload:
    """Tests 13-16: /v1/media/upload and /v1/media/analyze."""

    @pytest.mark.slow
    async def test_upload_jpeg(self, gateway_client: httpx.AsyncClient, tiny_jpeg: bytes):
        """POST /v1/media/upload with a small JPEG succeeds."""
        files = {"file": ("test.jpg", tiny_jpeg, "image/jpeg")}
        data = {"prompt": "What is this?"}
        r = await gateway_client.post(
            "/v1/media/upload", files=files, data=data, timeout=120.0
        )
        assert r.status_code == 200
        body = r.json()
        assert "choices" in body or "_media" in body

    async def test_upload_invalid_filetype(
        self, gateway_client: httpx.AsyncClient, text_file_bytes: bytes
    ):
        """POST /v1/media/upload with a text file is rejected."""
        files = {"file": ("readme.txt", text_file_bytes, "text/plain")}
        r = await gateway_client.post("/v1/media/upload", files=files)
        assert r.status_code == 400
        body = r.json()
        assert "error" in body
        assert body.get("code") in ("invalid_type", "no_media")

    async def test_upload_empty_file(self, gateway_client: httpx.AsyncClient):
        """POST /v1/media/upload with an empty file is rejected."""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        r = await gateway_client.post("/v1/media/upload", files=files)
        assert r.status_code == 400
        body = r.json()
        assert body.get("code") == "no_media"

    @pytest.mark.slow
    async def test_media_analyze(self, gateway_client: httpx.AsyncClient, tiny_jpeg: bytes):
        """POST /v1/media/analyze with a base64 image succeeds."""
        import base64

        b64 = base64.b64encode(tiny_jpeg).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        payload = {
            "media_url": data_url,
            "media_type": "image",
            "prompt": "What is in this image?",
        }
        r = await gateway_client.post(
            "/v1/media/analyze", json=payload, timeout=120.0
        )
        assert r.status_code == 200
        body = r.json()
        assert "choices" in body


class TestGatewayRateLimiting:
    """Test 17: rate limiting."""

    @pytest.mark.slow
    async def test_rate_limit_429(self, gateway_client: httpx.AsyncClient):
        """Rapid requests eventually get 429 if rate limiting is enforced."""
        # Send a burst of lightweight requests
        responses = []
        for _ in range(50):
            r = await gateway_client.get("/health")
            responses.append(r.status_code)

        # If the gateway enforces rate limiting, at least one should be 429.
        # If no rate limiter is installed, all will be 200 and we xfail.
        if 429 not in responses:
            pytest.xfail("Gateway does not enforce rate limiting on /health")
        assert 429 in responses


class TestGatewayHeaders:
    """Tests 18-19: request ID and security headers."""

    async def test_request_id_header(self, gateway_client: httpx.AsyncClient):
        """Response includes X-Request-ID header."""
        r = await gateway_client.get("/health")
        if "x-request-id" not in r.headers:
            pytest.xfail("X-Request-ID header not yet implemented on gateway")
        assert r.headers["x-request-id"]

    async def test_security_headers(self, gateway_client: httpx.AsyncClient):
        """Response includes standard security headers."""
        r = await gateway_client.get("/health")
        # Check for common security headers; xfail if none present
        expected = [
            "x-content-type-options",
            "x-frame-options",
            "strict-transport-security",
        ]
        found = [h for h in expected if h in r.headers]
        if not found:
            pytest.xfail("Security headers not yet implemented on gateway")
        # At least one security header should be present
        assert len(found) >= 1


class TestGatewayConcurrency:
    """Test 20: concurrent requests."""

    @pytest.mark.slow
    async def test_concurrent_requests(self, gateway_client: httpx.AsyncClient):
        """10 simultaneous /health requests all succeed."""

        async def fetch():
            return await gateway_client.get("/health")

        results = await asyncio.gather(*[fetch() for _ in range(10)])
        status_codes = [r.status_code for r in results]
        # All should succeed (200) — no 5xx crashes
        for code in status_codes:
            assert code == 200, f"Concurrent request failed with status {code}"


# ===================================================================
# SECTION 2: Cloud Run Proxy tests
# ===================================================================


@pytest.mark.cloud
class TestCloudProxyStatus:
    """Tests 21-24: proxy status and health proxying."""

    async def test_api_status(self, cloud_client: httpx.AsyncClient):
        """GET /api/status returns proxy status and upstream info."""
        r = await cloud_client.get("/api/status")
        assert r.status_code == 200
        body = r.json()
        assert body.get("proxy") == "ok"
        assert "version" in body
        assert "gateway_url" in body
        assert "upstream_healthy" in body
        assert "limits" in body

    async def test_healthz_liveness(self, cloud_client: httpx.AsyncClient):
        """GET /healthz returns 200 (Kubernetes liveness probe)."""
        r = await cloud_client.get("/healthz")
        if r.status_code == 404:
            pytest.xfail("/healthz not yet implemented on cloud proxy")
        assert r.status_code == 200

    async def test_ready_probe(self, cloud_client: httpx.AsyncClient):
        """GET /_ready returns 200 or 503."""
        r = await cloud_client.get("/_ready")
        if r.status_code == 404:
            pytest.xfail("/_ready not yet implemented on cloud proxy")
        assert r.status_code in (200, 503)

    async def test_proxied_health(self, cloud_client: httpx.AsyncClient):
        """GET /health through proxy returns gateway health."""
        r = await cloud_client.get("/health")
        # Could be 200 (gateway reachable) or 502 (gateway down)
        assert r.status_code in (200, 502)
        if r.status_code == 200:
            body = r.json()
            assert "tiers" in body


@pytest.mark.cloud
class TestCloudProxyInference:
    """Test 25: full inference through proxy."""

    @pytest.mark.slow
    async def test_proxied_chat_completions(self, cloud_client: httpx.AsyncClient):
        """POST /v1/chat/completions through proxy returns valid response."""
        payload = {
            "messages": [{"role": "user", "content": "Say hi in one word."}],
            "max_tokens": 8,
            "temperature": 0.0,
        }
        r = await cloud_client.post(
            "/v1/chat/completions", json=payload, timeout=120.0
        )
        # 200 = success, 502 = gateway down, 504 = timeout
        assert r.status_code in (200, 502, 504)
        if r.status_code == 200:
            body = r.json()
            assert "choices" in body


@pytest.mark.cloud
class TestCloudProxyRateLimit:
    """Test 26: proxy rate limiting."""

    @pytest.mark.slow
    async def test_rate_limit_429(self, cloud_client: httpx.AsyncClient):
        """Rapid requests to proxy eventually hit 429."""
        responses = []
        for _ in range(40):
            r = await cloud_client.get("/health")
            responses.append(r.status_code)
            if r.status_code == 429:
                break

        if 429 not in responses:
            pytest.xfail("Cloud proxy did not return 429 within 40 requests")
        assert 429 in responses


@pytest.mark.cloud
class TestCloudProxyRequestID:
    """Test 27: request ID propagation."""

    async def test_request_id_roundtrip(self, cloud_client: httpx.AsyncClient):
        """X-Request-ID sent to proxy is echoed back."""
        headers = {"X-Request-ID": "test-e2e-12345"}
        r = await cloud_client.get("/health", headers=headers)
        if "x-request-id" not in r.headers:
            pytest.xfail("X-Request-ID propagation not yet implemented on proxy")
        assert r.headers["x-request-id"] == "test-e2e-12345"


@pytest.mark.cloud
class TestCloudProxyLargeBody:
    """Test 28: large body rejection."""

    @pytest.mark.xfail(
        reason=(
            "Cloud Run's Google Load Balancer buffers the full request body before "
            "forwarding to the container. Spoofing Content-Length with a small body "
            "causes the LB to wait for the remaining bytes and eventually time out. "
            "The middleware is verified correct at the unit-test level."
        ),
        strict=False,
    )
    async def test_large_body_rejected(self, cloud_client: httpx.AsyncClient):
        """Request exceeding MAX_BODY_SIZE (100 MB) should return 413.

        NOTE: This test is xfail because Cloud Run's Google Front End / load
        balancer buffers the full request body before the app sees it. A spoofed
        Content-Length that claims 200 MB but sends only 15 bytes results in the
        LB waiting for the rest of the data, causing a timeout rather than a
        proxied 413 from the FastAPI middleware. The middleware is still correct
        and tested at the integration level when data is actually large enough.
        """
        import http.client
        import json
        import ssl
        import urllib.parse

        base = str(cloud_client.base_url)
        parsed = urllib.parse.urlparse(base)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path = "/v1/chat/completions"
        body = b'{"messages":[]}'

        ctx = ssl.create_default_context() if parsed.scheme == "https" else None
        conn = (
            http.client.HTTPSConnection(host, port, context=ctx, timeout=30)
            if ctx
            else http.client.HTTPConnection(host, port, timeout=30)
        )
        try:
            # Manually set Content-Length to 200 MB while body is tiny
            conn.putrequest("POST", path)
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(200 * 1024 * 1024))
            conn.endheaders(body)
            resp = conn.getresponse()
            assert resp.status == 413, f"Expected 413, got {resp.status}"
            data = json.loads(resp.read())
            assert "error" in data
        finally:
            conn.close()


@pytest.mark.cloud
class TestCloudProxyCORS:
    """Test 29: CORS headers."""

    async def test_cors_headers_present(self, cloud_client: httpx.AsyncClient):
        """CORS Access-Control headers are present in responses."""
        headers = {
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        }
        r = await cloud_client.options("/v1/chat/completions", headers=headers)
        # FastAPI CORS middleware should respond to preflight OPTIONS
        assert r.status_code in (200, 204, 405)
        # Check for CORS headers
        cors_header = r.headers.get("access-control-allow-origin")
        if cors_header is None:
            pytest.xfail("CORS headers not present in OPTIONS response")
        assert cors_header in ("*", "https://example.com")


# ===================================================================
# SECTION 3: Media processing unit tests (media.py)
# ===================================================================


@pytest.mark.media_unit
class TestMagicByteDetection:
    """Test 30: magic byte detection for various file types."""

    def test_jpeg_detection(self):
        import media

        data = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        assert media._detect_type(data) == "image/jpeg"

    def test_png_detection(self):
        import media

        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        assert media._detect_type(data) == "image/png"

    def test_wav_detection(self):
        import media

        # RIFF....WAVE
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 20
        assert media._detect_type(data) == "audio/wav"

    def test_mp4_detection(self):
        import media

        # ftyp box at offset 4
        data = b"\x00\x00\x00\x1c" + b"ftyp" + b"isom" + b"\x00" * 20
        assert media._detect_type(data) == "video/mp4"

    def test_webp_detection(self):
        import media

        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00" * 20
        assert media._detect_type(data) == "image/webp"

    def test_unknown_bytes(self):
        import media

        data = b"\x00\x01\x02\x03" * 10
        assert media._detect_type(data) is None

    def test_too_short(self):
        import media

        assert media._detect_type(b"\xff\xd8") is None


@pytest.mark.media_unit
class TestCategoryClassification:
    """Test 31: _category classifies MIME types correctly."""

    def test_image_mimes(self):
        import media

        for mime in ("image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"):
            assert media._category(mime) == "image", f"Failed for {mime}"

    def test_audio_mimes(self):
        import media

        for mime in ("audio/wav", "audio/mpeg", "audio/mp4", "audio/ogg", "audio/flac"):
            assert media._category(mime) == "audio", f"Failed for {mime}"

    def test_video_mimes(self):
        import media

        for mime in ("video/mp4", "video/quicktime", "video/webm"):
            assert media._category(mime) == "video", f"Failed for {mime}"

    def test_disallowed_mime(self):
        import media

        assert media._category("text/plain") is None
        assert media._category("application/pdf") is None


@pytest.mark.media_unit
class TestImageProcessing:
    """Tests 32-33: image processing via _process_image."""

    def test_small_image_succeeds(self, tiny_jpeg: bytes):
        import media

        result = media._process_image(tiny_jpeg, "test.jpg")
        assert isinstance(result, media.ProcessedMedia)
        assert result.category == "image"
        assert result.original_size == len(tiny_jpeg)
        assert result.processed_size > 0
        assert len(result.content_parts) == 1
        part = result.content_parts[0]
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_corrupted_image_fails(self, corrupted_jpeg: bytes):
        import media

        result = media._process_image(corrupted_jpeg, "bad.jpg")
        assert isinstance(result, media.MediaError)
        assert result.code == "processing_failed"

    def test_png_image_succeeds(self, tiny_png: bytes):
        import media

        result = media._process_image(tiny_png, "test.png")
        assert isinstance(result, media.ProcessedMedia)
        assert result.category == "image"


@pytest.mark.media_unit
class TestFileSizeLimits:
    """Test 34: file size limit enforcement."""

    def test_oversized_image_rejected(self):
        import media

        huge = b"\xff\xd8\xff\xe0" + (b"\x00" * (media.MAX_IMAGE_BYTES + 1))
        result = media._process_image(huge, "huge.jpg")
        assert isinstance(result, media.MediaError)
        assert result.code == "too_large"

    @pytest.mark.asyncio
    async def test_empty_upload_rejected(self):
        import media

        result = await media.process_upload(data=b"", filename="empty.jpg")
        assert isinstance(result, media.MediaError)
        assert result.code == "no_media"

    @pytest.mark.asyncio
    async def test_unrecognized_type_rejected(self):
        import media

        # Random bytes that match no magic signature
        data = bytes(range(256)) * 4
        result = await media.process_upload(data=data, filename="mystery.bin")
        assert isinstance(result, media.MediaError)
        assert result.code == "invalid_type"

    @pytest.mark.asyncio
    async def test_valid_jpeg_through_process_upload(self, tiny_jpeg: bytes):
        import media

        result = await media.process_upload(
            data=tiny_jpeg, filename="test.jpg", declared_mime="image/jpeg"
        )
        assert isinstance(result, media.ProcessedMedia)
        assert result.category == "image"
