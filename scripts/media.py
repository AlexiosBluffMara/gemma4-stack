"""
Server-side media processing for Gemma 4 inference.

Handles validation, conversion, and compression of images, audio, and video
into the exact formats that Gemma 4 (via mlx_vlm) natively accepts:

  Images  → JPEG/PNG, resized to fit 280-token budget (~1280px max edge)
  Audio   → 16 kHz mono WAV (Gemma 4's native audio input)
  Video   → 1fps JPEG frames extracted via ffmpeg (max 8 frames)

Security:
  - Magic-byte validation (not just MIME/extension)
  - File size hard limits
  - Temp files cleaned up automatically
  - No shell injection (subprocess with list args, never shell=True)
  - Filename sanitization
"""

import asyncio
import base64
import io
import logging
import os
import re
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

log = logging.getLogger("media")

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

MAX_IMAGE_BYTES = 20 * 1024 * 1024     # 20 MB raw upload
MAX_AUDIO_BYTES = 50 * 1024 * 1024     # 50 MB raw upload
MAX_VIDEO_BYTES = 10 * 1024 * 1024 * 1024  # 10 GB raw upload (streamed to disk, chunked processing)
MAX_AUDIO_DURATION_S = 30              # Gemma 4 default: 750 tokens × 40ms
MAX_VIDEO_FRAMES = 8                   # Default frame count for short videos
VIDEO_FPS = 1                          # Extract 1 frame per second
IMAGE_MAX_EDGE = 1280                  # Max longest edge for images
IMAGE_QUALITY = 85                     # JPEG quality
FRAME_QUALITY = 80                     # JPEG quality for video frames
FRAME_MAX_EDGE = 768                   # Slightly smaller for frames (multiple per request)

# ── Video chunking thresholds ────────────────────────────────────────────────
# Videos longer than this use the chunked multi-pass path instead of single-pass.
VIDEO_CHUNK_DIRECT_MAX_S: float = 60.0

# Per-tier temporal window size (seconds per chunk).
# Tuned to each model's memory footprint and typical context length.
VIDEO_CHUNK_DURATION_BY_TIER: dict[str, float] = {
    "fast":    30.0,   # E2B 2B params — short windows, minimal memory pressure
    "primary": 60.0,   # E4B 4B params — 1-minute windows, balanced
    "heavy":   120.0,  # 26B model     — 2-minute windows, richest context
}

# Maximum frames extracted per chunk, matched to each model's vision budget.
MAX_FRAMES_BY_TIER: dict[str, int] = {
    "fast":    4,    # E2B: 4 frames keeps token budget manageable
    "primary": 8,    # E4B: 8 frames is Gemma 4's practical sweet spot
    "heavy":   16,   # 26B: 16 frames for maximum visual fidelity
}

MAX_VIDEO_CHUNKS: int = 30           # Hard cap: at most 30 chunks per video
CHUNK_FRAME_OVERLAP_S: float = 1.0   # 1-second overlap between adjacent chunks
UPLOAD_STREAM_CHUNK_SIZE: int = 4 * 1024 * 1024  # 4 MB read chunk when streaming upload to disk

# ---------------------------------------------------------------------------
# Magic bytes for file type validation
# ---------------------------------------------------------------------------

MAGIC_BYTES = {
    # Images
    b"\xff\xd8\xff":          "image/jpeg",
    b"\x89PNG\r\n\x1a\n":    "image/png",
    b"RIFF":                  "image/webp",  # RIFF....WEBP (checked further below)
    b"GIF87a":                "image/gif",
    b"GIF89a":                "image/gif",
    b"\x00\x00\x01\x00":     "image/x-icon",
    b"BM":                    "image/bmp",
    # Audio
    b"fLaC":                  "audio/flac",
    b"OggS":                  "audio/ogg",
    b"ID3":                   "audio/mpeg",
    b"\xff\xfb":              "audio/mpeg",
    b"\xff\xf3":              "audio/mpeg",
    b"\xff\xf2":              "audio/mpeg",
    # Note: WAV starts with RIFF....WAVE — handled in _detect_type
    # Note: WebM/Matroska starts with 0x1A45DFA3 — handled in _detect_type
    # Note: MP4/MOV uses ftyp box — handled in _detect_type
}

# Allowed categories
ALLOWED_IMAGE_MIMES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp",
}
ALLOWED_AUDIO_MIMES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a", "audio/x-m4a", "audio/aac",
    "audio/ogg", "audio/opus", "audio/webm",
    "audio/flac", "audio/x-flac",
    "audio/caf", "audio/x-caf",  # iPhone native
}
ALLOWED_VIDEO_MIMES = {
    "video/mp4", "video/quicktime", "video/webm",
    "video/x-matroska", "video/avi", "video/x-msvideo",
    "video/3gpp", "video/3gpp2",
}


def _detect_type(data: bytes) -> Optional[str]:
    """Detect MIME type from magic bytes. Returns None if unrecognized."""
    if len(data) < 12:
        return None

    # RIFF container: could be WAV or WebP
    if data[:4] == b"RIFF":
        if data[8:12] == b"WAVE":
            return "audio/wav"
        if data[8:12] == b"WEBP":
            return "image/webp"
        return None

    # Matroska/WebM: 0x1A45DFA3
    if data[:4] == b"\x1a\x45\xdf\xa3":
        # Could be WebM video or WebM audio — check further with ffprobe later
        return "video/webm"

    # MP4/MOV: ftyp box
    if data[4:8] == b"ftyp":
        ftyp = data[8:12].decode("ascii", errors="ignore").lower()
        if ftyp in ("qt  ", "mqt "):
            return "video/quicktime"
        if ftyp.startswith("m4a"):
            return "audio/mp4"
        # isom, mp41, mp42, avc1, etc. → MP4 video
        return "video/mp4"

    # CAF (iPhone audio)
    if data[:4] == b"caff":
        return "audio/caf"

    # Standard magic byte check
    for magic, mime in MAGIC_BYTES.items():
        if data[: len(magic)] == magic:
            return mime

    return None


def _category(mime: str) -> Optional[str]:
    """Return 'image', 'audio', or 'video' from MIME, or None if disallowed."""
    if mime in ALLOWED_IMAGE_MIMES:
        return "image"
    if mime in ALLOWED_AUDIO_MIMES:
        return "audio"
    if mime in ALLOWED_VIDEO_MIMES:
        return "video"
    return None


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ProcessedMedia:
    """Result of processing a media file."""
    category: str                   # "image" | "audio" | "video"
    content_parts: list             # OpenAI multimodal content parts
    original_name: str = ""
    original_size: int = 0
    processed_size: int = 0         # Total bytes after conversion
    frame_count: int = 0            # For video: number of frames extracted
    duration_s: float = 0.0         # For audio/video: duration in seconds
    warnings: list = field(default_factory=list)
    temp_files: list = field(default_factory=list)  # Paths to clean up after request


@dataclass
class MediaError:
    """Error result."""
    error: str
    code: str  # "invalid_type", "too_large", "processing_failed", "no_media"


@dataclass
class VideoChunk:
    """One temporal segment of a long video, ready for a single model call."""
    index: int                    # 0-based segment index
    total: int                    # total number of segments
    start_s: float                # segment start time (seconds)
    end_s: float                  # segment end time (seconds)
    frames: list                  # base64-JPEG strings (one per extracted frame)
    frame_timestamps: list        # actual timestamp for each frame

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass
class ChunkedVideoMedia:
    """
    Return type for long videos that require multiple sequential model calls.

    The gateway iterates over `chunks`, sends each one to the model for a
    per-segment analysis, then performs a final aggregation pass.
    """
    chunks: list                  # list[VideoChunk]
    duration_s: float
    strategy: str = "chunked"     # "chunked" | future: "hierarchical"
    tier: str = "primary"
    chunk_duration_s: float = 0.0
    frames_per_chunk: int = 0
    original_name: str = ""
    original_size: int = 0
    warnings: list = field(default_factory=list)
    temp_files: list = field(default_factory=list)

    @property
    def total_frames(self) -> int:
        return sum(c.frame_count for c in self.chunks)


# Module-level semaphore for concurrent ffmpeg frame extraction.
# Lazily initialised inside async context to avoid event-loop issues.
_EXTRACT_SEM: Optional[asyncio.Semaphore] = None


def _get_extract_sem() -> asyncio.Semaphore:
    global _EXTRACT_SEM
    if _EXTRACT_SEM is None:
        _EXTRACT_SEM = asyncio.Semaphore(4)   # max 4 concurrent ffmpeg processes
    return _EXTRACT_SEM


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def _process_image(data: bytes, filename: str) -> ProcessedMedia | MediaError:
    """Validate, resize, and compress an image to JPEG for Gemma 4."""
    if len(data) > MAX_IMAGE_BYTES:
        return MediaError(
            f"Image too large: {len(data) / 1024 / 1024:.1f} MB (max {MAX_IMAGE_BYTES // 1024 // 1024} MB)",
            "too_large",
        )

    try:
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        return MediaError(f"Cannot open image: {e}", "processing_failed")

    # Convert to RGB (handles RGBA, palette, CMYK, etc.)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    warnings = []

    # Resize if needed — maintain aspect ratio, cap longest edge
    w, h = img.size
    max_edge = IMAGE_MAX_EDGE
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        warnings.append(f"Resized from {w}x{h} to {new_w}x{new_h}")

    # Strip EXIF (security: can contain GPS, device info, embedded thumbnails)
    clean_img = Image.new(img.mode, img.size)
    clean_img.putdata(list(img.getdata()))

    # Encode as JPEG
    buf = io.BytesIO()
    # Use PNG for images with transparency that were originally PNG
    out_format = "JPEG"
    clean_img.save(buf, format=out_format, quality=IMAGE_QUALITY, optimize=True)
    encoded = buf.getvalue()
    b64 = base64.b64encode(encoded).decode("ascii")

    content_parts = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
    }]

    return ProcessedMedia(
        category="image",
        content_parts=content_parts,
        original_name=filename,
        original_size=len(data),
        processed_size=len(encoded),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Audio processing — convert to 16kHz mono WAV via ffmpeg
# ---------------------------------------------------------------------------

async def _process_audio(data: bytes, filename: str) -> ProcessedMedia | MediaError:
    """Convert any audio to 16kHz mono WAV for Gemma 4's native audio input."""
    if len(data) > MAX_AUDIO_BYTES:
        return MediaError(
            f"Audio too large: {len(data) / 1024 / 1024:.1f} MB (max {MAX_AUDIO_BYTES // 1024 // 1024} MB)",
            "too_large",
        )

    warnings = []

    with tempfile.TemporaryDirectory(prefix="gemma4_audio_") as tmpdir:
        # Sanitize filename — only allow alphanum, dot, dash, underscore
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
        in_path = os.path.join(tmpdir, f"input_{safe_name}")
        out_path = os.path.join(tmpdir, "output.wav")

        with open(in_path, "wb") as f:
            f.write(data)

        # Probe duration first
        try:
            probe = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                in_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(probe.communicate(), timeout=15)
            duration = float(stdout.decode().strip())
        except Exception:
            duration = 0.0

        if duration > MAX_AUDIO_DURATION_S:
            warnings.append(
                f"Audio truncated from {duration:.1f}s to {MAX_AUDIO_DURATION_S}s "
                f"(Gemma 4 limit)"
            )

        # Convert to 16kHz mono WAV, truncate if needed
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-ar", "16000",       # 16 kHz sample rate (Gemma 4 native)
            "-ac", "1",           # mono
            "-sample_fmt", "s16", # 16-bit PCM
            "-f", "wav",
        ]
        if duration > MAX_AUDIO_DURATION_S:
            cmd.extend(["-t", str(MAX_AUDIO_DURATION_S)])
        cmd.append(out_path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            if proc.returncode != 0:
                err_msg = stderr.decode(errors="replace")[-500:]
                return MediaError(
                    f"Audio conversion failed: {err_msg}",
                    "processing_failed",
                )
        except asyncio.TimeoutError:
            return MediaError("Audio conversion timed out (60s)", "processing_failed")

        with open(out_path, "rb") as f:
            wav_data = f.read()

    # mlx_vlm's load_audio accepts file paths — save WAV to a persistent temp file
    # (it gets cleaned up after the request via the ProcessedMedia.temp_files list)
    import uuid as _uuid
    wav_path = os.path.join(tempfile.gettempdir(), f"gemma4_audio_{_uuid.uuid4().hex}.wav")
    with open(wav_path, "wb") as wf:
        wf.write(wav_data)

    actual_duration = min(duration, MAX_AUDIO_DURATION_S) if duration > 0 else 0

    # mlx_vlm server expects input_audio.data to be a loadable path/URL
    content_parts = [{
        "type": "input_audio",
        "input_audio": {"data": wav_path, "format": "wav"},
    }]

    return ProcessedMedia(
        category="audio",
        content_parts=content_parts,
        original_name=filename,
        original_size=len(data),
        processed_size=len(wav_data),
        duration_s=actual_duration,
        warnings=warnings,
        temp_files=[wav_path],
    )


# ---------------------------------------------------------------------------
# Video processing — extract frames via ffmpeg, convert to JPEG
# ---------------------------------------------------------------------------

async def _process_video(
    data: bytes,
    filename: str,
    max_frames: int = MAX_VIDEO_FRAMES,
) -> ProcessedMedia | MediaError:
    """Extract frames from video at 1fps, convert each to JPEG for Gemma 4.

    Args:
        max_frames: Maximum frames to extract. Defaults to MAX_VIDEO_FRAMES (8).
                    Callers can pass tier-specific values (4 for fast, 16 for heavy).
    """
    if len(data) > MAX_VIDEO_BYTES:
        return MediaError(
            f"Video too large: {len(data) / 1024 / 1024:.1f} MB (max {MAX_VIDEO_BYTES // 1024 // 1024} MB)",
            "too_large",
        )

    warnings = []

    with tempfile.TemporaryDirectory(prefix="gemma4_video_") as tmpdir:
        safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
        in_path = os.path.join(tmpdir, f"input_{safe_name}")

        with open(in_path, "wb") as f:
            f.write(data)

        # Probe duration and resolution
        try:
            probe = await asyncio.create_subprocess_exec(
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-show_entries", "stream=width,height,codec_type",
                "-of", "json",
                in_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(probe.communicate(), timeout=15)
            import json
            probe_data = json.loads(stdout.decode())
            duration = float(probe_data.get("format", {}).get("duration", 0))
        except Exception:
            duration = 0.0
            probe_data = {}

        if duration <= 0:
            return MediaError(
                "Cannot determine video duration. File may be corrupted.",
                "processing_failed",
            )

        # Calculate frame count: 1fps, capped at max_frames
        total_possible = int(duration)
        frame_count = min(total_possible, max_frames)

        if frame_count < 1:
            frame_count = 1  # At least try to get one frame

        if total_possible > max_frames:
            # Spread frames evenly across the video
            interval = duration / frame_count
            warnings.append(
                f"Video is {duration:.1f}s — sampling {frame_count} frames "
                f"evenly (1 every {interval:.1f}s)"
            )
        else:
            interval = 1.0  # 1fps

        # Build scale filter for frame size
        scale_filter = (
            f"scale='if(gt(iw,ih),{FRAME_MAX_EDGE},-2)'"
            f":'if(gt(ih,iw),{FRAME_MAX_EDGE},-2)'"
        )

        # Extract frames
        frame_paths = []
        for i in range(frame_count):
            timestamp = i * interval
            out_frame = os.path.join(tmpdir, f"frame_{i:03d}.jpg")

            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{timestamp:.2f}",
                "-i", in_path,
                "-vframes", "1",
                "-vf", f"scale='min({FRAME_MAX_EDGE},iw)':'min({FRAME_MAX_EDGE},ih)':force_original_aspect_ratio=decrease",
                "-q:v", "3",  # JPEG quality (2-5 is good, lower = better)
                out_frame,
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode == 0 and os.path.exists(out_frame):
                    frame_paths.append(out_frame)
            except asyncio.TimeoutError:
                warnings.append(f"Frame {i} extraction timed out")
            except Exception as e:
                warnings.append(f"Frame {i} extraction failed: {e}")

        if not frame_paths:
            # Fallback: try a simple single-frame extraction
            out_frame = os.path.join(tmpdir, "fallback_frame.jpg")
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", in_path,
                    "-vframes", "1", "-q:v", "3", out_frame,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode == 0 and os.path.exists(out_frame):
                    frame_paths.append(out_frame)
            except Exception:
                pass

        if not frame_paths:
            return MediaError(
                "Could not extract any frames from video. Format may be unsupported.",
                "processing_failed",
            )

        # Read frames, resize with Pillow for reliability, encode as base64
        content_parts = []
        total_size = 0

        # Add a text marker describing the video
        content_parts.append({
            "type": "text",
            "text": (
                f"[Video: {filename}, {duration:.1f}s, "
                f"{len(frame_paths)} frames extracted at "
                f"{'1fps' if total_possible <= max_frames else f'1/{interval:.1f}s'}]"
            ),
        })

        for fpath in frame_paths:
            try:
                img = Image.open(fpath)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Ensure size cap
                w, h = img.size
                if max(w, h) > FRAME_MAX_EDGE:
                    scale = FRAME_MAX_EDGE / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=FRAME_QUALITY, optimize=True)
                frame_bytes = buf.getvalue()
                total_size += len(frame_bytes)
                b64 = base64.b64encode(frame_bytes).decode("ascii")

                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            except Exception as e:
                warnings.append(f"Failed to encode frame {fpath}: {e}")

    return ProcessedMedia(
        category="video",
        content_parts=content_parts,
        original_name=filename,
        original_size=len(data),
        processed_size=total_size,
        frame_count=len(frame_paths),
        duration_s=duration,
        warnings=warnings,
    )


async def _process_video_from_path(
    path: str,
    filename: str,
    max_frames: int = MAX_VIDEO_FRAMES,
    file_size: int = 0,
) -> "ProcessedMedia | MediaError":
    """
    Same as _process_video but operates on an already-on-disk file path.

    Used by the path-based process_upload() API so we never load a video
    entirely into RAM — the gateway streams the upload directly to disk
    and passes the path here.
    """
    actual_size = file_size or os.path.getsize(path)
    warnings: list = []

    # Probe duration and resolution
    try:
        probe = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-show_entries", "stream=width,height,codec_type",
            "-of", "json",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(probe.communicate(), timeout=15)
        import json as _json
        probe_data = _json.loads(stdout.decode())
        duration = float(probe_data.get("format", {}).get("duration", 0))
    except Exception:
        duration = 0.0
        probe_data = {}

    if duration <= 0:
        return MediaError(
            "Cannot determine video duration. File may be corrupted.",
            "processing_failed",
        )

    total_possible = int(duration)
    frame_count = min(total_possible, max_frames)
    if frame_count < 1:
        frame_count = 1

    if total_possible > max_frames:
        interval = duration / frame_count
        warnings.append(
            f"Video is {duration:.1f}s — sampling {frame_count} frames "
            f"evenly (1 every {interval:.1f}s)"
        )
    else:
        interval = 1.0

    with tempfile.TemporaryDirectory(prefix="gemma4_video_") as tmpdir:
        frame_paths = []
        for i in range(frame_count):
            timestamp = i * interval
            out_frame = os.path.join(tmpdir, f"frame_{i:03d}.jpg")
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{timestamp:.2f}",
                "-i", path,
                "-vframes", "1",
                "-vf", f"scale='min({FRAME_MAX_EDGE},iw)':'min({FRAME_MAX_EDGE},ih)':force_original_aspect_ratio=decrease",
                "-q:v", "3",
                out_frame,
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode == 0 and os.path.exists(out_frame):
                    frame_paths.append(out_frame)
            except asyncio.TimeoutError:
                warnings.append(f"Frame {i} extraction timed out")
            except Exception as e:
                warnings.append(f"Frame {i} extraction failed: {e}")

        if not frame_paths:
            out_frame = os.path.join(tmpdir, "fallback_frame.jpg")
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ffmpeg", "-y", "-i", path,
                    "-vframes", "1", "-q:v", "3", out_frame,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
                if proc.returncode == 0 and os.path.exists(out_frame):
                    frame_paths.append(out_frame)
            except Exception:
                pass

        if not frame_paths:
            return MediaError(
                "Could not extract any frames from video. Format may be unsupported.",
                "processing_failed",
            )

        content_parts: list = []
        total_size = 0

        content_parts.append({
            "type": "text",
            "text": (
                f"[Video: {filename}, {duration:.1f}s, "
                f"{len(frame_paths)} frames extracted at "
                f"{'1fps' if total_possible <= max_frames else f'1/{interval:.1f}s'}]"
            ),
        })

        for fpath in frame_paths:
            try:
                img = Image.open(fpath)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if max(w, h) > FRAME_MAX_EDGE:
                    scale = FRAME_MAX_EDGE / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=FRAME_QUALITY, optimize=True)
                frame_bytes = buf.getvalue()
                total_size += len(frame_bytes)
                b64 = base64.b64encode(frame_bytes).decode("ascii")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            except Exception as e:
                warnings.append(f"Failed to encode frame {fpath}: {e}")

    return ProcessedMedia(
        category="video",
        content_parts=content_parts,
        original_name=filename,
        original_size=actual_size,
        processed_size=total_size,
        frame_count=len(frame_paths),
        duration_s=duration,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Video chunking helpers
# ---------------------------------------------------------------------------

async def _get_video_info(path: str) -> dict:
    """
    Return {duration_s, width, height, fps} from ffprobe.
    Returns an empty dict on failure — callers must handle gracefully.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        import json as _json
        info = _json.loads(stdout.decode())
        duration = float(info.get("format", {}).get("duration", 0))
        width = height = fps_f = 0.0
        for stream in info.get("streams", []):
            if stream.get("codec_type") == "video":
                width = float(stream.get("width", 0))
                height = float(stream.get("height", 0))
                r_fps = stream.get("r_frame_rate", "0/1")
                try:
                    num, den = r_fps.split("/")
                    fps_f = float(num) / float(den) if float(den) else 0.0
                except Exception:
                    fps_f = 0.0
                break
        return {"duration_s": duration, "width": width, "height": height, "fps": fps_f}
    except Exception as exc:
        log.debug("ffprobe failed for %s: %s", path, exc)
        return {}


async def _extract_one_frame(
    video_path: str,
    timestamp: float,
    out_path: str,
    max_edge: int = FRAME_MAX_EDGE,
) -> bool:
    """Extract a single JPEG frame at `timestamp` seconds. Returns True on success."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{timestamp:.3f}",
        "-i", video_path,
        "-vframes", "1",
        "-vf",
        (
            f"scale='min({max_edge},iw)':'min({max_edge},ih)'"
            f":force_original_aspect_ratio=decrease"
        ),
        "-q:v", "3",
        out_path,
    ]
    async with _get_extract_sem():
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=30)
            return proc.returncode == 0 and os.path.exists(out_path)
        except Exception as exc:
            log.debug("Frame extract failed at %.3fs: %s", timestamp, exc)
            return False


async def _extract_frames_for_window(
    video_path: str,
    start_s: float,
    end_s: float,
    n_frames: int,
    tmpdir: str,
    chunk_index: int,
) -> tuple:
    """
    Extract n_frames evenly spread across [start_s, end_s].

    Returns:
        (b64_frames: list[str], timestamps: list[float])
        where b64_frames contains base64-encoded JPEG strings.
    """
    if n_frames <= 0 or end_s <= start_s:
        return [], []

    window = end_s - start_s
    if n_frames == 1:
        timestamps = [start_s + window / 2.0]
    else:
        step = window / (n_frames - 1)
        timestamps = [start_s + step * i for i in range(n_frames)]

    frame_paths = [
        os.path.join(tmpdir, f"c{chunk_index:02d}_f{i:02d}.jpg")
        for i in range(n_frames)
    ]

    # Extract all frames concurrently (semaphore limits max parallelism)
    results = await asyncio.gather(
        *[_extract_one_frame(video_path, ts, fp) for ts, fp in zip(timestamps, frame_paths)],
        return_exceptions=True,
    )

    b64_frames: list = []
    actual_ts: list = []
    for ok, fp, ts in zip(results, frame_paths, timestamps):
        if ok is True and os.path.exists(fp):
            try:
                img = Image.open(fp)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                w, h = img.size
                if max(w, h) > FRAME_MAX_EDGE:
                    scale = FRAME_MAX_EDGE / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=FRAME_QUALITY, optimize=True)
                b64_frames.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                actual_ts.append(ts)
            except Exception as exc:
                log.debug("Frame encode failed: %s", exc)

    return b64_frames, actual_ts


def _compute_chunk_plan(
    duration_s: float,
    tier: str,
) -> tuple:
    """
    Compute chunking windows for a video based on tier capacity.

    Applies CHUNK_FRAME_OVERLAP_S overlap between consecutive chunks so the
    model sees some continuity at segment boundaries.

    Returns:
        (windows: list[tuple[float, float]], frames_per_chunk: int, chunk_dur: float)
    """
    chunk_dur = VIDEO_CHUNK_DURATION_BY_TIER.get(tier, 60.0)
    n_frames = MAX_FRAMES_BY_TIER.get(tier, 8)

    windows: list = []
    start = 0.0
    while start < duration_s:
        end = min(start + chunk_dur, duration_s)
        windows.append((start, end))
        next_start = start + chunk_dur - CHUNK_FRAME_OVERLAP_S
        if next_start >= duration_s:
            break
        start = next_start

    # Safety cap: sub-sample windows if we'd exceed MAX_VIDEO_CHUNKS
    if len(windows) > MAX_VIDEO_CHUNKS:
        step = len(windows) / MAX_VIDEO_CHUNKS
        sampled = [windows[int(i * step)] for i in range(MAX_VIDEO_CHUNKS)]
        # Always include the final chunk
        last_start = sampled[-1][0]
        sampled[-1] = (last_start, duration_s)
        windows = sampled

    return windows, n_frames, chunk_dur


async def process_video_chunked(
    path: str,
    filename: str,
    tier: str = "primary",
    file_size: int = 0,
) -> "ProcessedMedia | ChunkedVideoMedia | MediaError":
    """
    Smart video processor with dynamic chunking.

    Operates on an already-on-disk file path — the gateway streams the upload
    to disk first, so this function never loads video bytes into RAM.

    Routing logic
    ─────────────
    Short video (≤ VIDEO_CHUNK_DIRECT_MAX_S = 60 s)
        Single-pass: delegates to _process_video_from_path() with tier-tuned
        frame count. Returns ProcessedMedia — identical to the existing path.

    Long video (> 60 s)
        Chunked multi-pass: splits the video into overlapping temporal windows
        and returns ChunkedVideoMedia. The gateway calls the model once per
        chunk then performs a final aggregation pass.

    Frame budget per chunk (tuned to model size)
    ─────────────────────────────────────────────
    fast    (E2B, 2B params) →  4 frames / 30 s chunk
    primary (E4B, 4B params) →  8 frames / 60 s chunk
    heavy   (26B model)      → 16 frames / 120 s chunk
    """
    tier_key = tier if tier in MAX_FRAMES_BY_TIER else "primary"
    actual_size = file_size or os.path.getsize(path)
    warnings: list = []

    try:
        # Probe metadata — file is already on disk, no write step needed
        info = await _get_video_info(path)
        duration = info.get("duration_s", 0.0)

        if duration <= 0:
            return MediaError(
                "Cannot determine video duration — file may be corrupted or unsupported.",
                "processing_failed",
            )

        log.info(
            "Video processing: file=%s duration=%.1fs size=%.1fMB tier=%s",
            filename, duration, actual_size / 1024 / 1024, tier_key,
        )

        # ── SHORT VIDEO: single-pass ──────────────────────────────────────────
        if duration <= VIDEO_CHUNK_DIRECT_MAX_S:
            max_frames = MAX_FRAMES_BY_TIER.get(tier_key, MAX_VIDEO_FRAMES)
            return await _process_video_from_path(path, filename, max_frames=max_frames, file_size=actual_size)

        # ── LONG VIDEO: chunked multi-pass ────────────────────────────────────
        windows, n_frames, chunk_dur = _compute_chunk_plan(duration, tier_key)

        log.info(
            "Chunking %s → %d segments × ~%.0fs, %d frames/seg (%s tier)",
            filename, len(windows), chunk_dur, n_frames, tier_key,
        )
        warnings.append(
            f"Long video ({duration:.0f}s) split into {len(windows)} segment(s) "
            f"of ~{chunk_dur:.0f}s each, {n_frames} frames/segment ({tier_key} tier)"
        )

        with tempfile.TemporaryDirectory(prefix="gemma4_chunks_") as tmpdir:
            chunks: list = []
            for i, (start_s, end_s) in enumerate(windows):
                frames, ts_list = await _extract_frames_for_window(
                    path, start_s, end_s, n_frames, tmpdir, i,
                )
                if not frames:
                    warnings.append(f"Segment {i + 1}: no frames extracted, skipping")
                    continue
                chunks.append(VideoChunk(
                    index=i,
                    total=len(windows),
                    start_s=start_s,
                    end_s=end_s,
                    frames=frames,
                    frame_timestamps=ts_list,
                ))

        if not chunks:
            return MediaError(
                "Failed to extract frames from any video segment.",
                "processing_failed",
            )

        return ChunkedVideoMedia(
            chunks=chunks,
            duration_s=duration,
            strategy="chunked",
            tier=tier_key,
            chunk_duration_s=chunk_dur,
            frames_per_chunk=n_frames,
            original_name=filename,
            original_size=actual_size,
            warnings=warnings,
            temp_files=[],  # upload temp file is managed by the gateway's own finally block
        )

    except Exception as exc:
        log.exception("process_video_chunked failed for %s", filename)
        return MediaError(f"Video processing failed: {exc}", "processing_failed")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def process_upload(
    path: str,
    filename: str,
    declared_mime: Optional[str] = None,
    tier: str = "primary",
    file_size: int = 0,
) -> "ProcessedMedia | ChunkedVideoMedia | MediaError":
    """
    Process an uploaded media file end-to-end.

    Operates on an already-on-disk file path — the gateway streams the upload
    to disk before calling this function, so video files (up to 10 GB) are
    never fully loaded into RAM.  Images and audio (which are bounded to 20 MB
    and 50 MB respectively) are still read into memory for processing.

    1. Reads the first 16 bytes for magic-byte type detection
    2. Enforces size limits per category
    3. Converts to Gemma 4 native format
    4. Returns content ready for the chat API

    For short videos (≤60 s) and all images/audio → returns ProcessedMedia.
    For long videos (>60 s) → returns ChunkedVideoMedia (multiple model calls needed).

    Args:
        path:          Path to the already-written upload file on disk
        filename:      Original filename (used for logging, sanitised internally)
        declared_mime: Client-declared MIME type (used as fallback only)
        tier:          Inference tier — controls frame budget for video
        file_size:     Known byte count (skip stat() call if provided)

    Returns:
        ProcessedMedia, ChunkedVideoMedia, or MediaError
    """
    # Step 0: Read file header for magic-byte detection (16 bytes is enough for all formats)
    try:
        with open(path, "rb") as fh:
            header = fh.read(16)
    except OSError as exc:
        return MediaError(f"Cannot read upload file: {exc}", "processing_failed")

    if not header:
        return MediaError("Empty file", "no_media")

    actual_size = file_size or os.path.getsize(path)

    # Step 1: Detect type from magic bytes
    detected_mime = _detect_type(header)

    # Fallback to declared MIME if magic detection fails
    # (some formats like .caf or .m4a may not be caught by simple magic)
    if detected_mime is None and declared_mime:
        detected_mime = declared_mime.split(";")[0].strip().lower()
        log.warning(
            "Magic byte detection failed for %s, falling back to declared: %s",
            filename, detected_mime,
        )

    if detected_mime is None:
        return MediaError(
            f"Unrecognized file type for '{filename}'. "
            "Supported: JPEG, PNG, WebP, GIF, WAV, MP3, M4A, WebM, MP4, MOV",
            "invalid_type",
        )

    # Step 2: Categorize and validate
    cat = _category(detected_mime)

    # Special case: WebM could be audio-only or video
    if detected_mime == "video/webm" and declared_mime:
        declared_lower = declared_mime.lower()
        if "audio" in declared_lower:
            cat = "audio"

    if cat is None:
        return MediaError(
            f"Unsupported media type: {detected_mime}",
            "invalid_type",
        )

    log.info(
        "Processing %s: %s (%s, %.1f KB)",
        cat, filename, detected_mime, actual_size / 1024,
    )

    # Step 3: Process by category
    if cat == "image":
        # Images are small (≤ 20 MB) — safe to load into memory
        if actual_size > MAX_IMAGE_BYTES:
            return MediaError(
                f"Image too large: {actual_size / 1024 / 1024:.1f} MB "
                f"(max {MAX_IMAGE_BYTES // 1024 // 1024} MB)",
                "too_large",
            )
        with open(path, "rb") as fh:
            data = fh.read()
        return _process_image(data, filename)

    elif cat == "audio":
        # Audio is bounded at 50 MB — safe to load into memory
        if actual_size > MAX_AUDIO_BYTES:
            return MediaError(
                f"Audio too large: {actual_size / 1024 / 1024:.1f} MB "
                f"(max {MAX_AUDIO_BYTES // 1024 // 1024} MB)",
                "too_large",
            )
        with open(path, "rb") as fh:
            data = fh.read()
        return await _process_audio(data, filename)

    elif cat == "video":
        # Videos can be up to 10 GB — process from path, never load into RAM
        if actual_size > MAX_VIDEO_BYTES:
            return MediaError(
                f"Video too large: {actual_size / 1024 / 1024 / 1024:.2f} GB "
                f"(max {MAX_VIDEO_BYTES // 1024 // 1024 // 1024} GB)",
                "too_large",
            )
        return await process_video_chunked(path, filename, tier=tier, file_size=actual_size)

    else:
        return MediaError(f"Unhandled category: {cat}", "processing_failed")
