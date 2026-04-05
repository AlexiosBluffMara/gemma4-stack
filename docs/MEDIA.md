# Gemma 4 Stack — Media Processing Reference

**Module:** `scripts/media.py`
**Last updated:** 2026-04-04

---

## Overview

`media.py` is the server-side media processing pipeline. All uploaded files pass through it before reaching the model. It handles:

- **Type validation** — magic-byte detection, never trusting declared MIME or filename
- **Size enforcement** — per-category hard limits
- **Format conversion** — images to JPEG, audio to 16 kHz WAV, video to JPEG frames
- **Video chunking** — large videos split into overlapping temporal segments
- **Streaming compatibility** — works from file paths, never loads video bytes into RAM

---

## Constants & Limits

```python
# Size limits
MAX_IMAGE_BYTES = 20 * 1024 * 1024          # 20 MB
MAX_AUDIO_BYTES = 50 * 1024 * 1024          # 50 MB
MAX_VIDEO_BYTES = 10 * 1024 * 1024 * 1024   # 10 GB

# Streaming
UPLOAD_STREAM_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB per read chunk

# Image/frame dimensions
IMAGE_MAX_EDGE   = 1280   # px — longest edge for uploaded images
IMAGE_QUALITY    = 85     # JPEG quality for images
FRAME_MAX_EDGE   = 768    # px — longest edge for video frames
FRAME_QUALITY    = 80     # JPEG quality for frames

# Audio
MAX_AUDIO_DURATION_S = 30  # seconds — Gemma 4 audio limit

# Video (single-pass)
MAX_VIDEO_FRAMES = 8       # default frame count for short videos
VIDEO_FPS = 1              # default 1 frame per second

# Video chunking
VIDEO_CHUNK_DIRECT_MAX_S = 60.0  # videos ≤ this use single-pass
MAX_VIDEO_CHUNKS = 30            # hard cap on number of chunks
CHUNK_FRAME_OVERLAP_S = 1.0     # overlap between consecutive chunks (seconds)

# Per-tier chunking parameters
VIDEO_CHUNK_DURATION_BY_TIER = {
    "fast":    30.0,   # E2B: 30-second windows
    "primary": 60.0,   # E4B: 60-second windows
    "primary": 60.0,   # E4B: 60-second windows
    "heavy":   120.0,  # 26B: 120-second windows
}
MAX_FRAMES_BY_TIER = {
    "fast":    4,   # E2B: 4 frames per chunk
    "primary": 8,   # E4B: 8 frames per chunk
    "heavy":   16,  # 26B: 16 frames per chunk
}
```

---

## Public API

### `process_upload(path, filename, declared_mime, tier, file_size) → ProcessedMedia | ChunkedVideoMedia | MediaError`

The main entry point. Called by the gateway after streaming the upload to a temp file.

```python
async def process_upload(
    path: str,                       # Path to the already-on-disk upload file
    filename: str,                   # Original filename (for logging, sanitised internally)
    declared_mime: Optional[str],    # Client-declared MIME (fallback only)
    tier: str = "primary",           # Controls frame budget for video
    file_size: int = 0,              # Known size in bytes (skip os.stat if provided)
) -> ProcessedMedia | ChunkedVideoMedia | MediaError
```

**Processing flow:**

```
1. Read first 16 bytes → magic-byte type detection
2. Fallback to declared_mime if magic detection fails
3. Determine category (image / audio / video)
4. Enforce per-category size limit
5a. image  → read full file (≤20 MB) → _process_image()
5b. audio  → read full file (≤50 MB) → _process_audio()
5c. video  → pass path → process_video_chunked()
             (never reads full video into memory)
```

---

### `process_video_chunked(path, filename, tier, file_size) → ProcessedMedia | ChunkedVideoMedia | MediaError`

Routes video to single-pass or chunked processing based on duration.

```python
async def process_video_chunked(
    path: str,
    filename: str,
    tier: str = "primary",
    file_size: int = 0,
) -> ProcessedMedia | ChunkedVideoMedia | MediaError
```

**Routing:**
- Duration ≤ `VIDEO_CHUNK_DIRECT_MAX_S` (60s) → `_process_video_from_path()` → `ProcessedMedia`
- Duration > 60s → frame extraction per chunk → `ChunkedVideoMedia`

---

## Return Types

### `ProcessedMedia`

Returned for images, audio, and short videos (≤60s). The gateway sends this directly to the model in a single call.

```python
@dataclass
class ProcessedMedia:
    category: str          # "image" | "audio" | "video"
    content_parts: list    # OpenAI multimodal content_parts list
    original_name: str
    original_size: int     # bytes as received
    processed_size: int    # bytes after conversion (0 for video)
    frame_count: int       # for video: number of frames extracted
    duration_s: float      # for audio/video: duration in seconds
    warnings: list         # non-fatal processing notes
    temp_files: list       # paths to clean up after request (audio WAV path)
```

`content_parts` structure per category:

```python
# Image
[{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]

# Audio
[{"type": "input_audio", "input_audio": {"data": "/tmp/gemma4_audio_<uuid>.wav", "format": "wav"}}]

# Short video
[
  {"type": "text", "text": "[Video: clip.mp4, 45.2s, 8 frames extracted at 1fps]"},
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  # frame 0
  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},  # frame 1
  # ...
]
```

---

### `ChunkedVideoMedia`

Returned for long videos (>60s). The gateway iterates over chunks, calls the model once per chunk, then performs a final aggregation call.

```python
@dataclass
class ChunkedVideoMedia:
    chunks: list[VideoChunk]   # temporal segments
    duration_s: float          # total video duration
    strategy: str = "chunked"
    tier: str = "primary"
    chunk_duration_s: float    # seconds per chunk
    frames_per_chunk: int      # frames extracted per chunk
    original_name: str
    original_size: int         # bytes as received
    warnings: list
    temp_files: list           # empty [] — gateway manages upload temp file

    @property
    def total_frames(self) -> int:
        return sum(c.frame_count for c in self.chunks)
```

---

### `VideoChunk`

One temporal segment within a chunked video. Each chunk contains base64-encoded JPEG frames.

```python
@dataclass
class VideoChunk:
    index: int               # 0-based segment index
    total: int               # total number of segments
    start_s: float           # segment start time (seconds)
    end_s: float             # segment end time (seconds)
    frames: list[str]        # base64-encoded JPEG strings
    frame_timestamps: list[float]  # actual extraction timestamp per frame

    @property
    def frame_count(self) -> int: return len(self.frames)

    @property
    def duration_s(self) -> float: return self.end_s - self.start_s
```

---

### `MediaError`

Returned when processing fails. The gateway maps `code` to an HTTP status.

```python
@dataclass
class MediaError:
    error: str   # human-readable message
    code: str    # "invalid_type" | "too_large" | "processing_failed" | "no_media"
```

| Code | HTTP Status | Meaning |
|------|-------------|---------|
| `invalid_type` | 400 | Unrecognized or unsupported file type |
| `too_large` | 413 | File exceeds size limit |
| `processing_failed` | 422 | ffmpeg/Pillow error, corrupted file, zero-duration video |
| `no_media` | 400 | Empty file |

---

## Image Processing Detail

### Pipeline

```
Input file (≤20 MB) on disk
       │
bytes = open(path, "rb").read()
       │
Pillow Image.open(io.BytesIO(bytes))
       │
Convert mode to RGB (handles RGBA → RGB, L → RGB, palette → RGB)
       │
If max(width, height) > 1280px:
    scale = 1280 / max(w, h)
    new_size = (int(w*scale), int(h*scale))
    img = img.resize(new_size, Image.LANCZOS)
    warnings.append("Resized from WxH to W'xH'")
       │
Strip EXIF (security):
    clean = Image.new(mode, size)
    clean.putdata(list(img.getdata()))
    # Only pixel data, no metadata
       │
Encode: clean.save(buf, format="JPEG", quality=85, optimize=True)
       │
base64.b64encode(buf.getvalue())
       │
return ProcessedMedia(content_parts=[{"type": "image_url", "image_url": {...}}])
```

### Why JPEG for all images?

Gemma 4's vision encoder processes images as JPEG-compressed inputs. Re-encoding all images to JPEG:
1. Ensures consistent colour space (RGB, 8-bit per channel)
2. Strips metadata (EXIF, ICC profiles) that could waste context tokens
3. Reduces file size for large PNGs (PNG is lossless; JPEG at quality=85 is ~5× smaller)

**Exception:** For future PNG-with-transparency support, the format selection can be changed per-image.

---

## Audio Processing Detail

### Pipeline

```
Input file (≤50 MB) on disk
       │
bytes = open(path, "rb").read()
       │
Write to temp input file (sanitised filename)
       │
ffprobe → duration in seconds
       │
if duration > 30s:
    will truncate to 30s
    warnings.append("Audio truncated from Xs to 30s (Gemma 4 limit)")
       │
ffmpeg:
  -ar 16000        # 16 kHz sample rate (Gemma 4 native)
  -ac 1            # mono channel
  -sample_fmt s16  # 16-bit PCM
  -f wav           # WAV container
  [-t 30]          # truncate if needed
  → output.wav
       │
Write WAV to persistent temp file:
    /tmp/gemma4_audio_<uuid>.wav
    [persistent because mlx_vlm.load_audio() needs a path, not bytes]
       │
return ProcessedMedia(
    content_parts=[{"type": "input_audio", "input_audio": {"data": wav_path}}],
    temp_files=[wav_path],   # cleaned up by gateway finally block
)
```

### Why 16 kHz Mono WAV?

Gemma 4's audio encoder (based on AudioPaLM) natively operates on 16 kHz mono 16-bit PCM. Sending any other format causes the model to reject the input. The ffmpeg conversion handles:
- Sample rate resampling (48 kHz → 16 kHz, etc.)
- Channel downmix (stereo → mono)
- Format conversion (MP3, FLAC, M4A → PCM WAV)
- Container rewrapping (e.g., M4A container with AAC → WAV with PCM)

---

## Video Processing Detail

### Short Video Pipeline (`_process_video_from_path`)

For videos ≤60 seconds, frames are extracted in a single pass:

```
Video file on disk (path)
       │
ffprobe → {duration_s, width, height, fps}
       │
if duration ≤ max_frames:
    interval = 1.0  (1 fps)
else:
    interval = duration / max_frames  (evenly spread)
    warnings.append("Video is Xs — sampling N frames every Ys")
       │
TemporaryDirectory(prefix="gemma4_video_")
       │
for i in range(frame_count):
    timestamp = i * interval
    ffmpeg:
      -ss timestamp         # seek to timestamp
      -i path               # input
      -vframes 1            # extract exactly 1 frame
      -vf scale='min(768,iw)':'min(768,ih)':force_original_aspect_ratio=decrease
      -q:v 3                # JPEG quality level 3 (good quality)
      → frame_NNN.jpg
       │
    Pillow: open, ensure RGB, resize if needed, re-encode JPEG (quality=80)
    base64-encode → b64_frame string
       │
content_parts = [
    {"type": "text", "text": "[Video: name, Xs, N frames at rate]"},
    {"type": "image_url", ...},  # frame 0
    ...
]
       │
return ProcessedMedia(content_parts, frame_count=N, duration_s=X)
```

### Long Video Pipeline (`process_video_chunked`)

For videos >60 seconds, temporal windows are computed and frames extracted per window:

**Step 1: Compute chunk plan**

```python
def _compute_chunk_plan(duration_s: float, tier: str) -> tuple:
    chunk_dur = VIDEO_CHUNK_DURATION_BY_TIER[tier]  # 30/60/120
    n_frames  = MAX_FRAMES_BY_TIER[tier]             # 4/8/16

    windows = []
    start = 0.0
    while start < duration_s:
        end = min(start + chunk_dur, duration_s)
        windows.append((start, end))
        next_start = start + chunk_dur - CHUNK_FRAME_OVERLAP_S  # 1s overlap
        if next_start >= duration_s: break
        start = next_start

    # Safety: cap at MAX_VIDEO_CHUNKS (30)
    if len(windows) > 30:
        step = len(windows) / 30
        windows = [windows[int(i*step)] for i in range(30)]
        windows[-1] = (windows[-1][0], duration_s)  # include end

    return windows, n_frames, chunk_dur
```

**Example: 240s video, primary tier (60s chunks, 8 frames)**

```
windows = [
    (0.0,  60.0),   # chunk 0: 0s–60s
    (59.0, 119.0),  # chunk 1: 59s–119s  (1s overlap)
    (118.0, 178.0), # chunk 2: 118s–178s
    (177.0, 240.0), # chunk 3: 177s–240s
]
n_frames = 8
chunk_dur = 60.0
```

**Step 2: Extract frames for each window**

```python
async def _extract_frames_for_window(video_path, start_s, end_s, n_frames, tmpdir, chunk_index):
    window = end_s - start_s
    if n_frames == 1:
        timestamps = [start_s + window / 2.0]
    else:
        step = window / (n_frames - 1)
        timestamps = [start_s + step * i for i in range(n_frames)]
    # timestamps spread evenly across [start_s, end_s]

    # Extract all frames concurrently (semaphore limits to 4 simultaneous ffmpeg)
    results = await asyncio.gather(
        *[_extract_one_frame(video_path, ts, out_path) for ts, out_path in zip(timestamps, frame_paths)],
        return_exceptions=True,
    )

    # Pillow post-process each successful frame
    b64_frames, actual_timestamps = [], []
    for ok, fp, ts in zip(results, frame_paths, timestamps):
        if ok is True:
            img = Image.open(fp)
            # resize, re-encode, base64
            b64_frames.append(b64_encoded_jpeg)
            actual_timestamps.append(ts)

    return b64_frames, actual_timestamps
```

**Step 3: Assemble ChunkedVideoMedia**

```python
with TemporaryDirectory() as tmpdir:
    chunks = []
    for i, (start_s, end_s) in enumerate(windows):
        frames, ts_list = await _extract_frames_for_window(
            path, start_s, end_s, n_frames, tmpdir, i
        )
        if not frames:
            warnings.append(f"Segment {i+1}: no frames extracted, skipping")
            continue
        chunks.append(VideoChunk(
            index=i, total=len(windows),
            start_s=start_s, end_s=end_s,
            frames=frames, frame_timestamps=ts_list,
        ))
# tmpdir auto-cleaned; frames are already base64-encoded in memory

return ChunkedVideoMedia(
    chunks=chunks,
    duration_s=duration,
    tier=tier_key,
    chunk_duration_s=chunk_dur,
    frames_per_chunk=n_frames,
    original_name=filename,
    original_size=actual_size,
    warnings=warnings,
    temp_files=[],   # gateway manages upload temp file cleanup
)
```

---

## Frame Extraction Concurrency

Frame extraction uses `asyncio.Semaphore(4)` to prevent spawning too many ffmpeg processes simultaneously (which would contend for CPU, memory, and disk I/O):

```python
_EXTRACT_SEM: Optional[asyncio.Semaphore] = None

def _get_extract_sem() -> asyncio.Semaphore:
    global _EXTRACT_SEM
    if _EXTRACT_SEM is None:
        _EXTRACT_SEM = asyncio.Semaphore(4)  # max 4 concurrent ffmpeg
    return _EXTRACT_SEM

async def _extract_one_frame(video_path, timestamp, out_path, max_edge=FRAME_MAX_EDGE):
    cmd = ["ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", video_path,
           "-vframes", "1", "-vf", f"scale='min({max_edge},iw)'...", "-q:v", "3", out_path]
    async with _get_extract_sem():
        proc = await asyncio.create_subprocess_exec(*cmd, ...)
        await asyncio.wait_for(proc.wait(), timeout=30)
        return proc.returncode == 0 and os.path.exists(out_path)
```

Within a single chunk, all frame extractions happen concurrently (up to the semaphore limit). Chunks themselves are processed sequentially by the gateway.

---

## Security Hardening

### Magic-Byte Validation

```python
def _detect_type(data: bytes) -> Optional[str]:
    """Detect MIME from first 16 bytes. Returns None if unrecognized."""
    # Explicit RIFF container disambiguation (WAV vs WebP)
    if data[:4] == b"RIFF":
        if data[8:12] == b"WAVE": return "audio/wav"
        if data[8:12] == b"WEBP": return "image/webp"
        return None  # unknown RIFF

    # MP4/MOV: ftyp box at offset 4
    if data[4:8] == b"ftyp":
        ftyp = data[8:12].decode("ascii", errors="ignore").lower()
        if ftyp in ("qt  ", "mqt "): return "video/quicktime"
        if ftyp.startswith("m4a"):  return "audio/mp4"
        return "video/mp4"  # isom, mp41, avc1, ...

    # ... other checks ...
```

A declared `Content-Type: image/jpeg` for a file that starts with `ftyp mp4` will be detected as `video/mp4`. The declared MIME is only used as a last-resort fallback if magic detection completely fails (e.g., some `.caf` audio files).

### Filename Sanitization

Filenames are sanitized before use in any file path or model context:

```python
safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
# "../../etc/passwd" → ".._.._etc_passwd"
# "video file (1).mp4" → "video_file__1_.mp4"
```

### No Shell Injection

All subprocess calls use list arguments, never `shell=True`:

```python
# SAFE: list args
proc = await asyncio.create_subprocess_exec(
    "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", video_path, ...
)

# NEVER: shell=True
# subprocess.run(f"ffmpeg -i {filename}", shell=True)  ← injection risk
```

---

## Temp File Lifecycle

| Temp File | Created By | Path | Cleaned Up By |
|-----------|-----------|------|--------------|
| Upload temp | gateway streaming | `/tmp/gemma4_upload_<random>` | `gateway.py` finally block |
| Audio WAV | `_process_audio()` | `/tmp/gemma4_audio_<uuid>.wav` | `result.temp_files` in gateway finally |
| Video frames | ffmpeg per-chunk | `TemporaryDirectory(prefix="gemma4_chunks_")` | Python `with` block auto-cleanup |
| Short video frames | ffmpeg | `TemporaryDirectory(prefix="gemma4_video_")` | Python `with` block auto-cleanup |

The gateway's unified `finally` block ensures cleanup happens regardless of outcome:

```python
finally:
    if upload_tmp_path:
        try: os.remove(upload_tmp_path)
        except OSError: pass
    for tf in getattr(result, "temp_files", []):
        try: os.remove(tf)
        except OSError: pass
```

---

## Required System Dependencies

| Dependency | Version | Purpose |
|-----------|---------|---------|
| `ffmpeg` | ≥6.0 | Video frame extraction, audio conversion |
| `ffprobe` | ≥6.0 (bundled with ffmpeg) | Media duration/metadata probing |
| `Pillow` | ≥10.0 | Image processing, frame post-processing |

Install:

```bash
brew install ffmpeg      # includes ffprobe
pip install Pillow       # in gateway-venv
```

---

## Testing Media Processing

```bash
# Run just the media tests
cd tests
pytest test_e2e.py -v -k "media"

# Test with a real file
curl -X POST http://localhost:8080/v1/media/upload \
  -F "file=@/path/to/test.jpg;type=image/jpeg" \
  -F "prompt=What is in this image?"

# Test error handling — unsupported type
curl -X POST http://localhost:8080/v1/media/upload \
  -F "file=@/path/to/script.sh;type=application/x-sh" \
  -F "prompt=Analyze this"
# → 400 {"error": "Unrecognized file type...", "code": "invalid_type"}

# Test size limit
python3 -c "open('/tmp/bigfile.jpg', 'wb').write(b'FF' + b'x' * (21*1024*1024))"
curl -X POST http://localhost:8080/v1/media/upload \
  -F "file=@/tmp/bigfile.jpg;type=image/jpeg"
# → 413 {"error": "Image too large: 21.0 MB (max 20 MB)", "code": "too_large"}
```
