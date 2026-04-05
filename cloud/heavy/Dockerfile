# ============================================================
# Gemma 4 Heavy Tier — Cloud Run GPU (NVIDIA L4 24 GB VRAM)
# ============================================================
#
# Build stages:
#   1. builder  — compile llama-cpp-python with CUDA 12.4
#   2. runtime  — lean CUDA runtime image with only what's needed
#
# Resulting image: ~8 GB (CUDA runtime + compiled llama-cpp-python)
# Model (15.6 GB GGUF) is NOT baked in — downloaded from GCS on cold start.

# ---- Stage 1: Build llama-cpp-python with CUDA -------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Build tools + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-venv \
        build-essential \
        cmake \
        ninja-build \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3 point to 3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /build

# Compile llama-cpp-python with CUDA backend
# CMAKE_ARGS tells the build to enable GGML CUDA
ENV CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89"
ENV FORCE_CMAKE=1

RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install in a venv for clean copy to runtime stage
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Cloud Build has no GPU driver — only a stub libcuda.so (no .1 suffix).
# Create the versioned symlink so the linker can resolve libcuda.so.1 when
# llama-cpp-python's CUDA build links the final CLI binaries.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" >> /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

RUN pip install --no-cache-dir \
    "llama-cpp-python>=0.2.80" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.27.0" \
    "pydantic>=2.0" \
    "google-cloud-storage>=2.16.0" \
    "huggingface_hub[hf_transfer]>=0.21.0"


# ---- Stage 2: Runtime image -------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/opt/venv/bin:$PATH"
# Enable fast HuggingFace downloads (hf_transfer uses Rust internals)
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Only runtime CUDA libs needed
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        libgomp1 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Copy compiled venv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Non-root user for security
RUN useradd -m -u 1000 -s /bin/bash gemma
USER gemma
WORKDIR /home/gemma

# Application code
COPY --chown=gemma:gemma main.py .

# Cloud Run injects PORT env var; default to 8080
ENV PORT=8080
# GCS bucket where the GGUF model is staged
ENV GCS_BUCKET=gemma4good-models
ENV GCS_MODEL_BLOB=gemma-4-26b-q4.gguf
# GPU layers: -1 = all layers on GPU (L4 has 24 GB VRAM, model fits at ~15.6 GB)
ENV N_GPU_LAYERS=-1
ENV CTX_SIZE=8192
ENV N_PARALLEL=4

# Health check so Cloud Run knows the container is alive during model load
HEALTHCHECK --interval=10s --timeout=5s --start-period=300s --retries=60 \
    CMD curl -fsS http://localhost:${PORT}/healthz || exit 1

CMD ["python3", "main.py"]
