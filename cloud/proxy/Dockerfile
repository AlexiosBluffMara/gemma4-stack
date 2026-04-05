# ---- Builder stage ----
FROM python:3.12.8-slim AS builder

WORKDIR /build

# Pin pip version for reproducibility
RUN pip install --no-cache-dir pip==24.3.1

COPY proxy/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- Runtime stage ----
FROM python:3.12.8-slim

LABEL maintainer="gemma4good"
LABEL description="Gemma 4 Cloud Proxy - routes requests to on-prem gateway via Tailscale"
LABEL version="1.0"

# Python environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --no-create-home --shell /sbin/nologin appuser

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY proxy/*.py ./

# Copy web UI as static files
COPY web/ /app/static/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
