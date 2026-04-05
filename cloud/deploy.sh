#!/usr/bin/env bash
# Deploy the Gemma 4 Cloud Proxy to Google Cloud Run.
#
# Usage:
#   ./deploy.sh <PROJECT_ID> [API_KEY]
#
# The proxy connects to the Mac Mini gateway via Tailscale Funnel:
#   https://miniapple.scylla-betta.ts.net
#
# Cloud Run → (public internet) → Tailscale Funnel → Mac Mini gateway
#
set -euo pipefail

PROJECT_ID="${1:?Usage: ./deploy.sh <PROJECT_ID> [API_KEY]}"
API_KEY="${2:-}"

REGION="us-central1"
REPO="gemma4"
IMAGE="gemma4-proxy"
SERVICE="gemma4-proxy"
GATEWAY_URL="https://miniapple.scylla-betta.ts.net"
TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"

echo "============================================"
echo "  Gemma 4 Cloud Proxy — Deploy to Cloud Run"
echo "============================================"
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Gateway:  ${GATEWAY_URL}"
echo "  API Key:  ${API_KEY:+(set)}"
echo ""

# 1. Enable required APIs
echo "==> Enabling required GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}" \
  --quiet

# 2. Create Artifact Registry repo
echo "==> Creating Artifact Registry repository..."
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet 2>/dev/null || echo "    (already exists)"

# 3. Configure Docker auth for Artifact Registry
echo "==> Configuring Docker auth..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# 4. Build and push the Docker image
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "==> Building Docker image..."
docker build -t "${TAG}" -f "${SCRIPT_DIR}/proxy/Dockerfile" "${SCRIPT_DIR}"

echo "==> Pushing to Artifact Registry..."
docker push "${TAG}"

# 5. Deploy to Cloud Run
echo "==> Deploying to Cloud Run..."
ENV_VARS="GATEWAY_URL=${GATEWAY_URL}"
if [ -n "${API_KEY}" ]; then
  ENV_VARS="${ENV_VARS},API_KEY=${API_KEY}"
fi

gcloud run deploy "${SERVICE}" \
  --image="${TAG}" \
  --platform=managed \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --allow-unauthenticated \
  --set-env-vars="${ENV_VARS}" \
  --memory=512Mi \
  --cpu=1 \
  --port=8080 \
  --max-instances=3 \
  --min-instances=0 \
  --timeout=300 \
  --concurrency=80 \
  --quiet

# 6. Print the public URL
URL=$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "============================================"
echo "  Deployed successfully!"
echo ""
echo "  Cloud Run URL:  ${URL}"
echo "  Gateway URL:    ${GATEWAY_URL}"
echo ""
echo "  Test it:"
echo "    curl ${URL}/health"
echo "    curl ${URL}/api/status"
echo "============================================"
