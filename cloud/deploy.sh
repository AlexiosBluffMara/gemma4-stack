#!/usr/bin/env bash
# Deploy the Gemma 4 Cloud Proxy to Google Cloud Run.
# Usage: ./deploy.sh <PROJECT_ID> [GATEWAY_URL] [API_KEY]
set -euo pipefail

PROJECT_ID="${1:?Usage: ./deploy.sh <PROJECT_ID> [GATEWAY_URL] [API_KEY]}"
GATEWAY_URL="${2:-http://100.75.223.113:8080}"
API_KEY="${3:-}"

REGION="us-central1"
REPO="gemma4"
IMAGE="gemma4-proxy"
SERVICE="gemma4-proxy"
TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:latest"

echo "==> Project:  ${PROJECT_ID}"
echo "==> Region:   ${REGION}"
echo "==> Gateway:  ${GATEWAY_URL}"
echo ""

# 1. Enable required APIs
echo "==> Enabling required GCP APIs..."
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT_ID}" \
  --quiet

# 2. Create Artifact Registry repo (ignore error if it already exists)
echo "==> Creating Artifact Registry repository (if needed)..."
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet 2>/dev/null || true

# 3. Configure Docker auth for Artifact Registry
echo "==> Configuring Docker auth..."
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# 4. Build and push the Docker image
echo "==> Building Docker image..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
docker build -t "${TAG}" -f "${SCRIPT_DIR}/proxy/Dockerfile" "${SCRIPT_DIR}"

echo "==> Pushing Docker image..."
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
  --quiet

# 6. Print the public URL
URL=$(gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --format="value(status.url)")

echo ""
echo "=========================================="
echo "  Deployed successfully!"
echo "  URL: ${URL}"
echo "=========================================="
