#!/usr/bin/env bash
# =============================================================
# Deploy Gemma 4 Heavy Tier to Cloud Run GPU (NVIDIA L4)
# =============================================================
#
# What this script does:
#   1. Uploads the GGUF model to GCS (only needed once, ~15.6 GB)
#   2. Creates Secret Manager secrets (heavy-api-key, hf-token)
#   3. Builds the Docker image via Cloud Build (multi-stage, CUDA)
#   4. Pushes to Artifact Registry
#   5. Deploys the Cloud Run GPU service
#   6. Prints the service URL + updates the gateway's HEAVY_URL
#
# Usage:
#   export HEAVY_API_KEY="your-random-secret-here"
#   export HF_TOKEN="hf_xxxx"                      # HuggingFace read token
#   bash cloud/heavy/deploy.sh [GCP_PROJECT_ID]
#
# Prerequisites:
#   - gcloud CLI authenticated: gcloud auth login
#   - Billing enabled on the project
#   - Artifact Registry and Cloud Run APIs enabled
#   - A local copy of the GGUF model OR let the container download it
# =============================================================

set -euo pipefail

PROJECT="${1:-gemma4good}"
REGION="us-central1"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT}/gemma4/gemma4-heavy"
SERVICE_NAME="gemma4-heavy"
GCS_BUCKET="${PROJECT}-models"
GCS_BLOB="gemma-4-26b-q4.gguf"

# Local GGUF path — set this if you have the model locally for faster upload
# If not set, the container will download it from HF on first cold start.
LOCAL_MODEL_PATH="${LOCAL_MODEL_PATH:-}"

HF_TOKEN="${HF_TOKEN:-}"
HEAVY_API_KEY="${HEAVY_API_KEY:-}"

# Generate a random API key if not provided
if [ -z "$HEAVY_API_KEY" ]; then
    HEAVY_API_KEY=$(openssl rand -hex 32)
    echo "Generated HEAVY_API_KEY: $HEAVY_API_KEY"
    echo "SAVE THIS — you'll need to add it to the gateway LaunchAgent plist."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo "  Deploying Gemma 4 Heavy Tier"
echo "  Project: $PROJECT"
echo "  Region:  $REGION"
echo "  Service: $SERVICE_NAME"
echo "================================================"
echo ""

# ── Step 1: Enable required APIs ────────────────────────────────────────────
echo "[1/6] Enabling required GCP APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    storage.googleapis.com \
    --project="$PROJECT" \
    --quiet

# ── Step 2: Create Artifact Registry repo (idempotent) ──────────────────────
echo "[2/6] Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories describe gemma4 \
    --location="$REGION" \
    --project="$PROJECT" \
    --quiet 2>/dev/null \
|| gcloud artifacts repositories create gemma4 \
    --repository-format=docker \
    --location="$REGION" \
    --description="Gemma 4 inference stack images" \
    --project="$PROJECT" \
    --quiet

# ── Step 3: Upload model to GCS (only if local copy provided) ───────────────
echo "[3/6] Checking GCS model staging..."
GCS_URI="gs://${GCS_BUCKET}/${GCS_BLOB}"

# Create bucket if needed
gcloud storage buckets describe "gs://${GCS_BUCKET}" \
    --project="$PROJECT" \
    --quiet 2>/dev/null \
|| gcloud storage buckets create "gs://${GCS_BUCKET}" \
    --location="$REGION" \
    --project="$PROJECT" \
    --uniform-bucket-level-access \
    --quiet

if [ -n "$LOCAL_MODEL_PATH" ] && [ -f "$LOCAL_MODEL_PATH" ]; then
    # Check if model already uploaded (by comparing size)
    REMOTE_SIZE=$(gcloud storage objects describe "${GCS_URI}" \
        --format='value(size)' 2>/dev/null || echo "0")
    LOCAL_SIZE=$(stat -f%z "$LOCAL_MODEL_PATH" 2>/dev/null || stat -c%s "$LOCAL_MODEL_PATH")

    if [ "$REMOTE_SIZE" = "$LOCAL_SIZE" ]; then
        echo "  Model already in GCS (size matches: ${LOCAL_SIZE} bytes). Skipping upload."
    else
        echo "  Uploading model to ${GCS_URI}..."
        echo "  Size: $(du -sh "$LOCAL_MODEL_PATH" | cut -f1)"
        gcloud storage cp "$LOCAL_MODEL_PATH" "${GCS_URI}" \
            --project="$PROJECT"
        echo "  Upload complete."
    fi
else
    # No local model — container will download from HuggingFace on first cold start
    REMOTE_EXISTS=$(gcloud storage objects describe "${GCS_URI}" \
        --format='value(name)' 2>/dev/null || echo "")
    if [ -z "$REMOTE_EXISTS" ]; then
        echo "  No local model and no GCS copy found."
        echo "  The container will download from HuggingFace on first cold start (~80-120s)."
        echo "  To pre-stage: set LOCAL_MODEL_PATH=/path/to/gemma-4-26B-A4B-it-Q4_K_M.gguf"
        echo ""
    else
        echo "  Model already in GCS: ${GCS_URI}"
    fi
fi

# ── Step 4: Create/update Secret Manager secrets ────────────────────────────
echo "[4/6] Setting up Secret Manager secrets..."

_upsert_secret() {
    local name="$1" value="$2"
    if gcloud secrets describe "$name" --project="$PROJECT" --quiet 2>/dev/null; then
        echo "$value" | gcloud secrets versions add "$name" \
            --data-file=- \
            --project="$PROJECT" \
            --quiet
        echo "  Updated secret: $name"
    else
        echo "$value" | gcloud secrets create "$name" \
            --data-file=- \
            --replication-policy=automatic \
            --project="$PROJECT" \
            --quiet
        echo "  Created secret: $name"
    fi
}

_upsert_secret "heavy-api-key" "$HEAVY_API_KEY"

if [ -n "$HF_TOKEN" ]; then
    _upsert_secret "hf-token" "$HF_TOKEN"
else
    # Create empty placeholder so the secret reference in service.yaml doesn't fail
    gcloud secrets describe "hf-token" --project="$PROJECT" --quiet 2>/dev/null \
    || echo "" | gcloud secrets create "hf-token" \
        --data-file=- \
        --replication-policy=automatic \
        --project="$PROJECT" \
        --quiet
    echo "  hf-token left empty (no HF_TOKEN provided — container will use GCS)"
fi

# Grant Cloud Run SA access to secrets
CR_SA="${PROJECT}@appspot.gserviceaccount.com"
# Try the compute SA if the App Engine one doesn't exist
CR_SA_COMPUTE="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"-compute@developer.gserviceaccount.com

for SA in "$CR_SA" "$CR_SA_COMPUTE"; do
    gcloud projects add-iam-policy-binding "$PROJECT" \
        --member="serviceAccount:${SA}" \
        --role="roles/secretmanager.secretAccessor" \
        --quiet 2>/dev/null || true
done

# Grant GCS read access to Cloud Run SA
for SA in "$CR_SA" "$CR_SA_COMPUTE"; do
    gcloud storage buckets add-iam-policy-binding "gs://${GCS_BUCKET}" \
        --member="serviceAccount:${SA}" \
        --role="roles/storage.objectViewer" \
        --quiet 2>/dev/null || true
done

# ── Step 5: Build Docker image via Cloud Build ───────────────────────────────
echo "[5/6] Building Docker image with Cloud Build..."
echo "  This compiles llama-cpp-python with CUDA 12.4 — expect 10-15 minutes."
echo ""
gcloud builds submit \
    --tag="${REGISTRY}:latest" \
    --project="$PROJECT" \
    --machine-type=E2_HIGHCPU_32 \
    --timeout=3600s \
    "$SCRIPT_DIR"

echo "  Image pushed: ${REGISTRY}:latest"

# ── Step 6: Deploy Cloud Run GPU service ─────────────────────────────────────
echo "[6/6] Deploying Cloud Run GPU service..."
gcloud run services replace \
    "${SCRIPT_DIR}/service.yaml" \
    --region="$REGION" \
    --project="$PROJECT" \
    --quiet

# Allow unauthenticated access — auth is handled by HEAVY_API_KEY header check
gcloud run services add-iam-policy-binding "$SERVICE_NAME" \
    --region="$REGION" \
    --project="$PROJECT" \
    --member="allUsers" \
    --role="roles/run.invoker" \
    --quiet

# Get the service URL
HEAVY_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --project="$PROJECT" \
    --format='value(status.url)')

echo ""
echo "================================================"
echo "  Heavy tier deployed!"
echo ""
echo "  URL:          $HEAVY_URL"
echo "  HEAVY_API_KEY: $HEAVY_API_KEY"
echo ""
echo "  Add these to your gateway LaunchAgent plist:"
echo "    <key>HEAVY_URL</key><string>${HEAVY_URL}</string>"
echo "    <key>HEAVY_API_KEY</key><string>${HEAVY_API_KEY}</string>"
echo ""
echo "  Then restart the gateway:"
echo "    launchctl unload ~/Library/LaunchAgents/com.gemma4.gateway.plist"
echo "    launchctl load  ~/Library/LaunchAgents/com.gemma4.gateway.plist"
echo ""
echo "  Test:"
echo "    curl -X POST ${HEAVY_URL}/v1/chat/completions \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -H 'Authorization: Bearer ${HEAVY_API_KEY}' \\"
echo "      -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":50}'"
echo "================================================"
