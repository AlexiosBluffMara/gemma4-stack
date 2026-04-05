#!/usr/bin/env bash
# Deploy the budget alert Cloud Function
set -euo pipefail

PROJECT_ID="${1:-gemma4good}"
REGION="us-central1"

echo "Deploying budget alert function..."

gcloud functions deploy budget-alert \
  --gen2 \
  --runtime=python312 \
  --trigger-topic=gemma4-budget-alerts \
  --entry-point=handle_budget_alert \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --memory=256Mi \
  --timeout=60s \
  --set-env-vars="GCP_PROJECT=${PROJECT_ID},REGION=${REGION},SERVICE_NAME=gemma4-proxy" \
  --quiet

echo "Done. Function will auto-shutdown Cloud Run when budget is exceeded."
