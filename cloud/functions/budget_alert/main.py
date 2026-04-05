"""Budget alert handler for Gemma 4 Cloud Run service.

Triggered by Pub/Sub messages from Cloud Billing budget notifications.
- At 90%: logs a warning
- At 100%: scales Cloud Run service to 0 max instances (effectively stops it)
- Can be manually re-enabled by updating the service's maxScale annotation

Deploy:
  gcloud functions deploy budget-alert \
    --gen2 --runtime=python312 \
    --trigger-topic=gemma4-budget-alerts \
    --entry-point=handle_budget_alert \
    --region=us-central1 \
    --project=gemma4good
"""

import base64
import json
import logging
import os

import functions_framework
from google.cloud import run_v2

PROJECT_ID = os.environ.get("GCP_PROJECT", "gemma4good")
REGION = os.environ.get("REGION", "us-central1")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "gemma4-proxy")
SHUTDOWN_THRESHOLD = float(os.environ.get("SHUTDOWN_THRESHOLD", "1.0"))

logger = logging.getLogger(__name__)


@functions_framework.cloud_event
def handle_budget_alert(cloud_event):
    """Handle a budget alert Pub/Sub message."""
    # Decode the Pub/Sub message
    data = base64.b64decode(cloud_event.data["message"]["data"])
    budget_notification = json.loads(data)

    cost_amount = budget_notification.get("costAmount", 0)
    budget_amount = budget_notification.get("budgetAmount", 0)
    alert_threshold = budget_notification.get("alertThresholdExceeded", 0)

    logger.info(
        f"Budget alert: ${cost_amount:.2f} / ${budget_amount:.2f} "
        f"(threshold: {alert_threshold * 100:.0f}%)"
    )

    if alert_threshold >= SHUTDOWN_THRESHOLD:
        logger.warning(
            f"Budget threshold {alert_threshold * 100:.0f}% exceeded! "
            f"Scaling down {SERVICE_NAME}..."
        )
        _scale_to_zero()
    elif alert_threshold >= 0.9:
        logger.warning(
            f"Budget at {alert_threshold * 100:.0f}% - approaching limit. "
            f"Service will be stopped at {SHUTDOWN_THRESHOLD * 100:.0f}%."
        )


def _scale_to_zero():
    """Scale the Cloud Run service to 0 max instances."""
    try:
        client = run_v2.ServicesClient()
        service_name = f"projects/{PROJECT_ID}/locations/{REGION}/services/{SERVICE_NAME}"

        # Get current service
        service = client.get_service(name=service_name)

        # Update max scale to 0
        service.template.scaling.max_instance_count = 0

        # Update the service
        update_mask = {"paths": ["template.scaling.max_instance_count"]}
        operation = client.update_service(
            service=service,
            update_mask=update_mask,
        )
        result = operation.result()
        logger.info(f"Service {SERVICE_NAME} scaled to 0. URI: {result.uri}")

    except Exception as e:
        logger.error(f"Failed to scale down {SERVICE_NAME}: {e}", exc_info=True)
