"""Google Cloud Storage module for persisting media files and AI insights."""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from typing import Optional

from google.cloud import storage

logger = logging.getLogger("storage")

BUCKET_NAME = os.environ.get("GCS_BUCKET", "gemma4good-media")


class StorageClient:
    """Async-safe wrapper around Google Cloud Storage."""

    def __init__(self):
        self._client: Optional[storage.Client] = None
        self._bucket: Optional[storage.Bucket] = None

    def _ensure_client(self):
        if self._client is None:
            self._client = storage.Client()
            self._bucket = self._client.bucket(BUCKET_NAME)

    async def upload_media(
        self,
        user_id: str,
        file_data: bytes,
        filename: str,
        content_type: str,
        category: str,
    ) -> dict:
        """Upload raw media to GCS. Returns {gcs_uri, signed_url, size_bytes}."""
        try:
            self._ensure_client()
            ext = PurePosixPath(filename).suffix or ".bin"
            media_id = uuid.uuid4().hex
            date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            blob_path = f"uploads/{user_id}/{date_prefix}/{media_id}{ext}"

            loop = asyncio.get_running_loop()

            def _upload():
                blob = self._bucket.blob(blob_path)
                blob.upload_from_string(file_data, content_type=content_type)
                blob.metadata = {"category": category, "original_name": filename}
                blob.patch()
                signed_url = blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(hours=1),
                    method="GET",
                )
                return signed_url

            signed_url = await loop.run_in_executor(None, _upload)
            gcs_uri = f"gs://{BUCKET_NAME}/{blob_path}"
            return {
                "gcs_uri": gcs_uri,
                "signed_url": signed_url,
                "size_bytes": len(file_data),
                "media_id": media_id,
            }
        except Exception:
            logger.exception("Failed to upload media for user %s", user_id)
            return {
                "gcs_uri": "",
                "signed_url": "",
                "size_bytes": len(file_data),
                "media_id": "",
            }

    async def upload_insight(
        self,
        user_id: str,
        conversation_id: str,
        message_id: str,
        insight_data: dict,
    ) -> str:
        """Upload AI response JSON to GCS. Returns gcs_uri."""
        try:
            self._ensure_client()
            blob_path = (
                f"insights/{user_id}/{conversation_id}/{message_id}.json"
            )
            payload = json.dumps(insight_data, ensure_ascii=False, default=str)

            loop = asyncio.get_running_loop()

            def _upload():
                blob = self._bucket.blob(blob_path)
                blob.upload_from_string(
                    payload, content_type="application/json"
                )

            await loop.run_in_executor(None, _upload)
            return f"gs://{BUCKET_NAME}/{blob_path}"
        except Exception:
            logger.exception("Failed to upload insight for user %s", user_id)
            return ""

    async def upload_thumbnail(
        self, user_id: str, media_id: str, thumb_data: bytes
    ) -> str:
        """Upload a thumbnail image. Returns gcs_uri."""
        try:
            self._ensure_client()
            blob_path = f"thumbnails/{user_id}/{media_id}_thumb.jpg"

            loop = asyncio.get_running_loop()

            def _upload():
                blob = self._bucket.blob(blob_path)
                blob.upload_from_string(
                    thumb_data, content_type="image/jpeg"
                )

            await loop.run_in_executor(None, _upload)
            return f"gs://{BUCKET_NAME}/{blob_path}"
        except Exception:
            logger.exception(
                "Failed to upload thumbnail for user %s", user_id
            )
            return ""

    async def get_signed_url(
        self, gcs_uri: str, expiration_minutes: int = 60
    ) -> str:
        """Generate a signed download URL for a GCS object."""
        try:
            self._ensure_client()
            # Parse gs://bucket/path
            path = gcs_uri.removeprefix(f"gs://{BUCKET_NAME}/")

            loop = asyncio.get_running_loop()

            def _sign():
                blob = self._bucket.blob(path)
                return blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(minutes=expiration_minutes),
                    method="GET",
                )

            return await loop.run_in_executor(None, _sign)
        except Exception:
            logger.exception("Failed to generate signed URL for %s", gcs_uri)
            return ""

    async def delete_object(self, gcs_uri: str) -> bool:
        """Delete an object from GCS. Returns True if deleted."""
        try:
            self._ensure_client()
            path = gcs_uri.removeprefix(f"gs://{BUCKET_NAME}/")

            loop = asyncio.get_running_loop()

            def _delete():
                blob = self._bucket.blob(path)
                blob.delete()

            await loop.run_in_executor(None, _delete)
            return True
        except Exception:
            logger.exception("Failed to delete %s", gcs_uri)
            return False

    async def get_user_storage_bytes(self, user_id: str) -> int:
        """Calculate total storage used by a user (approximate)."""
        try:
            self._ensure_client()
            prefix = f"uploads/{user_id}/"

            loop = asyncio.get_running_loop()

            def _sum_sizes():
                total = 0
                for blob in self._client.list_blobs(
                    BUCKET_NAME, prefix=prefix
                ):
                    total += blob.size or 0
                return total

            return await loop.run_in_executor(None, _sum_sizes)
        except Exception:
            logger.exception(
                "Failed to calculate storage for user %s", user_id
            )
            return 0
