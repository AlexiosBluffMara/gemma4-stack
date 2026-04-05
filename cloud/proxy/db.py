"""Firestore module for user profiles, conversations, and usage tracking."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from google.cloud import firestore

logger = logging.getLogger("db")

# Default daily limits for new users
_DEFAULT_LIMITS = {
    "daily_text": 100,
    "daily_media": 20,
    "monthly_storage_mb": 500,
}


class Database:
    """Async-safe Firestore wrapper for user data and usage tracking."""

    def __init__(self):
        self._client: Optional[firestore.AsyncClient] = None

    def _ensure_client(self):
        if self._client is None:
            self._client = firestore.AsyncClient()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    async def get_or_create_user(
        self,
        uid: str,
        email: str,
        display_name: str = "",
        photo_url: str = "",
    ) -> dict:
        """Get or create a user document. Returns user data dict."""
        try:
            self._ensure_client()
            doc_ref = self._client.collection("users").document(uid)
            doc = await doc_ref.get()

            if doc.exists:
                await doc_ref.update(
                    {"last_active": datetime.now(timezone.utc)}
                )
                data = doc.to_dict()
                data["uid"] = uid
                return data

            # Create new user
            user_data = {
                "uid": uid,
                "email": email,
                "display_name": display_name,
                "photo_url": photo_url,
                "limits": _DEFAULT_LIMITS.copy(),
                "created_at": datetime.now(timezone.utc),
                "last_active": datetime.now(timezone.utc),
                "monthly_text_count": 0,
                "monthly_media_count": 0,
                "monthly_tokens": 0,
            }
            await doc_ref.set(user_data)
            return user_data
        except Exception:
            logger.exception("Failed to get/create user %s", uid)
            return {
                "uid": uid,
                "email": email,
                "display_name": display_name,
                "limits": _DEFAULT_LIMITS.copy(),
            }

    async def get_user(self, uid: str) -> Optional[dict]:
        """Get user by UID. Returns None if not found."""
        try:
            self._ensure_client()
            doc = await self._client.collection("users").document(uid).get()
            if doc.exists:
                data = doc.to_dict()
                data["uid"] = uid
                return data
            return None
        except Exception:
            logger.exception("Failed to get user %s", uid)
            return None

    # ------------------------------------------------------------------
    # Usage Tracking
    # ------------------------------------------------------------------

    async def check_usage_limit(
        self, uid: str, event_type: str
    ) -> tuple[bool, int]:
        """Check if user is within daily usage limits.

        Returns (allowed, remaining_count).
        event_type: 'text' or 'media'
        """
        try:
            self._ensure_client()
            user = await self.get_user(uid)
            if user is None:
                return False, 0

            limit_key = f"daily_{event_type}"
            daily_limit = user.get("limits", _DEFAULT_LIMITS).get(
                limit_key, _DEFAULT_LIMITS.get(limit_key, 0)
            )

            # Count today's events
            today_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            query = (
                self._client.collection("usage_events")
                .where(filter=firestore.FieldFilter("uid", "==", uid))
                .where(
                    filter=firestore.FieldFilter(
                        "event_type", "==", event_type
                    )
                )
                .where(
                    filter=firestore.FieldFilter(
                        "created_at", ">=", today_start
                    )
                )
            )
            docs = [doc async for doc in query.stream()]
            used = len(docs)
            remaining = max(0, daily_limit - used)
            return remaining > 0, remaining
        except Exception:
            logger.exception(
                "Failed to check usage limit for user %s", uid
            )
            # Fail open: allow the request if we can't check
            return True, -1

    async def record_usage(
        self,
        uid: str,
        event_type: str,
        tier: str,
        tokens: int = 0,
        latency_ms: int = 0,
        media_size_bytes: int = 0,
        media_gcs_uri: str = "",
    ) -> str:
        """Record a usage event. Returns event ID."""
        try:
            self._ensure_client()
            event_data = {
                "uid": uid,
                "event_type": event_type,
                "tier": tier,
                "tokens": tokens,
                "latency_ms": latency_ms,
                "media_size_bytes": media_size_bytes,
                "media_gcs_uri": media_gcs_uri,
                "created_at": datetime.now(timezone.utc),
            }
            _, doc_ref = await self._client.collection(
                "usage_events"
            ).add(event_data)

            # Increment monthly counters on user doc
            user_ref = self._client.collection("users").document(uid)
            count_field = (
                "monthly_text_count"
                if event_type == "text"
                else "monthly_media_count"
            )
            await user_ref.update(
                {
                    count_field: firestore.Increment(1),
                    "monthly_tokens": firestore.Increment(tokens),
                }
            )
            return doc_ref.id
        except Exception:
            logger.exception("Failed to record usage for user %s", uid)
            return ""

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def create_conversation(self, uid: str, title: str = "") -> str:
        """Create a new conversation. Returns conversation_id."""
        try:
            self._ensure_client()
            now = datetime.now(timezone.utc)
            conv_data = {
                "uid": uid,
                "title": title or "New conversation",
                "created_at": now,
                "updated_at": now,
                "message_count": 0,
            }
            _, doc_ref = await self._client.collection(
                "conversations"
            ).add(conv_data)
            return doc_ref.id
        except Exception:
            logger.exception(
                "Failed to create conversation for user %s", uid
            )
            return ""

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        media_refs: list[str] = None,
        routing: dict = None,
        tokens_used: int = 0,
    ) -> str:
        """Add a message to a conversation. Returns message_id."""
        try:
            self._ensure_client()
            now = datetime.now(timezone.utc)
            msg_data = {
                "role": role,
                "content": content,
                "media_refs": media_refs or [],
                "routing": routing or {},
                "tokens_used": tokens_used,
                "created_at": now,
            }
            conv_ref = self._client.collection("conversations").document(
                conversation_id
            )
            _, msg_ref = await conv_ref.collection("messages").add(msg_data)

            # Update conversation metadata
            await conv_ref.update(
                {
                    "updated_at": now,
                    "message_count": firestore.Increment(1),
                }
            )
            return msg_ref.id
        except Exception:
            logger.exception(
                "Failed to add message to conversation %s", conversation_id
            )
            return ""

    async def get_conversations(
        self, uid: str, limit: int = 20
    ) -> list[dict]:
        """Get recent conversations for a user, ordered by updated_at desc."""
        try:
            self._ensure_client()
            query = (
                self._client.collection("conversations")
                .where(filter=firestore.FieldFilter("uid", "==", uid))
                .order_by("updated_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            results = []
            async for doc in query.stream():
                data = doc.to_dict()
                data["id"] = doc.id
                results.append(data)
            return results
        except Exception:
            logger.exception(
                "Failed to get conversations for user %s", uid
            )
            return []

    async def get_messages(
        self, conversation_id: str, limit: int = 50
    ) -> list[dict]:
        """Get messages in a conversation, ordered by created_at."""
        try:
            self._ensure_client()
            query = (
                self._client.collection("conversations")
                .document(conversation_id)
                .collection("messages")
                .order_by("created_at")
                .limit(limit)
            )
            results = []
            async for doc in query.stream():
                data = doc.to_dict()
                data["id"] = doc.id
                results.append(data)
            return results
        except Exception:
            logger.exception(
                "Failed to get messages for conversation %s", conversation_id
            )
            return []

    # ------------------------------------------------------------------
    # Usage Stats
    # ------------------------------------------------------------------

    async def get_usage_stats(self, uid: str, days: int = 30) -> dict:
        """Get usage statistics for a user over the last N days."""
        try:
            self._ensure_client()
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            query = (
                self._client.collection("usage_events")
                .where(filter=firestore.FieldFilter("uid", "==", uid))
                .where(
                    filter=firestore.FieldFilter("created_at", ">=", cutoff)
                )
                .order_by("created_at")
            )

            total_requests = 0
            text_requests = 0
            media_requests = 0
            tokens_generated = 0
            storage_bytes = 0
            by_day: dict[str, dict] = {}

            async for doc in query.stream():
                data = doc.to_dict()
                total_requests += 1
                evt = data.get("event_type", "")
                if evt == "text":
                    text_requests += 1
                elif evt == "media":
                    media_requests += 1
                tokens_generated += data.get("tokens", 0)
                storage_bytes += data.get("media_size_bytes", 0)

                day_key = data.get("created_at", cutoff).strftime("%Y-%m-%d")
                day = by_day.setdefault(
                    day_key,
                    {"requests": 0, "tokens": 0, "media_bytes": 0},
                )
                day["requests"] += 1
                day["tokens"] += data.get("tokens", 0)
                day["media_bytes"] += data.get("media_size_bytes", 0)

            return {
                "total_requests": total_requests,
                "text_requests": text_requests,
                "media_requests": media_requests,
                "tokens_generated": tokens_generated,
                "storage_bytes": storage_bytes,
                "by_day": [
                    {"date": k, **v}
                    for k, v in sorted(by_day.items())
                ],
            }
        except Exception:
            logger.exception("Failed to get usage stats for user %s", uid)
            return {
                "total_requests": 0,
                "text_requests": 0,
                "media_requests": 0,
                "tokens_generated": 0,
                "storage_bytes": 0,
                "by_day": [],
            }
