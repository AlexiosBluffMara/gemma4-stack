"""Firebase Authentication module for token verification."""

import logging
from typing import NamedTuple, Optional

import firebase_admin
from firebase_admin import auth as firebase_auth

logger = logging.getLogger("auth")


class UserInfo(NamedTuple):
    uid: str
    email: str
    display_name: str
    photo_url: str


# Sentinel for anonymous/fallback access when auth is unavailable
ANONYMOUS_USER = UserInfo(
    uid="anonymous",
    email="",
    display_name="Anonymous",
    photo_url="",
)


class AuthService:
    """Firebase Auth token verification."""

    _initialized = False
    _available = False

    @classmethod
    def initialize(cls):
        """Initialize Firebase Admin SDK. Call once at startup.

        Uses Application Default Credentials on Cloud Run
        (the service account is auto-detected).
        Falls back to allowing anonymous access if initialization fails.
        """
        if cls._initialized:
            return
        cls._initialized = True
        try:
            firebase_admin.initialize_app()
            cls._available = True
            logger.info("Firebase Auth initialized successfully")
        except Exception as exc:
            cls._available = False
            logger.warning(
                "Firebase Auth unavailable — anonymous access enabled: %s", exc
            )

    @classmethod
    def verify_token(cls, id_token: str) -> Optional[UserInfo]:
        """Verify a Firebase ID token. Returns UserInfo or None.

        If Firebase Auth is not available (e.g. local dev without
        credentials), returns ANONYMOUS_USER with a warning.
        """
        if not cls._initialized:
            cls.initialize()

        if not cls._available:
            logger.debug(
                "Auth unavailable, returning anonymous user"
            )
            return ANONYMOUS_USER

        try:
            decoded = firebase_auth.verify_id_token(id_token)
            return UserInfo(
                uid=decoded["uid"],
                email=decoded.get("email", ""),
                display_name=decoded.get("name", ""),
                photo_url=decoded.get("picture", ""),
            )
        except Exception as exc:
            logger.warning("Token verification failed: %s", exc)
            return None

    @classmethod
    def extract_token(cls, authorization_header: str) -> Optional[str]:
        """Extract Bearer token from Authorization header."""
        if not authorization_header:
            return None
        parts = authorization_header.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
        return None
