"""
BridgeTwin — Offline-first bridge inspection and digital twin module.

Implements the field inspection workflow described in the BridgeTwin project:
  - AASHTO-compliant defect tagging using a local SQLite database
  - Multimodal crack/defect detection prompt assembly for Gemma 4 vision models
  - Inspection record management (create, save, sync-ready export)
  - Geographic stamping, timestamping, and legally-grounded condition tagging

AASHTO Reference:
  AASHTO Manual for Bridge Element Inspection, 2nd Edition.
  Condition States (CS):
    CS 1 — Good condition (no defect / minor surface defects)
    CS 2 — Fair condition (minor cracking / surface deterioration)
    CS 3 — Poor condition (moderate cracking / spalling / exposed rebar)
    CS 4 — Severe / Imminent Failure

This module is designed to run fully offline on Gemma 4 E2B / E4B (on-device).
When network is available, call ``export_sync_payload()`` to build a JSON
package ready for upload to the municipal documentation server.

Usage (edge device, offline):
    from bridge_twin import InspectionSession

    session = InspectionSession(inspector_id="eng-001", bridge_id="IL-0047")
    record  = session.new_record(lat=40.5142, lon=-88.9906)
    record.set_image_base64(b64_image_string)
    # Ask Gemma 4 (via gateway or on-device) to classify the image; parse result:
    session.apply_detection(record, gemma_response_text)
    session.save()
    payload = session.export_sync_payload()   # upload when back online
"""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# AASHTO condition-state catalogue
# ---------------------------------------------------------------------------

# (element_code, element_name, condition_state, description, severity_label)
_AASHTO_RAW: list[tuple[str, str, int, str, str]] = [
    # Concrete elements
    ("RC-CON", "Reinforced Concrete Column / Pier",    1, "Sound concrete, no defects observed.", "Good"),
    ("RC-CON", "Reinforced Concrete Column / Pier",    2, "Cracking ≤ 0.01 in. wide, minor delamination.", "Fair"),
    ("RC-CON", "Reinforced Concrete Column / Pier",    3, "Crack > 0.01 in., spall ≥ 1 in. deep, or delamination.", "Poor"),
    ("RC-CON", "Reinforced Concrete Column / Pier",    4, "Exposed rebar, section loss > 25 %, or collapse risk.", "Severe"),
    ("RC-DEC", "Reinforced Concrete Deck",             1, "Deck surface sound; no cracking or scaling.", "Good"),
    ("RC-DEC", "Reinforced Concrete Deck",             2, "Fine surface cracks (map cracking), minor scaling.", "Fair"),
    ("RC-DEC", "Reinforced Concrete Deck",             3, "Spalling ≥ 1 in. deep, delamination, exposed rebar.", "Poor"),
    ("RC-DEC", "Reinforced Concrete Deck",             4, "Through-depth cracks, severe section loss.", "Severe"),
    ("RC-BM",  "Reinforced Concrete Beam / Girder",   1, "No cracks or staining; good surface condition.", "Good"),
    ("RC-BM",  "Reinforced Concrete Beam / Girder",   2, "Minor flexural cracks (< 0.01 in.).", "Fair"),
    ("RC-BM",  "Reinforced Concrete Beam / Girder",   3, "Shear or flexural cracks > 0.01 in., spall, delamination.", "Poor"),
    ("RC-BM",  "Reinforced Concrete Beam / Girder",   4, "Exposed / corroded rebar, severe section loss.", "Severe"),
    # Steel elements
    ("ST-BM",  "Steel Beam / Girder",                 1, "No section loss; paint intact.", "Good"),
    ("ST-BM",  "Steel Beam / Girder",                 2, "Moderate corrosion; paint failure; < 10 % section loss.", "Fair"),
    ("ST-BM",  "Steel Beam / Girder",                 3, "Heavy corrosion; 10–25 % section loss.", "Poor"),
    ("ST-BM",  "Steel Beam / Girder",                 4, "Active corrosion; > 25 % section loss; laminar cracking.", "Severe"),
    # Timber elements
    ("TM-STR", "Timber Stringer / Stringer Beam",     1, "Sound wood; no decay or insect damage.", "Good"),
    ("TM-STR", "Timber Stringer / Stringer Beam",     2, "Surface checks ≤ 1/3 depth; minor decay at ends.", "Fair"),
    ("TM-STR", "Timber Stringer / Stringer Beam",     3, "Splits > 1/3 depth, moderate decay, section loss < 25 %.", "Poor"),
    ("TM-STR", "Timber Stringer / Stringer Beam",     4, "Severe decay; section loss > 25 %; punky / hollow.", "Severe"),
    # Protective coatings / paint systems
    ("PC-GEN", "Paint / Protective Coating (General)", 1, "Intact coating, no rust bleed.", "Good"),
    ("PC-GEN", "Paint / Protective Coating (General)", 2, "Minor rust bleeding ≤ 10 % of surface.", "Fair"),
    ("PC-GEN", "Paint / Protective Coating (General)", 3, "Rust bleeding 10–50 % of surface area.", "Poor"),
    ("PC-GEN", "Paint / Protective Coating (General)", 4, "Rust bleeding > 50 % or coating delamination.", "Severe"),
    # Joints
    ("JT-EXP", "Expansion Joint",                     1, "Seal intact; no debris accumulation.", "Good"),
    ("JT-EXP", "Expansion Joint",                     2, "Minor seal damage; minor debris.", "Fair"),
    ("JT-EXP", "Expansion Joint",                     3, "Seal failure, significant debris, minor spall at interface.", "Poor"),
    ("JT-EXP", "Expansion Joint",                     4, "Joint missing / fully failed; section loss at interface.", "Severe"),
]

# Keyword → AASHTO element code mapping used by auto-detection
_KEYWORD_TO_ELEMENT: dict[str, str] = {
    "column":    "RC-CON",
    "pier":      "RC-CON",
    "pylon":     "RC-CON",
    "deck":      "RC-DEC",
    "slab":      "RC-DEC",
    "beam":      "RC-BM",
    "girder":    "RC-BM",
    "steel":     "ST-BM",
    "timber":    "TM-STR",
    "wood":      "TM-STR",
    "paint":     "PC-GEN",
    "coating":   "PC-GEN",
    "joint":     "JT-EXP",
    "expansion": "JT-EXP",
    "rebar":     "RC-CON",
    "spall":     "RC-CON",
    "crack":     "RC-CON",
}

# Keywords that hint at a particular condition state
_CS_KEYWORDS: dict[int, list[str]] = {
    1: ["good", "sound", "intact", "no defect", "no crack", "clean"],
    2: ["minor", "fair", "slight", "fine crack", "surface crack", "map crack"],
    3: ["spall", "delamination", "exposed rebar", "moderate", "poor", "0.01", "deep crack", "rust"],
    4: ["severe", "collapse", "section loss", "imminent", "punky", "hollow", "missing"],
}

# ---------------------------------------------------------------------------
# AASHTO SQLite database
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH = Path(tempfile.gettempdir()) / "bridge_twin_aashto.db"


def _init_aashto_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialise (or open) the local AASHTO SQLite database.

    The database is created in memory if *db_path* is ``:memory:``.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS aashto_conditions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            element_code     TEXT    NOT NULL,
            element_name     TEXT    NOT NULL,
            condition_state  INTEGER NOT NULL CHECK(condition_state BETWEEN 1 AND 4),
            description      TEXT    NOT NULL,
            severity_label   TEXT    NOT NULL,
            UNIQUE(element_code, condition_state)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inspection_records (
            id               TEXT    PRIMARY KEY,
            bridge_id        TEXT    NOT NULL,
            inspector_id     TEXT    NOT NULL,
            timestamp_utc    TEXT    NOT NULL,
            latitude         REAL,
            longitude        REAL,
            element_code     TEXT,
            condition_state  INTEGER,
            severity_label   TEXT,
            image_b64        TEXT,
            notes            TEXT,
            synced           INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()

    # Seed AASHTO condition states (idempotent — IGNORE on conflict)
    conn.executemany(
        """INSERT OR IGNORE INTO aashto_conditions
           (element_code, element_name, condition_state, description, severity_label)
           VALUES (?, ?, ?, ?, ?)""",
        _AASHTO_RAW,
    )
    conn.commit()
    return conn


def lookup_condition(
    element_code: str,
    condition_state: int,
    db_path: Path = DEFAULT_DB_PATH,
) -> Optional[dict[str, Any]]:
    """Return the AASHTO condition-state record for a given element and CS."""
    conn = _init_aashto_db(db_path)
    row = conn.execute(
        "SELECT * FROM aashto_conditions WHERE element_code=? AND condition_state=?",
        (element_code.upper(), condition_state),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return dict(row)


def list_elements(db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, Any]]:
    """Return the distinct bridge elements defined in the AASHTO catalogue."""
    conn = _init_aashto_db(db_path)
    rows = conn.execute(
        "SELECT DISTINCT element_code, element_name FROM aashto_conditions ORDER BY element_code"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Defect detection helpers
# ---------------------------------------------------------------------------

def build_detection_prompt(image_description: str = "") -> str:
    """Return the system prompt for Gemma 4 bridge defect detection.

    The prompt instructs the model to:
    1. Identify the structural element visible in the image.
    2. Rate its condition using AASHTO condition states 1–4.
    3. List specific defects (crack width, spalling depth, exposed rebar, etc.).
    4. Respond in strict JSON so results can be parsed programmatically.
    """
    return (
        "You are a certified bridge inspection AI assistant grounded in the "
        "AASHTO Manual for Bridge Element Inspection (2nd Edition). "
        "When provided with an image of a bridge structural element, respond "
        "ONLY with a JSON object using exactly this schema:\n"
        '{\n'
        '  "element_type": "<one of: column, pier, deck, beam, girder, steel, timber, joint, coating>",\n'
        '  "condition_state": <integer 1–4>,\n'
        '  "severity": "<Good|Fair|Poor|Severe>",\n'
        '  "defects": ["<defect 1>", "<defect 2>"],\n'
        '  "notes": "<free-form technical notes>"\n'
        '}\n\n'
        "Condition state key:\n"
        "  CS 1 = Good  — no defects or very minor surface blemishes\n"
        "  CS 2 = Fair  — minor cracking / surface deterioration\n"
        "  CS 3 = Poor  — moderate cracking / spalling ≥ 1 in. deep / exposed rebar\n"
        "  CS 4 = Severe — section loss > 25 %, collapse risk, imminent failure\n"
        + (f"\nContext: {image_description}" if image_description else "")
    )


def parse_detection_response(response_text: str) -> Optional[dict[str, Any]]:
    """Extract the JSON detection payload from a Gemma 4 model response.

    Handles both bare JSON and JSON embedded in markdown code fences.
    Returns ``None`` if no valid JSON matching the schema is found.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", response_text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.rstrip("`").strip()

    # Try direct parse first
    candidates = [cleaned]

    # Also try to extract the first {...} block
    match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "condition_state" in obj:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def infer_condition_state_from_text(text: str) -> tuple[str, int]:
    """Heuristically derive (element_code, condition_state) from free-form text.

    Used as a fallback when the model response cannot be parsed as JSON.
    Returns ("RC-CON", 1) if nothing is detected.
    """
    lower = text.lower()

    # Detect element type
    element_code = "RC-CON"  # default
    for kw, code in _KEYWORD_TO_ELEMENT.items():
        if kw in lower:
            element_code = code
            break

    # Detect condition state — higher states take priority
    condition_state = 1
    for cs in (4, 3, 2, 1):
        if any(kw in lower for kw in _CS_KEYWORDS[cs]):
            condition_state = cs
            break

    return element_code, condition_state


# ---------------------------------------------------------------------------
# Inspection record
# ---------------------------------------------------------------------------

@dataclass
class InspectionRecord:
    """A single field observation linked to an AASHTO condition state."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    bridge_id: str = ""
    inspector_id: str = ""
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    element_code: Optional[str] = None
    condition_state: Optional[int] = None
    severity_label: Optional[str] = None
    image_b64: Optional[str] = None  # base64-encoded JPEG/PNG
    notes: str = ""
    synced: bool = False

    # ------------------------------------------------------------------
    def set_image_bytes(self, raw: bytes) -> None:
        """Attach an image from raw bytes (JPEG or PNG)."""
        self.image_b64 = base64.b64encode(raw).decode("ascii")

    def set_image_base64(self, b64: str) -> None:
        """Attach an image already encoded as a base64 string."""
        self.image_b64 = b64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InspectionRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Inspection session
# ---------------------------------------------------------------------------

class InspectionSession:
    """Manages a collection of field inspection records for a single bridge.

    Designed to operate entirely offline.  All records are persisted to the
    local SQLite database.  Call ``export_sync_payload()`` to get a JSON
    bundle suitable for upload to the municipal server once connectivity is
    restored.

    Args:
        bridge_id:    Municipal bridge identifier (e.g. "IL-0047").
        inspector_id: Badge / login ID of the field engineer.
        db_path:      Path to the local SQLite database.
                      Defaults to a shared file in the OS temp directory so
                      that multiple sessions for the same bridge accumulate.
    """

    def __init__(
        self,
        bridge_id: str,
        inspector_id: str,
        db_path: Path = DEFAULT_DB_PATH,
    ) -> None:
        self.bridge_id = bridge_id
        self.inspector_id = inspector_id
        self.db_path = db_path
        self._conn = _init_aashto_db(db_path)
        self._records: list[InspectionRecord] = []

    # ------------------------------------------------------------------
    # Record management
    # ------------------------------------------------------------------

    def new_record(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> InspectionRecord:
        """Create and register a new blank inspection record."""
        rec = InspectionRecord(
            bridge_id=self.bridge_id,
            inspector_id=self.inspector_id,
            latitude=lat,
            longitude=lon,
        )
        self._records.append(rec)
        return rec

    def apply_detection(
        self,
        record: InspectionRecord,
        gemma_response: str,
    ) -> InspectionRecord:
        """Parse a Gemma 4 detection response and update the record in-place.

        Falls back to heuristic keyword analysis if the JSON cannot be parsed.
        The record is cross-referenced against the local AASHTO database to
        attach the canonical condition-state description.

        Args:
            record:          The InspectionRecord to update.
            gemma_response:  Raw text returned by the Gemma 4 model.

        Returns:
            The updated record (same object, mutated in-place).
        """
        parsed = parse_detection_response(gemma_response)

        if parsed is not None:
            # Map natural-language element type to AASHTO code
            elem_type = str(parsed.get("element_type", "")).lower()
            element_code = _KEYWORD_TO_ELEMENT.get(elem_type, "RC-CON")
            condition_state = int(parsed.get("condition_state", 1))
            severity_label = str(parsed.get("severity", "Good"))
            notes_parts: list[str] = []
            if parsed.get("defects"):
                notes_parts.append("Defects: " + ", ".join(str(d) for d in parsed["defects"]))
            if parsed.get("notes"):
                notes_parts.append(str(parsed["notes"]))
            record.notes = " | ".join(notes_parts) if notes_parts else record.notes
        else:
            # Fallback: heuristic keyword analysis
            element_code, condition_state = infer_condition_state_from_text(gemma_response)
            severity_label = {1: "Good", 2: "Fair", 3: "Poor", 4: "Severe"}.get(
                condition_state, "Good"
            )

        record.element_code = element_code
        record.condition_state = condition_state
        record.severity_label = severity_label
        return record

    def save(self) -> int:
        """Persist all in-memory records to the local SQLite database.

        Returns:
            Number of records written (new + updated).
        """
        written = 0
        for rec in self._records:
            self._conn.execute(
                """INSERT OR REPLACE INTO inspection_records
                   (id, bridge_id, inspector_id, timestamp_utc, latitude, longitude,
                    element_code, condition_state, severity_label, image_b64, notes, synced)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec.id,
                    rec.bridge_id,
                    rec.inspector_id,
                    rec.timestamp_utc,
                    rec.latitude,
                    rec.longitude,
                    rec.element_code,
                    rec.condition_state,
                    rec.severity_label,
                    rec.image_b64,
                    rec.notes,
                    int(rec.synced),
                ),
            )
            written += 1
        self._conn.commit()
        return written

    def load_unsynced(self) -> list[InspectionRecord]:
        """Load all unsynced records for this bridge from the database."""
        rows = self._conn.execute(
            "SELECT * FROM inspection_records WHERE bridge_id=? AND synced=0",
            (self.bridge_id,),
        ).fetchall()
        return [InspectionRecord(**dict(r)) for r in rows]

    def mark_synced(self, record_ids: list[str]) -> None:
        """Mark the given record IDs as successfully synced."""
        if not record_ids:
            return
        placeholders = ",".join("?" * len(record_ids))
        self._conn.execute(
            f"UPDATE inspection_records SET synced=1 WHERE id IN ({placeholders})",
            record_ids,
        )
        self._conn.commit()

    def export_sync_payload(self) -> dict[str, Any]:
        """Build a sync-ready JSON payload containing all unsynced records.

        This payload is uploaded to the municipal documentation server once
        network connectivity is restored.

        Returns:
            A dict with keys:
              ``bridge_id``, ``inspector_id``, ``exported_at``,
              ``record_count``, ``records``
        """
        unsynced = self.load_unsynced()
        return {
            "bridge_id": self.bridge_id,
            "inspector_id": self.inspector_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "record_count": len(unsynced),
            "records": [r.to_dict() for r in unsynced],
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "InspectionSession":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Gemma 4 multimodal message builder
# ---------------------------------------------------------------------------

def build_vision_message(
    image_b64: str,
    image_mime: str = "image/jpeg",
    context: str = "",
) -> list[dict[str, Any]]:
    """Build the ``messages`` payload for a Gemma 4 vision inference call.

    The payload is OpenAI-compatible and works with the gateway's
    ``POST /v1/chat/completions`` endpoint.

    Args:
        image_b64:  Base64-encoded image string.
        image_mime: MIME type of the image (default: ``image/jpeg``).
        context:    Optional free-form context passed as a user text turn.

    Returns:
        A list suitable for the ``messages`` field of a chat-completions request.
    """
    system_prompt = build_detection_prompt(context)
    data_url = f"data:{image_mime};base64,{image_b64}"
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": (
                        context
                        or "Please inspect this bridge element and classify the condition state."
                    ),
                },
            ],
        },
    ]
