"""
GemmaZeroWaste — Pantry Scanner Module

Uses Gemma 4 multimodal vision to identify pantry items from camera images,
then extracts structured inventory via native function calling.

Designed for on-device, offline-first inference (no images leave the device).
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("zerowaste.scanner")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

FOOD_CATEGORIES = [
    "grain", "protein", "dairy", "fruit", "vegetable",
    "legume", "canned", "condiment", "snack", "beverage",
    "frozen", "oil", "spice", "baking", "other",
]

UNIT_CHOICES = [
    "oz", "lb", "g", "kg", "ml", "L", "cup", "tbsp", "tsp",
    "each", "can", "bag", "box", "jar", "bottle", "bunch",
]


@dataclass
class PantryItem:
    """A single identified pantry item."""
    name: str
    category: str = "other"
    quantity: float = 1.0
    unit: str = "each"
    estimated_expiry_days: int | None = None
    confidence: float = 0.0

    @property
    def expiry_date(self) -> date | None:
        if self.estimated_expiry_days is not None:
            return date.today() + timedelta(days=self.estimated_expiry_days)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "quantity": self.quantity,
            "unit": self.unit,
            "estimated_expiry_days": self.estimated_expiry_days,
            "expiry_date": str(self.expiry_date) if self.expiry_date else None,
            "confidence": self.confidence,
        }


@dataclass
class PantryInventory:
    """Complete pantry inventory from one or more scans."""
    items: list[PantryItem] = field(default_factory=list)
    scan_count: int = 0

    def add_items(self, new_items: list[PantryItem]) -> None:
        """Merge new items, incrementing quantity for duplicates."""
        existing = {item.name.lower(): item for item in self.items}
        for item in new_items:
            key = item.name.lower()
            if key in existing:
                existing[key].quantity += item.quantity
            else:
                self.items.append(item)
                existing[key] = item
        self.scan_count += 1

    @property
    def by_category(self) -> dict[str, list[PantryItem]]:
        cats: dict[str, list[PantryItem]] = {}
        for item in self.items:
            cats.setdefault(item.category, []).append(item)
        return cats

    @property
    def expiring_soon(self) -> list[PantryItem]:
        """Items expiring within 3 days."""
        cutoff = date.today() + timedelta(days=3)
        return [
            i for i in self.items
            if i.expiry_date is not None and i.expiry_date <= cutoff
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "scan_count": self.scan_count,
            "total_items": len(self.items),
            "categories": list(self.by_category.keys()),
            "expiring_soon": [i.to_dict() for i in self.expiring_soon],
        }


# ---------------------------------------------------------------------------
# Gemma 4 function-calling tool definitions
# ---------------------------------------------------------------------------

PANTRY_SCAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "register_pantry_items",
            "description": (
                "Register identified food items from a pantry shelf image. "
                "Call this once with ALL items visible in the image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "List of identified pantry items",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Item name (e.g. 'brown rice', 'canned black beans')",
                                },
                                "category": {
                                    "type": "string",
                                    "enum": FOOD_CATEGORIES,
                                    "description": "Food category",
                                },
                                "quantity": {
                                    "type": "number",
                                    "description": "Estimated count or amount visible",
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": UNIT_CHOICES,
                                    "description": "Unit of measurement",
                                },
                                "estimated_expiry_days": {
                                    "type": "integer",
                                    "description": (
                                        "Estimated days until expiry based on item type. "
                                        "Fresh produce: 3-7, dairy: 7-14, canned: 365+, "
                                        "grains: 180+. Use null if unknown."
                                    ),
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Detection confidence 0.0-1.0",
                                },
                            },
                            "required": ["name", "category", "quantity", "unit"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    }
]

SCAN_SYSTEM_PROMPT = (
    "You are a pantry inventory assistant for GemmaZeroWaste, an app that helps "
    "families in food deserts reduce waste and optimize nutrition. "
    "When shown a photo of a pantry shelf or food items:\n"
    "1. Identify EVERY visible food item.\n"
    "2. Estimate quantities and appropriate units.\n"
    "3. Classify into food categories.\n"
    "4. Estimate days until expiry based on item type.\n"
    "5. Call the register_pantry_items function with ALL items.\n"
    "Be thorough — missing items means wasted food."
)


# ---------------------------------------------------------------------------
# Scanner class
# ---------------------------------------------------------------------------

class PantryScanner:
    """Scans pantry images via Gemma 4 multimodal + function calling."""

    def __init__(
        self,
        gateway_url: str = "http://localhost:8080",
        tier: str = "primary",
        timeout: float = 60.0,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.tier = tier
        self.timeout = timeout
        self._inventory = PantryInventory()

    @property
    def inventory(self) -> PantryInventory:
        return self._inventory

    def _encode_image(self, image_path: str | Path) -> str:
        """Read and base64-encode an image file."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        data = path.read_bytes()
        return base64.b64encode(data).decode("ascii")

    async def scan_image(self, image_path: str | Path) -> list[PantryItem]:
        """
        Scan a single pantry image and return identified items.

        Uses Gemma 4 multimodal vision + function calling to identify items
        and extract structured inventory data.
        """
        b64 = self._encode_image(image_path)
        suffix = Path(image_path).suffix.lower()
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
        }.get(suffix, "image/jpeg")

        payload = {
            "model": "gemma-4",
            "messages": [
                {"role": "system", "content": SCAN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Scan this pantry image. Identify all food items, "
                                "estimate quantities and expiry, then call "
                                "register_pantry_items with the full inventory."
                            ),
                        },
                    ],
                },
            ],
            "tools": PANTRY_SCAN_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "register_pantry_items"}},
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        if self.tier != "auto":
            payload["_tier"] = self.tier

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.gateway_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()

        return self._parse_response(resp.json())

    def scan_image_from_base64(self, b64_data: str) -> dict[str, Any]:
        """
        Synchronous helper: build the request payload for a base64 image.

        Returns the chat completions payload (caller sends it).
        """
        return {
            "model": "gemma-4",
            "messages": [
                {"role": "system", "content": SCAN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Scan this pantry image and call "
                                "register_pantry_items with all items."
                            ),
                        },
                    ],
                },
            ],
            "tools": PANTRY_SCAN_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "register_pantry_items"}},
            "temperature": 0.1,
            "max_tokens": 2048,
        }

    def _parse_response(self, response: dict[str, Any]) -> list[PantryItem]:
        """Extract PantryItems from a Gemma 4 function-calling response."""
        items: list[PantryItem] = []

        choices = response.get("choices", [])
        if not choices:
            logger.warning("No choices in scan response")
            return items

        message = choices[0].get("message", {})

        # Try tool_calls first (OpenAI function-calling format)
        tool_calls = message.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            if fn.get("name") == "register_pantry_items":
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)
                items.extend(self._items_from_args(args))

        # Fallback: parse JSON from content if no tool_calls
        if not items:
            content = message.get("content", "")
            items = self._extract_items_from_text(content)

        self._inventory.add_items(items)
        return items

    def _items_from_args(self, args: dict[str, Any]) -> list[PantryItem]:
        """Convert function-call arguments to PantryItem list."""
        result: list[PantryItem] = []
        for raw in args.get("items", []):
            cat = raw.get("category", "other")
            if cat not in FOOD_CATEGORIES:
                cat = "other"
            unit = raw.get("unit", "each")
            if unit not in UNIT_CHOICES:
                unit = "each"
            result.append(PantryItem(
                name=raw.get("name", "unknown"),
                category=cat,
                quantity=float(raw.get("quantity", 1)),
                unit=unit,
                estimated_expiry_days=raw.get("estimated_expiry_days"),
                confidence=float(raw.get("confidence", 0.8)),
            ))
        return result

    def _extract_items_from_text(self, text: str) -> list[PantryItem]:
        """Fallback: extract items from free-text or embedded JSON."""
        items: list[PantryItem] = []
        # Try to find JSON in the text
        json_match = re.search(r'\{[\s\S]*"items"\s*:\s*\[[\s\S]*\]\s*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._items_from_args(data)
            except json.JSONDecodeError:
                pass

        # Last resort: line-by-line extraction (only lines starting with list markers)
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped and stripped[0] in "-•*":
                name = stripped.lstrip("-•* ").strip()
                if name and len(name) > 1:
                    items.append(PantryItem(name=name[:80], confidence=0.3))
        return items

    def clear_inventory(self) -> None:
        """Reset the inventory."""
        self._inventory = PantryInventory()
