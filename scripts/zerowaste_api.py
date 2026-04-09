"""
GemmaZeroWaste — API Module

FastAPI application combining pantry scanning, bulk-buy optimization,
recipe generation, and impact tracking into a single edge-native API.

Integrates with the existing Gemma 4 gateway for multimodal inference.
All processing runs locally — no data leaves the device.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from bulk_optimizer import (
    BulkOptimizer,
    BULK_CATALOG,
    DEFAULT_BUDGET_WEEKLY,
    DEFAULT_HOUSEHOLD_SIZE,
    OptimizationResult,
)
from impact_tracker import (
    CHICAGO_FOOD_DESERT_STATS,
    ImpactTracker,
    ImpactMetrics,
)
from pantry_scanner import PantryInventory, PantryItem, PantryScanner
from recipe_engine import MealPlan, RecipeEngine

logger = logging.getLogger("zerowaste.api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GATEWAY_URL = os.getenv("GEMMA_GATEWAY_URL", "http://localhost:8080")
DEFAULT_TIER = os.getenv("GEMMA_TIER", "primary")
WEB_DIR = Path(__file__).resolve().parent.parent / "cloud" / "web"


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    """Shared state across requests."""

    def __init__(self) -> None:
        self.scanner = PantryScanner(gateway_url=GATEWAY_URL, tier=DEFAULT_TIER)
        self.optimizer = BulkOptimizer()
        self.recipe_engine = RecipeEngine(
            gateway_url=GATEWAY_URL, tier=DEFAULT_TIER
        )
        self.impact_tracker = ImpactTracker()
        self.last_optimization: OptimizationResult | None = None
        self.last_meal_plan: MealPlan | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("GemmaZeroWaste API starting — gateway=%s tier=%s",
                GATEWAY_URL, DEFAULT_TIER)
    yield
    logger.info("GemmaZeroWaste API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GemmaZeroWaste",
    description=(
        "On-device multimodal nutrition guardian for Chicago food deserts. "
        "Powered by Gemma 4 edge inference — all processing happens locally."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints: Pantry Scanning
# ---------------------------------------------------------------------------

@app.post("/api/scan")
async def scan_pantry(
    file: UploadFile = File(...),
    tier: str = Form(DEFAULT_TIER),
):
    """
    Scan a pantry image and identify food items.

    Uses Gemma 4 multimodal vision + function calling to extract
    structured inventory from a photo.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are accepted")

    # Stream upload to temp file
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(file.filename or "img.jpg").suffix
    )
    try:
        data = await file.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(413, "Image must be under 20 MB")
        tmp.write(data)
        tmp.close()

        state.scanner.tier = tier
        items = await state.scanner.scan_image(tmp.name)

        return JSONResponse({
            "status": "ok",
            "items": [i.to_dict() for i in items],
            "inventory": state.scanner.inventory.to_dict(),
        })
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Scan failed")
        raise HTTPException(500, f"Scan failed: {exc}") from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)


@app.get("/api/inventory")
async def get_inventory():
    """Return the current pantry inventory."""
    return JSONResponse(state.scanner.inventory.to_dict())


@app.delete("/api/inventory")
async def clear_inventory():
    """Reset the pantry inventory."""
    state.scanner.clear_inventory()
    return JSONResponse({"status": "cleared"})


@app.post("/api/inventory/manual")
async def add_manual_item(
    name: str = Form(...),
    category: str = Form("other"),
    quantity: float = Form(1.0),
    unit: str = Form("each"),
    expiry_days: int | None = Form(None),
):
    """Manually add an item to the pantry inventory."""
    item = PantryItem(
        name=name,
        category=category,
        quantity=quantity,
        unit=unit,
        estimated_expiry_days=expiry_days,
        confidence=1.0,
    )
    state.scanner.inventory.add_items([item])
    return JSONResponse({
        "status": "added",
        "item": item.to_dict(),
        "inventory": state.scanner.inventory.to_dict(),
    })


# ---------------------------------------------------------------------------
# Endpoints: Bulk-Buy Optimization
# ---------------------------------------------------------------------------

@app.post("/api/optimize")
async def optimize_bulk_buy(
    budget: float = Form(DEFAULT_BUDGET_WEEKLY),
    household_size: int = Form(DEFAULT_HOUSEHOLD_SIZE),
    planning_days: int = Form(7),
    dietary_restrictions: str = Form(""),
):
    """
    Generate an optimized bulk-buy shopping list.

    Uses linear programming to minimize cost while meeting nutrition
    targets and reducing waste. Considers existing pantry inventory.
    """
    restrictions = [
        r.strip() for r in dietary_restrictions.split(",") if r.strip()
    ]

    # Build existing inventory from scanner
    existing: dict[str, int] = {}
    for item in state.scanner.inventory.items:
        # Map pantry items to catalog items (simple name matching)
        for cat_id, cat_data in BULK_CATALOG.items():
            if item.name.lower() in cat_data["name"].lower():
                existing[cat_id] = int(item.quantity)

    optimizer = BulkOptimizer(
        budget=budget,
        household_size=household_size,
        planning_days=planning_days,
    )

    result = optimizer.optimize(
        existing_inventory=existing,
        dietary_restrictions=restrictions,
    )
    state.last_optimization = result

    return JSONResponse(result.to_dict())


@app.get("/api/catalog")
async def get_catalog():
    """Return the bulk item catalog with prices and nutrition info."""
    return JSONResponse({
        item_id: {
            "name": data["name"],
            "cost": data["cost"],
            "unit": data["unit"],
            "servings": data["servings"],
            "shelf_life_days": data["shelf_life_days"],
            "category": data["category"],
            "cost_per_serving": round(data["cost"] / data["servings"], 3),
        }
        for item_id, data in BULK_CATALOG.items()
    })


# ---------------------------------------------------------------------------
# Endpoints: Recipe Generation
# ---------------------------------------------------------------------------

@app.post("/api/recipes")
async def generate_recipes(
    household_size: int = Form(3),
    days: int = Form(7),
    dietary_restrictions: str = Form(""),
    tier: str = Form(DEFAULT_TIER),
):
    """
    Generate a zero-waste meal plan from current inventory + bulk items.

    Uses Gemma 4 function calling to create recipes that prioritize
    expiring items and minimize food waste.
    """
    pantry_items = [i.to_dict() for i in state.scanner.inventory.items]
    bulk_items = (
        [i.to_dict() for i in state.last_optimization.items]
        if state.last_optimization
        else []
    )
    restrictions = [
        r.strip() for r in dietary_restrictions.split(",") if r.strip()
    ]

    state.recipe_engine.tier = tier
    plan = await state.recipe_engine.generate_meal_plan(
        pantry_items=pantry_items,
        bulk_items=bulk_items,
        household_size=household_size,
        days=days,
        dietary_restrictions=restrictions or None,
    )
    state.last_meal_plan = plan

    return JSONResponse(plan.to_dict())


# ---------------------------------------------------------------------------
# Endpoints: Impact Tracking
# ---------------------------------------------------------------------------

@app.post("/api/impact/record")
async def record_impact(
    servings_consumed: int = Form(...),
    items_wasted: int = Form(0),
):
    """Record consumption data for impact tracking."""
    inventory = state.scanner.inventory
    optimization = state.last_optimization

    session = state.impact_tracker.record_session(
        household_size=(
            optimization.household_size if optimization else DEFAULT_HOUSEHOLD_SIZE
        ),
        total_spent=optimization.total_cost if optimization else 0,
        servings_planned=(
            sum(i.total_servings for i in optimization.items)
            if optimization else 0
        ),
        servings_consumed=servings_consumed,
        pantry_items_used=len(inventory.items),
        pantry_items_wasted=items_wasted,
        recipes_generated=(
            len(state.last_meal_plan.recipes) if state.last_meal_plan else 0
        ),
    )

    return JSONResponse(session.to_dict())


@app.get("/api/impact")
async def get_impact():
    """Return aggregated impact metrics."""
    nutrition_coverage = (
        state.last_optimization.nutrition_coverage
        if state.last_optimization
        else {}
    )

    metrics = state.impact_tracker.compute_metrics(
        nutrition_coverage=nutrition_coverage
    )

    return JSONResponse(metrics.to_dict())


@app.get("/api/impact/summary")
async def get_impact_summary():
    """Return a human-readable impact summary."""
    summary = state.impact_tracker.generate_impact_summary()
    return JSONResponse({"summary": summary})


@app.get("/api/impact/chicago")
async def get_chicago_context():
    """Return Chicago food desert contextual data."""
    return JSONResponse({
        "food_desert_stats": CHICAGO_FOOD_DESERT_STATS,
        "description": (
            "Chicago has 37 neighborhoods classified as food deserts, "
            "where 633,631 residents (23% of the city) lack reliable access "
            "to affordable, nutritious food within reasonable distance. "
            "GemmaZeroWaste targets these communities with on-device AI "
            "that works offline, requires no data plan, and keeps all "
            "personal data local."
        ),
    })


# ---------------------------------------------------------------------------
# Endpoints: System
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Health check."""
    return JSONResponse({
        "status": "ok",
        "service": "GemmaZeroWaste",
        "version": "1.0.0",
        "gateway": GATEWAY_URL,
        "tier": DEFAULT_TIER,
        "inventory_items": len(state.scanner.inventory.items),
        "sessions_recorded": len(state.impact_tracker.sessions),
    })


@app.get("/zerowaste")
async def serve_zerowaste_ui():
    """Serve the ZeroWaste PWA."""
    html_path = WEB_DIR / "zerowaste.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    raise HTTPException(404, "ZeroWaste UI not found")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "zerowaste_api:app",
        host="0.0.0.0",
        port=8090,
        reload=True,
    )
