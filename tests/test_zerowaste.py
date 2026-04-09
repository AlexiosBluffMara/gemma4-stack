"""
GemmaZeroWaste — Unit Tests

Tests for the pantry scanner, bulk optimizer, recipe engine, and impact tracker.
All tests run offline without a Gemma 4 gateway.

Run:  cd tests && python3 -m pytest test_zerowaste.py -v
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add scripts/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from bulk_optimizer import (
    BULK_CATALOG,
    BulkBuyItem,
    BulkOptimizer,
    DAILY_NUTRITION_TARGETS,
    DEFAULT_BUDGET_WEEKLY,
    DEFAULT_HOUSEHOLD_SIZE,
    DEFAULT_PLANNING_DAYS,
    OptimizationResult,
)
from impact_tracker import (
    CHICAGO_FOOD_DESERT_STATS,
    CO2E_PER_KG_FOOD_WASTE,
    US_AVG_FOOD_WASTE_PCT,
    ImpactMetrics,
    ImpactSession,
    ImpactTracker,
)
from pantry_scanner import (
    FOOD_CATEGORIES,
    PANTRY_SCAN_TOOLS,
    PantryInventory,
    PantryItem,
    PantryScanner,
    SCAN_SYSTEM_PROMPT,
    UNIT_CHOICES,
)
from recipe_engine import (
    Ingredient,
    MealPlan,
    Recipe,
    RecipeEngine,
)


# ===========================================================================
# Pantry Scanner Tests
# ===========================================================================

class TestPantryItem:
    """Test PantryItem data model."""

    def test_basic_creation(self):
        item = PantryItem(name="Brown Rice", category="grain", quantity=2.0, unit="bag")
        assert item.name == "Brown Rice"
        assert item.category == "grain"
        assert item.quantity == 2.0
        assert item.unit == "bag"

    def test_expiry_date_calculation(self):
        item = PantryItem(name="Milk", estimated_expiry_days=7)
        expected = date.today() + timedelta(days=7)
        assert item.expiry_date == expected

    def test_expiry_date_none(self):
        item = PantryItem(name="Canned Beans")
        assert item.expiry_date is None

    def test_to_dict(self):
        item = PantryItem(
            name="Eggs", category="protein",
            quantity=12, unit="each",
            estimated_expiry_days=14, confidence=0.95,
        )
        d = item.to_dict()
        assert d["name"] == "Eggs"
        assert d["category"] == "protein"
        assert d["quantity"] == 12
        assert d["confidence"] == 0.95
        assert d["expiry_date"] is not None

    def test_default_values(self):
        item = PantryItem(name="Unknown")
        assert item.category == "other"
        assert item.quantity == 1.0
        assert item.unit == "each"
        assert item.confidence == 0.0


class TestPantryInventory:
    """Test PantryInventory management."""

    def test_add_items(self):
        inv = PantryInventory()
        items = [
            PantryItem(name="Rice", quantity=2),
            PantryItem(name="Beans", quantity=3),
        ]
        inv.add_items(items)
        assert len(inv.items) == 2
        assert inv.scan_count == 1

    def test_merge_duplicates(self):
        inv = PantryInventory()
        inv.add_items([PantryItem(name="Rice", quantity=2)])
        inv.add_items([PantryItem(name="rice", quantity=1)])
        assert len(inv.items) == 1
        assert inv.items[0].quantity == 3
        assert inv.scan_count == 2

    def test_by_category(self):
        inv = PantryInventory()
        inv.add_items([
            PantryItem(name="Rice", category="grain"),
            PantryItem(name="Oats", category="grain"),
            PantryItem(name="Milk", category="dairy"),
        ])
        cats = inv.by_category
        assert len(cats["grain"]) == 2
        assert len(cats["dairy"]) == 1

    def test_expiring_soon(self):
        inv = PantryInventory()
        inv.add_items([
            PantryItem(name="Bananas", estimated_expiry_days=2),
            PantryItem(name="Rice", estimated_expiry_days=180),
            PantryItem(name="Milk", estimated_expiry_days=3),
        ])
        expiring = inv.expiring_soon
        assert len(expiring) == 2
        names = {i.name for i in expiring}
        assert "Bananas" in names
        assert "Milk" in names

    def test_to_dict(self):
        inv = PantryInventory()
        inv.add_items([PantryItem(name="Test")])
        d = inv.to_dict()
        assert d["total_items"] == 1
        assert d["scan_count"] == 1
        assert isinstance(d["items"], list)


class TestPantryScannerParsing:
    """Test the scanner's response parsing (no gateway needed)."""

    def test_parse_function_call_response(self):
        scanner = PantryScanner()
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "register_pantry_items",
                            "arguments": json.dumps({
                                "items": [
                                    {
                                        "name": "Brown Rice",
                                        "category": "grain",
                                        "quantity": 2,
                                        "unit": "bag",
                                        "estimated_expiry_days": 180,
                                        "confidence": 0.92,
                                    },
                                    {
                                        "name": "Canned Black Beans",
                                        "category": "canned",
                                        "quantity": 4,
                                        "unit": "can",
                                        "confidence": 0.88,
                                    },
                                ]
                            }),
                        }
                    }]
                }
            }]
        }
        items = scanner._parse_response(response)
        assert len(items) == 2
        assert items[0].name == "Brown Rice"
        assert items[0].category == "grain"
        assert items[1].name == "Canned Black Beans"
        assert items[1].quantity == 4

    def test_parse_empty_response(self):
        scanner = PantryScanner()
        items = scanner._parse_response({"choices": []})
        assert items == []

    def test_parse_text_fallback(self):
        scanner = PantryScanner()
        response = {
            "choices": [{
                "message": {
                    "content": "I see:\n- Brown Rice\n- Black Beans\n- Oatmeal"
                }
            }]
        }
        items = scanner._parse_response(response)
        assert len(items) == 3

    def test_parse_invalid_category_fallback(self):
        scanner = PantryScanner()
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "register_pantry_items",
                            "arguments": json.dumps({
                                "items": [{
                                    "name": "Mystery Food",
                                    "category": "not_a_real_category",
                                    "quantity": 1,
                                    "unit": "each",
                                }]
                            }),
                        }
                    }]
                }
            }]
        }
        items = scanner._parse_response(response)
        assert items[0].category == "other"

    def test_scan_tools_schema(self):
        """Verify the function-calling tool schema is well-formed."""
        assert len(PANTRY_SCAN_TOOLS) == 1
        tool = PANTRY_SCAN_TOOLS[0]
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] == "register_pantry_items"
        assert "items" in fn["parameters"]["properties"]

    def test_system_prompt_contains_key_instructions(self):
        assert "GemmaZeroWaste" in SCAN_SYSTEM_PROMPT
        assert "food desert" in SCAN_SYSTEM_PROMPT.lower() or "waste" in SCAN_SYSTEM_PROMPT.lower()

    def test_food_categories_valid(self):
        assert "grain" in FOOD_CATEGORIES
        assert "protein" in FOOD_CATEGORIES
        assert "vegetable" in FOOD_CATEGORIES
        assert len(FOOD_CATEGORIES) >= 10


# ===========================================================================
# Bulk Optimizer Tests
# ===========================================================================

class TestBulkCatalog:
    """Test the bulk item catalog data."""

    def test_catalog_has_items(self):
        assert len(BULK_CATALOG) >= 10

    def test_all_items_have_required_fields(self):
        for item_id, data in BULK_CATALOG.items():
            assert "name" in data, f"{item_id} missing name"
            assert "cost" in data, f"{item_id} missing cost"
            assert "servings" in data, f"{item_id} missing servings"
            assert "nutrition_per_serving" in data, f"{item_id} missing nutrition"
            assert "shelf_life_days" in data, f"{item_id} missing shelf_life"
            assert "category" in data, f"{item_id} missing category"
            assert data["cost"] > 0
            assert data["servings"] > 0

    def test_nutrition_has_required_nutrients(self):
        for item_id, data in BULK_CATALOG.items():
            nutr = data["nutrition_per_serving"]
            for nutrient in ["calories", "protein", "fiber"]:
                assert nutrient in nutr, f"{item_id} missing {nutrient}"

    def test_categories_are_valid(self):
        for item_id, data in BULK_CATALOG.items():
            assert data["category"] in FOOD_CATEGORIES, \
                f"{item_id} has invalid category: {data['category']}"

    def test_diverse_categories(self):
        categories = {data["category"] for data in BULK_CATALOG.values()}
        assert len(categories) >= 5, f"Only {len(categories)} categories in catalog"


def _pulp_available() -> bool:
    try:
        import pulp  # noqa: F401
        return True
    except ImportError:
        return False


class TestBulkOptimizer:
    """Test the LP optimizer."""

    def test_greedy_fallback_basic(self):
        """Test greedy optimizer (no PuLP dependency needed)."""
        optimizer = BulkOptimizer(budget=50.0, household_size=3, planning_days=7)
        result = optimizer._greedy_fallback(None, None)
        assert result.solver_status == "greedy_fallback"
        assert result.total_cost <= 50.0
        assert len(result.items) > 0

    def test_greedy_respects_budget(self):
        optimizer = BulkOptimizer(budget=10.0, household_size=2, planning_days=7)
        result = optimizer._greedy_fallback(None, None)
        assert result.total_cost <= 10.0

    def test_greedy_dietary_restrictions(self):
        optimizer = BulkOptimizer(budget=50.0)
        result = optimizer._greedy_fallback(None, ["dairy", "protein"])
        for item in result.items:
            assert item.category not in ("dairy", "protein")

    def test_optimization_result_to_dict(self):
        result = OptimizationResult(
            total_cost=45.50,
            budget=58.50,
            household_size=3,
            planning_days=7,
        )
        d = result.to_dict()
        assert d["total_cost"] == 45.50
        assert d["budget_remaining"] == 13.0
        assert "cost_per_person_per_day" in d

    def test_cost_per_person_per_day(self):
        result = OptimizationResult(
            total_cost=42.0,
            household_size=3,
            planning_days=7,
        )
        # 42 / (3 * 7) = 2.0
        assert result.cost_per_person_per_day == 2.0

    def test_waste_risk_classification(self):
        optimizer = BulkOptimizer(planning_days=7)
        result = optimizer._greedy_fallback(None, None)
        for item in result.items:
            assert item.waste_risk in ("low", "medium", "high")

    def test_nutrition_coverage_computed(self):
        optimizer = BulkOptimizer(budget=50.0)
        result = optimizer._greedy_fallback(None, None)
        assert len(result.nutrition_coverage) > 0
        for nutrient, pct in result.nutrition_coverage.items():
            assert 0 <= pct <= 100, f"{nutrient} coverage out of range: {pct}"

    def test_estimated_waste_reasonable(self):
        optimizer = BulkOptimizer(budget=50.0)
        result = optimizer._greedy_fallback(None, None)
        assert 0 <= result.estimated_waste_pct <= 50

    @pytest.mark.skipif(
        not _pulp_available(),
        reason="PuLP not installed",
    )
    def test_lp_solver_optimal(self):
        """Test full LP solver when PuLP is available."""
        optimizer = BulkOptimizer(budget=58.50, household_size=3, planning_days=7)
        result = optimizer.optimize()
        assert result.solver_status == "optimal"
        assert result.total_cost <= 58.50
        assert len(result.items) > 0

    @pytest.mark.skipif(
        not _pulp_available(),
        reason="PuLP not installed",
    )
    def test_lp_with_existing_inventory(self):
        optimizer = BulkOptimizer(budget=58.50)
        result = optimizer.optimize(
            existing_inventory={"oats_5lb": 30, "dried_black_beans_2lb": 10}
        )
        assert result.solver_status == "optimal"

    def test_empty_catalog(self):
        optimizer = BulkOptimizer(catalog={}, budget=50.0)
        result = optimizer.optimize(dietary_restrictions=["everything"])
        assert "infeasible" in result.solver_status or len(result.items) == 0

    def test_default_constants(self):
        assert DEFAULT_BUDGET_WEEKLY > 0
        assert DEFAULT_HOUSEHOLD_SIZE > 0
        assert DEFAULT_PLANNING_DAYS > 0


# ===========================================================================
# Recipe Engine Tests
# ===========================================================================

class TestRecipeModels:
    """Test recipe data models."""

    def test_ingredient_creation(self):
        ing = Ingredient(name="Rice", quantity=2, unit="cup")
        assert ing.available is True
        d = ing.to_dict()
        assert d["name"] == "Rice"

    def test_recipe_total_time(self):
        recipe = Recipe(name="Test", prep_time_min=10, cook_time_min=25)
        assert recipe.total_time_min == 35

    def test_recipe_to_dict(self):
        recipe = Recipe(
            name="Bean Stew",
            servings=6,
            ingredients=[Ingredient(name="Beans", quantity=2, unit="cup")],
            instructions=["Cook beans", "Add spices"],
        )
        d = recipe.to_dict()
        assert d["name"] == "Bean Stew"
        assert d["servings"] == 6
        assert len(d["ingredients"]) == 1
        assert len(d["instructions"]) == 2

    def test_meal_plan_to_dict(self):
        plan = MealPlan(
            recipes=[Recipe(name="Test")],
            days=7,
            household_size=3,
        )
        d = plan.to_dict()
        assert d["total_recipes"] == 1
        assert d["days"] == 7


class TestRecipeEngineParsing:
    """Test recipe engine response parsing."""

    def test_parse_function_call(self):
        engine = RecipeEngine()
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "function": {
                            "name": "create_meal_plan",
                            "arguments": json.dumps({
                                "recipes": [{
                                    "name": "Rice and Beans Bowl",
                                    "servings": 4,
                                    "prep_time_min": 10,
                                    "cook_time_min": 25,
                                    "difficulty": "easy",
                                    "ingredients": [
                                        {"name": "brown rice", "quantity": 2, "unit": "cup"},
                                        {"name": "black beans", "quantity": 1, "unit": "can"},
                                    ],
                                    "instructions": [
                                        "Cook rice according to package",
                                        "Heat beans with spices",
                                        "Combine and serve",
                                    ],
                                    "waste_reduction_note": "Uses expiring rice first",
                                    "cultural_note": "Latin American comfort food",
                                }]
                            }),
                        }
                    }]
                }
            }]
        }
        plan = engine._parse_response(
            response, household_size=3, days=7,
            pantry_items=[{"name": "brown rice"}, {"name": "black beans"}],
        )
        assert len(plan.recipes) == 1
        r = plan.recipes[0]
        assert r.name == "Rice and Beans Bowl"
        assert r.servings == 4
        assert len(r.ingredients) == 2
        assert len(r.instructions) == 3

    def test_parse_empty_response(self):
        engine = RecipeEngine()
        plan = engine._parse_response({"choices": []}, 3, 7, [])
        assert len(plan.recipes) == 0

    def test_build_payload(self):
        engine = RecipeEngine()
        payload = engine.build_meal_plan_payload(
            pantry_items=[{"name": "rice", "quantity": 2, "unit": "bag"}],
            household_size=4,
            days=5,
        )
        assert payload["model"] == "gemma-4"
        assert len(payload["tools"]) == 1
        assert "5-day" in payload["messages"][1]["content"]


# ===========================================================================
# Impact Tracker Tests
# ===========================================================================

class TestImpactTracker:
    """Test impact metrics calculation."""

    def test_record_session(self):
        tracker = ImpactTracker()
        session = tracker.record_session(
            household_size=3,
            total_spent=45.0,
            servings_planned=63,
            servings_consumed=57,
            pantry_items_used=10,
            pantry_items_wasted=1,
            recipes_generated=5,
        )
        assert session.household_size == 3
        assert session.total_spent == 45.0
        assert session.waste_diverted_kg > 0  # reduced waste vs average

    def test_compute_metrics_empty(self):
        tracker = ImpactTracker()
        metrics = tracker.compute_metrics()
        assert metrics.food_waste_pct == 0
        assert metrics.co2e_avoided_kg == 0

    def test_compute_metrics_with_data(self):
        tracker = ImpactTracker()
        tracker.record_session(
            household_size=3,
            total_spent=45.0,
            servings_planned=63,
            servings_consumed=57,
            pantry_items_used=10,
            pantry_items_wasted=1,
            recipes_generated=5,
        )
        metrics = tracker.compute_metrics(
            nutrition_coverage={"calories": 85.0, "protein": 90.0, "fiber": 70.0}
        )
        assert metrics.food_waste_pct > 0
        assert metrics.food_waste_pct < US_AVG_FOOD_WASTE_PCT
        assert metrics.co2e_avoided_kg > 0
        assert metrics.equivalent_car_miles > 0
        assert metrics.nutrition_score > 0
        assert metrics.total_spent == 45.0

    def test_waste_reduction_calculation(self):
        tracker = ImpactTracker()
        # Perfect consumption: 0% waste
        tracker.record_session(
            household_size=3,
            total_spent=50.0,
            servings_planned=63,
            servings_consumed=63,  # zero waste
            pantry_items_used=10,
            pantry_items_wasted=0,
            recipes_generated=5,
        )
        metrics = tracker.compute_metrics()
        assert metrics.food_waste_pct == 0
        assert metrics.waste_reduction_vs_avg == US_AVG_FOOD_WASTE_PCT

    def test_co2e_calculation(self):
        tracker = ImpactTracker()
        tracker.record_session(
            household_size=3,
            total_spent=50.0,
            servings_planned=100,
            servings_consumed=90,
            pantry_items_used=10,
            pantry_items_wasted=2,
            recipes_generated=5,
        )
        metrics = tracker.compute_metrics()
        # Waste diverted = expected_waste - actual_waste (in servings) * 0.25 kg
        expected_waste = 100 * (US_AVG_FOOD_WASTE_PCT / 100)
        actual_waste = 100 - 90
        diverted_servings = expected_waste - actual_waste
        diverted_kg = diverted_servings * 0.25
        expected_co2 = diverted_kg * CO2E_PER_KG_FOOD_WASTE
        assert abs(metrics.co2e_avoided_kg - expected_co2) < 0.01

    def test_impact_summary_readable(self):
        tracker = ImpactTracker()
        tracker.record_session(
            household_size=3,
            total_spent=45.0,
            servings_planned=63,
            servings_consumed=57,
            pantry_items_used=10,
            pantry_items_wasted=1,
            recipes_generated=5,
        )
        summary = tracker.generate_impact_summary()
        assert "GemmaZeroWaste" in summary
        assert "CO₂" in summary
        assert "Waste" in summary or "waste" in summary

    def test_chicago_stats_present(self):
        assert CHICAGO_FOOD_DESERT_STATS["population_in_food_deserts"] > 0
        assert CHICAGO_FOOD_DESERT_STATS["neighborhoods_affected"] > 0

    def test_metrics_to_dict(self):
        metrics = ImpactMetrics(
            food_waste_pct=10.0,
            co2e_avoided_kg=5.0,
            total_spent=45.0,
        )
        d = metrics.to_dict()
        assert "waste_reduction" in d
        assert "climate_impact" in d
        assert "nutrition" in d
        assert "cost" in d
        assert "equity" in d
        assert "community_scale" in d

    def test_session_to_dict(self):
        session = ImpactSession(
            household_size=3,
            total_spent=45.0,
            total_servings_planned=63,
            total_servings_consumed=57,
        )
        d = session.to_dict()
        assert d["household_size"] == 3
        assert d["total_spent"] == 45.0

    def test_multiple_sessions_aggregate(self):
        tracker = ImpactTracker()
        for _ in range(4):  # 4 weeks
            tracker.record_session(
                household_size=3,
                total_spent=50.0,
                servings_planned=63,
                servings_consumed=58,
                pantry_items_used=12,
                pantry_items_wasted=1,
                recipes_generated=5,
            )
        metrics = tracker.compute_metrics()
        assert metrics.total_spent == 200.0
        assert len(tracker.sessions) == 4


# ===========================================================================
# Integration Tests (data flow between modules)
# ===========================================================================

class TestModuleIntegration:
    """Test data flow between ZeroWaste modules."""

    def test_scanner_to_optimizer_flow(self):
        """Items from scanner feed into optimizer."""
        inv = PantryInventory()
        inv.add_items([
            PantryItem(name="Oats", category="grain", quantity=30, unit="each"),
            PantryItem(name="Black Beans", category="legume", quantity=10, unit="each"),
        ])

        optimizer = BulkOptimizer(budget=40.0, household_size=3)
        result = optimizer._greedy_fallback(
            existing_inventory={"oats_5lb": 30},
            dietary_restrictions=None,
        )
        assert result.total_cost <= 40.0
        assert len(result.items) > 0

    def test_optimizer_to_recipe_payload(self):
        """Optimizer results feed into recipe engine payload."""
        optimizer = BulkOptimizer(budget=50.0)
        opt_result = optimizer._greedy_fallback(None, None)

        engine = RecipeEngine()
        pantry_items = [{"name": "existing rice", "quantity": 2, "unit": "bag"}]
        bulk_items = [i.to_dict() for i in opt_result.items]

        payload = engine.build_meal_plan_payload(
            pantry_items=pantry_items,
            bulk_items=bulk_items,
            household_size=3,
            days=7,
        )
        assert "existing rice" in payload["messages"][1]["content"]
        assert any(
            i.name in payload["messages"][1]["content"]
            for i in opt_result.items
        )

    def test_full_pipeline_data_flow(self):
        """Test the complete data pipeline (scanner → optimizer → impact)."""
        # 1. Scanner produces inventory
        inv = PantryInventory()
        inv.add_items([
            PantryItem(name="Rice", category="grain", quantity=5, unit="bag"),
            PantryItem(name="Beans", category="legume", quantity=3, unit="can"),
            PantryItem(name="Broccoli", category="vegetable", quantity=2,
                       unit="bag", estimated_expiry_days=5),
        ])

        # 2. Optimizer generates shopping list
        optimizer = BulkOptimizer(budget=50.0, household_size=3)
        opt_result = optimizer._greedy_fallback(None, None)

        # 3. Impact tracker records results
        tracker = ImpactTracker()
        session = tracker.record_session(
            household_size=3,
            total_spent=opt_result.total_cost,
            servings_planned=sum(i.total_servings for i in opt_result.items),
            servings_consumed=int(
                sum(i.total_servings for i in opt_result.items) * 0.9
            ),
            pantry_items_used=len(inv.items),
            pantry_items_wasted=0,
            recipes_generated=5,
        )

        metrics = tracker.compute_metrics(
            nutrition_coverage=opt_result.nutrition_coverage,
        )

        # Verify full pipeline
        assert opt_result.total_cost <= 50.0
        assert session.waste_diverted_kg >= 0
        assert metrics.co2e_avoided_kg >= 0
        assert metrics.total_spent == opt_result.total_cost
