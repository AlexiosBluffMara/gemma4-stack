"""
GemmaZeroWaste — Bulk-Buy Optimizer Module

Linear programming solver for cost/nutrition/waste-minimized bulk purchasing.

Formulation:  min  cᵀx
              s.t. Ax ≥ b     (nutrition minimums)
                   x  ≤ u     (storage / waste caps)
                   x  ≥ 0

Uses PuLP (Apache 2.0) for LP solving — runs fully offline, no cloud needed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("zerowaste.optimizer")

# ---------------------------------------------------------------------------
# Nutrition & cost data (Chicago bulk-store representative prices, April 2026)
# ---------------------------------------------------------------------------

# USDA-aligned daily recommended intake per adult (simplified)
DAILY_NUTRITION_TARGETS = {
    "calories":     2000,   # kcal
    "protein":      50,     # g
    "fiber":        28,     # g
    "vitamin_c":    90,     # mg
    "calcium":      1000,   # mg
    "iron":         18,     # mg
}

# Bulk items available at Chicago-area stores (Costco, ALDI, Pete's Fresh Market)
# Each entry: (cost_per_unit, unit, shelf_life_days, nutrition_per_unit)
BULK_CATALOG: dict[str, dict[str, Any]] = {
    "brown_rice_5lb": {
        "name": "Brown Rice (5 lb bag)",
        "cost": 6.99,
        "unit": "bag",
        "shelf_life_days": 180,
        "servings": 50,
        "nutrition_per_serving": {
            "calories": 215, "protein": 5, "fiber": 3.5,
            "vitamin_c": 0, "calcium": 20, "iron": 1.0,
        },
        "category": "grain",
    },
    "dried_black_beans_2lb": {
        "name": "Dried Black Beans (2 lb)",
        "cost": 3.49,
        "unit": "bag",
        "shelf_life_days": 365,
        "servings": 24,
        "nutrition_per_serving": {
            "calories": 227, "protein": 15, "fiber": 15,
            "vitamin_c": 0, "calcium": 46, "iron": 3.6,
        },
        "category": "legume",
    },
    "canned_tomatoes_6pk": {
        "name": "Canned Diced Tomatoes (6-pack)",
        "cost": 7.99,
        "unit": "pack",
        "shelf_life_days": 730,
        "servings": 21,
        "nutrition_per_serving": {
            "calories": 25, "protein": 1, "fiber": 1,
            "vitamin_c": 12, "calcium": 35, "iron": 0.5,
        },
        "category": "canned",
    },
    "frozen_broccoli_3lb": {
        "name": "Frozen Broccoli (3 lb bag)",
        "cost": 5.49,
        "unit": "bag",
        "shelf_life_days": 240,
        "servings": 12,
        "nutrition_per_serving": {
            "calories": 55, "protein": 4, "fiber": 5,
            "vitamin_c": 100, "calcium": 60, "iron": 1.0,
        },
        "category": "vegetable",
    },
    "frozen_chicken_5lb": {
        "name": "Frozen Chicken Breast (5 lb)",
        "cost": 14.99,
        "unit": "bag",
        "shelf_life_days": 270,
        "servings": 20,
        "nutrition_per_serving": {
            "calories": 165, "protein": 31, "fiber": 0,
            "vitamin_c": 0, "calcium": 15, "iron": 1.0,
        },
        "category": "protein",
    },
    "eggs_60ct": {
        "name": "Eggs (60 count)",
        "cost": 12.99,
        "unit": "box",
        "shelf_life_days": 35,
        "servings": 60,
        "nutrition_per_serving": {
            "calories": 78, "protein": 6, "fiber": 0,
            "vitamin_c": 0, "calcium": 28, "iron": 0.9,
        },
        "category": "protein",
    },
    "oats_5lb": {
        "name": "Rolled Oats (5 lb)",
        "cost": 5.99,
        "unit": "bag",
        "shelf_life_days": 365,
        "servings": 60,
        "nutrition_per_serving": {
            "calories": 150, "protein": 5, "fiber": 4,
            "vitamin_c": 0, "calcium": 20, "iron": 2.0,
        },
        "category": "grain",
    },
    "peanut_butter_40oz": {
        "name": "Peanut Butter (40 oz jar)",
        "cost": 7.49,
        "unit": "jar",
        "shelf_life_days": 180,
        "servings": 40,
        "nutrition_per_serving": {
            "calories": 190, "protein": 7, "fiber": 2,
            "vitamin_c": 0, "calcium": 15, "iron": 0.6,
        },
        "category": "protein",
    },
    "bananas_3lb": {
        "name": "Bananas (3 lb bunch)",
        "cost": 1.49,
        "unit": "bunch",
        "shelf_life_days": 7,
        "servings": 6,
        "nutrition_per_serving": {
            "calories": 105, "protein": 1, "fiber": 3,
            "vitamin_c": 10, "calcium": 6, "iron": 0.3,
        },
        "category": "fruit",
    },
    "whole_milk_gal": {
        "name": "Whole Milk (1 gallon)",
        "cost": 4.29,
        "unit": "gallon",
        "shelf_life_days": 14,
        "servings": 16,
        "nutrition_per_serving": {
            "calories": 150, "protein": 8, "fiber": 0,
            "vitamin_c": 2, "calcium": 300, "iron": 0.1,
        },
        "category": "dairy",
    },
    "canned_tuna_12pk": {
        "name": "Canned Tuna (12-pack)",
        "cost": 11.99,
        "unit": "pack",
        "shelf_life_days": 1095,
        "servings": 12,
        "nutrition_per_serving": {
            "calories": 100, "protein": 22, "fiber": 0,
            "vitamin_c": 0, "calcium": 10, "iron": 1.5,
        },
        "category": "protein",
    },
    "frozen_mixed_veg_4lb": {
        "name": "Frozen Mixed Vegetables (4 lb)",
        "cost": 6.29,
        "unit": "bag",
        "shelf_life_days": 240,
        "servings": 16,
        "nutrition_per_serving": {
            "calories": 65, "protein": 3, "fiber": 4,
            "vitamin_c": 8, "calcium": 25, "iron": 0.8,
        },
        "category": "vegetable",
    },
    "tortillas_30ct": {
        "name": "Flour Tortillas (30 count)",
        "cost": 4.99,
        "unit": "pack",
        "shelf_life_days": 21,
        "servings": 30,
        "nutrition_per_serving": {
            "calories": 140, "protein": 4, "fiber": 1,
            "vitamin_c": 0, "calcium": 40, "iron": 1.5,
        },
        "category": "grain",
    },
    "sweet_potatoes_5lb": {
        "name": "Sweet Potatoes (5 lb bag)",
        "cost": 4.49,
        "unit": "bag",
        "shelf_life_days": 30,
        "servings": 10,
        "nutrition_per_serving": {
            "calories": 112, "protein": 2, "fiber": 4,
            "vitamin_c": 30, "calcium": 40, "iron": 0.7,
        },
        "category": "vegetable",
    },
}

# Chicago food desert context:  average household size = 3.2, SNAP benefit ~$234/mo
DEFAULT_HOUSEHOLD_SIZE = 3
DEFAULT_BUDGET_WEEKLY = 58.50  # ~$234/month / 4 weeks
DEFAULT_PLANNING_DAYS = 7


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class BulkBuyItem:
    """A recommended purchase."""
    item_id: str
    name: str
    quantity: int
    cost: float
    total_servings: int
    shelf_life_days: int
    category: str
    waste_risk: str  # "low", "medium", "high"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "name": self.name,
            "quantity": self.quantity,
            "cost": round(self.cost, 2),
            "total_servings": self.total_servings,
            "shelf_life_days": self.shelf_life_days,
            "category": self.category,
            "waste_risk": self.waste_risk,
        }


@dataclass
class OptimizationResult:
    """Complete bulk-buy plan."""
    items: list[BulkBuyItem] = field(default_factory=list)
    total_cost: float = 0.0
    budget: float = 0.0
    household_size: int = 3
    planning_days: int = 7
    nutrition_coverage: dict[str, float] = field(default_factory=dict)
    estimated_waste_pct: float = 0.0
    solver_status: str = "unsolved"

    @property
    def budget_remaining(self) -> float:
        return max(0, self.budget - self.total_cost)

    @property
    def cost_per_person_per_day(self) -> float:
        if self.household_size > 0 and self.planning_days > 0:
            return self.total_cost / (self.household_size * self.planning_days)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "total_cost": round(self.total_cost, 2),
            "budget": round(self.budget, 2),
            "budget_remaining": round(self.budget_remaining, 2),
            "cost_per_person_per_day": round(self.cost_per_person_per_day, 2),
            "household_size": self.household_size,
            "planning_days": self.planning_days,
            "nutrition_coverage": {
                k: round(v, 1) for k, v in self.nutrition_coverage.items()
            },
            "estimated_waste_pct": round(self.estimated_waste_pct, 1),
            "solver_status": self.solver_status,
        }


# ---------------------------------------------------------------------------
# LP Optimizer
# ---------------------------------------------------------------------------

class BulkOptimizer:
    """
    Linear programming optimizer for bulk grocery purchases.

    Minimizes cost while meeting nutrition targets for a household
    over a planning period, subject to budget and waste constraints.
    """

    def __init__(
        self,
        catalog: dict[str, dict[str, Any]] | None = None,
        budget: float = DEFAULT_BUDGET_WEEKLY,
        household_size: int = DEFAULT_HOUSEHOLD_SIZE,
        planning_days: int = DEFAULT_PLANNING_DAYS,
    ):
        self.catalog = catalog if catalog is not None else BULK_CATALOG
        self.budget = budget
        self.household_size = household_size
        self.planning_days = planning_days

    def optimize(
        self,
        existing_inventory: dict[str, int] | None = None,
        dietary_restrictions: list[str] | None = None,
    ) -> OptimizationResult:
        """
        Solve the bulk-buy LP.

        Args:
            existing_inventory: item_id → servings already in pantry
            dietary_restrictions: categories to exclude (e.g. ["dairy"])

        Returns:
            OptimizationResult with recommended purchases.
        """
        try:
            from pulp import (
                LpMinimize, LpProblem, LpVariable, lpSum, LpStatus,
                PULP_CBC_CMD,
            )
        except ImportError:
            logger.warning("PuLP not installed; falling back to greedy optimizer")
            return self._greedy_fallback(existing_inventory, dietary_restrictions)

        existing = existing_inventory or {}
        excluded_cats = set(dietary_restrictions or [])

        # Filter catalog
        available = {
            k: v for k, v in self.catalog.items()
            if v["category"] not in excluded_cats
        }

        if not available:
            return OptimizationResult(solver_status="infeasible_no_items")

        # Total servings needed
        total_servings_needed = self.household_size * self.planning_days * 3  # 3 meals/day

        # Decision variables: integer quantity of each bulk item to buy
        prob = LpProblem("GemmaZeroWaste_BulkBuy", LpMinimize)
        x = {}
        for item_id in available:
            x[item_id] = LpVariable(f"buy_{item_id}", lowBound=0, upBound=10, cat="Integer")

        # Objective: minimize total cost
        prob += lpSum(
            available[i]["cost"] * x[i] for i in available
        ), "TotalCost"

        # Budget constraint
        prob += lpSum(
            available[i]["cost"] * x[i] for i in available
        ) <= self.budget, "BudgetCap"

        # Nutrition constraints: meet daily targets × household × days
        for nutrient, daily_target in DAILY_NUTRITION_TARGETS.items():
            total_target = daily_target * self.household_size * self.planning_days
            # Credit from existing inventory
            existing_credit = sum(
                existing.get(i, 0)
                * available[i]["nutrition_per_serving"].get(nutrient, 0)
                for i in available if i in existing
            )
            prob += lpSum(
                available[i]["servings"]
                * available[i]["nutrition_per_serving"].get(nutrient, 0)
                * x[i]
                for i in available
            ) >= max(0, total_target * 0.7 - existing_credit), f"Min_{nutrient}"

        # Variety: at least 3 categories
        categories = set(v["category"] for v in available.values())
        for cat in categories:
            cat_items = [i for i in available if available[i]["category"] == cat]
            if cat_items:
                prob += lpSum(x[i] for i in cat_items) >= 0, f"Cat_{cat}_min"

        # Solve
        solver = PULP_CBC_CMD(msg=0, timeLimit=10)
        prob.solve(solver)

        status = LpStatus[prob.status]
        if status != "Optimal":
            logger.warning("LP solver status: %s — trying greedy fallback", status)
            return self._greedy_fallback(existing_inventory, dietary_restrictions)

        # Build result
        result = OptimizationResult(
            budget=self.budget,
            household_size=self.household_size,
            planning_days=self.planning_days,
            solver_status="optimal",
        )

        for item_id, var in x.items():
            qty = int(var.varValue or 0)
            if qty > 0:
                item_data = available[item_id]
                total_srvs = qty * item_data["servings"]
                shelf = item_data["shelf_life_days"]
                waste_risk = (
                    "high" if shelf < self.planning_days
                    else "medium" if shelf < self.planning_days * 4
                    else "low"
                )
                result.items.append(BulkBuyItem(
                    item_id=item_id,
                    name=item_data["name"],
                    quantity=qty,
                    cost=round(qty * item_data["cost"], 2),
                    total_servings=total_srvs,
                    shelf_life_days=shelf,
                    category=item_data["category"],
                    waste_risk=waste_risk,
                ))
                result.total_cost += qty * item_data["cost"]

        # Compute nutrition coverage
        result.nutrition_coverage = self._compute_nutrition_coverage(
            result.items, existing
        )

        # Estimate waste
        result.estimated_waste_pct = self._estimate_waste(result.items)

        return result

    def _compute_nutrition_coverage(
        self,
        items: list[BulkBuyItem],
        existing: dict[str, int],
    ) -> dict[str, float]:
        """Calculate % of nutrition targets met by the plan."""
        coverage: dict[str, float] = {}
        for nutrient, daily_target in DAILY_NUTRITION_TARGETS.items():
            total_target = daily_target * self.household_size * self.planning_days
            if total_target == 0:
                coverage[nutrient] = 100.0
                continue

            supplied = 0.0
            for item in items:
                cat_data = self.catalog.get(item.item_id, {})
                per_serving = cat_data.get("nutrition_per_serving", {}).get(nutrient, 0)
                supplied += item.total_servings * per_serving

            # Add existing inventory
            for item_id, servings in existing.items():
                cat_data = self.catalog.get(item_id, {})
                per_serving = cat_data.get("nutrition_per_serving", {}).get(nutrient, 0)
                supplied += servings * per_serving

            coverage[nutrient] = min(100.0, (supplied / total_target) * 100)

        return coverage

    def _estimate_waste(self, items: list[BulkBuyItem]) -> float:
        """Estimate % of purchased food that will go to waste."""
        total_servings = sum(i.total_servings for i in items)
        if total_servings == 0:
            return 0.0

        needed = self.household_size * self.planning_days * 3
        waste_servings = 0.0
        for item in items:
            if item.shelf_life_days < self.planning_days:
                # Perishable items: assume 20% waste
                waste_servings += item.total_servings * 0.20
            elif item.total_servings > needed * 0.5:
                # Over-purchased non-perishable: 5% waste
                waste_servings += item.total_servings * 0.05

        return (waste_servings / total_servings) * 100

    def _greedy_fallback(
        self,
        existing_inventory: dict[str, int] | None,
        dietary_restrictions: list[str] | None,
    ) -> OptimizationResult:
        """
        Simple greedy approach when PuLP is unavailable.

        Selects items by cost-per-serving, respecting budget.
        """
        excluded_cats = set(dietary_restrictions or [])
        available = {
            k: v for k, v in self.catalog.items()
            if v["category"] not in excluded_cats
        }

        # Rank by cost per serving (ascending)
        ranked = sorted(
            available.items(),
            key=lambda kv: kv[1]["cost"] / max(1, kv[1]["servings"]),
        )

        result = OptimizationResult(
            budget=self.budget,
            household_size=self.household_size,
            planning_days=self.planning_days,
            solver_status="greedy_fallback",
        )

        remaining_budget = self.budget
        needed_servings = self.household_size * self.planning_days * 3
        total_servings = 0
        categories_seen: set[str] = set()

        for item_id, item_data in ranked:
            if remaining_budget <= 0 or total_servings >= needed_servings * 1.2:
                break

            qty = max(1, min(
                math.floor(remaining_budget / item_data["cost"]),
                3,  # max 3 of any single bulk item
            ))

            if qty * item_data["cost"] > remaining_budget:
                continue

            shelf = item_data["shelf_life_days"]
            waste_risk = (
                "high" if shelf < self.planning_days
                else "medium" if shelf < self.planning_days * 4
                else "low"
            )

            srvs = qty * item_data["servings"]
            result.items.append(BulkBuyItem(
                item_id=item_id,
                name=item_data["name"],
                quantity=qty,
                cost=round(qty * item_data["cost"], 2),
                total_servings=srvs,
                shelf_life_days=shelf,
                category=item_data["category"],
                waste_risk=waste_risk,
            ))
            remaining_budget -= qty * item_data["cost"]
            total_servings += srvs
            result.total_cost += qty * item_data["cost"]
            categories_seen.add(item_data["category"])

        result.nutrition_coverage = self._compute_nutrition_coverage(
            result.items, existing_inventory or {}
        )
        result.estimated_waste_pct = self._estimate_waste(result.items)
        return result
