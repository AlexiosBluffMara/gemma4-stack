"""
GemmaZeroWaste — Recipe Engine Module

Generates zero-waste meal plans from pantry inventory using Gemma 4 function
calling.  Prioritizes items expiring soonest to minimize food waste.

Runs fully on-device — no recipe data sent to cloud services.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("zerowaste.recipes")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Ingredient:
    """Recipe ingredient with pantry cross-reference."""
    name: str
    quantity: float
    unit: str
    pantry_item_name: str | None = None  # matched pantry item
    available: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "unit": self.unit,
            "pantry_item_name": self.pantry_item_name,
            "available": self.available,
        }


@dataclass
class Recipe:
    """A generated recipe."""
    name: str
    servings: int = 4
    prep_time_min: int = 15
    cook_time_min: int = 30
    difficulty: str = "easy"  # easy, medium, hard
    ingredients: list[Ingredient] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    nutrition_per_serving: dict[str, float] = field(default_factory=dict)
    waste_reduction_note: str = ""
    cultural_note: str = ""

    @property
    def total_time_min(self) -> int:
        return self.prep_time_min + self.cook_time_min

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "servings": self.servings,
            "prep_time_min": self.prep_time_min,
            "cook_time_min": self.cook_time_min,
            "total_time_min": self.total_time_min,
            "difficulty": self.difficulty,
            "ingredients": [i.to_dict() for i in self.ingredients],
            "instructions": self.instructions,
            "nutrition_per_serving": self.nutrition_per_serving,
            "waste_reduction_note": self.waste_reduction_note,
            "cultural_note": self.cultural_note,
        }


@dataclass
class MealPlan:
    """Multi-day meal plan with waste tracking."""
    recipes: list[Recipe] = field(default_factory=list)
    days: int = 7
    household_size: int = 3
    estimated_waste_pct: float = 0.0
    pantry_utilization_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recipes": [r.to_dict() for r in self.recipes],
            "days": self.days,
            "household_size": self.household_size,
            "total_recipes": len(self.recipes),
            "estimated_waste_pct": round(self.estimated_waste_pct, 1),
            "pantry_utilization_pct": round(self.pantry_utilization_pct, 1),
        }


# ---------------------------------------------------------------------------
# Gemma 4 function-calling tool definitions
# ---------------------------------------------------------------------------

RECIPE_GEN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_meal_plan",
            "description": (
                "Generate a zero-waste meal plan using available pantry items. "
                "Prioritize items expiring soonest. Include culturally diverse, "
                "budget-friendly recipes suitable for multi-generational households."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "recipes": {
                        "type": "array",
                        "description": "List of recipes for the meal plan",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "Recipe name",
                                },
                                "servings": {
                                    "type": "integer",
                                    "description": "Number of servings",
                                },
                                "prep_time_min": {
                                    "type": "integer",
                                    "description": "Preparation time in minutes",
                                },
                                "cook_time_min": {
                                    "type": "integer",
                                    "description": "Cooking time in minutes",
                                },
                                "difficulty": {
                                    "type": "string",
                                    "enum": ["easy", "medium", "hard"],
                                },
                                "ingredients": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "quantity": {"type": "number"},
                                            "unit": {"type": "string"},
                                            "pantry_item_name": {
                                                "type": "string",
                                                "description": "Matching pantry item name (if available)",
                                            },
                                        },
                                        "required": ["name", "quantity", "unit"],
                                    },
                                },
                                "instructions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Step-by-step cooking instructions",
                                },
                                "nutrition_per_serving": {
                                    "type": "object",
                                    "properties": {
                                        "calories": {"type": "number"},
                                        "protein": {"type": "number"},
                                        "fiber": {"type": "number"},
                                    },
                                },
                                "waste_reduction_note": {
                                    "type": "string",
                                    "description": "How this recipe helps reduce food waste",
                                },
                                "cultural_note": {
                                    "type": "string",
                                    "description": "Cultural origin or inspiration for the dish",
                                },
                            },
                            "required": ["name", "servings", "ingredients", "instructions"],
                        },
                    },
                },
                "required": ["recipes"],
            },
        },
    }
]

RECIPE_SYSTEM_PROMPT = (
    "You are a zero-waste meal planning assistant for GemmaZeroWaste. "
    "Your goal is to help families in Chicago food deserts make the most "
    "of their pantry items with minimal waste.\n\n"
    "Guidelines:\n"
    "1. PRIORITIZE items expiring soonest — use them first.\n"
    "2. Create simple, budget-friendly recipes (under 30 min prep preferred).\n"
    "3. Include culturally diverse options (Mexican, African-American soul food, "
    "Asian, Mediterranean — reflecting Chicago's diverse communities).\n"
    "4. Each recipe should be family-sized (4-6 servings).\n"
    "5. Include a waste reduction note explaining how the recipe minimizes waste.\n"
    "6. Ensure nutritional balance across the plan.\n"
    "7. Use ONLY ingredients from the provided pantry + bulk items.\n"
    "8. Call the create_meal_plan function with your complete plan."
)


# ---------------------------------------------------------------------------
# Recipe Engine
# ---------------------------------------------------------------------------

class RecipeEngine:
    """Generates zero-waste meal plans via Gemma 4 function calling."""

    def __init__(
        self,
        gateway_url: str = "http://localhost:8080",
        tier: str = "primary",
        timeout: float = 90.0,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.tier = tier
        self.timeout = timeout

    async def generate_meal_plan(
        self,
        pantry_items: list[dict[str, Any]],
        bulk_items: list[dict[str, Any]] | None = None,
        household_size: int = 3,
        days: int = 7,
        dietary_restrictions: list[str] | None = None,
    ) -> MealPlan:
        """
        Generate a zero-waste meal plan from pantry + bulk items.

        Args:
            pantry_items: Current pantry inventory (from PantryScanner)
            bulk_items: Recommended bulk purchases (from BulkOptimizer)
            household_size: Number of people in household
            days: Planning period in days
            dietary_restrictions: e.g. ["dairy-free", "vegetarian"]

        Returns:
            MealPlan with recipes optimized for zero waste.
        """
        # Build context about available ingredients
        inventory_text = self._format_inventory(pantry_items, bulk_items)
        restrictions = ", ".join(dietary_restrictions) if dietary_restrictions else "none"

        prompt = (
            f"Create a {days}-day meal plan for a family of {household_size} "
            f"using ONLY these available ingredients:\n\n{inventory_text}\n\n"
            f"Dietary restrictions: {restrictions}\n\n"
            f"Focus on using items expiring soonest first. "
            f"Generate enough recipes to cover {days} days of meals "
            f"(breakfast, lunch, dinner). Call create_meal_plan with your plan."
        )

        payload = {
            "model": "gemma-4",
            "messages": [
                {"role": "system", "content": RECIPE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "tools": RECIPE_GEN_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "create_meal_plan"}},
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        if self.tier != "auto":
            payload["_tier"] = self.tier

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.gateway_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()

        return self._parse_response(
            resp.json(), household_size, days, pantry_items
        )

    def build_meal_plan_payload(
        self,
        pantry_items: list[dict[str, Any]],
        bulk_items: list[dict[str, Any]] | None = None,
        household_size: int = 3,
        days: int = 7,
        dietary_restrictions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build the request payload without sending it (for offline use)."""
        inventory_text = self._format_inventory(pantry_items, bulk_items)
        restrictions = ", ".join(dietary_restrictions) if dietary_restrictions else "none"

        prompt = (
            f"Create a {days}-day meal plan for a family of {household_size} "
            f"using ONLY these available ingredients:\n\n{inventory_text}\n\n"
            f"Dietary restrictions: {restrictions}\n\n"
            f"Focus on using items expiring soonest first. "
            f"Generate enough recipes to cover {days} days of meals."
        )

        return {
            "model": "gemma-4",
            "messages": [
                {"role": "system", "content": RECIPE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "tools": RECIPE_GEN_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "create_meal_plan"}},
            "temperature": 0.7,
            "max_tokens": 4096,
        }

    def _format_inventory(
        self,
        pantry_items: list[dict[str, Any]],
        bulk_items: list[dict[str, Any]] | None,
    ) -> str:
        """Format inventory into readable text for the prompt."""
        lines = ["## Current Pantry:"]
        for item in pantry_items:
            expiry = item.get("expiry_date", "unknown")
            lines.append(
                f"- {item['name']}: {item.get('quantity', '?')} {item.get('unit', '')} "
                f"(expires: {expiry})"
            )

        if bulk_items:
            lines.append("\n## New Bulk Purchases:")
            for item in bulk_items:
                lines.append(
                    f"- {item['name']}: {item.get('total_servings', '?')} servings "
                    f"(shelf life: {item.get('shelf_life_days', '?')} days)"
                )

        return "\n".join(lines)

    def _parse_response(
        self,
        response: dict[str, Any],
        household_size: int,
        days: int,
        pantry_items: list[dict[str, Any]],
    ) -> MealPlan:
        """Parse Gemma 4 function-calling response into a MealPlan."""
        plan = MealPlan(
            days=days,
            household_size=household_size,
        )

        choices = response.get("choices", [])
        if not choices:
            return plan

        message = choices[0].get("message", {})

        # Try tool_calls
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            if fn.get("name") == "create_meal_plan":
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)
                plan.recipes = self._recipes_from_args(args)

        # Fallback: parse from content
        if not plan.recipes:
            content = message.get("content", "")
            plan.recipes = self._extract_recipes_from_text(content)

        # Calculate utilization
        pantry_names = {i["name"].lower() for i in pantry_items}
        used_items: set[str] = set()
        for recipe in plan.recipes:
            for ing in recipe.ingredients:
                if ing.pantry_item_name:
                    used_items.add(ing.pantry_item_name.lower())

        if pantry_names:
            plan.pantry_utilization_pct = (
                len(used_items & pantry_names) / len(pantry_names)
            ) * 100

        return plan

    def _recipes_from_args(self, args: dict[str, Any]) -> list[Recipe]:
        """Convert function-call arguments to Recipe list."""
        recipes: list[Recipe] = []
        for raw in args.get("recipes", []):
            ingredients = []
            for ing in raw.get("ingredients", []):
                ingredients.append(Ingredient(
                    name=ing.get("name", ""),
                    quantity=float(ing.get("quantity", 1)),
                    unit=ing.get("unit", ""),
                    pantry_item_name=ing.get("pantry_item_name"),
                ))
            nutrition = raw.get("nutrition_per_serving", {})
            recipes.append(Recipe(
                name=raw.get("name", "Unnamed Recipe"),
                servings=int(raw.get("servings", 4)),
                prep_time_min=int(raw.get("prep_time_min", 15)),
                cook_time_min=int(raw.get("cook_time_min", 30)),
                difficulty=raw.get("difficulty", "easy"),
                ingredients=ingredients,
                instructions=raw.get("instructions", []),
                nutrition_per_serving={
                    "calories": nutrition.get("calories", 0),
                    "protein": nutrition.get("protein", 0),
                    "fiber": nutrition.get("fiber", 0),
                },
                waste_reduction_note=raw.get("waste_reduction_note", ""),
                cultural_note=raw.get("cultural_note", ""),
            ))
        return recipes

    def _extract_recipes_from_text(self, text: str) -> list[Recipe]:
        """Fallback: extract recipes from free text."""
        recipes: list[Recipe] = []
        # Try JSON extraction
        import re
        json_match = re.search(
            r'\{[\s\S]*"recipes"\s*:\s*\[[\s\S]*\]\s*\}', text
        )
        if json_match:
            try:
                data = json.loads(json_match.group())
                return self._recipes_from_args(data)
            except json.JSONDecodeError:
                pass

        # Minimal fallback: create one recipe from text
        if text.strip():
            recipes.append(Recipe(
                name="Generated Meal Suggestion",
                instructions=[text.strip()[:500]],
            ))
        return recipes
