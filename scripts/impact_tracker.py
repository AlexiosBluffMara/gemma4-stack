"""
GemmaZeroWaste — Impact Tracker Module

Calculates and tracks quantifiable environmental and social impact metrics:
- Food waste reduction (kg diverted from landfill)
- GHG emissions avoided (kg CO₂e from methane prevention)
- Nutrition improvement (% of USDA targets met)
- Cost savings vs. non-optimized shopping
- Chicago food desert equity metrics

All calculations are local — no external API calls needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger("zerowaste.impact")

# ---------------------------------------------------------------------------
# Constants (sourced from EPA, USDA, ReFED, City of Chicago Open Data)
# ---------------------------------------------------------------------------

# EPA: average US household wastes 31.9% of food purchased
US_AVG_FOOD_WASTE_PCT = 31.9

# ReFED: 1 kg food waste = 2.5 kg CO₂e (includes methane from landfill)
CO2E_PER_KG_FOOD_WASTE = 2.5

# USDA: average cost of wasted food per US household = $1,500/year
US_AVG_ANNUAL_FOOD_WASTE_COST = 1500.0

# Chicago food desert stats (source: Chicago Health Atlas, USDA ERS)
CHICAGO_FOOD_DESERT_STATS = {
    "population_in_food_deserts": 633_631,  # ~23% of Chicago
    "neighborhoods_affected": 37,
    "avg_distance_to_grocery_mi": 1.6,
    "snap_participation_pct": 42.0,
    "diet_related_disease_rate_pct": 38.5,  # diabetes + obesity
    "median_household_income": 32_500,
    "avg_household_size": 3.2,
}

# Nutrition scoring weights (higher = more important for food desert communities)
NUTRITION_WEIGHTS = {
    "calories":   0.15,
    "protein":    0.25,
    "fiber":      0.20,
    "vitamin_c":  0.15,
    "calcium":    0.15,
    "iron":       0.10,
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ImpactSession:
    """A single tracking session (one week of meal planning)."""
    start_date: str = ""
    end_date: str = ""
    household_size: int = 3
    total_spent: float = 0.0
    total_servings_planned: int = 0
    total_servings_consumed: int = 0
    pantry_items_used: int = 0
    pantry_items_wasted: int = 0
    recipes_generated: int = 0
    waste_diverted_kg: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "household_size": self.household_size,
            "total_spent": round(self.total_spent, 2),
            "total_servings_planned": self.total_servings_planned,
            "total_servings_consumed": self.total_servings_consumed,
            "pantry_items_used": self.pantry_items_used,
            "pantry_items_wasted": self.pantry_items_wasted,
            "recipes_generated": self.recipes_generated,
            "waste_diverted_kg": round(self.waste_diverted_kg, 2),
        }


@dataclass
class ImpactMetrics:
    """Aggregated impact metrics across all sessions."""
    # Waste reduction
    food_waste_pct: float = 0.0  # actual waste %
    waste_reduction_vs_avg: float = 0.0  # % reduction vs US average
    waste_diverted_kg: float = 0.0  # total kg food saved from waste

    # Climate impact
    co2e_avoided_kg: float = 0.0  # kg CO₂e prevented
    equivalent_car_miles: float = 0.0  # equivalent car miles avoided
    equivalent_trees_month: float = 0.0  # equivalent tree-months of carbon capture

    # Nutrition
    nutrition_score: float = 0.0  # 0-100 composite score
    nutrition_coverage: dict[str, float] = field(default_factory=dict)

    # Cost
    total_spent: float = 0.0
    estimated_savings: float = 0.0  # vs. non-optimized shopping
    cost_per_person_per_day: float = 0.0

    # Equity
    snap_budget_efficiency: float = 0.0  # % of SNAP benefit utilized effectively
    meals_per_snap_dollar: float = 0.0

    # Community
    households_equivalent: float = 0.0  # equivalent households if scaled

    def to_dict(self) -> dict[str, Any]:
        return {
            "waste_reduction": {
                "food_waste_pct": round(self.food_waste_pct, 1),
                "waste_reduction_vs_avg": round(self.waste_reduction_vs_avg, 1),
                "waste_diverted_kg": round(self.waste_diverted_kg, 2),
            },
            "climate_impact": {
                "co2e_avoided_kg": round(self.co2e_avoided_kg, 2),
                "equivalent_car_miles": round(self.equivalent_car_miles, 1),
                "equivalent_trees_month": round(self.equivalent_trees_month, 2),
            },
            "nutrition": {
                "score": round(self.nutrition_score, 1),
                "coverage": {
                    k: round(v, 1)
                    for k, v in self.nutrition_coverage.items()
                },
            },
            "cost": {
                "total_spent": round(self.total_spent, 2),
                "estimated_savings": round(self.estimated_savings, 2),
                "cost_per_person_per_day": round(self.cost_per_person_per_day, 2),
            },
            "equity": {
                "snap_budget_efficiency": round(self.snap_budget_efficiency, 1),
                "meals_per_snap_dollar": round(self.meals_per_snap_dollar, 2),
            },
            "community_scale": {
                "households_equivalent": round(self.households_equivalent, 1),
                "chicago_food_desert_context": CHICAGO_FOOD_DESERT_STATS,
            },
        }


# ---------------------------------------------------------------------------
# Impact Tracker
# ---------------------------------------------------------------------------

class ImpactTracker:
    """
    Tracks and quantifies the real-world impact of GemmaZeroWaste usage.

    All calculations are deterministic and run locally — no cloud calls.
    """

    def __init__(self):
        self.sessions: list[ImpactSession] = []

    def record_session(
        self,
        household_size: int,
        total_spent: float,
        servings_planned: int,
        servings_consumed: int,
        pantry_items_used: int,
        pantry_items_wasted: int,
        recipes_generated: int,
        planning_days: int = 7,
    ) -> ImpactSession:
        """Record a meal planning session."""
        # Estimate waste diverted: what would have been wasted without planning
        # US avg wastes 31.9% of food; assume planning reduces this to actual waste rate
        expected_waste_servings = servings_planned * (US_AVG_FOOD_WASTE_PCT / 100)
        actual_waste_servings = servings_planned - servings_consumed
        diverted_servings = max(0, expected_waste_servings - actual_waste_servings)
        # Rough conversion: 1 serving ≈ 0.25 kg food
        waste_diverted_kg = diverted_servings * 0.25

        today = date.today()
        session = ImpactSession(
            start_date=str(today),
            end_date=str(today),
            household_size=household_size,
            total_spent=total_spent,
            total_servings_planned=servings_planned,
            total_servings_consumed=servings_consumed,
            pantry_items_used=pantry_items_used,
            pantry_items_wasted=pantry_items_wasted,
            recipes_generated=recipes_generated,
            waste_diverted_kg=waste_diverted_kg,
        )
        self.sessions.append(session)
        return session

    def compute_metrics(
        self,
        nutrition_coverage: dict[str, float] | None = None,
    ) -> ImpactMetrics:
        """Compute aggregate impact metrics across all sessions."""
        if not self.sessions:
            return ImpactMetrics()

        metrics = ImpactMetrics()

        total_planned = sum(s.total_servings_planned for s in self.sessions)
        total_consumed = sum(s.total_servings_consumed for s in self.sessions)
        total_wasted_items = sum(s.pantry_items_wasted for s in self.sessions)
        total_used_items = sum(s.pantry_items_used for s in self.sessions)
        total_spent = sum(s.total_spent for s in self.sessions)
        total_waste_kg = sum(s.waste_diverted_kg for s in self.sessions)
        total_hh_size = sum(s.household_size for s in self.sessions)
        num_sessions = len(self.sessions)
        avg_hh = total_hh_size / num_sessions if num_sessions else 3

        # Waste reduction
        if total_planned > 0:
            actual_waste_pct = (
                (total_planned - total_consumed) / total_planned
            ) * 100
            metrics.food_waste_pct = max(0, actual_waste_pct)
            metrics.waste_reduction_vs_avg = max(
                0, US_AVG_FOOD_WASTE_PCT - metrics.food_waste_pct
            )
        metrics.waste_diverted_kg = total_waste_kg

        # Climate impact
        metrics.co2e_avoided_kg = total_waste_kg * CO2E_PER_KG_FOOD_WASTE
        # EPA: average passenger vehicle emits 0.404 kg CO₂ per mile
        metrics.equivalent_car_miles = metrics.co2e_avoided_kg / 0.404
        # Average tree absorbs ~21.77 kg CO₂ per year = ~1.81 kg/month
        if metrics.co2e_avoided_kg > 0:
            metrics.equivalent_trees_month = metrics.co2e_avoided_kg / 1.81

        # Nutrition
        if nutrition_coverage:
            metrics.nutrition_coverage = nutrition_coverage
            weighted_score = sum(
                nutrition_coverage.get(n, 0) * w
                for n, w in NUTRITION_WEIGHTS.items()
            )
            metrics.nutrition_score = min(100, weighted_score)

        # Cost
        metrics.total_spent = total_spent
        # Estimate savings: US avg wastes $1500/yr = $28.85/wk
        # We saved (waste_reduction_vs_avg / 100) of that
        weekly_savings = (
            (US_AVG_ANNUAL_FOOD_WASTE_COST / 52)
            * (metrics.waste_reduction_vs_avg / 100)
        )
        metrics.estimated_savings = weekly_savings * num_sessions

        if avg_hh > 0 and num_sessions > 0:
            metrics.cost_per_person_per_day = total_spent / (avg_hh * num_sessions * 7)

        # Equity
        snap_monthly = 234.0  # avg SNAP benefit per household
        if total_spent > 0 and total_consumed > 0:
            metrics.meals_per_snap_dollar = total_consumed / total_spent
            # Efficiency: % of budget that became consumed meals (not waste)
            metrics.snap_budget_efficiency = min(
                100, (total_consumed / total_planned) * 100
            ) if total_planned > 0 else 0

        # Community scale: if all food desert households used this
        if total_waste_kg > 0 and avg_hh > 0:
            per_household_kg = total_waste_kg / num_sessions
            total_households = (
                CHICAGO_FOOD_DESERT_STATS["population_in_food_deserts"]
                / CHICAGO_FOOD_DESERT_STATS["avg_household_size"]
            )
            metrics.households_equivalent = total_households

        return metrics

    def generate_impact_summary(
        self,
        metrics: ImpactMetrics | None = None,
    ) -> str:
        """Generate a human-readable impact summary for the dashboard."""
        if metrics is None:
            metrics = self.compute_metrics()

        lines = [
            "🌿 GemmaZeroWaste Impact Report",
            "=" * 40,
            "",
            "📊 Waste Reduction:",
            f"   Food waste: {metrics.food_waste_pct:.1f}% "
            f"(vs. US avg {US_AVG_FOOD_WASTE_PCT}%)",
            f"   Reduction: {metrics.waste_reduction_vs_avg:.1f} percentage points",
            f"   Food saved: {metrics.waste_diverted_kg:.1f} kg",
            "",
            "🌍 Climate Impact:",
            f"   CO₂e avoided: {metrics.co2e_avoided_kg:.1f} kg",
            f"   = {metrics.equivalent_car_miles:.0f} car miles avoided",
            f"   = {metrics.equivalent_trees_month:.1f} tree-months of carbon capture",
            "",
            "💰 Cost:",
            f"   Total spent: ${metrics.total_spent:.2f}",
            f"   Estimated savings: ${metrics.estimated_savings:.2f}",
            f"   Per person/day: ${metrics.cost_per_person_per_day:.2f}",
            "",
            "🏘️ Community Scale (if adopted across Chicago food deserts):",
            f"   {CHICAGO_FOOD_DESERT_STATS['population_in_food_deserts']:,} "
            f"residents across {CHICAGO_FOOD_DESERT_STATS['neighborhoods_affected']} "
            f"neighborhoods",
        ]

        if metrics.co2e_avoided_kg > 0 and metrics.households_equivalent > 0:
            scaled_co2 = (
                metrics.co2e_avoided_kg
                / max(1, len(self.sessions))
                * metrics.households_equivalent
            )
            lines.append(
                f"   Potential annual CO₂e reduction: "
                f"{scaled_co2 * 52:,.0f} kg ({scaled_co2 * 52 / 1000:,.1f} tonnes)"
            )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemma 4 function-calling tools for impact analysis
# ---------------------------------------------------------------------------

IMPACT_ANALYSIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "report_impact",
            "description": (
                "Report the environmental and social impact of the user's "
                "meal planning session. Explain metrics in accessible language."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Human-readable impact summary",
                    },
                    "key_stat": {
                        "type": "string",
                        "description": "Single most impactful statistic to highlight",
                    },
                    "motivation": {
                        "type": "string",
                        "description": "Encouraging message about the user's impact",
                    },
                },
                "required": ["summary", "key_stat", "motivation"],
            },
        },
    }
]
