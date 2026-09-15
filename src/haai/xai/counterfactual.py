from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from ..config import Config


@dataclass
class Counterfactual:
    change: str
    cost: float
    original_exposure: float
    new_exposure: float
    achieved: bool
    verified: bool = False
    verified_exposure: float = float("nan")

    def describe(self) -> str:
        if not self.achieved:
            return "no single admissible change reduces exposure below the medium band"
        return (
            f"{self.change.replace('_', ' ')} would reduce exposure from "
            f"{self.original_exposure:.3f} to {self.new_exposure:.3f} at operational cost "
            f"{self.cost:.2f}"
        )


def admissible_changes(
    config: Config, controls: Sequence[str], zone: str, zone_reach: Dict[str, float]
) -> List[Tuple[str, Dict[str, float]]]:
    candidates: List[Tuple[str, Dict[str, float]]] = []
    for family in config.org.control_families:
        if family not in controls:
            candidates.append((f"activate_{family}", {"control_coverage_delta": 1.0}))
    candidates.append(("remove_external_reachability", {"external_reachable": 0.0}))
    zones = sorted(zone_reach)
    if zone in zones:
        position = zones.index(zone)
        for offset in (-1, 1):
            neighbour = position + offset
            if 0 <= neighbour < len(zones):
                candidates.append(("relocate_zone", {"zone_reach": zone_reach[zones[neighbour]]}))
    candidates.append(("apply_patch", {"sw_match": 0.0}))
    return candidates[: config.xai.counterfactual_max_candidates]


def apply_change(
    row: Dict[str, float], change: Dict[str, float], control_families: int
) -> Dict[str, float]:
    updated = dict(row)
    for key, value in change.items():
        if key == "control_coverage_delta":
            step = 1.0 / max(control_families, 1)
            updated["control_coverage"] = min(
                1.0, float(updated.get("control_coverage", 0.0)) + step
            )
        else:
            updated[key] = value
    return updated


def search_counterfactual(
    config: Config,
    row: Dict[str, float],
    controls: Sequence[str],
    zone_reach: Dict[str, float],
    recompute: Callable[[Dict[str, float]], float],
    lambda_balance: float | None = None,
) -> Counterfactual:
    original = float(recompute(row))
    target = config.xai.counterfactual_medium_band
    balance = config.xai.counterfactual_lambda if lambda_balance is None else lambda_balance
    costs = config.xai.counterfactual_costs
    best: Counterfactual | None = None
    best_objective = float("inf")
    for name, change in admissible_changes(config, controls, str(row.get("zone", "")), zone_reach):
        candidate = apply_change(row, change, len(config.org.control_families))
        value = float(recompute(candidate))
        cost = float(costs.get(name, 0.5))
        objective = (value - target) + balance * cost if value > target else balance * cost
        if objective < best_objective:
            best_objective = objective
            best = Counterfactual(
                change=name,
                cost=cost,
                original_exposure=original,
                new_exposure=value,
                achieved=value <= target < original,
            )
    if best is None:
        return Counterfactual("none", 0.0, original, original, False)
    return best


def counterfactual_frame(items: Sequence[Counterfactual]) -> "object":
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "change": item.change,
                "cost": item.cost,
                "original_exposure": round(item.original_exposure, 4),
                "surrogate_exposure": round(item.new_exposure, 4),
                "achieved_on_surrogate": item.achieved,
                "verified_on_model": item.verified,
                "model_exposure": round(item.verified_exposure, 4)
                if np.isfinite(item.verified_exposure)
                else float("nan"),
            }
            for item in items
        ]
    )
