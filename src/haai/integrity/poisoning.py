from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from ..config import Config


@dataclass
class PoisonedBatch:
    vectors: np.ndarray
    poisoned_mask: np.ndarray
    rate: float
    provenance: str


def recorded_tampering(telemetry: pd.DataFrame, columns: Tuple[str, ...]) -> PoisonedBatch | None:
    if "tampered" not in telemetry.columns:
        return None
    mask = telemetry["tampered"].to_numpy(dtype=bool)
    if not mask.any():
        return None
    vectors = telemetry.loc[:, list(columns)].to_numpy(dtype=float)
    return PoisonedBatch(
        vectors=vectors,
        poisoned_mask=mask,
        rate=float(mask.mean()),
        provenance="recorded tampering labels",
    )


def inject_attack_model(
    config: Config,
    vectors: np.ndarray,
    epss_column: int,
    cvss_column: int,
    poc_column: int | None,
    rate: float,
    rng: np.random.Generator,
) -> PoisonedBatch:
    poisoned = np.array(vectors, dtype=float, copy=True)
    low_cvss, high_cvss = config.integrity.poison_cvss_band
    cvss = poisoned[:, cvss_column]
    eligible = np.flatnonzero((cvss >= low_cvss) & (cvss <= high_cvss))
    target_count = int(round(rate * len(poisoned)))
    if len(eligible) >= target_count:
        selected = rng.choice(eligible, size=target_count, replace=False)
    else:
        remainder = np.setdiff1d(np.arange(len(poisoned)), eligible)
        extra = rng.choice(
            remainder, size=min(target_count - len(eligible), len(remainder)), replace=False
        )
        selected = np.concatenate([eligible, extra])
    mask = np.zeros(len(poisoned), dtype=bool)
    mask[selected] = True
    low_epss, high_epss = config.integrity.poisoned_epss_bounds
    poisoned[selected, epss_column] = rng.uniform(low_epss, high_epss, size=len(selected))
    if poc_column is not None:
        poisoned[selected, poc_column] = 1.0
    return PoisonedBatch(
        vectors=poisoned,
        poisoned_mask=mask,
        rate=rate,
        provenance="injected attack model (config.integrity)",
    )
