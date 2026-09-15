from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class WilcoxonResult:
    statistic: float
    p_value: float
    n: int
    exact_bound: bool
    direction: str

    def format(self) -> str:
        if self.exact_bound:
            return f"p = {self.p_value:.1e} (exact bound at n={self.n})"
        return f"p = {self.p_value:.3g} (n={self.n})"


def minimum_attainable_p(n: int) -> float:
    return 2.0 / (2.0**n)


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> WilcoxonResult:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    differences = a - b
    n = len(differences)
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return WilcoxonResult(0.0, 1.0, n, False, "none")
    statistic, p_value = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    all_same_direction = bool(np.all(nonzero > 0) or np.all(nonzero < 0))
    bound = minimum_attainable_p(len(nonzero))
    exact_bound = all_same_direction and len(nonzero) == n
    if exact_bound:
        p_value = max(float(p_value), bound)
    direction = "positive" if float(np.mean(differences)) > 0 else "negative"
    return WilcoxonResult(float(statistic), float(p_value), n, exact_bound, direction)


def summarise(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def format_mean_std(values: Sequence[float], decimals: int = 3) -> str:
    mean, std = summarise(values)
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"
