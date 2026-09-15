from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import List, Tuple

import numpy as np

from .surrogate import ReferenceSurrogate


def _coalition_masks(n_features: int) -> List[Tuple[int, ...]]:
    masks: List[Tuple[int, ...]] = []
    indices = list(range(n_features))
    for size in range(n_features + 1):
        masks.extend(combinations(indices, size))
    return masks


def exact_shapley(
    surrogate: ReferenceSurrogate, instance: np.ndarray, background: np.ndarray | None = None
) -> np.ndarray:
    fields = surrogate.fields
    n = len(fields)
    baseline = surrogate.marginal_expectation() if background is None else background
    coalitions = _coalition_masks(n)
    index = {coalition: position for position, coalition in enumerate(coalitions)}
    matrix = np.tile(baseline, (len(coalitions), 1))
    for position, coalition in enumerate(coalitions):
        for feature in coalition:
            matrix[position, feature] = instance[feature]
    values = surrogate.predict(matrix)

    phi = np.zeros(n)
    for feature in range(n):
        others = [i for i in range(n) if i != feature]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                with_feature = tuple(sorted(subset + (feature,)))
                phi[feature] += weight * (
                    values[index[with_feature]] - values[index[tuple(sorted(subset))]]
                )
    return phi
