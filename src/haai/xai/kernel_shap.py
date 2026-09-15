from __future__ import annotations

from math import comb

import numpy as np

from .surrogate import ReferenceSurrogate


def _shapley_kernel(n: int, size: int) -> float:
    if size == 0 or size == n:
        return 1e6
    return (n - 1) / (comb(n, size) * size * (n - size))


def kernel_shap(
    surrogate: ReferenceSurrogate,
    instance: np.ndarray,
    n_coalitions: int,
    rng: np.random.Generator,
    background: np.ndarray | None = None,
) -> np.ndarray:
    n = len(surrogate.fields)
    baseline = surrogate.marginal_expectation() if background is None else background
    sizes = rng.integers(1, n, size=n_coalitions)
    masks = np.zeros((n_coalitions, n), dtype=bool)
    for row, size in enumerate(sizes):
        masks[row, rng.choice(n, size=int(size), replace=False)] = True
    masks = np.vstack([np.zeros(n, dtype=bool), np.ones(n, dtype=bool), masks])

    synthetic = np.where(masks, instance, baseline)
    values = surrogate.predict(synthetic)
    weights = np.array([_shapley_kernel(n, int(mask.sum())) for mask in masks])

    design = masks.astype(float)
    base_value = values[0]
    full_value = values[1]
    target = values - base_value

    weighted = design * weights[:, None]
    gram = design.T @ weighted
    gram += np.eye(n) * 1e-8
    solution = np.linalg.solve(gram, weighted.T @ target)
    correction = (full_value - base_value - solution.sum()) / n
    return solution + correction
