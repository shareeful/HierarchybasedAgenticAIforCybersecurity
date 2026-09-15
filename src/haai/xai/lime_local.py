from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import Ridge

from .surrogate import ReferenceSurrogate


def gower_distance(instance: np.ndarray, neighbours: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    safe = np.where(ranges > 0, ranges, 1.0)
    return np.abs(neighbours - instance).mean(axis=1) if safe.size == 0 else (
        np.abs(neighbours - instance) / safe
    ).mean(axis=1)


def lime_explain(
    surrogate: ReferenceSurrogate,
    instance: np.ndarray,
    n_neighbours: int,
    kernel_width: float,
    n_nonzero: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    reference = surrogate.reference
    n_features = reference.shape[1]
    neighbours = np.empty((n_neighbours, n_features))
    for feature in range(n_features):
        neighbours[:, feature] = rng.choice(reference[:, feature], size=n_neighbours, replace=True)
    keep = rng.random((n_neighbours, n_features)) < 0.5
    neighbours = np.where(keep, np.tile(instance, (n_neighbours, 1)), neighbours)
    neighbours[0] = instance

    ranges = reference.max(axis=0) - reference.min(axis=0)
    distance = gower_distance(instance, neighbours, ranges)
    weights = np.exp(-(distance**2) / (kernel_width**2))
    responses = surrogate.predict(neighbours)

    centred = neighbours - instance
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(centred, responses, sample_weight=weights)
    coefficients = ridge.coef_
    if n_nonzero < n_features:
        keep_index = np.argsort(-np.abs(coefficients))[:n_nonzero]
        restricted = np.zeros_like(centred)
        restricted[:, keep_index] = centred[:, keep_index]
        ridge = Ridge(alpha=1.0, fit_intercept=True)
        ridge.fit(restricted, responses, sample_weight=weights)
        coefficients = ridge.coef_
    fidelity = float(ridge.score(centred, responses, sample_weight=weights))
    return coefficients, fidelity
