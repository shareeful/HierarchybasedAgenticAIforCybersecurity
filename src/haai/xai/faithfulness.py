from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .surrogate import ReferenceSurrogate


@dataclass
class DeletionResult:
    technique: str
    top_k_delta: float
    random_delta: float
    evaluated_on: str = "surrogate"
    instances: int = 0

    def ratio(self) -> float:
        return self.top_k_delta / self.random_delta if self.random_delta else float("inf")


def deletion_test(
    surrogate: ReferenceSurrogate,
    instances: np.ndarray,
    attributions: np.ndarray,
    top_k: int,
    rng: np.random.Generator,
    technique: str,
) -> DeletionResult:
    baseline = surrogate.marginal_expectation()
    original = surrogate.predict(instances)
    n_features = instances.shape[1]

    ranked = np.argsort(-np.abs(attributions), axis=1)[:, :top_k]
    masked_top = instances.copy()
    for row in range(len(instances)):
        masked_top[row, ranked[row]] = baseline[ranked[row]]
    top_delta = float(np.mean(np.abs(surrogate.predict(masked_top) - original)))

    masked_random = instances.copy()
    for row in range(len(instances)):
        choice = rng.choice(n_features, size=min(top_k, n_features), replace=False)
        masked_random[row, choice] = baseline[choice]
    random_delta = float(np.mean(np.abs(surrogate.predict(masked_random) - original)))

    return DeletionResult(technique=technique, top_k_delta=top_delta, random_delta=random_delta)


def model_deletion_test(
    predict: Callable[[np.ndarray], np.ndarray],
    instances: np.ndarray,
    attributions: np.ndarray,
    baseline: np.ndarray,
    top_k: int,
    rng: np.random.Generator,
    technique: str,
) -> DeletionResult:
    original = predict(instances)
    n_features = instances.shape[1]
    ranked = np.argsort(-np.abs(attributions), axis=1)[:, :top_k]
    masked_top = instances.copy()
    for row in range(len(instances)):
        masked_top[row, ranked[row]] = baseline[ranked[row]]
    top_delta = float(np.mean(np.abs(predict(masked_top) - original)))
    masked_random = instances.copy()
    for row in range(len(instances)):
        choice = rng.choice(n_features, size=min(top_k, n_features), replace=False)
        masked_random[row, choice] = baseline[choice]
    random_delta = float(np.mean(np.abs(predict(masked_random) - original)))
    return DeletionResult(
        technique=technique,
        top_k_delta=top_delta,
        random_delta=random_delta,
        evaluated_on="model",
        instances=len(instances),
    )


def counterfactual_validity(results: Sequence) -> float:
    proposed = [r for r in results if r.achieved]
    if not proposed:
        return float("nan")
    return float(np.mean([bool(r.verified) for r in proposed]))
