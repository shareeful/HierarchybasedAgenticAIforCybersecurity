from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from ..config import Config
from ..logging_utils import get_logger

LOGGER = get_logger()


@dataclass
class ReferenceSurrogate:
    fields: Tuple[str, ...]
    model: HistGradientBoostingRegressor
    reference: np.ndarray
    r2: float
    name: str

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return self.model.predict(np.atleast_2d(matrix))

    def marginal_expectation(self) -> np.ndarray:
        return self.reference.mean(axis=0)


class RollingSurrogate:
    def __init__(self, config: Config, name: str, fields: Sequence[str], rng: np.random.Generator):
        self.config = config
        self.name = name
        self.fields = tuple(fields)
        self.rng = rng
        self.window: Deque[Tuple[np.ndarray, float]] = deque(maxlen=config.xai.reference_window)
        self.surrogate: ReferenceSurrogate | None = None
        self.cycles_since_refit = 0
        self.refits = 0

    def observe(self, matrix: np.ndarray, scores: np.ndarray) -> None:
        for row, score in zip(np.atleast_2d(matrix), np.atleast_1d(scores)):
            self.window.append((np.asarray(row, dtype=float), float(score)))
        self.cycles_since_refit += len(np.atleast_1d(scores))

    def maybe_refit(self, force: bool = False) -> ReferenceSurrogate | None:
        if len(self.window) < 64:
            return self.surrogate
        due = self.cycles_since_refit >= self.config.xai.surrogate_retrain_interval
        if not (force or due or self.surrogate is None):
            return self.surrogate
        matrix = np.vstack([row for row, _ in self.window])
        target = np.asarray([score for _, score in self.window])
        split = int(0.8 * len(matrix))
        model = HistGradientBoostingRegressor(
            max_iter=self.config.xai.surrogate_estimators,
            max_depth=self.config.xai.surrogate_max_depth,
            learning_rate=self.config.xai.surrogate_learning_rate,
            l2_regularization=1.0,
            random_state=int(self.rng.integers(0, 2**31 - 1)),
        ).fit(matrix[:split], target[:split])
        held_out = r2_score(target[split:], model.predict(matrix[split:])) if split < len(matrix) else 1.0
        self.surrogate = ReferenceSurrogate(
            fields=self.fields,
            model=model,
            reference=matrix,
            r2=float(held_out),
            name=self.name,
        )
        self.cycles_since_refit = 0
        self.refits += 1
        return self.surrogate


def build_surrogate(
    config: Config,
    family: str,
    fields: Sequence[str],
    matrix: np.ndarray,
    target: np.ndarray,
    rng: np.random.Generator,
) -> ReferenceSurrogate:
    split = max(int(0.8 * len(matrix)), 1)
    seed = int(rng.integers(0, 2**31 - 1))
    if family == "gradient_boosting":
        model = HistGradientBoostingRegressor(
            max_iter=config.xai.surrogate_estimators,
            max_depth=config.xai.surrogate_max_depth,
            learning_rate=config.xai.surrogate_learning_rate,
            l2_regularization=1.0,
            random_state=seed,
        )
    elif family == "random_forest":
        model = RandomForestRegressor(
            n_estimators=config.xai.surrogate_estimators,
            max_depth=config.xai.surrogate_max_depth,
            random_state=seed,
            n_jobs=1,
        )
    elif family == "ridge":
        model = Ridge(alpha=1.0)
    else:
        raise ValueError(f"unknown surrogate family: {family}")
    model.fit(matrix[:split], target[:split])
    held_out = (
        float(r2_score(target[split:], model.predict(matrix[split:])))
        if split < len(matrix)
        else float("nan")
    )
    return ReferenceSurrogate(
        fields=tuple(fields), model=model, reference=matrix, r2=held_out, name=family
    )
