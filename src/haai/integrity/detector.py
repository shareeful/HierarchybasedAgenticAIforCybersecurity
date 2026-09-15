from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

from ..config import Config
from ..logging_utils import get_logger

LOGGER = get_logger()


def anomaly_score(forest: IsolationForest, vectors: np.ndarray) -> np.ndarray:
    return -forest.score_samples(vectors)


class IsolationForestVerifier:
    def __init__(self, config: Config, agent: str, rng: np.random.Generator):
        self.config = config
        self.agent = agent
        self.rng = rng
        self.window: Deque[np.ndarray] = deque(maxlen=config.integrity.observation_window)
        self.forest: IsolationForest | None = None
        self.threshold = config.integrity.threshold_grid_start
        self.refits = 0

    def prime(self, vectors: np.ndarray) -> "IsolationForestVerifier":
        for vector in vectors[-self.config.integrity.observation_window :]:
            self.window.append(np.asarray(vector, dtype=float))
        self._refit()
        return self

    def _refit(self) -> None:
        if len(self.window) < 32:
            return
        integrity = self.config.integrity
        matrix = np.vstack(self.window)
        self.forest = IsolationForest(
            n_estimators=integrity.n_estimators,
            max_samples=min(integrity.max_samples, len(matrix)),
            random_state=int(self.rng.integers(0, 2**31 - 1)),
            n_jobs=1,
        ).fit(matrix)
        self.refits += 1

    def calibrate(self, clean_vectors: np.ndarray) -> float:
        integrity = self.config.integrity
        if self.forest is None:
            self.prime(clean_vectors)
        scores = anomaly_score(self.forest, clean_vectors)
        grid = np.arange(
            integrity.threshold_grid_start,
            integrity.threshold_grid_stop + integrity.threshold_grid_step,
            integrity.threshold_grid_step,
        )
        chosen = grid[-1]
        for candidate in grid:
            if float((scores > candidate).mean()) < integrity.false_positive_ceiling:
                chosen = candidate
                break
        self.threshold = float(chosen)
        LOGGER.info(
            "Isolation Forest threshold for %s calibrated to delta=%.3f (clean FPR=%.4f)",
            self.agent,
            self.threshold,
            float((scores > self.threshold).mean()),
        )
        return self.threshold

    def score(self, vectors: np.ndarray) -> np.ndarray:
        if self.forest is None:
            self.prime(vectors)
        return anomaly_score(self.forest, np.atleast_2d(vectors))

    def verify(self, vectors: np.ndarray, refit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        scores = self.score(vectors)
        flags = scores > self.threshold
        if refit:
            for vector, flag in zip(vectors, flags):
                if not flag:
                    self.window.append(vector)
            if len(self.window) >= self.config.integrity.observation_window:
                self._refit()
        return scores, flags


class FixedThresholdDetector:
    def __init__(self, config: Config, agent: str):
        self.config = config
        self.agent = agent
        self.threshold = np.zeros(0)

    def calibrate(self, clean_vectors: np.ndarray) -> "FixedThresholdDetector":
        self.threshold = np.quantile(
            clean_vectors, self.config.integrity.fixed_threshold_quantile, axis=0
        )
        return self

    def verify(self, vectors: np.ndarray, refit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        exceed = (vectors > self.threshold).any(axis=1)
        scores = (vectors / np.maximum(self.threshold, 1e-9)).max(axis=1)
        return scores, exceed


class RollingZScoreDetector:
    def __init__(self, config: Config, agent: str):
        self.config = config
        self.agent = agent
        self.window: Deque[np.ndarray] = deque(maxlen=config.integrity.observation_window)

    def calibrate(self, clean_vectors: np.ndarray) -> "RollingZScoreDetector":
        for vector in clean_vectors[-self.config.integrity.observation_window :]:
            self.window.append(np.asarray(vector, dtype=float))
        return self

    def verify(self, vectors: np.ndarray, refit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        vectors = np.atleast_2d(np.asarray(vectors, dtype=float))
        if len(self.window) < 16:
            for vector in vectors:
                self.window.append(vector)
            return np.zeros(len(vectors)), np.zeros(len(vectors), dtype=bool)
        reference = np.vstack(self.window)
        mean = reference.mean(axis=0)
        std = reference.std(axis=0) + 1e-9
        z = np.abs((vectors - mean) / std)
        scores = z.max(axis=1)
        flags = scores > self.config.integrity.zscore_threshold
        if refit:
            for vector, flag in zip(vectors, flags):
                if not flag:
                    self.window.append(vector)
        return scores, flags
