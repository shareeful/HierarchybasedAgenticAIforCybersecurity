from __future__ import annotations

from collections import deque
from typing import Deque, Hashable, Sequence, Tuple

import numpy as np

from .base import BanditPolicy


class DiscountedUCB(BanditPolicy):
    def __init__(
        self,
        arms: Sequence[Hashable],
        rng: np.random.Generator,
        gamma: float = 0.85,
        epsilon: float = 1e-6,
    ):
        super().__init__(arms, rng)
        self.gamma = gamma
        self.epsilon = epsilon
        self.weighted_reward = np.zeros(len(self.arms), dtype=float)
        self.weighted_count = np.zeros(len(self.arms), dtype=float)
        self.discounted_total = 0.0

    def _grow(self) -> None:
        self.weighted_reward = np.append(self.weighted_reward, 0.0)
        self.weighted_count = np.append(self.weighted_count, 0.0)

    def _mean(self) -> np.ndarray:
        return self.weighted_reward / np.maximum(self.weighted_count, self.epsilon)

    def _bonus(self) -> np.ndarray:
        total = max(self.discounted_total, 1.0)
        return np.sqrt(2.0 * np.log(total) / np.maximum(self.weighted_count, self.epsilon))

    def select(self) -> Hashable:
        return self.arms[int(np.argmax(self._mean() + self._bonus()))]

    def update(self, arm: Hashable, reward: float) -> None:
        position = self.ensure_arm(arm)
        self.weighted_reward *= self.gamma
        self.weighted_count *= self.gamma
        self.discounted_total = self.discounted_total * self.gamma + 1.0
        self.weighted_reward[position] += float(np.clip(reward, 0.0, 1.0))
        self.weighted_count[position] += 1.0
        self.t += 1

    def estimate(self, arm: Hashable) -> Tuple[float, float]:
        position = self.ensure_arm(arm)
        mean = self.weighted_reward[position] / max(self.weighted_count[position], self.epsilon)
        total = max(self.discounted_total, 1.0)
        bonus = np.sqrt(2.0 * np.log(total) / max(self.weighted_count[position], self.epsilon))
        return float(mean), float(bonus / (1.0 + bonus))

    def reliability(self, arm: Hashable) -> float:
        mean, bonus = self.estimate(arm)
        return float(np.clip(mean * (1.0 - bonus), 0.0, 1.0))


class SlidingWindowUCB(BanditPolicy):
    def __init__(self, arms: Sequence[Hashable], rng: np.random.Generator, tau: int = 80):
        super().__init__(arms, rng)
        self.tau = tau
        self.window: Deque[Tuple[int, float]] = deque(maxlen=tau)

    def _grow(self) -> None:
        return None

    def _statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        totals = np.zeros(len(self.arms), dtype=float)
        counts = np.zeros(len(self.arms), dtype=float)
        for position, reward in self.window:
            totals[position] += reward
            counts[position] += 1.0
        return totals, counts

    def scores(self) -> np.ndarray:
        totals, counts = self._statistics()
        means = np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0)
        bonus = np.where(
            counts > 0,
            np.sqrt(2.0 * np.log(max(self.tau, 2)) / np.maximum(counts, 1.0)),
            np.inf,
        )
        return means + bonus

    def select(self) -> Hashable:
        values = self.scores()
        best = np.flatnonzero(values == values.max())
        return self.arms[int(self.rng.choice(best))]

    def update(self, arm: Hashable, reward: float) -> None:
        position = self.ensure_arm(arm)
        self.window.append((position, float(reward)))
        self.t += 1

    def reliability(self, arm: Hashable) -> float:
        totals, counts = self._statistics()
        position = self.ensure_arm(arm)
        if counts[position] == 0:
            return 0.0
        return float(np.clip(totals[position] / counts[position], 0.0, 1.0))


class StaticPolicy(BanditPolicy):
    def __init__(self, arms: Sequence[Hashable], rng: np.random.Generator, fixed_arm: Hashable | None = None):
        super().__init__(arms, rng)
        self.fixed_arm = fixed_arm

    def _grow(self) -> None:
        return None

    def select(self) -> Hashable:
        if self.fixed_arm is not None:
            return self.fixed_arm
        return self.arms[int(self.rng.integers(0, len(self.arms)))]

    def update(self, arm: Hashable, reward: float) -> None:
        self.t += 1

    def reliability(self, arm: Hashable) -> float:
        return 0.5
