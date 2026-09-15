from __future__ import annotations

from typing import Dict, Hashable, Sequence, Tuple

import numpy as np

from .base import BanditPolicy


class ThompsonSampling(BanditPolicy):
    def __init__(
        self,
        arms: Sequence[Hashable],
        rng: np.random.Generator,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        super().__init__(arms, rng)
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.alpha = np.full(len(self.arms), prior_alpha, dtype=float)
        self.beta = np.full(len(self.arms), prior_beta, dtype=float)

    def _grow(self) -> None:
        self.alpha = np.append(self.alpha, self.prior_alpha)
        self.beta = np.append(self.beta, self.prior_beta)

    def sample(self) -> np.ndarray:
        return self.rng.beta(self.alpha, self.beta)

    def select(self) -> Hashable:
        return self.arms[int(np.argmax(self.sample()))]

    def update(self, arm: Hashable, reward: float) -> None:
        position = self.ensure_arm(arm)
        reward = float(np.clip(reward, 0.0, 1.0))
        self.alpha[position] += reward
        self.beta[position] += 1.0 - reward
        self.t += 1

    def posterior(self, arm: Hashable) -> Tuple[float, float]:
        position = self.ensure_arm(arm)
        a = self.alpha[position]
        b = self.beta[position]
        mean = a / (a + b)
        variance = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
        return float(mean), float(np.sqrt(variance))

    def reliability(self, arm: Hashable) -> float:
        mean, std = self.posterior(arm)
        normalised_std = min(std / 0.5, 1.0)
        return float(np.clip(mean * (1.0 - normalised_std), 0.0, 1.0))

    def reliability_ranking(self) -> Dict[Hashable, float]:
        samples = self.sample()
        return {arm: float(samples[i]) for i, arm in enumerate(self.arms)}
