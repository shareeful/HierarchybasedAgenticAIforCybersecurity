from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable, Sequence

import numpy as np


class BanditPolicy(ABC):
    def __init__(self, arms: Sequence[Hashable], rng: np.random.Generator):
        self.arms = list(arms)
        self.index = {arm: i for i, arm in enumerate(self.arms)}
        self.rng = rng
        self.t = 0

    def ensure_arm(self, arm: Hashable) -> int:
        if arm not in self.index:
            self.index[arm] = len(self.arms)
            self.arms.append(arm)
            self._grow()
        return self.index[arm]

    @abstractmethod
    def _grow(self) -> None: ...

    @abstractmethod
    def select(self) -> Hashable: ...

    @abstractmethod
    def update(self, arm: Hashable, reward: float) -> None: ...

    @abstractmethod
    def reliability(self, arm: Hashable) -> float: ...
