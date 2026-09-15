from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..config import Config


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=float)
        self.actions = np.zeros(capacity, dtype=int)
        self.rewards = np.zeros(capacity, dtype=float)
        self.size = 0
        self.cursor = 0

    def add(self, state: np.ndarray, action: int, reward: float) -> None:
        self.states[self.cursor] = state
        self.actions[self.cursor] = action
        self.rewards[self.cursor] = reward
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        index = rng.integers(0, self.size, size=min(batch, self.size))
        return self.states[index], self.actions[index], self.rewards[index]


class DeepQNetworkPolicy:
    def __init__(self, config: Config, state_dim: int, rng: np.random.Generator, horizon: int = 1_000):
        evaluation = config.evaluation
        self.rng = rng
        self.n_actions = len(config.bandit.actions)
        self.actions = list(config.bandit.actions)
        hidden = evaluation.dqn_hidden_units
        scale_in = np.sqrt(2.0 / state_dim)
        scale_h = np.sqrt(2.0 / hidden)
        self.w1 = rng.normal(0.0, scale_in, size=(state_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, scale_h, size=(hidden, hidden))
        self.b2 = np.zeros(hidden)
        self.w3 = rng.normal(0.0, scale_h, size=(hidden, self.n_actions))
        self.b3 = np.zeros(self.n_actions)
        self.target = self._weights()
        self.buffer = ReplayBuffer(evaluation.dqn_replay_capacity, state_dim)
        self.learning_rate = evaluation.dqn_learning_rate
        self.batch_size = evaluation.dqn_batch_size
        self.target_refresh = evaluation.dqn_target_refresh
        self.epsilon_start = evaluation.dqn_epsilon_start
        self.epsilon_end = evaluation.dqn_epsilon_end
        self.total_steps = 0
        self.horizon = horizon

    def _weights(self):
        return (
            self.w1.copy(), self.b1.copy(), self.w2.copy(),
            self.b2.copy(), self.w3.copy(), self.b3.copy(),
        )

    @staticmethod
    def _forward(state: np.ndarray, weights) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        w1, b1, w2, b2, w3, b3 = weights
        h1 = np.maximum(state @ w1 + b1, 0.0)
        h2 = np.maximum(h1 @ w2 + b2, 0.0)
        return h1, h2, h2 @ w3 + b3

    def q_values(self, state: np.ndarray) -> np.ndarray:
        _, _, q = self._forward(np.atleast_2d(state), self._weights())
        return q[0]

    @property
    def epsilon(self) -> float:
        progress = min(self.total_steps / max(self.horizon, 1), 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select(self, state: np.ndarray) -> int:
        self.total_steps += 1
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_values(state)))

    def update(self, state: np.ndarray, action: int, reward: float) -> None:
        self.buffer.add(state, action, reward)
        if self.buffer.size < self.batch_size:
            return
        states, actions, rewards = self.buffer.sample(self.batch_size, self.rng)
        h1, h2, q = self._forward(states, self._weights())
        rows = np.arange(len(states))
        error = np.zeros_like(q)
        error[rows, actions] = q[rows, actions] - rewards
        error /= len(states)

        grad_w3 = h2.T @ error
        grad_b3 = error.sum(axis=0)
        delta2 = (error @ self.w3.T) * (h2 > 0)
        grad_w2 = h1.T @ delta2
        grad_b2 = delta2.sum(axis=0)
        delta1 = (delta2 @ self.w2.T) * (h1 > 0)
        grad_w1 = states.T @ delta1
        grad_b1 = delta1.sum(axis=0)

        self.w3 -= self.learning_rate * grad_w3
        self.b3 -= self.learning_rate * grad_b3
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1

        if self.total_steps % self.target_refresh == 0:
            self.target = self._weights()
