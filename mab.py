import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class ThompsonSamplingMAB:
    n_arms: int
    arm_labels: List[str]
    alpha: np.ndarray = field(default_factory=lambda: None)
    beta: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)

    def select_arm(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        binary_reward = 1 if reward >= 0.5 else 0
        self.alpha[arm] += binary_reward
        self.beta[arm] += (1 - binary_reward)

    def get_confidence(self, arm: int) -> float:
        total = self.alpha[arm] + self.beta[arm]
        if total <= 2:
            return 0.5
        mean = self.alpha[arm] / total
        variance = (self.alpha[arm] * self.beta[arm]) / (total ** 2 * (total + 1))
        return float(np.clip(1.0 - np.sqrt(variance) * 4, 0.1, 1.0))

    def get_state(self) -> Dict:
        return {
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "arm_labels": self.arm_labels
        }

    def load_state(self, state: Dict):
        self.alpha = np.array(state["alpha"])
        self.beta = np.array(state["beta"])


@dataclass
class DiscountedUCBMAB:
    n_arms: int
    arm_labels: List[str]
    gamma: float = 0.85
    sum_rewards: np.ndarray = field(default_factory=lambda: None)
    sum_weights: np.ndarray = field(default_factory=lambda: None)
    t: int = 0

    def __post_init__(self):
        self.sum_rewards = np.zeros(self.n_arms)
        self.sum_weights = np.ones(self.n_arms) * 0.1

    def select_arm(self) -> int:
        self.t += 1
        if self.t <= self.n_arms:
            return self.t - 1
        means = self.sum_rewards / np.maximum(self.sum_weights, 1e-9)
        log_term = np.log(max(self.t, 1))
        exploration = np.sqrt(2 * log_term / np.maximum(self.sum_weights, 1e-9))
        ucb_values = means + exploration
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.sum_rewards = self.gamma * self.sum_rewards
        self.sum_weights = self.gamma * self.sum_weights
        self.sum_rewards[arm] += reward
        self.sum_weights[arm] += 1.0

    def get_confidence(self, arm: int) -> float:
        if self.sum_weights[arm] < 1:
            return 0.5
        mean = self.sum_rewards[arm] / self.sum_weights[arm]
        log_term = np.log(max(self.t, 1))
        uncertainty = np.sqrt(2 * log_term / max(self.sum_weights[arm], 1))
        confidence = float(np.clip(mean / (mean + uncertainty + 1e-9), 0.1, 1.0))
        return confidence

    def get_state(self) -> Dict:
        return {
            "sum_rewards": self.sum_rewards.tolist(),
            "sum_weights": self.sum_weights.tolist(),
            "t": self.t,
            "gamma": self.gamma,
            "arm_labels": self.arm_labels
        }

    def load_state(self, state: Dict):
        self.sum_rewards = np.array(state["sum_rewards"])
        self.sum_weights = np.array(state["sum_weights"])
        self.t = state["t"]
        self.gamma = state["gamma"]


@dataclass
class SlidingWindowUCBMAB:
    n_arms: int
    arm_labels: List[str]
    tau: int = 80
    window: List[Dict] = field(default_factory=list)
    t: int = 0

    def _compute_window_stats(self):
        counts = np.zeros(self.n_arms)
        rewards = np.zeros(self.n_arms)
        for entry in self.window:
            arm = entry["arm"]
            counts[arm] += 1
            rewards[arm] += entry["reward"]
        return counts, rewards

    def select_arm(self) -> int:
        self.t += 1
        counts, rewards = self._compute_window_stats()
        if np.any(counts == 0):
            unplayed = np.where(counts == 0)[0]
            return int(unplayed[0])
        means = rewards / np.maximum(counts, 1e-9)
        log_tau = np.log(max(min(self.t, self.tau), 1))
        exploration = np.sqrt(2 * log_tau / np.maximum(counts, 1e-9))
        ucb_values = means + exploration
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float):
        self.window.append({"arm": arm, "reward": reward})
        if len(self.window) > self.tau:
            self.window.pop(0)

    def get_confidence(self, arm: int) -> float:
        counts, rewards = self._compute_window_stats()
        if counts[arm] < 1:
            return 0.5
        mean = rewards[arm] / counts[arm]
        log_tau = np.log(max(min(self.t, self.tau), 1))
        uncertainty = np.sqrt(2 * log_tau / max(counts[arm], 1))
        confidence = float(np.clip(mean / (mean + uncertainty + 1e-9), 0.1, 1.0))
        return confidence

    def get_state(self) -> Dict:
        return {
            "window": self.window,
            "t": self.t,
            "tau": self.tau,
            "arm_labels": self.arm_labels
        }

    def load_state(self, state: Dict):
        self.window = state["window"]
        self.t = state["t"]
        self.tau = state["tau"]


def build_vulnerability_arms() -> List[str]:
    cwe_types = ["CWE-89", "CWE-79", "CWE-22", "CWE-287", "CWE-119", "CWE-200", "CWE-502", "OTHER"]
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    arms = []
    for cwe in cwe_types:
        for q in quartiles:
            arms.append(f"{cwe}_{q}")
    return arms


def get_epss_quartile(epss: float) -> str:
    if epss < 0.25:
        return "Q1"
    elif epss < 0.50:
        return "Q2"
    elif epss < 0.75:
        return "Q3"
    return "Q4"


def get_vulnerability_arm(cwe: str, epss: float, arm_labels: List[str]) -> int:
    quartile = get_epss_quartile(epss)
    matched_cwe = "OTHER"
    known_cwes = ["CWE-89", "CWE-79", "CWE-22", "CWE-287", "CWE-119", "CWE-200", "CWE-502"]
    for k in known_cwes:
        if k in cwe:
            matched_cwe = k
            break
    target = f"{matched_cwe}_{quartile}"
    if target in arm_labels:
        return arm_labels.index(target)
    return 0


def build_contextual_arms() -> List[str]:
    zones = ["DMZ", "cloud", "internal", "isolated"]
    criticalities = ["critical", "high", "medium", "low"]
    arms = []
    for z in zones:
        for c in criticalities:
            arms.append(f"{z}_{c}")
    return arms


def get_contextual_arm(zone: str, criticality: str, arm_labels: List[str]) -> int:
    target = f"{zone}_{criticality}"
    if target in arm_labels:
        return arm_labels.index(target)
    return 0


def build_supervisor_arms() -> List[str]:
    return ["patch", "compensate", "defer", "accept"]


def get_supervisor_arm(action: str, arm_labels: List[str]) -> int:
    if action in arm_labels:
        return arm_labels.index(action)
    return 0
