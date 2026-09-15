from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..bandits.dqn import DeepQNetworkPolicy
from ..bandits.thompson import ThompsonSampling
from ..bandits.ucb import DiscountedUCB, SlidingWindowUCB, StaticPolicy
from ..config import Config
from ..data.outcomes import RecordedOutcomes, ReplayEvaluator, reward_profile_table
from ..logging_utils import get_logger
from ..metrics import cumulative_regret, recovery_cycles
from ..seeds import SeedBook
from ..stats import format_mean_std

LOGGER = get_logger()

POLICY_ORDER: Tuple[str, ...] = (
    "static",
    "thompson",
    "discounted_ucb",
    "sliding_window_ucb",
    "dqn",
)

POLICY_DISPLAY: Dict[str, str] = {
    "static": "Static policy",
    "thompson": "Thompson Sampling",
    "discounted_ucb": "Discounted UCB",
    "sliding_window_ucb": "Sliding Window UCB",
    "dqn": "Deep Q-Network",
}


@dataclass
class PolicyTrace:
    accepted: int
    reward: np.ndarray
    best_reward: np.ndarray
    matched_best: np.ndarray
    positions: np.ndarray

    def regret(self) -> np.ndarray:
        return cumulative_regret(self.best_reward, self.reward)


@dataclass
class Experiment2Result:
    table11: pd.DataFrame
    table12: pd.DataFrame
    tau_sensitivity: pd.DataFrame
    gamma_sensitivity: pd.DataFrame
    regret_curves: Dict[str, np.ndarray]
    accuracy_curves: Dict[str, np.ndarray]
    coverage: pd.DataFrame
    change_point: int
    per_run: pd.DataFrame


def _state_vector(context: pd.Series) -> np.ndarray:
    return np.asarray(
        [
            float(context.get("cvss", 0.0)) / 10.0,
            float(context.get("epss", 0.0)),
            float(context.get("criticality", 0.0)),
            float(context.get("zone_reach", 0.0)),
            float(context.get("control_coverage", 0.0)),
            float(context.get("sw_match", 0.0)),
        ],
        dtype=float,
    )


def _make_policy(name: str, config: Config, rng: np.random.Generator, tau: int, gamma: float, horizon: int):
    actions = list(config.bandit.actions)
    if name == "thompson":
        return ThompsonSampling(actions, rng, config.bandit.thompson_prior_alpha, config.bandit.thompson_prior_beta)
    if name == "discounted_ucb":
        return DiscountedUCB(actions, rng, gamma, config.bandit.exploration_guard_epsilon)
    if name == "sliding_window_ucb":
        return SlidingWindowUCB(actions, rng, tau)
    if name == "static":
        return StaticPolicy(actions, rng, actions[0])
    if name == "dqn":
        return DeepQNetworkPolicy(config, 6, rng, horizon)
    raise ValueError(f"unknown policy: {name}")


def replay_policy(
    name: str,
    config: Config,
    evaluator: ReplayEvaluator,
    rng: np.random.Generator,
    tau: int | None = None,
    gamma: float | None = None,
) -> PolicyTrace:
    horizon = len(evaluator.outcomes.log)
    policy = _make_policy(
        name,
        config,
        rng,
        tau if tau is not None else config.bandit.sliding_window_tau,
        gamma if gamma is not None else config.bandit.discount_gamma,
        horizon,
    )
    actions = list(config.bandit.actions)
    rewards: List[float] = []
    best: List[float] = []
    matched: List[bool] = []
    positions: List[int] = []
    for step in evaluator.steps():
        if name == "dqn":
            state = _state_vector(step.context)
            index = policy.select(state)
            proposed = actions[index]
        else:
            proposed = policy.select()
        if proposed != step.logged_action:
            continue
        if name == "dqn":
            policy.update(_state_vector(step.context), actions.index(proposed), step.reward)
        else:
            policy.update(proposed, step.reward)
        rewards.append(step.reward)
        best.append(step.best_reward)
        matched.append(proposed == step.best_action)
        positions.append(step.index)
    return PolicyTrace(
        accepted=len(rewards),
        reward=np.asarray(rewards, dtype=float),
        best_reward=np.asarray(best, dtype=float),
        matched_best=np.asarray(matched, dtype=bool),
        positions=np.asarray(positions, dtype=int),
    )


def _smooth(values: np.ndarray, window: int = 25) -> np.ndarray:
    if len(values) < window or window <= 1:
        return np.asarray(values, dtype=float)
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=float), kernel, mode="same")


def run_experiment2(
    config: Config, outcomes: RecordedOutcomes, n_runs: int | None = None
) -> Experiment2Result:
    n_runs = n_runs or config.evaluation.n_runs
    book = SeedBook(config.evaluation.master_seed)
    base = ReplayEvaluator(config, outcomes)
    coverage = base.coverage()
    total = len(outcomes.log)
    change_point = int(round(config.bandit.change_point_quantile * total))

    rows: List[dict] = []
    regret_curves: Dict[str, np.ndarray] = {}
    accuracy_curves: Dict[str, np.ndarray] = {}

    for name in POLICY_ORDER:
        collected_regret: List[np.ndarray] = []
        collected_accuracy: List[np.ndarray] = []
        for index in range(n_runs):
            rng = book.stream(f"experiment2:{name}", index)
            order = rng.permutation(total) if index else np.arange(total)
            evaluator = ReplayEvaluator(config, outcomes, order)
            trace = replay_policy(name, config, evaluator, rng)
            if trace.accepted < config.bandit.minimum_replay_events:
                LOGGER.warning(
                    "policy %s accepted only %d replay events in run %d; the recorded ledger "
                    "does not cover this policy densely enough to score it",
                    name,
                    trace.accepted,
                    index,
                )
                continue
            split = int(round(config.bandit.change_point_quantile * trace.accepted))
            accuracy = trace.matched_best.astype(float)
            rows.append(
                {
                    "policy": name,
                    "run": index,
                    "accepted": trace.accepted,
                    "mean_reward": float(trace.reward.mean()),
                    "final_regret": float(trace.regret()[-1]),
                    "accuracy_pre": float(accuracy[:split].mean()) if split else float("nan"),
                    "accuracy_post": float(accuracy[split:].mean()) if split < trace.accepted else float("nan"),
                    "recovery": int(recovery_cycles(_smooth(accuracy), split)),
                }
            )
            collected_regret.append(trace.regret())
            collected_accuracy.append(_smooth(accuracy))
        if collected_regret:
            length = min(len(item) for item in collected_regret)
            regret_curves[name] = np.mean([item[:length] for item in collected_regret], axis=0)
            accuracy_curves[name] = np.mean([item[:length] for item in collected_accuracy], axis=0)
        LOGGER.info("replay complete for policy %s", name)

    per_run = pd.DataFrame(rows)
    if not len(per_run):
        raise ValueError(
            "offline replay accepted no events for any policy; the remediation ledger is too "
            "small or too concentrated on a single action to evaluate a bandit against it"
        )
    table_rows: List[dict] = []
    for name in POLICY_ORDER:
        subset = per_run[per_run["policy"] == name]
        if not len(subset):
            continue
        table_rows.append(
            {
                "Policy": POLICY_DISPLAY[name],
                "raw_key": name,
                "accepted_events": int(subset["accepted"].mean()),
                "mean_reward": format_mean_std(subset["mean_reward"]),
                "final_regret": format_mean_std(subset["final_regret"], 2),
                "accuracy_pre": format_mean_std(subset["accuracy_pre"]),
                "accuracy_post": format_mean_std(subset["accuracy_post"]),
                "recovery_cycles": format_mean_std(subset["recovery"], 1),
                "recovery_mean": round(float(subset["recovery"].mean()), 1),
            }
        )
    table11 = pd.DataFrame(table_rows)

    tau_rows: List[dict] = []
    for tau in config.bandit.sliding_window_tau_grid:
        values = []
        for index in range(max(n_runs // 4, 1)):
            rng = book.stream(f"experiment2:tau:{tau}", index)
            evaluator = ReplayEvaluator(config, outcomes, rng.permutation(total))
            trace = replay_policy("sliding_window_ucb", config, evaluator, rng, tau=tau)
            if trace.accepted:
                values.append(float(trace.matched_best.mean()))
        tau_rows.append(
            {
                "tau": tau,
                "accuracy_mean": round(float(np.mean(values)), 4) if values else float("nan"),
                "accuracy_std": round(float(np.std(values)), 4) if values else float("nan"),
            }
        )
    tau_sensitivity = pd.DataFrame(tau_rows)

    gamma_rows: List[dict] = []
    for gamma in config.bandit.discount_gamma_grid:
        values = []
        for index in range(max(n_runs // 4, 1)):
            rng = book.stream(f"experiment2:gamma:{gamma}", index)
            evaluator = ReplayEvaluator(config, outcomes, rng.permutation(total))
            trace = replay_policy("discounted_ucb", config, evaluator, rng, gamma=gamma)
            if trace.accepted:
                values.append(float(trace.matched_best.mean()))
        gamma_rows.append(
            {
                "gamma": gamma,
                "accuracy_mean": round(float(np.mean(values)), 4) if values else float("nan"),
                "accuracy_std": round(float(np.std(values)), 4) if values else float("nan"),
            }
        )
    gamma_sensitivity = pd.DataFrame(gamma_rows)

    return Experiment2Result(
        table11=table11,
        table12=reward_profile_table(config, outcomes),
        tau_sensitivity=tau_sensitivity,
        gamma_sensitivity=gamma_sensitivity,
        regret_curves=regret_curves,
        accuracy_curves=accuracy_curves,
        coverage=coverage,
        change_point=change_point,
        per_run=per_run,
    )
