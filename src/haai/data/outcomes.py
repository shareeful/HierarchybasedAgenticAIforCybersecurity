from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger

LOGGER = get_logger()


@dataclass
class RecordedOutcomes:
    log: pd.DataFrame
    actions: Tuple[str, ...]
    reward_column: str

    def __len__(self) -> int:
        return len(self.log)

    def action_counts(self) -> pd.Series:
        return self.log["action"].value_counts()

    def mean_reward_by_action(self) -> pd.Series:
        return self.log.groupby("action")[self.reward_column].mean()


def composite_reward(config: Config, log: pd.DataFrame, profile: str | None = None) -> np.ndarray:
    if profile is None:
        weights = (
            config.reward.alpha_exposure,
            config.reward.alpha_exploitation,
            config.reward.alpha_analyst,
            config.reward.beta_disruption,
        )
    else:
        weights = config.reward.profiles[profile]
    alpha_exposure, alpha_exploitation, alpha_analyst, beta_disruption = weights
    residual = log["residual_exposure"].to_numpy(dtype=float)
    exploited = log["exploited_after"].to_numpy(dtype=float)
    accepted = log["analyst_accepted"].to_numpy(dtype=float)
    disruption = log["disruption_observed"].to_numpy(dtype=float)
    return (
        alpha_exposure * (1.0 - residual)
        + alpha_exploitation * (1.0 - exploited)
        + alpha_analyst * accepted
        - beta_disruption * disruption
    )


def build_recorded_outcomes(
    config: Config, log: pd.DataFrame, pairs: pd.DataFrame | None = None
) -> RecordedOutcomes:
    frame = log.copy()
    known = set(config.bandit.actions)
    unknown = sorted(set(frame["action"]) - known)
    if unknown:
        raise ValueError(
            f"remediation_outcomes.csv contains actions outside the configured set {known}: {unknown}"
        )
    frame["reward"] = composite_reward(config, frame)
    if pairs is not None:
        context = pairs.drop_duplicates(subset=["cve_id", "asset_id"])
        frame = frame.merge(context, on=["cve_id", "asset_id"], how="inner")
        if len(frame) == 0:
            raise ValueError(
                "the remediation ledger does not intersect the vulnerability-asset pairs derived "
                "from the feeds and the software inventory"
            )
    LOGGER.info(
        "recorded outcomes: %d decisions over %d actions, mean reward %.3f",
        len(frame),
        frame["action"].nunique(),
        float(frame["reward"].mean()),
    )
    return RecordedOutcomes(
        log=frame.reset_index(drop=True), actions=config.bandit.actions, reward_column="reward"
    )


def stratum_keys(config: Config, frame: pd.DataFrame) -> np.ndarray:
    buckets = np.asarray(config.bandit.criticality_buckets, dtype=float)
    criticality = np.digitize(frame["criticality"].to_numpy(dtype=float), buckets)
    quartile = frame["epss_quartile"].to_numpy(dtype=int) if "epss_quartile" in frame else np.zeros(len(frame), dtype=int)
    zone = frame["zone"].astype(str).to_numpy()
    return np.array([f"{z}|{c}|{q}" for z, c, q in zip(zone, criticality, quartile)])


def empirical_best_actions(
    outcomes: RecordedOutcomes, strata: np.ndarray, minimum_support: int
) -> Dict[str, str]:
    frame = outcomes.log.assign(stratum=strata)
    table = frame.groupby(["stratum", "action"])["reward"].agg(["mean", "size"])
    best: Dict[str, str] = {}
    for stratum, group in table.groupby(level=0):
        supported = group[group["size"] >= minimum_support]
        pool = supported if len(supported) else group
        best[str(stratum)] = str(pool["mean"].idxmax()[1])
    return best


@dataclass
class ReplayStep:
    index: int
    context: pd.Series
    stratum: str
    logged_action: str
    reward: float
    best_action: str
    best_reward: float


class ReplayEvaluator:
    def __init__(self, config: Config, outcomes: RecordedOutcomes, order: np.ndarray | None = None):
        self.config = config
        self.outcomes = outcomes
        self.strata = stratum_keys(config, outcomes.log)
        self.best = empirical_best_actions(outcomes, self.strata, config.bandit.minimum_stratum_support)
        frame = outcomes.log.assign(stratum=self.strata)
        means = frame.groupby(["stratum", "action"])["reward"].mean()
        self.stratum_action_reward = {
            (str(stratum), str(action)): float(value) for (stratum, action), value in means.items()
        }
        self.order = np.arange(len(outcomes.log)) if order is None else np.asarray(order)

    def best_reward(self, stratum: str) -> float:
        action = self.best.get(stratum)
        if action is None:
            return 0.0
        return self.stratum_action_reward.get((stratum, action), 0.0)

    def steps(self) -> Iterator[ReplayStep]:
        log = self.outcomes.log
        for position in self.order:
            stratum = str(self.strata[position])
            yield ReplayStep(
                index=int(position),
                context=log.iloc[position],
                stratum=stratum,
                logged_action=str(log["action"].iat[position]),
                reward=float(log["reward"].iat[position]),
                best_action=self.best.get(stratum, str(log["action"].iat[position])),
                best_reward=self.best_reward(stratum),
            )

    def coverage(self) -> pd.DataFrame:
        frame = self.outcomes.log.assign(stratum=self.strata)
        counts = frame.groupby("stratum")["action"].nunique()
        return pd.DataFrame(
            {
                "strata": [len(counts)],
                "strata_with_all_actions": [int((counts == len(self.outcomes.actions)).sum())],
                "median_actions_per_stratum": [float(counts.median())],
                "decisions": [len(frame)],
            }
        )


def reward_profile_table(config: Config, outcomes: RecordedOutcomes) -> pd.DataFrame:
    rows = []
    for profile in config.reward.profiles:
        reward = composite_reward(config, outcomes.log, profile)
        frame = outcomes.log.assign(profile_reward=reward)
        ranking = frame.groupby("action")["profile_reward"].mean().sort_values(ascending=False)
        rows.append(
            {
                "profile": profile,
                "weights": str(config.reward.profiles[profile]),
                "mean_reward": round(float(reward.mean()), 4),
                "best_action": str(ranking.index[0]),
                "best_action_reward": round(float(ranking.iloc[0]), 4),
                "action_ranking": " > ".join(str(name) for name in ranking.index),
            }
        )
    return pd.DataFrame(rows)
