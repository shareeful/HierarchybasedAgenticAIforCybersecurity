from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..agents.llm import AgentModels
from ..config import Config
from ..data.usecase import UseCase, response_times
from ..logging_utils import get_logger
from ..metrics import classification_metrics, select_threshold
from ..pipeline.features import aggregate_to_records
from ..pipeline.orchestrator import AssessmentTrace
from ..agents.supervisor_agent import select_actions
from .systems import SystemRun

LOGGER = get_logger()


@dataclass
class Experiment5Result:
    table16: pd.DataFrame
    table17: pd.DataFrame
    action_distribution: pd.DataFrame
    confusion: pd.DataFrame
    agreement: Dict[str, float]
    agreement_by_subsystem: pd.DataFrame
    operator_targets: pd.DataFrame
    matched_decisions: pd.DataFrame
    trace: AssessmentTrace


def _risk_level(value: float) -> str:
    if value >= 0.75:
        return "critical"
    if value >= 0.5:
        return "high"
    if value >= 0.25:
        return "medium"
    return "low"


def _actions_from_scores(
    config: Config, scores: np.ndarray, regression: np.ndarray
) -> np.ndarray:
    actions, _ = select_actions(config, np.asarray(scores, dtype=float), regression)
    return actions


def run_experiment5(
    config: Config,
    run: SystemRun,
    use_case: UseCase,
    use_case_pairs: pd.DataFrame,
    models: AgentModels,
) -> Experiment5Result:
    pipeline = run.pipeline
    trace = pipeline.assess(use_case_pairs, key="use_case_vulnerability")
    frame = trace.frame()

    decisions = use_case.decisions.copy()
    matched = decisions.merge(
        frame.loc[
            :,
            [
                "cve_id",
                "asset_id",
                "subsystem",
                "risk_score",
                "exposure_score",
                "regression_risk",
                "joint_risk",
                "flat_risk",
                "action",
                "control_type",
                "cvss",
                "epss",
                "cwe",
            ],
        ],
        on=["cve_id", "asset_id"],
        how="inner",
        suffixes=("_analyst", "_system"),
    )
    if len(matched) == 0:
        raise ValueError(
            "none of the recorded analyst decisions could be matched to a vulnerability-asset "
            "pair derived from the pilot inventory and the vulnerability feeds"
        )
    LOGGER.info(
        "matched %d of %d recorded analyst decisions to pipeline output",
        len(matched),
        len(decisions),
    )

    flat_actions = _actions_from_scores(
        config, matched["flat_risk"].to_numpy(dtype=float), matched["regression_risk"].to_numpy(dtype=float)
    )
    cvss_actions = _actions_from_scores(
        config,
        matched["cvss"].to_numpy(dtype=float) / 10.0,
        np.zeros(len(matched), dtype=float),
    )
    analyst = matched["action_analyst"].to_numpy(dtype=object)
    system = matched["action_system"].to_numpy(dtype=object)

    agreement = {
        "proposed": float(np.mean(system == analyst)),
        "flat_multi_agent": float(np.mean(flat_actions == analyst)),
        "static_cvss": float(np.mean(cvss_actions == analyst)),
        "matched_decisions": int(len(matched)),
        "recorded_decisions": int(len(decisions)),
    }

    confusion = pd.crosstab(
        pd.Series(analyst, name="analyst"), pd.Series(system, name="system")
    ).reindex(index=list(config.bandit.actions), columns=list(config.bandit.actions), fill_value=0)

    by_subsystem = (
        pd.DataFrame(
            {
                "subsystem": matched["subsystem"],
                "agree": (system == analyst).astype(float),
            }
        )
        .groupby("subsystem")
        .agg(decisions=("agree", "size"), agreement=("agree", "mean"))
        .reset_index()
    )

    action_distribution = (
        pd.DataFrame(
            {
                "analyst": pd.Series(analyst).value_counts(),
                "proposed": pd.Series(system).value_counts(),
                "flat_multi_agent": pd.Series(flat_actions).value_counts(),
                "static_cvss": pd.Series(cvss_actions).value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
        .reindex(list(config.bandit.actions))
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"index": "action"})
    )

    table16_rows: List[dict] = []
    unique_records = trace.unique
    for name in config.evaluation.baselines:
        model = run.fitted_baselines.get(name)
        if name == "proposed":
            records = aggregate_to_records(use_case_pairs.assign(score=trace.joint_risk), "score")
        elif name == "flat_multi_agent":
            records = aggregate_to_records(use_case_pairs.assign(score=trace.flat_risk), "score")
        elif name == "single_agent" and model is not None:
            records = aggregate_to_records(
                use_case_pairs.assign(score=model.score(use_case_pairs)), "score"
            )
        elif model is not None:
            records = unique_records.loc[:, ["cve_id", "label"]].assign(
                score=model.score(unique_records)
            )
        else:
            continue
        threshold = select_threshold(
            records["label"].to_numpy(dtype=int), records["score"].to_numpy(dtype=float)
        )
        measured = classification_metrics(
            records["label"].to_numpy(dtype=int), records["score"].to_numpy(dtype=float), threshold
        )
        table16_rows.append(
            {
                "system": config.evaluation.baseline_display[name],
                "use_case_f1": round(measured.f1, 4),
                "use_case_auc": round(measured.auc, 4),
                "use_case_precision": round(measured.precision, 4),
                "use_case_recall": round(measured.recall, 4),
                "records": measured.support,
            }
        )
    table16 = pd.DataFrame(table16_rows)

    worked = frame.sort_values("joint_risk", ascending=False).head(10)
    table17 = pd.DataFrame(
        {
            "Subsystem": worked["subsystem"].to_numpy(),
            "CVE": worked["cve_id"].to_numpy(),
            "Weakness class": worked["cwe"].to_numpy(),
            "Urgency": np.round(worked["risk_score"].to_numpy(dtype=float), 3),
            "Exposure": np.round(worked["exposure_score"].to_numpy(dtype=float), 3),
            "Joint risk": np.round(worked["joint_risk"].to_numpy(dtype=float), 3),
            "Risk level": [_risk_level(value) for value in worked["joint_risk"]],
            "Recommended action": worked["action"].to_numpy(),
            "Control": worked["control_type"].to_numpy(),
        }
    )

    targets = use_case.operator_targets.copy()
    if "achieved" in targets.columns:
        targets["achieved"] = pd.to_numeric(targets["achieved"], errors="coerce")
        targets["target_met"] = targets["achieved"] >= targets["target"]
        targets["source"] = "recorded post-deployment measurement"
    else:
        targets["achieved"] = np.nan
        targets["target_met"] = pd.NA
        targets["source"] = (
            "not measurable from the supplied records: operator_targets.csv carries no "
            "'achieved' column, and a post-deployment measurement cannot be derived from "
            "pre-deployment data"
        )
    timings = response_times(use_case)
    if len(timings):
        targets = targets.assign(
            recorded_median_acknowledge_seconds=float(timings["acknowledge_seconds"].median())
        )

    return Experiment5Result(
        table16=table16,
        table17=table17,
        action_distribution=action_distribution,
        confusion=confusion,
        agreement=agreement,
        agreement_by_subsystem=by_subsystem,
        operator_targets=targets,
        matched_decisions=matched,
        trace=trace,
    )
