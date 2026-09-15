from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..agents.confidence import (
    ConfidenceCalibrator,
    arm_reliability,
    contextual_arm,
    vulnerability_arm,
)
from ..agents.contextual_agent import ContextualAgent
from ..agents.llm import AgentModels, CallLog
from ..agents.supervisor_agent import SupervisorAgent, SupervisorDecision
from ..agents.schemas import ParseTelemetry
from ..agents.vulnerability_agent import AgentOutput, VulnerabilityAgent
from ..bandits.thompson import ThompsonSampling
from ..bandits.ucb import DiscountedUCB
from ..config import Config
from ..integrity.detector import IsolationForestVerifier
from ..logging_utils import get_logger

LOGGER = get_logger()


@dataclass
class AssessmentTrace:
    pairs: pd.DataFrame
    unique: pd.DataFrame
    vulnerability_unique: AgentOutput
    vulnerability: AgentOutput
    contextual: AgentOutput
    confidence_va: np.ndarray
    confidence_ca: np.ndarray
    flag_va: np.ndarray
    flag_ca: np.ndarray
    decision: SupervisorDecision
    joint_risk: np.ndarray
    flat_risk: np.ndarray

    def frame(self) -> pd.DataFrame:
        return self.pairs.assign(
            risk_score=self.vulnerability.score,
            exposure_score=self.contextual.score,
            regression_risk=self.contextual.secondary,
            confidence_va=self.confidence_va,
            confidence_ca=self.confidence_ca,
            flag_va=self.flag_va,
            flag_ca=self.flag_ca,
            joint_risk=self.joint_risk,
            flat_risk=self.flat_risk,
            action=self.decision.action,
            action_utility=self.decision.action_utility,
            control_type=self.decision.control_type,
            decision_source=self.decision.source,
        )


class HierarchicalPipeline:
    def __init__(
        self,
        config: Config,
        models: AgentModels,
        rng: np.random.Generator,
        telemetry: ParseTelemetry | None = None,
    ):
        self.config = config
        self.models = models
        self.rng = rng
        self.telemetry = telemetry or ParseTelemetry()
        self.log: CallLog = models.log
        self.vulnerability_agent = VulnerabilityAgent(
            config, models.vulnerability, self.telemetry, self.log
        )
        self.contextual_agent = ContextualAgent(config, models.contextual, self.telemetry, self.log)
        self.supervisor_agent = SupervisorAgent(config, models.supervisor, self.telemetry, self.log)
        self.thompson = ThompsonSampling(
            [], rng, config.bandit.thompson_prior_alpha, config.bandit.thompson_prior_beta
        )
        self.discounted = DiscountedUCB(
            [], rng, config.bandit.discount_gamma, config.bandit.exploration_guard_epsilon
        )
        self.calibrator_va = ConfidenceCalibrator("vulnerability_agent")
        self.calibrator_ca = ConfidenceCalibrator("contextual_agent")
        self.verifier_va = IsolationForestVerifier(config, "vulnerability_agent", rng)
        self.verifier_ca = IsolationForestVerifier(config, "contextual_agent", rng)
        self.cache: Dict[str, AgentOutput] = {}
        self.fitted = False

    def _vulnerability_arms(self, frame: pd.DataFrame) -> List[Tuple[str, int]]:
        return [
            vulnerability_arm(cwe, quartile)
            for cwe, quartile in zip(frame["cwe"].astype(str), frame["epss_quartile"].astype(int))
        ]

    def _contextual_arms(self, pairs: pd.DataFrame) -> List[Tuple[str, int]]:
        buckets = self.config.bandit.criticality_buckets
        return [
            contextual_arm(zone, criticality, buckets)
            for zone, criticality in zip(
                pairs["zone"].astype(str), pairs["criticality"].astype(float)
            )
        ]

    def _reliability(self, policy, arms: Sequence) -> np.ndarray:
        return np.asarray([arm_reliability(policy, arm) for arm in arms], dtype=float)

    def assess_vulnerabilities(
        self, frame: pd.DataFrame, need_attention: bool = False, key: str | None = None
    ) -> AgentOutput:
        if key is not None and key in self.cache:
            return self.cache[key]
        output = self.vulnerability_agent.assess(frame, need_attention=need_attention)
        if key is not None:
            self.cache[key] = output
        return output

    def integrity_vectors_va(self, frame: pd.DataFrame, output: AgentOutput) -> np.ndarray:
        return np.column_stack(
            [
                frame["cvss"].to_numpy(dtype=float),
                frame["epss"].to_numpy(dtype=float),
                frame["poc_available"].to_numpy(dtype=float),
                output.score,
                output.reported_confidence,
            ]
        )

    def integrity_vectors_ca(self, pairs: pd.DataFrame, output: AgentOutput) -> np.ndarray:
        return np.column_stack(
            [
                pairs["criticality"].to_numpy(dtype=float),
                pairs["zone_reach"].to_numpy(dtype=float),
                pairs["control_coverage"].to_numpy(dtype=float),
                output.score,
                output.secondary,
                output.reported_confidence,
            ]
        )

    def fit(self, tuning_pairs: pd.DataFrame) -> "HierarchicalPipeline":
        unique = tuning_pairs.drop_duplicates(subset="cve_id").reset_index(drop=True)
        vulnerability = self.assess_vulnerabilities(unique, key="tuning_vulnerability")
        risk_by_cve = dict(zip(unique["cve_id"], vulnerability.score))
        risk_scores = tuning_pairs["cve_id"].map(risk_by_cve).to_numpy(dtype=float)
        contextual = self.contextual_agent.assess(tuning_pairs, risk_scores)

        va_arms = self._vulnerability_arms(unique)
        ca_arms = self._contextual_arms(tuning_pairs)
        correct_va = (
            np.abs(vulnerability.score - unique["label"].to_numpy(dtype=float)) < 0.5
        ).astype(float)
        for arm, reward in zip(va_arms, correct_va):
            self.thompson.update(arm, float(reward))
        exposure_truth = tuning_pairs["exposure_ground_truth"].to_numpy(dtype=float)
        correct_ca = (np.abs(contextual.score - exposure_truth) < 0.25).astype(float)
        for arm, reward in zip(ca_arms, correct_ca):
            self.discounted.update(arm, float(reward))

        self.calibrator_va.fit(
            vulnerability.reported_confidence,
            vulnerability.sequence_logprob,
            self._reliability(self.thompson, va_arms),
            correct_va,
        )
        self.calibrator_ca.fit(
            contextual.reported_confidence,
            contextual.sequence_logprob,
            self._reliability(self.discounted, ca_arms),
            correct_ca,
        )

        self.verifier_va.prime(self.integrity_vectors_va(unique, vulnerability))
        self.verifier_va.calibrate(self.integrity_vectors_va(unique, vulnerability))
        self.verifier_ca.prime(self.integrity_vectors_ca(tuning_pairs, contextual))
        self.verifier_ca.calibrate(self.integrity_vectors_ca(tuning_pairs, contextual))

        labels = tuning_pairs["label"].to_numpy(dtype=float)
        confidence_va = self.calibrator_va.transform(
            np.asarray([vulnerability.reported_confidence[i] for i in range(len(unique))]),
            vulnerability.sequence_logprob,
            self._reliability(self.thompson, va_arms),
        )
        confidence_by_cve = dict(zip(unique["cve_id"], confidence_va))
        self.tuning_reference = {
            "risk_by_cve": risk_by_cve,
            "confidence_by_cve": confidence_by_cve,
            "labels": labels,
        }
        self.fitted = True
        LOGGER.info(
            "pipeline fitted on %d tuning pairs (%d unique records); VA calibration ECE %.4f, "
            "CA calibration ECE %.4f",
            len(tuning_pairs),
            len(unique),
            self.calibrator_va.report.ece_after if self.calibrator_va.report else float("nan"),
            self.calibrator_ca.report.ece_after if self.calibrator_ca.report else float("nan"),
        )
        return self

    def assess(
        self,
        pairs: pd.DataFrame,
        apply_integrity: bool = True,
        use_confidence_weighting: bool = True,
        use_mab: bool = True,
        generate_explanations: bool = False,
        need_attention: bool = False,
        key: str | None = None,
    ) -> AssessmentTrace:
        unique = pairs.drop_duplicates(subset="cve_id").reset_index(drop=True)
        vulnerability = self.assess_vulnerabilities(
            unique, need_attention=need_attention, key=key
        )
        risk_by_cve = dict(zip(unique["cve_id"], vulnerability.score))
        risk_scores = pairs["cve_id"].map(risk_by_cve).to_numpy(dtype=float)
        contextual = self.contextual_agent.assess(
            pairs, risk_scores, need_attention=need_attention
        )

        va_arms = self._vulnerability_arms(unique)
        ca_arms = self._contextual_arms(pairs)
        reliability_va = (
            self._reliability(self.thompson, va_arms) if use_mab else np.ones(len(unique))
        )
        reliability_ca = (
            self._reliability(self.discounted, ca_arms) if use_mab else np.ones(len(pairs))
        )
        confidence_va_unique = self.calibrator_va.transform(
            vulnerability.reported_confidence, vulnerability.sequence_logprob, reliability_va
        )
        confidence_ca = self.calibrator_ca.transform(
            contextual.reported_confidence, contextual.sequence_logprob, reliability_ca
        )
        confidence_map = dict(zip(unique["cve_id"], confidence_va_unique))
        confidence_va = pairs["cve_id"].map(confidence_map).to_numpy(dtype=float)
        if not use_confidence_weighting:
            confidence_va = np.ones(len(pairs))
            confidence_ca = np.ones(len(pairs))

        if apply_integrity:
            _, flag_va_unique = self.verifier_va.verify(
                self.integrity_vectors_va(unique, vulnerability), refit=False
            )
            flag_map = dict(zip(unique["cve_id"], flag_va_unique))
            flag_va = pairs["cve_id"].map(flag_map).to_numpy(dtype=bool)
            _, flag_ca = self.verifier_ca.verify(
                self.integrity_vectors_ca(pairs, contextual), refit=False
            )
        else:
            flag_va = np.zeros(len(pairs), dtype=bool)
            flag_ca = np.zeros(len(pairs), dtype=bool)

        flag_va = flag_va | pairs["cve_id"].map(
            dict(zip(unique["cve_id"], vulnerability.flags))
        ).to_numpy(dtype=bool)
        flag_ca = flag_ca | contextual.flags

        expanded = AgentOutput(
            score=risk_scores,
            secondary=np.zeros(len(pairs)),
            reported_confidence=pairs["cve_id"]
            .map(dict(zip(unique["cve_id"], vulnerability.reported_confidence)))
            .to_numpy(dtype=float),
            sequence_logprob=pairs["cve_id"]
            .map(dict(zip(unique["cve_id"], vulnerability.sequence_logprob)))
            .to_numpy(dtype=float),
            flags=flag_va,
            parsed=vulnerability.parsed,
            prompts=vulnerability.prompts,
            completions=vulnerability.completions,
            attention=vulnerability.attention,
            fields=vulnerability.fields,
        )

        decision = self.supervisor_agent.decide(
            pairs,
            expanded.score,
            contextual.score,
            contextual.secondary,
            confidence_va,
            confidence_ca,
            flag_va,
            flag_ca,
            generate_explanations=generate_explanations,
        )
        flat = 0.5 * expanded.score + 0.5 * contextual.score
        return AssessmentTrace(
            pairs=pairs.reset_index(drop=True),
            unique=unique,
            vulnerability_unique=vulnerability,
            vulnerability=expanded,
            contextual=contextual,
            confidence_va=confidence_va,
            confidence_ca=confidence_ca,
            flag_va=flag_va,
            flag_ca=flag_ca,
            decision=decision,
            joint_risk=decision.joint_risk,
            flat_risk=flat,
        )

    def update_policies(self, trace: AssessmentTrace, rewards: np.ndarray) -> None:
        unique = trace.pairs.drop_duplicates(subset="cve_id")
        va_arms = self._vulnerability_arms(unique)
        ca_arms = self._contextual_arms(trace.pairs)
        by_cve = (
            pd.DataFrame({"cve_id": trace.pairs["cve_id"], "reward": rewards})
            .groupby("cve_id")["reward"]
            .mean()
        )
        for arm, cve_id in zip(va_arms, unique["cve_id"]):
            self.thompson.update(arm, float(by_cve.get(cve_id, 0.0)))
        for arm, reward in zip(ca_arms, np.asarray(rewards, dtype=float)):
            self.discounted.update(arm, float(reward))
