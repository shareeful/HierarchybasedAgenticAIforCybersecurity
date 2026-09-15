from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .llm import CallLog, Completion, LanguageModel
from .parsing import ParsedOutput, parse_completion
from .prompts import RenderedPrompt, render_supervisor_prompt
from .schemas import ParseTelemetry, supervisor_schema

LOGGER = get_logger()


def joint_risk_eq23(
    risk: np.ndarray, exposure: np.ndarray, confidence_va: np.ndarray, confidence_ca: np.ndarray
) -> np.ndarray:
    risk = np.asarray(risk, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    cva = np.clip(np.asarray(confidence_va, dtype=float), 1e-6, None)
    cca = np.clip(np.asarray(confidence_ca, dtype=float), 1e-6, None)
    return (cva * risk + cca * exposure) / (cva + cca)


def action_utility(
    config: Config, joint: np.ndarray, regression: np.ndarray, action: str
) -> np.ndarray:
    eta = config.bandit.risk_removal_fraction[action]
    kappa = config.bandit.disruption_cost_factor[action] * np.asarray(regression, dtype=float)
    return config.bandit.omega_risk * eta * np.asarray(joint, dtype=float) - config.bandit.omega_cost * kappa


def utility_matrix(config: Config, joint: np.ndarray, regression: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [action_utility(config, joint, regression, action) for action in config.bandit.actions]
    )


def recommend_control(config: Config, cwe: Sequence[str]) -> List[str]:
    access = set(config.controls.access_control_cwes)
    validation = set(config.controls.input_validation_cwes)
    boundary = set(config.controls.network_boundary_cwes)
    recommendations: List[str] = []
    for value in cwe:
        token = str(value)
        if token in access:
            recommendations.append("access_control")
        elif token in validation:
            recommendations.append("input_validation")
        elif token in boundary:
            recommendations.append("network_segmentation")
        else:
            recommendations.append("patch_management")
    return recommendations


@dataclass
class TrustDecision:
    joint_risk: np.ndarray
    source: np.ndarray
    escalated: np.ndarray


def apply_trust_policy(
    joint: np.ndarray,
    risk: np.ndarray,
    exposure: np.ndarray,
    flag_va: np.ndarray,
    flag_ca: np.ndarray,
) -> TrustDecision:
    flag_va = np.asarray(flag_va, dtype=bool)
    flag_ca = np.asarray(flag_ca, dtype=bool)
    both = flag_va & flag_ca
    only_va = flag_va & ~flag_ca
    only_ca = flag_ca & ~flag_va
    resolved = np.asarray(joint, dtype=float).copy()
    resolved[only_va] = np.asarray(exposure, dtype=float)[only_va]
    resolved[only_ca] = np.asarray(risk, dtype=float)[only_ca]
    source = np.full(len(resolved), "both", dtype=object)
    source[only_va] = "contextual_only"
    source[only_ca] = "vulnerability_only"
    source[both] = "human_review"
    return TrustDecision(joint_risk=resolved, source=source, escalated=both)


def select_actions(
    config: Config,
    joint: np.ndarray,
    regression: np.ndarray,
    cycles_open: np.ndarray | None = None,
    bias: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    matrix = utility_matrix(config, joint, regression)
    if bias is not None:
        matrix = matrix + np.asarray(bias, dtype=float)
    chosen = np.argmax(matrix, axis=1)
    actions = np.asarray(config.bandit.actions, dtype=object)[chosen]
    if cycles_open is not None:
        overdue = np.asarray(cycles_open) >= config.bandit.escalation_window
        actions = np.where(overdue, "patch", actions)
    utilities = matrix[np.arange(len(chosen)), chosen]
    return actions, utilities


@dataclass
class SupervisorDecision:
    joint_risk: np.ndarray
    action: np.ndarray
    action_utility: np.ndarray
    reference_action: np.ndarray
    control_type: np.ndarray
    source: np.ndarray
    escalated: np.ndarray
    parsed: List[ParsedOutput]
    completions: List[Completion]
    adherence: float
    technical_explanation: List[str]
    plain_explanation: List[str]


class SupervisorAgent:
    name = "supervisor_agent"

    def __init__(self, config: Config, model: LanguageModel, telemetry: ParseTelemetry, log: CallLog):
        self.config = config
        self.model = model
        self.telemetry = telemetry
        self.log = log
        self.schema = supervisor_schema(config.bandit.actions, config.controls.taxonomy)

    def decide(
        self,
        pairs: pd.DataFrame,
        risk: np.ndarray,
        exposure: np.ndarray,
        regression: np.ndarray,
        confidence_va: np.ndarray,
        confidence_ca: np.ndarray,
        flag_va: np.ndarray,
        flag_ca: np.ndarray,
        vulnerability_evidence: Sequence[Dict[str, object]] | None = None,
        contextual_evidence: Sequence[Dict[str, object]] | None = None,
        cycles_open: np.ndarray | None = None,
        generate_explanations: bool = True,
        batch_size: int | None = None,
    ) -> SupervisorDecision:
        joint = joint_risk_eq23(risk, exposure, confidence_va, confidence_ca)
        trust = apply_trust_policy(joint, risk, exposure, flag_va, flag_ca)
        reference_action, utilities = select_actions(
            self.config, trust.joint_risk, regression, cycles_open
        )
        reference_action = np.where(trust.escalated, "defer", reference_action)
        controls = np.asarray(recommend_control(self.config, pairs["cwe"].astype(str)), dtype=object)

        parsed_all: List[ParsedOutput] = []
        completions_all: List[Completion] = []
        actions = reference_action.copy()
        technical: List[str] = []
        plain: List[str] = []
        agreed = 0

        if not generate_explanations:
            return SupervisorDecision(
                joint_risk=trust.joint_risk,
                action=actions,
                action_utility=utilities,
                reference_action=reference_action,
                control_type=controls,
                source=trust.source,
                escalated=trust.escalated,
                parsed=parsed_all,
                completions=completions_all,
                adherence=float("nan"),
                technical_explanation=technical,
                plain_explanation=plain,
            )

        prompts: List[RenderedPrompt] = []
        for position in range(len(pairs)):
            va_package = {
                "risk_score": round(float(risk[position]), 3),
                "confidence": round(float(confidence_va[position]), 3),
            }
            if vulnerability_evidence is not None:
                va_package.update(vulnerability_evidence[position])
            ca_package = {
                "exposure_score": round(float(exposure[position]), 3),
                "regression_risk": round(float(regression[position]), 3),
                "confidence": round(float(confidence_ca[position]), 3),
            }
            if contextual_evidence is not None:
                ca_package.update(contextual_evidence[position])
            prompts.append(
                render_supervisor_prompt(
                    va_package, ca_package, bool(flag_va[position]), bool(flag_ca[position])
                )
            )

        size = batch_size or self.config.runtime.inference_batch_size
        for start in range(0, len(prompts), size):
            chunk = prompts[start : start + size]
            completions = self.model.generate(
                [item.text for item in chunk], agent=self.name, log=self.log
            )
            for offset, (prompt, completion) in enumerate(zip(chunk, completions)):
                position = start + offset

                def retry(prompt_text=prompt.text) -> str:
                    again = self.model.generate(
                        [prompt_text + "\n\nRespond with the JSON object only."],
                        agent=self.name,
                        log=self.log,
                    )
                    return again[0].text

                parsed = parse_completion(
                    completion.text,
                    self.schema,
                    self.telemetry,
                    retry=retry,
                    score_field="joint_risk",
                    reasoning_field="technical_explanation",
                    max_retries=self.config.fidelity.repair_retry_attempts,
                )
                parsed_all.append(parsed)
                completions_all.append(completion)
                emitted = str(parsed.values["action"])
                if emitted == reference_action[position]:
                    agreed += 1
                if not parsed.flagged() and emitted in self.config.bandit.actions:
                    actions[position] = emitted
                emitted_control = str(parsed.values["control_type"])
                if emitted_control in self.config.controls.taxonomy:
                    controls[position] = emitted_control
                technical.append(str(parsed.values["technical_explanation"]))
                plain.append(str(parsed.values["plain_explanation"]))
            LOGGER.info(
                "%s decided %d/%d pairs", self.name, min(start + size, len(prompts)), len(prompts)
            )

        actions = np.where(trust.escalated, "defer", actions)
        adherence = agreed / len(prompts) if prompts else float("nan")
        return SupervisorDecision(
            joint_risk=trust.joint_risk,
            action=actions,
            action_utility=utilities,
            reference_action=reference_action,
            control_type=controls,
            source=trust.source,
            escalated=trust.escalated,
            parsed=parsed_all,
            completions=completions_all,
            adherence=float(adherence),
            technical_explanation=technical,
            plain_explanation=plain,
        )
