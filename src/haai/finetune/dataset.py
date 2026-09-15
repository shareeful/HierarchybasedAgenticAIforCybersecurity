from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..agents.contextual_agent import render_rows as render_contextual_rows
from ..agents.prompts import render_supervisor_prompt
from ..agents.vulnerability_agent import render_rows as render_vulnerability_rows
from ..config import Config
from ..logging_utils import get_logger

LOGGER = get_logger()

DECIMALS = 3


@dataclass
class InstructionExample:
    prompt: str
    completion: str
    targets: Dict[str, float]
    weight: float


@dataclass
class InstructionDataset:
    name: str
    examples: List[InstructionExample]
    numeric_fields: Tuple[str, ...]
    enum_field: str | None = None
    enum_values: Tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.examples)

    def write_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for example in self.examples:
                handle.write(
                    json.dumps(
                        {
                            "prompt": example.prompt,
                            "completion": example.completion,
                            "targets": example.targets,
                            "weight": example.weight,
                        }
                    )
                    + "\n"
                )
        LOGGER.info("wrote %d examples to %s", len(self.examples), path)
        return path


def inverse_frequency_weights(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    unique, counts = np.unique(values, return_counts=True)
    lookup = {key: float(len(values) / (len(unique) * count)) for key, count in zip(unique, counts)}
    return np.asarray([lookup[value] for value in values], dtype=float)


def _number(value: float) -> str:
    return f"{float(np.clip(value, 0.0, 1.0)):.{DECIMALS}f}"


def build_vulnerability_dataset(config: Config, frame: pd.DataFrame) -> InstructionDataset:
    prompts = render_vulnerability_rows(frame)
    labels = frame["label"].to_numpy(dtype=float)
    weights = (
        inverse_frequency_weights(labels)
        if config.lora.inverse_frequency_loss_weighting
        else np.ones(len(labels))
    )
    confidence = np.where(
        frame["kev"].to_numpy(dtype=bool) | frame["poc_available"].to_numpy(dtype=bool), 0.9, 0.6
    )
    examples: List[InstructionExample] = []
    for position, prompt in enumerate(prompts):
        risk = float(labels[position])
        record = frame.iloc[position]
        reasoning = (
            f"EPSS {float(record['epss']):.4f} with CVSS {float(record['cvss']):.1f}; "
            f"{'a public exploit is indexed' if bool(record['poc_available']) else 'no public exploit is indexed'}; "
            f"weakness {record['cwe']}"
            + (f" maps to {record['attack_technique']}" if record.get("attack_technique") else "")
            + "."
        )
        completion = json.dumps(
            {
                "risk_score": float(_number(risk)),
                "confidence": float(_number(confidence[position])),
                "reasoning": reasoning,
            }
        )
        examples.append(
            InstructionExample(
                prompt=prompt.text,
                completion=completion,
                targets={"risk_score": risk, "confidence": float(confidence[position])},
                weight=float(weights[position]),
            )
        )
    return InstructionDataset("vulnerability_agent", examples, ("risk_score", "confidence"))


def build_contextual_dataset(
    config: Config, pairs: pd.DataFrame, risk_scores: np.ndarray
) -> InstructionDataset:
    prompts = render_contextual_rows(pairs, risk_scores)
    exposure = pairs["exposure_ground_truth"].to_numpy(dtype=float)
    regression = pairs["patch_disruption_history"].to_numpy(dtype=float)
    buckets = np.digitize(exposure, np.asarray([0.25, 0.5, 0.75]))
    weights = (
        inverse_frequency_weights(buckets)
        if config.lora.inverse_frequency_loss_weighting
        else np.ones(len(exposure))
    )
    examples: List[InstructionExample] = []
    for position, prompt in enumerate(prompts):
        record = pairs.iloc[position]
        reasoning = (
            f"asset {record['asset_id']} in zone {record['zone']} at criticality "
            f"{float(record['criticality']):.2f} with control coverage "
            f"{float(record['control_coverage']):.2f}; software match strength "
            f"{float(record['sw_match']):.2f}."
        )
        completion = json.dumps(
            {
                "exposure_score": float(_number(exposure[position])),
                "regression_risk": float(_number(regression[position])),
                "confidence": float(_number(0.5 + 0.4 * float(record["sw_match"]))),
                "reasoning": reasoning,
            }
        )
        examples.append(
            InstructionExample(
                prompt=prompt.text,
                completion=completion,
                targets={
                    "exposure_score": float(exposure[position]),
                    "regression_risk": float(regression[position]),
                },
                weight=float(weights[position]),
            )
        )
    return InstructionDataset(
        "contextual_agent", examples, ("exposure_score", "regression_risk", "confidence")
    )


def build_supervisor_dataset(
    config: Config,
    pairs: pd.DataFrame,
    risk: np.ndarray,
    exposure: np.ndarray,
    regression: np.ndarray,
    recorded_actions: Sequence[str],
    control_types: Sequence[str],
) -> InstructionDataset:
    from ..agents.supervisor_agent import action_utility, joint_risk_eq23

    confidence = np.full(len(pairs), 0.8)
    joint = joint_risk_eq23(risk, exposure, confidence, confidence)
    actions = np.asarray(recorded_actions, dtype=object)
    weights = (
        inverse_frequency_weights(actions)
        if config.lora.inverse_frequency_loss_weighting
        else np.ones(len(actions))
    )
    examples: List[InstructionExample] = []
    for position in range(len(pairs)):
        action = str(actions[position])
        utility = float(action_utility(config, joint[position : position + 1], regression[position : position + 1], action)[0])
        va_package = {
            "risk_score": round(float(risk[position]), 3),
            "confidence": round(float(confidence[position]), 3),
        }
        ca_package = {
            "exposure_score": round(float(exposure[position]), 3),
            "regression_risk": round(float(regression[position]), 3),
            "confidence": round(float(confidence[position]), 3),
        }
        prompt = render_supervisor_prompt(va_package, ca_package, False, False)
        completion = json.dumps(
            {
                "joint_risk": float(_number(joint[position])),
                "action_utility": round(utility, 3),
                "action": action,
                "control_type": str(control_types[position]),
                "technical_explanation": (
                    f"confidence-weighted joint risk {joint[position]:.3f} with regression risk "
                    f"{float(regression[position]):.3f}; recorded remediation action was {action}."
                ),
                "plain_explanation": (
                    f"The recorded response for this vulnerability on this asset was to {action}."
                ),
            }
        )
        examples.append(
            InstructionExample(
                prompt=prompt.text,
                completion=completion,
                targets={"joint_risk": float(joint[position])},
                weight=float(weights[position]),
            )
        )
    return InstructionDataset(
        "supervisor_agent",
        examples,
        ("joint_risk",),
        enum_field="action",
        enum_values=tuple(config.bandit.actions),
    )


def dataset_statistics(datasets: Sequence[InstructionDataset]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        prompt_lengths = [len(example.prompt) for example in dataset.examples]
        completion_lengths = [len(example.completion) for example in dataset.examples]
        rows.append(
            {
                "dataset": dataset.name,
                "examples": len(dataset),
                "numeric_targets": ", ".join(dataset.numeric_fields),
                "enum_target": dataset.enum_field or "",
                "mean_prompt_characters": int(np.mean(prompt_lengths)) if prompt_lengths else 0,
                "mean_completion_characters": int(np.mean(completion_lengths)) if completion_lengths else 0,
            }
        )
    return pd.DataFrame(rows)


def read_jsonl(path: str | Path) -> List[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
