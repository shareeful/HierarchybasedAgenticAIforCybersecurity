from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    bounds: Tuple[float, float] | None = None
    enum: Tuple[str, ...] | None = None
    default: object = None
    required: bool = True


@dataclass(frozen=True)
class OutputSchema:
    name: str
    fields: Tuple[FieldSpec, ...]

    @property
    def required_names(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.required)

    def spec(self, name: str) -> FieldSpec | None:
        for candidate in self.fields:
            if candidate.name == name:
                return candidate
        return None


VULNERABILITY_SCHEMA = OutputSchema(
    name="vulnerability_agent",
    fields=(
        FieldSpec("risk_score", "float", bounds=(0.0, 1.0), default=0.5),
        FieldSpec("confidence", "float", bounds=(0.0, 1.0), default=0.3),
        FieldSpec("reasoning", "string", default=""),
    ),
)

CONTEXTUAL_SCHEMA = OutputSchema(
    name="contextual_agent",
    fields=(
        FieldSpec("exposure_score", "float", bounds=(0.0, 1.0), default=0.5),
        FieldSpec("regression_risk", "float", bounds=(0.0, 1.0), default=0.5),
        FieldSpec("confidence", "float", bounds=(0.0, 1.0), default=0.3),
        FieldSpec("reasoning", "string", default=""),
    ),
)


def supervisor_schema(actions: Sequence[str], controls: Sequence[str]) -> OutputSchema:
    return OutputSchema(
        name="supervisor_agent",
        fields=(
            FieldSpec("joint_risk", "float", bounds=(0.0, 1.0), default=0.5),
            FieldSpec("action_utility", "float", default=0.0),
            FieldSpec("action", "enum", enum=tuple(actions), default=actions[1]),
            FieldSpec("control_type", "enum", enum=tuple(controls), default=controls[-1]),
            FieldSpec("technical_explanation", "string", default=""),
            FieldSpec("plain_explanation", "string", default=""),
        ),
    )


@dataclass
class ParseTelemetry:
    completions: int = 0
    malformed_json: int = 0
    repaired_without_retry: int = 0
    retried: int = 0
    retry_recovered: int = 0
    unrecoverable: int = 0
    schema_violations: int = 0
    clamped_values: int = 0
    imputed_fields: int = 0
    semantic_contradictions: int = 0
    human_review_flags: int = 0
    per_agent: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, agent: str, key: str, amount: int = 1) -> None:
        bucket = self.per_agent.setdefault(agent, {})
        bucket[key] = bucket.get(key, 0) + amount
        setattr(self, key, getattr(self, key) + amount)

    def rate(self, key: str) -> float:
        return getattr(self, key) / self.completions if self.completions else 0.0

    def as_rows(self) -> list[dict]:
        rows = []
        for agent, bucket in sorted(self.per_agent.items()):
            total = bucket.get("completions", 0)
            rows.append(
                {
                    "agent": agent,
                    "completions": total,
                    "malformed_json_rate": bucket.get("malformed_json", 0) / total if total else 0.0,
                    "repaired_without_retry": bucket.get("repaired_without_retry", 0),
                    "retry_recovered": bucket.get("retry_recovered", 0),
                    "unrecoverable": bucket.get("unrecoverable", 0),
                    "schema_violation_rate": bucket.get("schema_violations", 0) / total if total else 0.0,
                    "clamped_values": bucket.get("clamped_values", 0),
                    "imputed_fields": bucket.get("imputed_fields", 0),
                    "semantic_contradiction_rate": bucket.get("semantic_contradictions", 0) / total
                    if total
                    else 0.0,
                    "human_review_flags": bucket.get("human_review_flags", 0),
                }
            )
        return rows
