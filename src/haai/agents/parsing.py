from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .schemas import FieldSpec, OutputSchema, ParseTelemetry

_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_UNQUOTED_KEY = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")
_SINGLE_QUOTED = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")
_BARE_STRING_VALUE = re.compile(r"(:\s*)([A-Za-z][A-Za-z0-9 _\-/]*?)(\s*[,}])")
_NUMBER_TOKEN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_FENCE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)

INTENSIFIERS: Tuple[Tuple[str, float], ...] = (
    ("critical", 0.85),
    ("actively exploited", 0.85),
    ("severe", 0.80),
    ("urgent", 0.78),
    ("high", 0.70),
    ("elevated", 0.62),
    ("significant", 0.62),
    ("moderate", 0.45),
    ("medium", 0.45),
    ("limited", 0.30),
    ("low", 0.25),
    ("minimal", 0.18),
    ("negligible", 0.12),
    ("no evidence", 0.10),
)

CONTRADICTION_MARGIN = 0.35


@dataclass
class ParsedOutput:
    values: Dict[str, object]
    raw: str
    repaired: bool = False
    retried: bool = False
    schema_violation: bool = False
    clamped: Tuple[str, ...] = ()
    imputed: Tuple[str, ...] = ()
    contradiction: bool = False
    human_review: bool = False
    failure_modes: Tuple[str, ...] = ()

    def flagged(self) -> bool:
        return self.human_review or self.contradiction


def extract_json_object(text: str) -> str | None:
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for position in range(start, len(text)):
        character = text[position]
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : position + 1]
    return text[start:]


def repair_json(text: str) -> str:
    candidate = extract_json_object(text)
    if candidate is None:
        return text
    candidate = candidate.strip()
    candidate = _SINGLE_QUOTED.sub(lambda m: '"' + m.group(1).replace('"', '\\"') + '"', candidate)
    candidate = _UNQUOTED_KEY.sub(r'\1"\2"\3', candidate)
    candidate = _TRAILING_COMMA.sub(r"\1", candidate)
    candidate = candidate.replace("True", "true").replace("False", "false").replace("None", "null")
    candidate = _BARE_STRING_VALUE.sub(
        lambda m: m.group(1) + _quote_if_needed(m.group(2)) + m.group(3), candidate
    )
    opens = candidate.count("{")
    closes = candidate.count("}")
    if opens > closes:
        candidate += "}" * (opens - closes)
    open_brackets = candidate.count("[")
    close_brackets = candidate.count("]")
    if open_brackets > close_brackets:
        candidate += "]" * (open_brackets - close_brackets)
    if candidate.count('"') % 2 == 1:
        candidate = candidate.rstrip("}") + '"' + "}" * (candidate.count("{") - candidate.count("}") + 1)
    return candidate


def _quote_if_needed(token: str) -> str:
    stripped = token.strip()
    if stripped in ("true", "false", "null"):
        return stripped
    if _NUMBER_TOKEN.fullmatch(stripped):
        return stripped
    return f'"{stripped}"'


def _coerce(spec: FieldSpec, value: object) -> Tuple[object, bool, bool]:
    clamped = False
    violated = False
    if spec.kind == "float":
        try:
            number = float(value)
        except (TypeError, ValueError):
            match = _NUMBER_TOKEN.search(str(value))
            if match is None:
                return spec.default, False, True
            number = float(match.group(0))
            violated = True
        if spec.bounds is not None:
            low, high = spec.bounds
            if number < low or number > high:
                number = min(max(number, low), high)
                clamped = True
                violated = True
        return number, clamped, violated
    if spec.kind == "enum":
        token = str(value).strip().lower().replace(" ", "_")
        if spec.enum and token not in spec.enum:
            for option in spec.enum or ():
                if option in token or token in option:
                    return option, False, True
            return spec.default, False, True
        return token, False, False
    return str(value), False, False


def semantic_contradiction(reasoning: str, score: float) -> bool:
    text = (reasoning or "").lower()
    matches = [strength for token, strength in INTENSIFIERS if token in text]
    if not matches:
        return False
    implied = max(matches) if score >= 0.5 else min(matches)
    return abs(implied - score) > CONTRADICTION_MARGIN


def parse_completion(
    completion: str,
    schema: OutputSchema,
    telemetry: ParseTelemetry,
    retry: Callable[[], str] | None = None,
    score_field: str | None = None,
    reasoning_field: str = "reasoning",
    max_retries: int = 1,
) -> ParsedOutput:
    telemetry.record(schema.name, "completions")
    failure_modes: List[str] = []
    raw = completion
    repaired = False
    retried = False
    payload: Dict[str, object] | None = None

    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        telemetry.record(schema.name, "malformed_json")
        failure_modes.append("malformed_json")
        try:
            payload = json.loads(repair_json(raw))
            repaired = True
            telemetry.record(schema.name, "repaired_without_retry")
        except (json.JSONDecodeError, TypeError):
            payload = None

    attempts = 0
    while payload is None and retry is not None and attempts < max_retries:
        attempts += 1
        retried = True
        telemetry.record(schema.name, "retried")
        raw = retry()
        try:
            payload = json.loads(raw)
            telemetry.record(schema.name, "retry_recovered")
        except (json.JSONDecodeError, TypeError):
            try:
                payload = json.loads(repair_json(raw))
                repaired = True
                telemetry.record(schema.name, "retry_recovered")
            except (json.JSONDecodeError, TypeError):
                payload = None

    if not isinstance(payload, dict):
        telemetry.record(schema.name, "unrecoverable")
        telemetry.record(schema.name, "human_review_flags")
        failure_modes.append("unrecoverable")
        values = {spec.name: spec.default for spec in schema.fields}
        telemetry.record(schema.name, "imputed_fields", len(values))
        return ParsedOutput(
            values=values,
            raw=raw,
            repaired=repaired,
            retried=retried,
            schema_violation=True,
            imputed=tuple(values),
            human_review=True,
            failure_modes=tuple(failure_modes),
        )

    values: Dict[str, object] = {}
    clamped: List[str] = []
    imputed: List[str] = []
    violated = False
    for spec in schema.fields:
        if spec.name not in payload or payload[spec.name] is None:
            values[spec.name] = spec.default
            if spec.required:
                imputed.append(spec.name)
                violated = True
            continue
        coerced, was_clamped, was_violated = _coerce(spec, payload[spec.name])
        values[spec.name] = coerced
        if was_clamped:
            clamped.append(spec.name)
        violated = violated or was_violated

    if violated:
        telemetry.record(schema.name, "schema_violations")
        failure_modes.append("schema_violation")
    if clamped:
        telemetry.record(schema.name, "clamped_values", len(clamped))
    if imputed:
        telemetry.record(schema.name, "imputed_fields", len(imputed))
        for spec in schema.fields:
            if spec.name in imputed and spec.name == "confidence":
                values["confidence"] = spec.default

    human_review = bool(imputed)
    contradiction = False
    if score_field is not None and reasoning_field in values:
        contradiction = semantic_contradiction(
            str(values.get(reasoning_field, "")), float(values.get(score_field, 0.5))
        )
        if contradiction:
            telemetry.record(schema.name, "semantic_contradictions")
            failure_modes.append("semantic_contradiction")
    if human_review:
        telemetry.record(schema.name, "human_review_flags")

    return ParsedOutput(
        values=values,
        raw=raw,
        repaired=repaired,
        retried=retried,
        schema_violation=violated,
        clamped=tuple(clamped),
        imputed=tuple(imputed),
        contradiction=contradiction,
        human_review=human_review,
        failure_modes=tuple(failure_modes),
    )
