from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

VULNERABILITY_SYSTEM = (
    "You are a cybersecurity threat analyst. Analyse the provided vulnerability record "
    "and assess its real-world exploitation urgency."
)

VULNERABILITY_REASONING = (
    "Reason over all signals to assess real-world exploitation urgency. Ground all reasoning "
    "in the provided data. Do not infer exploit details absent from the input."
)

VULNERABILITY_OUTPUT = (
    '{\n  "risk_score" : float [0,1],\n  "confidence"  : float [0,1],\n  "reasoning"   : string\n}'
)

VULNERABILITY_FIELD_LABELS: Tuple[Tuple[str, str], ...] = (
    ("cve_id", "CVE-ID"),
    ("cvss", "CVSS"),
    ("epss", "EPSS"),
    ("poc_available", "PoC Available"),
    ("cwe", "CWE"),
    ("attack_technique", "ATT&CK"),
    ("affected_software", "Affected SW"),
)

CONTEXTUAL_SYSTEM = (
    "You are a network security analyst with access to the organisation's infrastructure "
    "records. Assess the organisation's specific exposure to the identified vulnerability."
)

CONTEXTUAL_REASONING = (
    "Evaluate whether this asset is exposed to the identified vulnerability. Consider network "
    "reachability from the asset's zone and its topological adjacencies, business criticality, "
    "controls, and patch history."
)

CONTEXTUAL_OUTPUT = (
    '{\n  "exposure_score"  : float [0,1],\n  "regression_risk" : float [0,1],\n'
    '  "confidence"      : float [0,1],\n  "reasoning"       : string\n}'
)

CONTEXTUAL_FIELD_LABELS: Tuple[Tuple[str, str], ...] = (
    ("cve_id", "CVE-ID"),
    ("risk_score_input", "Risk Score"),
    ("asset_id", "Asset ID"),
    ("zone", "Zone"),
    ("adjacent_zones", "Adjacencies"),
    ("external_reachable", "Reachable"),
    ("criticality", "Criticality"),
    ("installed_software", "Installed SW"),
    ("controls", "Controls"),
    ("patch_history", "Patch Hist."),
)

SUPERVISOR_SYSTEM = (
    "You are a senior security risk manager. Receive structured assessments from two "
    "specialist agents and produce the final risk decision and explanation."
)

SUPERVISOR_REASONING = (
    "Synthesise both assessments weighted by confidence. If either agent is flagged, apply "
    "conservative policy. Select the action balancing risk reduction and operational safety. "
    "Recommend a control category."
)

SUPERVISOR_OUTPUT = (
    '{\n  "joint_risk"              : float [0,1],\n  "action_utility"          : float,\n'
    '  "action"                  : enum [patch|compensate|defer|accept],\n'
    '  "control_type"            : string,\n  "technical_explanation"   : string,\n'
    '  "plain_explanation"       : string\n}'
)


@dataclass
class RenderedPrompt:
    text: str
    field_spans: Dict[str, Tuple[int, int]]

    def span_lengths(self) -> Dict[str, int]:
        return {name: end - start for name, (start, end) in self.field_spans.items()}


def _render_block(
    system: str,
    labels: Sequence[Tuple[str, str]],
    values: Dict[str, object],
    reasoning: str,
    output_format: str,
    note: str,
) -> RenderedPrompt:
    header = f"SYSTEM: {system}\n\nINPUT FORMAT:\n"
    parts: List[str] = [header]
    cursor = len(header)
    spans: Dict[str, Tuple[int, int]] = {}
    width = max(len(label) for _, label in labels)
    for key, label in labels:
        prefix = f"{label.ljust(width)} : "
        rendered = str(values.get(key, ""))
        start = cursor + len(prefix)
        end = start + len(rendered)
        spans[key] = (start, end)
        line = prefix + rendered + "\n"
        parts.append(line)
        cursor += len(line)
    tail = (
        f"\nREASONING INSTRUCTION:\n{reasoning}\n\nOUTPUT FORMAT (JSON):\n{output_format}\n"
        f"\nNOTE: {note}\n"
    )
    parts.append(tail)
    return RenderedPrompt(text="".join(parts), field_spans=spans)


def render_vulnerability_prompt(values: Dict[str, object]) -> RenderedPrompt:
    return _render_block(
        VULNERABILITY_SYSTEM,
        VULNERABILITY_FIELD_LABELS,
        values,
        VULNERABILITY_REASONING,
        VULNERABILITY_OUTPUT,
        "the agent emits no attribution values. The SHAP and LIME artefacts attached to this "
        "assessment are computed externally over the fields above by the explanation module, "
        "not written by the model.",
    )


def render_contextual_prompt(values: Dict[str, object]) -> RenderedPrompt:
    return _render_block(
        CONTEXTUAL_SYSTEM,
        CONTEXTUAL_FIELD_LABELS,
        values,
        CONTEXTUAL_REASONING,
        CONTEXTUAL_OUTPUT,
        "attention weights and the counterfactual are not requested from the model. They are "
        "computed externally from the decoder attention tensors and by discrete search over "
        "admissible environmental changes.",
    )


def render_supervisor_prompt(
    vulnerability_package: Dict[str, object],
    contextual_package: Dict[str, object],
    flag_va: bool,
    flag_ca: bool,
) -> RenderedPrompt:
    body = (
        f"SYSTEM: {SUPERVISOR_SYSTEM}\n\nINPUT FORMAT:\n"
        f"Vulnerability Agent Package: {vulnerability_package}\n"
        f"  [ risk_score, confidence, top-3 SHAP fields (computed), LIME summary ]\n"
        f"Contextual Agent Package: {contextual_package}\n"
        f"  [ exposure_score, regression_risk, confidence, top-3 attention fields (computed), "
        f"counterfactual ]\n"
        f"Trust Status: VA={'FLAGGED' if flag_va else 'OK'}, CA={'FLAGGED' if flag_ca else 'OK'}\n"
        f"\nREASONING INSTRUCTION:\n{SUPERVISOR_REASONING}\n"
        f"\nOUTPUT FORMAT (JSON):\n{SUPERVISOR_OUTPUT}\n"
    )
    return RenderedPrompt(text=body, field_spans={})
