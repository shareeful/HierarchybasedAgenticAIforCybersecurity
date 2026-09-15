from __future__ import annotations

import pandas as pd

ADVERSARY_SCOPE = (
    (
        "Knowledge",
        "Black-box and grey-box adversaries, including knowledge of agent roles and prompt fields",
        "White-box adversaries holding the adapted weight matrices",
    ),
    (
        "Objective",
        "Risk inflation, risk deflation, assessment spoofing",
        "Model extraction, training-time backdoor insertion",
    ),
    (
        "Resource",
        "Manipulation of up to 30% of records reaching one task agent, sustained across cycles",
        "Simultaneous control of both task agents; retrospective alteration of exploitation evidence",
    ),
    (
        "Surface",
        "Public intelligence feeds; the upward channel between the task tier and the supervisory tier",
        "Compromise of the host on which the agents execute",
    ),
)

MITIGATION_MAP = (
    ("Crude manipulation (implausible exploit probability)", "Phase 1 validation and z>3 outlier filtering",
     "haai.data.corpus._outlier_filter"),
    ("Distributional manipulation surviving validation", "Isolation Forest verification over the multivariate output vector",
     "haai.integrity.detector.IsolationForestVerifier"),
    ("Partial degradation of one agent", "Confidence-weighted synthesis with category-conditional reliability",
     "haai.agents.supervisor_agent.joint_risk_eq23"),
    ("Detected compromise", "Conservative fallback on the trusted agent and Human-in-the-Loop alert",
     "haai.agents.supervisor_agent.SupervisorAgent.decide"),
    ("Slow poisoning beneath the detection threshold", "Residual risk, stated openly", "out of scope"),
    ("White-box adversary holding adapter weights", "Residual risk, stated openly", "out of scope"),
)


def scope_table() -> pd.DataFrame:
    return pd.DataFrame(ADVERSARY_SCOPE, columns=["Dimension", "Within scope", "Out of scope"])


def mitigation_table() -> pd.DataFrame:
    return pd.DataFrame(MITIGATION_MAP, columns=["Threat", "Mechanism", "Implementation"])
