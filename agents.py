import json
import numpy as np
from typing import Dict, List, Tuple

from mab import (
    ThompsonSamplingMAB, DiscountedUCBMAB, SlidingWindowUCBMAB,
    build_vulnerability_arms, build_contextual_arms, build_supervisor_arms,
    get_vulnerability_arm, get_contextual_arm
)


def _build_shap_values(features: Dict, risk_score: float) -> Dict[str, float]:
    weights = {"epss": 0.35, "poc_available": 0.25, "cvss": 0.20, "attack_technique": 0.10, "cwe": 0.10}
    return {k: round(risk_score * w, 4) for k, w in weights.items() if k in features}


def _build_lime_explanation(cve_id: str, risk_score: float, top_feature: str) -> str:
    level = "high" if risk_score > 0.7 else "moderate" if risk_score > 0.4 else "low"
    return (
        f"{cve_id} received a {level} risk score primarily because "
        f"{top_feature} indicates elevated exploitation likelihood "
        f"relative to other recently assessed vulnerabilities."
    )


def _build_attention_weights(asset_features: Dict) -> Dict[str, float]:
    z = asset_features.get("zone_score", 0.5) * 0.45
    c = asset_features.get("criticality_score", 0.5) * 0.35
    ctrl = (1 - min(asset_features.get("num_controls", 0) / 5, 1)) * 0.20
    total = max(z + c + ctrl, 1e-9)
    return {
        "network_zone": round(z / total, 4),
        "asset_criticality": round(c / total, 4),
        "control_coverage": round(ctrl / total, 4)
    }


def _build_counterfactual(asset_features: Dict, exposure: float) -> str:
    attn = _build_attention_weights(asset_features)
    top_factor = max(attn, key=attn.get)
    suggestions = {
        "network_zone": "moving this asset from the current network zone to an isolated segment",
        "asset_criticality": "deploying additional compensating controls to reduce effective criticality",
        "control_coverage": "adding at least two additional protective controls to this asset"
    }
    return (
        f"Exposure could be reduced by approximately {round(exposure * 0.35, 3)} by "
        f"{suggestions.get(top_factor, 'improving security controls')}."
    )


def _load_llm(model_name: str):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                        max_new_tokens=512, temperature=0.1, do_sample=True)
        return pipe
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        return None


def _llm_parse(pipe, prompt: str, fallback: Dict) -> Dict:
    if pipe is None:
        return fallback
    try:
        output = pipe(prompt)[0]["generated_text"]
        start, end = output.rfind("{"), output.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(output[start:end])
    except Exception:
        pass
    return fallback


class VulnerabilityAgent:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", use_local: bool = False):
        self.model_name = model_name
        self.pipe = _load_llm(model_name) if use_local else None
        arm_labels = build_vulnerability_arms()
        self.mab = ThompsonSamplingMAB(n_arms=len(arm_labels), arm_labels=arm_labels)

    def _rule_based_risk(self, features: Dict) -> float:
        risk = (features.get("cvss", 0) / 10.0) * 0.30 + \
               features.get("epss", 0) * 0.45 + \
               features.get("poc_available", 0) * 0.25
        return float(np.clip(risk, 0.0, 1.0))

    def _build_prompt(self, features: Dict) -> str:
        return (
            f"You are a cybersecurity threat analyst. Assess the vulnerability exploitation urgency.\n"
            f"CVE-ID: {features.get('cve_id')}\nCVSS: {features.get('cvss')}\n"
            f"EPSS: {features.get('epss')}\nPoC: {features.get('poc_available')}\n"
            f"CWE: {features.get('cwe')}\nATT&CK: {features.get('attack_technique')}\n"
            f"Return JSON: {{risk_score: float, confidence: float}}"
        )

    def assess(self, features: Dict) -> Dict:
        risk_score = self._rule_based_risk(features)
        if self.pipe is not None:
            parsed = _llm_parse(self.pipe, self._build_prompt(features), {})
            risk_score = float(np.clip(parsed.get("risk_score", risk_score), 0, 1))
        arm = get_vulnerability_arm(features.get("cwe", ""), features.get("epss", 0), self.mab.arm_labels)
        confidence = self.mab.get_confidence(arm)
        shap_values = _build_shap_values(features, risk_score)
        top_feature = max(shap_values, key=shap_values.get) if shap_values else "epss"
        return {
            "risk_score": round(risk_score, 4),
            "shap_values": shap_values,
            "lime_explanation": _build_lime_explanation(features.get("cve_id", ""), risk_score, top_feature),
            "confidence": round(confidence, 4),
            "arm": arm
        }

    def update_policy(self, arm: int, reward: float):
        self.mab.update(arm, reward)


class ContextualAwarenessAgent:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", use_local: bool = False):
        self.model_name = model_name
        self.pipe = _load_llm(model_name) if use_local else None
        arm_labels = build_contextual_arms()
        self.mab = DiscountedUCBMAB(n_arms=len(arm_labels), arm_labels=arm_labels)

    def _rule_based_exposure(self, asset_features: Dict) -> Tuple[float, float]:
        exposure = (asset_features.get("zone_score", 0.5) * 0.45 +
                    asset_features.get("criticality_score", 0.5) * 0.35 +
                    (1 - min(asset_features.get("num_controls", 0) / 5.0, 1)) * 0.20)
        return float(np.clip(exposure, 0, 1)), float(np.clip(asset_features.get("disruption_rate", 0.1), 0, 1))

    def _build_prompt(self, vuln: Dict, asset: Dict) -> str:
        return (
            f"You are a network security analyst. Assess organisational exposure.\n"
            f"CVE: {vuln.get('cve_id')} Risk: {vuln.get('risk_score', 0)}\n"
            f"Asset: {asset.get('asset_id')} Zone: {asset.get('zone')} Crit: {asset.get('criticality')}\n"
            f"Return JSON: {{exposure_score: float, regression_risk: float, confidence: float}}"
        )

    def assess(self, vuln_features: Dict, asset_features: Dict) -> Dict:
        exposure, regression_risk = self._rule_based_exposure(asset_features)
        if self.pipe is not None:
            parsed = _llm_parse(self.pipe, self._build_prompt(vuln_features, asset_features), {})
            exposure = float(np.clip(parsed.get("exposure_score", exposure), 0, 1))
            regression_risk = float(np.clip(parsed.get("regression_risk", regression_risk), 0, 1))
        arm = get_contextual_arm(
            asset_features.get("zone", "internal"),
            asset_features.get("criticality", "medium"),
            self.mab.arm_labels
        )
        confidence = self.mab.get_confidence(arm)
        return {
            "exposure_score": round(exposure, 4),
            "attention_weights": _build_attention_weights(asset_features),
            "counterfactual": _build_counterfactual(asset_features, exposure),
            "regression_risk": round(regression_risk, 4),
            "confidence": round(confidence, 4),
            "arm": arm
        }

    def update_policy(self, arm: int, reward: float):
        self.mab.update(arm, reward)


class SupervisorAgent:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", use_local: bool = False):
        self.model_name = model_name
        self.pipe = _load_llm(model_name) if use_local else None
        self.lambda_penalty = 0.4
        arm_labels = build_supervisor_arms()
        self.mab = SlidingWindowUCBMAB(n_arms=len(arm_labels), arm_labels=arm_labels)
        self.isolation_window: List[Dict] = []
        self.window_size = 50

    def _check_integrity(self, va_output: Dict, ca_output: Dict, delta: float = 0.75) -> Tuple[bool, bool]:
        entry = {
            "va_risk": va_output.get("risk_score", 0),
            "va_conf": va_output.get("confidence", 0.5),
            "ca_exp": ca_output.get("exposure_score", 0),
            "ca_reg": ca_output.get("regression_risk", 0),
            "ca_conf": ca_output.get("confidence", 0.5)
        }
        va_flag, ca_flag = False, False
        if len(self.isolation_window) >= 10:
            try:
                from sklearn.ensemble import IsolationForest
                history = np.array([[o["va_risk"], o["va_conf"], o["ca_exp"], o["ca_reg"], o["ca_conf"]]
                                    for o in self.isolation_window[-50:]])
                current = np.array([[entry["va_risk"], entry["va_conf"],
                                     entry["ca_exp"], entry["ca_reg"], entry["ca_conf"]]])
                clf = IsolationForest(contamination=0.1, random_state=42, n_estimators=50)
                clf.fit(history)
                score = clf.decision_function(current)[0]
                anomaly = float(np.clip(0.5 - score * 0.5, 0, 1))
                va_flag = anomaly > delta
                ca_flag = anomaly > delta
            except Exception:
                pass
        self.isolation_window.append(entry)
        if len(self.isolation_window) > self.window_size:
            self.isolation_window.pop(0)
        return va_flag, ca_flag

    def _compute_joint_risk(self, va: Dict, ca: Dict) -> float:
        c_va, c_ca = va.get("confidence", 0.5), ca.get("confidence", 0.5)
        denom = c_va + c_ca
        if denom < 1e-9:
            return (va.get("risk_score", 0) + ca.get("exposure_score", 0)) / 2
        return float(np.clip(
            (c_va * va.get("risk_score", 0) + c_ca * ca.get("exposure_score", 0)) / denom, 0, 1
        ))

    def _compute_adjusted_risk(self, r_joint: float, regression_risk: float) -> float:
        return float(np.clip(r_joint * (1 - self.lambda_penalty * regression_risk), 0, 1))

    def _select_action(self, r_adj: float) -> Tuple[str, int]:
        if r_adj > 0.75:
            action = "patch"
        elif r_adj > 0.5:
            action = "compensate"
        elif r_adj > 0.3:
            action = "defer"
        else:
            action = "accept"
        arm = self.mab.arm_labels.index(action)
        return action, arm

    def _recommend_control(self, cwe: str, exposure: float, zone: str) -> str:
        cwe_map = {
            "CWE-89": "input_validation", "CWE-79": "input_validation",
            "CWE-22": "access_control", "CWE-287": "access_control",
            "CWE-798": "access_control", "CWE-200": "access_control",
            "CWE-502": "patch_management", "CWE-78": "patch_management",
            "CWE-119": "patch_management"
        }
        for k, v in cwe_map.items():
            if k in cwe:
                return v
        if zone in ["DMZ", "cloud"] and exposure > 0.6:
            return "network_segmentation"
        return "logging_and_monitoring"

    def _generate_explanations(self, va: Dict, ca: Dict, r_joint: float, r_adj: float,
                                action: str, control: str, cve_id: str) -> Tuple[str, str]:
        reasons = {
            "patch": "adjusted risk exceeds immediate action threshold",
            "compensate": "risk is significant but regression risk moderates urgency",
            "defer": "adjusted risk does not justify immediate action",
            "accept": "current risk level is within acceptable tolerance"
        }
        technical = (
            f"CVE {cve_id}: risk={va.get('risk_score', 0):.3f}, "
            f"exposure={ca.get('exposure_score', 0):.3f}, "
            f"joint={r_joint:.3f}, adjusted={r_adj:.3f}. "
            f"Action: {action} — {reasons.get(action, '')}. "
            f"Control: {control}. {ca.get('counterfactual', '')}"
        )
        risk_level = "high" if va.get("risk_score", 0) > 0.7 else "moderate" if va.get("risk_score", 0) > 0.4 else "low"
        exp_level = "heavily" if ca.get("exposure_score", 0) > 0.7 else "partially" if ca.get("exposure_score", 0) > 0.4 else "minimally"
        plain = (
            f"Vulnerability {cve_id} poses a {risk_level} threat and our systems are {exp_level} exposed. "
            f"Recommended action: {action.upper()}. "
            f"Suggested measure: {control.replace('_', ' ')}."
        )
        return technical, plain

    def decide(self, va_output: Dict, ca_output: Dict, vuln_features: Dict, asset_features: Dict) -> Dict:
        va_flagged, ca_flagged = self._check_integrity(va_output, ca_output)
        if va_flagged and ca_flagged:
            return {
                "status": "ESCALATED",
                "message": "Both agents flagged. Decision escalated to human review.",
                "va_flagged": True, "ca_flagged": True,
                "action": None, "control_type": None,
                "technical_explanation": "Automated action halted due to dual integrity failure.",
                "plain_explanation": "Both AI agents produced anomalous outputs. Human review required."
            }
        eff_va = va_output if not va_flagged else {
            "risk_score": 0.5, "confidence": 0.3, "shap_values": {},
            "lime_explanation": "Conservative estimate — agent flagged."
        }
        eff_ca = ca_output if not ca_flagged else {
            "exposure_score": 0.5, "regression_risk": 0.3,
            "confidence": 0.3, "counterfactual": "Conservative estimate — agent flagged."
        }
        r_joint = self._compute_joint_risk(eff_va, eff_ca)
        r_adj = self._compute_adjusted_risk(r_joint, eff_ca.get("regression_risk", 0))
        action, arm = self._select_action(r_adj)
        control = self._recommend_control(
            vuln_features.get("cwe", ""),
            eff_ca.get("exposure_score", 0),
            asset_features.get("zone", "internal")
        )
        cve_id = vuln_features.get("cve_id", "UNKNOWN")
        tech, plain = self._generate_explanations(eff_va, eff_ca, r_joint, r_adj, action, control, cve_id)
        return {
            "status": "DECIDED",
            "cve_id": cve_id,
            "joint_risk": round(r_joint, 4),
            "adjusted_risk": round(r_adj, 4),
            "action": action,
            "action_arm": arm,
            "control_type": control,
            "va_flagged": va_flagged,
            "ca_flagged": ca_flagged,
            "technical_explanation": tech,
            "plain_explanation": plain,
            "va_output": eff_va,
            "ca_output": eff_ca
        }

    def update_policy(self, arm: int, reward: float):
        self.mab.update(arm, reward)
