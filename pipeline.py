import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from agents import VulnerabilityAgent, ContextualAwarenessAgent, SupervisorAgent
from data import (
    VulnerabilityRecord, AssetRecord,
    build_vulnerability_feature_vector, build_asset_feature_vector,
    generate_exposure_pairs, validate_records,
    fetch_nvd_cves, fetch_epss_scores, fetch_cisa_kev, parse_nvd_record
)


@dataclass
class OutcomeSignals:
    s_p: int = 0
    s_e: int = 0
    s_r: float = 0.0
    s_d: int = 0


@dataclass
class RewardConfig:
    alpha_r: float = 0.30
    alpha_e: float = 0.35
    alpha_d: float = 0.20
    beta_p: float = 0.15


class RewardComputer:
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()

    def compute(self, risk_score: float, outcomes: OutcomeSignals) -> float:
        reward = (
            self.config.alpha_r * (risk_score * outcomes.s_r) +
            self.config.alpha_e * (1 - outcomes.s_e) +
            self.config.alpha_d * outcomes.s_d -
            self.config.beta_p * (1 - outcomes.s_p)
        )
        return float(np.clip(reward, 0.0, 1.0))


class AuditTrail:
    def __init__(self):
        self.records: List[Dict] = []

    def log(self, entry: Dict):
        entry["timestamp"] = datetime.utcnow().isoformat()
        self.records.append(entry)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            self.records = json.load(f)


class HierVulExPipeline:
    def __init__(
        self,
        use_local_models: bool = False,
        va_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        ca_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        sa_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    ):
        self.va = VulnerabilityAgent(model_name=va_model, use_local=use_local_models)
        self.ca = ContextualAwarenessAgent(model_name=ca_model, use_local=use_local_models)
        self.sa = SupervisorAgent(model_name=sa_model, use_local=use_local_models)
        self.reward_computer = RewardComputer()
        self.audit = AuditTrail()
        self.decision_history: List[Dict] = []

    def run_single(self, vuln_record: VulnerabilityRecord, asset_record: AssetRecord) -> Dict:
        vuln_features = build_vulnerability_feature_vector(vuln_record)
        asset_features = build_asset_feature_vector(asset_record)
        va_output = self.va.assess(vuln_features)
        ca_output = self.ca.assess(vuln_features, asset_features)
        decision = self.sa.decide(va_output, ca_output, vuln_features, asset_features)
        trace = {"vuln_features": vuln_features, "asset_features": asset_features,
                 "va_output": va_output, "ca_output": ca_output, "decision": decision}
        self.audit.log(trace)
        self.decision_history.append(trace)
        return decision

    def run_batch(self, vulnerabilities: List[VulnerabilityRecord], assets: List[AssetRecord]) -> List[Dict]:
        pairs = generate_exposure_pairs(vulnerabilities, assets)
        return [self.run_single(pair.vulnerability, pair.asset) for pair in pairs]

    def observe_outcome_and_update(self, decision: Dict, outcomes: OutcomeSignals) -> float:
        reward = self.reward_computer.compute(decision.get("joint_risk", 0.5), outcomes)
        self.va.update_policy(decision.get("va_output", {}).get("arm", 0), reward)
        self.ca.update_policy(decision.get("ca_output", {}).get("arm", 0), reward)
        self.sa.update_policy(decision.get("action_arm", 0), reward)
        self.audit.log({"type": "policy_update", "cve_id": decision.get("cve_id"),
                        "reward": round(reward, 4),
                        "outcomes": {"s_p": outcomes.s_p, "s_e": outcomes.s_e,
                                     "s_r": outcomes.s_r, "s_d": outcomes.s_d}})
        return reward

    def save_state(self, path: str):
        state = {"va_mab": self.va.mab.get_state(), "ca_mab": self.ca.mab.get_state(),
                 "sa_mab": self.sa.mab.get_state(), "isolation_window": self.sa.isolation_window}
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        with open(path, "r") as f:
            state = json.load(f)
        self.va.mab.load_state(state["va_mab"])
        self.ca.mab.load_state(state["ca_mab"])
        self.sa.mab.load_state(state["sa_mab"])
        self.sa.isolation_window = state.get("isolation_window", [])


def collect_and_prepare_data(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    assets: List[AssetRecord] = None
):
    print(f"Fetching CVE records from NVD ({start_date} to {end_date})...")
    raw_cves = fetch_nvd_cves(start_date, end_date, results_per_page=100)
    print(f"Retrieved {len(raw_cves)} raw CVE records.")
    vulnerabilities = [r for raw in raw_cves if (r := parse_nvd_record(raw)) is not None]
    cve_ids = [v.cve_id for v in vulnerabilities]
    print(f"Fetching EPSS scores for {len(cve_ids)} CVEs...")
    epss_scores = fetch_epss_scores(cve_ids)
    for v in vulnerabilities:
        v.epss = epss_scores.get(v.cve_id, 0.0)
    print("Fetching CISA KEV catalogue...")
    kev_ids = fetch_cisa_kev()
    for v in vulnerabilities:
        v.label = 1 if v.cve_id in kev_ids else 0
    print(f"Labelled {sum(v.label for v in vulnerabilities)} CVEs as exploited.")
    if assets is None:
        assets = []
    vulnerabilities, assets = validate_records(vulnerabilities, assets)
    print(f"Validated: {len(vulnerabilities)} vulnerabilities, {len(assets)} assets.")
    return vulnerabilities, assets, kev_ids
