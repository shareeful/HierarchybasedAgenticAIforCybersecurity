from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class CorpusConfig:
    year_range: Tuple[int, int] = (2019, 2024)
    uncertain_epss_floor: float = 0.10
    negatives_per_positive: int = 10
    train_fraction: float = 0.80
    finetune_fraction_of_train: float = 0.80
    completeness_required: Tuple[str, ...] = ("cvss", "epss", "cwe", "affected_software")
    outlier_z_threshold: float = 3.0


@dataclass(frozen=True)
class OrgEnvConfig:
    partial_version_match: float = 0.5
    control_families: Tuple[str, ...] = (
        "network_segmentation",
        "host_firewall",
        "endpoint_protection",
        "patch_management",
    )
    exposure_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "sw_match": 0.35,
            "reach": 0.25,
            "criticality": 0.20,
            "control_gap": 0.15,
            "epss": 0.05,
        }
    )
    max_assets_per_cve: int = 0


@dataclass(frozen=True)
class UseCaseConfig:
    dependency_weight: float = 0.22
    alert_evidence_weight: float = 0.45


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 3e-4
    lr_scheduler: str = "linear"
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    max_sequence_length: int = 1_024
    epochs: int = 10
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    task_tier_base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    supervisor_base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    alternative_task_models: Tuple[str, ...] = (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Llama-3.2-3B-Instruct",
    )
    load_in_4bit: bool = True
    inference_temperature: float = 0.0
    max_new_tokens: int = 320
    vulnerability_loss_alpha: float = 0.6
    contextual_loss_beta: float = 0.7
    inverse_frequency_loss_weighting: bool = True


@dataclass(frozen=True)
class BanditConfig:
    thompson_prior_alpha: float = 1.0
    thompson_prior_beta: float = 1.0
    epss_quartiles: int = 4
    discount_gamma: float = 0.85
    discount_gamma_grid: Tuple[float, ...] = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99)
    criticality_buckets: Tuple[float, ...] = (0.4, 0.7)
    sliding_window_tau: int = 80
    sliding_window_tau_grid: Tuple[int, ...] = (20, 40, 60, 80, 100, 140, 200)
    exploration_guard_epsilon: float = 1e-6
    cross_validation_folds: int = 5
    minimum_stratum_support: int = 5
    minimum_replay_events: int = 10
    actions: Tuple[str, ...] = ("patch", "compensate", "defer", "accept")
    risk_removal_fraction: Dict[str, float] = field(
        default_factory=lambda: {
            "patch": 1.0,
            "compensate": 0.6,
            "defer": 0.1,
            "accept": 0.0,
        }
    )
    disruption_cost_factor: Dict[str, float] = field(
        default_factory=lambda: {
            "patch": 1.0,
            "compensate": 0.3,
            "defer": 0.0,
            "accept": 0.0,
        }
    )
    omega_risk: float = 1.0
    omega_cost: float = 0.4
    omega_cost_grid: Tuple[float, ...] = (0.1, 0.2, 0.4, 0.6, 0.8)
    escalation_window: int = 3
    replay_passes: int = 1
    change_point_quantile: float = 0.5


@dataclass(frozen=True)
class RewardConfig:
    alpha_exposure: float = 0.35
    alpha_exploitation: float = 0.35
    alpha_analyst: float = 0.15
    beta_disruption: float = 0.15
    profiles: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "default_balanced": (0.35, 0.35, 0.15, 0.15),
            "risk_averse": (0.20, 0.45, 0.10, 0.25),
            "throughput_oriented": (0.50, 0.25, 0.15, 0.10),
            "zero_exploitation": (0.55, 0.00, 0.20, 0.25),
            "zero_disruption": (0.40, 0.40, 0.20, 0.00),
        }
    )


@dataclass(frozen=True)
class IntegrityConfig:
    n_estimators: int = 200
    max_samples: int = 256
    observation_window: int = 500
    false_positive_ceiling: float = 0.08
    threshold_grid_start: float = 0.40
    threshold_grid_stop: float = 0.90
    threshold_grid_step: float = 0.005
    poisoning_rates: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    poisoned_epss_bounds: Tuple[float, float] = (0.75, 0.99)
    poison_cvss_band: Tuple[float, float] = (4.0, 7.0)
    zscore_threshold: float = 2.0
    fixed_threshold_quantile: float = 0.97


@dataclass(frozen=True)
class ExplainabilityConfig:
    reference_window: int = 2_000
    surrogate_estimators: int = 200
    surrogate_max_depth: int = 6
    surrogate_learning_rate: float = 0.08
    surrogate_families: Tuple[str, ...] = ("gradient_boosting", "random_forest", "ridge")
    vulnerability_fields: Tuple[str, ...] = (
        "cvss",
        "epss",
        "poc_available",
        "cwe_risk_prior",
        "attack_technique_prior",
        "affected_software_breadth",
    )
    contextual_fields: Tuple[str, ...] = (
        "sw_match",
        "zone_reach",
        "external_reachable",
        "criticality",
        "control_coverage",
        "adjacency_degree",
        "patch_disruption_history",
        "patch_recency",
        "risk_score_input",
        "epss",
    )
    kernel_shap_coalitions: int = 1_024
    lime_neighbours: int = 500
    lime_kernel_width: float = 0.75
    lime_nonzero_coefficients: int = 5
    counterfactual_lambda: float = 1.0
    counterfactual_lambda_range: Tuple[float, float] = (0.5, 5.0)
    counterfactual_medium_band: float = 0.4
    counterfactual_max_candidates: int = 40
    counterfactual_costs: Dict[str, float] = field(
        default_factory=lambda: {
            "activate_network_segmentation": 0.30,
            "activate_host_firewall": 0.15,
            "activate_endpoint_protection": 0.20,
            "activate_patch_management": 0.25,
            "remove_external_reachability": 0.55,
            "relocate_zone": 0.70,
            "apply_patch": 0.45,
        }
    )
    attention_layers_pooled: int = 4
    surrogate_retrain_interval: int = 100
    deletion_test_top_k: int = 3
    model_query_budget: int = 200


@dataclass(frozen=True)
class AgentFidelityConfig:
    repair_retry_attempts: int = 1
    imputed_score: float = 0.5
    imputed_confidence: float = 0.3
    rag_neighbours: int = 8


@dataclass(frozen=True)
class EvaluationConfig:
    n_runs: int = 20
    master_seed: int = 20240501
    significance_level: float = 0.05
    scalability_fractions: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
    baselines: Tuple[str, ...] = (
        "cvss_only",
        "cvss_epss_calibrated",
        "xgboost",
        "securebert_cve",
        "rag_llm",
        "single_agent",
        "flat_multi_agent",
        "proposed",
    )
    baseline_display: Dict[str, str] = field(
        default_factory=lambda: {
            "cvss_only": "CVSS-only",
            "cvss_epss_calibrated": "CVSS+EPSS (cal.)",
            "xgboost": "XGBoost",
            "securebert_cve": "SecureBERT-CVE",
            "rag_llm": "RAG-LLM",
            "single_agent": "Single-Agent",
            "flat_multi_agent": "Flat Multi-Agent",
            "proposed": "Proposed",
        }
    )
    ablations: Tuple[str, ...] = (
        "full",
        "no_supervisor",
        "no_mab",
        "no_integrity_check",
        "no_confidence_weighting",
    )
    ablation_display: Dict[str, str] = field(
        default_factory=lambda: {
            "full": "Full system (Proposed)",
            "no_supervisor": "No supervisor",
            "no_mab": "No MAB learning",
            "no_integrity_check": "No integrity check",
            "no_confidence_weighting": "No confidence weighting",
        }
    )
    dqn_hidden_units: int = 64
    dqn_replay_capacity: int = 10_000
    dqn_target_refresh: int = 200
    dqn_epsilon_start: float = 0.30
    dqn_epsilon_end: float = 0.05
    dqn_learning_rate: float = 1e-3
    dqn_batch_size: int = 64
    dqn_discount: float = 0.90


@dataclass(frozen=True)
class ControlMappingConfig:
    taxonomy: Tuple[str, ...] = (
        "access_control",
        "network_segmentation",
        "input_validation",
        "patch_management",
        "logging_and_monitoring",
    )
    access_control_cwes: Tuple[str, ...] = (
        "CWE-287", "CWE-269", "CWE-284", "CWE-306", "CWE-862", "CWE-863", "CWE-522",
    )
    input_validation_cwes: Tuple[str, ...] = (
        "CWE-94", "CWE-502", "CWE-89", "CWE-79", "CWE-78", "CWE-77", "CWE-611", "CWE-20",
    )
    network_boundary_cwes: Tuple[str, ...] = (
        "CWE-400", "CWE-406", "CWE-918", "CWE-441", "CWE-1327",
    )


@dataclass(frozen=True)
class BaselineConfig:
    securebert_model: str = "ehsanaghaei/SecureBERT"
    securebert_epochs: int = 3
    securebert_learning_rate: float = 2e-5
    securebert_batch_size: int = 16
    securebert_max_length: int = 256
    retrieval_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_max_length: int = 256
    retrieval_batch_size: int = 64
    xgboost_estimators: int = 400
    xgboost_max_depth: int = 5
    xgboost_learning_rate: float = 0.08
    xgboost_subsample: float = 0.9


@dataclass(frozen=True)
class RuntimeConfig:
    data_dir: str = ""
    results_dir: str = "results"
    adapter_dir: str = "adapters"
    inference_batch_size: int = 8
    timing_sample: int = 200
    evaluation_sample: int = 0
    figure_dpi: int = 200
    figure_format: str = "png"
    verbose: bool = True


@dataclass(frozen=True)
class Config:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    org: OrgEnvConfig = field(default_factory=OrgEnvConfig)
    use_case: UseCaseConfig = field(default_factory=UseCaseConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    integrity: IntegrityConfig = field(default_factory=IntegrityConfig)
    xai: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    fidelity: AgentFidelityConfig = field(default_factory=AgentFidelityConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    controls: ControlMappingConfig = field(default_factory=ControlMappingConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


DEFAULT_CONFIG = Config()


def get_config(**overrides) -> Config:
    if not overrides:
        return Config()
    base = Config()
    sections: Dict[str, Dict[str, object]] = {}
    for key, value in overrides.items():
        if "." not in key:
            raise KeyError(f"override '{key}' must be of the form 'section.field'")
        section, field_name = key.split(".", 1)
        if section not in base.__dataclass_fields__:
            raise KeyError(f"unknown configuration section '{section}'")
        if field_name not in getattr(base, section).__dataclass_fields__:
            raise KeyError(f"unknown configuration field '{section}.{field_name}'")
        sections.setdefault(section, {})[field_name] = value
    kwargs = {}
    for name in base.__dataclass_fields__:
        current = getattr(base, name)
        if name in sections:
            kwargs[name] = type(current)(**{**asdict(current), **sections[name]})
        else:
            kwargs[name] = current
    return Config(**kwargs)
