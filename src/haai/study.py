from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .agents.confidence import calibration_table
from .agents.llm import AgentModels, CompletionCache, load_agent_models
from .agents.schemas import ParseTelemetry
from .config import Config, get_config
from .data.outcomes import build_recorded_outcomes
from .experiments.experiment1 import (
    Experiment1Result,
    label_noise_sensitivity,
    labelling_sensitivity,
    run_experiment1,
)
from .experiments.experiment2 import Experiment2Result, run_experiment2
from .experiments.experiment3 import Experiment3Result, run_experiment3
from .experiments.experiment4 import Experiment4Result, run_experiment4
from .experiments.experiment5 import Experiment5Result, run_experiment5
from .experiments.explainability import ExplainabilityResult, run_explainability
from .experiments.reproduction import (
    DATASET_DEPENDENT_NOTE,
    PAPER_VALUES,
    ComparisonRow,
    build_comparison,
    summarise_comparison,
)
from .experiments.systems import SystemRun, run_systems
from .finetune.dataset import (
    InstructionDataset,
    build_contextual_dataset,
    build_supervisor_dataset,
    build_vulnerability_dataset,
    dataset_statistics,
)
from .logging_utils import get_logger, timed
from .pipeline.environment import Environment, build_environment, make_pairs
from .seeds import SeedBook, set_global_determinism

LOGGER = get_logger()


def configure(**overrides) -> Config:
    return get_config(**overrides) if overrides else get_config()


@dataclass
class Study:
    config: Config
    environment: Optional[Environment] = None
    models: Optional[AgentModels] = None
    cache: Optional[CompletionCache] = None
    telemetry: ParseTelemetry = field(default_factory=ParseTelemetry)
    tuning_pairs: Optional[pd.DataFrame] = None
    evaluation_pairs: Optional[pd.DataFrame] = None
    use_case_pairs: Optional[pd.DataFrame] = None
    ledger_pairs: Optional[pd.DataFrame] = None
    base_run: Optional[SystemRun] = None
    experiment1: Optional[Experiment1Result] = None
    experiment2: Optional[Experiment2Result] = None
    experiment3: Optional[Experiment3Result] = None
    experiment4: Optional[Experiment4Result] = None
    experiment5: Optional[Experiment5Result] = None
    explainability: Optional[ExplainabilityResult] = None
    finetune_datasets: Dict[str, InstructionDataset] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def build(self, root: str | Path | None = None) -> Environment:
        set_global_determinism(self.config.evaluation.master_seed)
        with timed("environment construction from recorded data", LOGGER):
            self.environment = build_environment(self.config, root)
        corpus = self.environment.corpus
        self.tables["table8"] = corpus.characteristics()
        self.tables["validation"] = corpus.validation.as_frame()
        self.tables["feed_summary"] = self.environment.feeds.as_summary()
        self.tables["provenance"] = self.environment.provenance()

        with timed("vulnerability-asset pairing", LOGGER):
            self.tuning_pairs = make_pairs(
                self.config, corpus.reward_tuning, self.environment.organisation
            )
            self.evaluation_pairs = make_pairs(
                self.config, corpus.evaluation, self.environment.organisation
            )
            self.use_case_pairs = make_pairs(
                self.config, corpus.labelled, self.environment.use_case.organisation
            )
            self.ledger_pairs = make_pairs(
                self.config, corpus.labelled, self.environment.organisation
            )
        if self.config.runtime.evaluation_sample:
            limit = self.config.runtime.evaluation_sample
            rng = np.random.default_rng(self.config.evaluation.master_seed)
            for name in ("tuning_pairs", "evaluation_pairs"):
                frame = getattr(self, name)
                if len(frame) > limit:
                    index = np.sort(rng.permutation(len(frame))[:limit])
                    setattr(self, name, frame.iloc[index].reset_index(drop=True))
        self.tables["table7"] = self.environment.use_case.summary(len(self.use_case_pairs))
        self.tables["table5"] = self.environment.use_case.subsystem_table()
        self.tables["table6"] = self.environment.use_case.scenario_table()
        LOGGER.info(
            "pairs: %d tuning, %d evaluation, %d use case",
            len(self.tuning_pairs),
            len(self.evaluation_pairs),
            len(self.use_case_pairs),
        )
        return self.environment

    def load_models(self) -> AgentModels:
        self.cache = CompletionCache()
        with timed("loading the agent language models", LOGGER):
            self.models = load_agent_models(self.config, self.config.evaluation.master_seed, self.cache)
        return self.models

    def build_finetuning_datasets(self, output_dir: str | Path = "results/finetuning") -> pd.DataFrame:
        assert self.environment is not None and self.tuning_pairs is not None
        corpus = self.environment.corpus
        finetune_pairs = make_pairs(self.config, corpus.finetune, self.environment.organisation)
        outcomes = self.environment.outcomes.merge(
            self.ledger_pairs, on=["cve_id", "asset_id"], how="inner"
        )
        datasets = {
            "vulnerability_agent": build_vulnerability_dataset(self.config, corpus.finetune),
            "contextual_agent": build_contextual_dataset(
                self.config,
                finetune_pairs,
                finetune_pairs["risk_target"].to_numpy(dtype=float),
            ),
        }
        if len(outcomes):
            from .agents.supervisor_agent import recommend_control

            datasets["supervisor_agent"] = build_supervisor_dataset(
                self.config,
                outcomes,
                outcomes["risk_target"].to_numpy(dtype=float),
                outcomes["exposure_ground_truth"].to_numpy(dtype=float),
                outcomes["patch_disruption_history"].to_numpy(dtype=float),
                outcomes["action"].tolist(),
                recommend_control(self.config, outcomes["cwe"].astype(str)),
            )
        else:
            LOGGER.warning(
                "the remediation ledger does not intersect the tuning pairs; the supervisor "
                "instruction set cannot be built from recorded actions"
            )
        directory = Path(output_dir)
        for name, dataset in datasets.items():
            dataset.write_jsonl(directory / f"{name}.jsonl")
        self.finetune_datasets = datasets
        statistics = dataset_statistics(list(datasets.values()))
        self.tables["finetuning_datasets"] = statistics
        return statistics

    def run_base(self) -> SystemRun:
        assert self.environment is not None
        if self.models is None:
            self.load_models()
        book = SeedBook(self.config.evaluation.master_seed)
        corpus = self.environment.corpus
        self.base_run = run_systems(
            self.config,
            self.models,
            self.tuning_pairs,
            self.evaluation_pairs,
            corpus.reward_tuning,
            corpus.evaluation,
            book.stream("systems", 0),
            telemetry=self.telemetry,
        )
        self.tables["calibration"] = calibration_table(
            {
                "vulnerability_agent": self.base_run.pipeline.calibrator_va,
                "contextual_agent": self.base_run.pipeline.calibrator_ca,
            }
        )
        self.tables["supervisor_adherence"] = pd.DataFrame(
            [
                {
                    "quantity": "supervisor action equals the Eq. 24 argmax",
                    "value": self.base_run.trace.decision.adherence,
                }
            ]
        )
        return self.base_run

    def run_experiment1(self, n_runs: int | None = None) -> Experiment1Result:
        assert self.base_run is not None
        self.experiment1 = run_experiment1(
            self.config, self.base_run, self.evaluation_pairs, n_runs
        )
        self.tables["table9"] = self.experiment1.table9
        self.tables["table10"] = self.experiment1.table10
        self.tables["parse_telemetry"] = self.experiment1.parse_telemetry
        self.tables["significance"] = self.experiment1.significance
        self.tables["metric_consistency"] = self.experiment1.consistency
        self.tables["label_noise"] = label_noise_sensitivity(self.config, self.base_run)
        return self.experiment1

    def run_labelling_sensitivity(self, budget: int = 1_500) -> pd.DataFrame:
        assert self.base_run is not None and self.environment is not None
        table = labelling_sensitivity(
            self.config,
            self.base_run.pipeline,
            self.environment.corpus,
            self.environment.organisation,
            budget,
        )
        self.tables["labelling_sensitivity"] = table
        return table

    def run_experiment2(self, n_runs: int | None = None) -> Experiment2Result:
        assert self.environment is not None
        outcomes = build_recorded_outcomes(
            self.config, self.environment.outcomes, self.ledger_pairs
        )
        self.experiment2 = run_experiment2(self.config, outcomes, n_runs)
        self.tables["table11"] = self.experiment2.table11
        self.tables["table12"] = self.experiment2.table12
        self.tables["replay_coverage"] = self.experiment2.coverage
        self.tables["tau_sensitivity"] = self.experiment2.tau_sensitivity
        self.tables["gamma_sensitivity"] = self.experiment2.gamma_sensitivity
        return self.experiment2

    def run_experiment3(self, n_runs: int | None = None) -> Experiment3Result:
        assert self.base_run is not None and self.environment is not None
        self.experiment3 = run_experiment3(
            self.config, self.base_run, self.environment.telemetry, self.evaluation_pairs, n_runs
        )
        self.tables["table14"] = self.experiment3.table14
        self.tables["integrity_detection"] = self.experiment3.detection_curves
        self.tables["integrity_quality"] = self.experiment3.quality_curves
        return self.experiment3

    def run_experiment4(self, n_runs: int | None = None) -> Experiment4Result:
        assert self.base_run is not None
        self.experiment4 = run_experiment4(self.config, self.base_run, self.evaluation_pairs, n_runs)
        self.tables["table15"] = self.experiment4.table15
        self.tables["call_log"] = self.experiment4.call_log
        for model in (
            self.base_run.pipeline.models.vulnerability,
            self.base_run.pipeline.models.contextual,
            self.base_run.pipeline.models.supervisor,
        ):
            model.cache = self.cache
        return self.experiment4

    def run_experiment5(self) -> Experiment5Result:
        assert self.base_run is not None and self.environment is not None
        self.experiment5 = run_experiment5(
            self.config,
            self.base_run,
            self.environment.use_case,
            self.use_case_pairs,
            self.models,
        )
        self.tables["table16"] = self.experiment5.table16
        self.tables["table17"] = self.experiment5.table17
        self.tables["analyst_agreement"] = pd.DataFrame(
            [{"quantity": key, "value": value} for key, value in self.experiment5.agreement.items()]
        )
        self.tables["agreement_by_subsystem"] = self.experiment5.agreement_by_subsystem
        self.tables["operator_targets"] = self.experiment5.operator_targets
        self.tables["action_distribution"] = self.experiment5.action_distribution
        return self.experiment5

    def run_explainability(self, sample_size: int = 64) -> ExplainabilityResult:
        assert self.base_run is not None
        self.explainability = run_explainability(
            self.config, self.base_run, self.evaluation_pairs, sample_size
        )
        self.tables["explainability_faithfulness"] = self.explainability.faithfulness
        self.tables["table13"] = self.explainability.surrogate_fidelity
        self.tables["counterfactuals"] = self.explainability.counterfactuals
        self.tables["attention"] = self.explainability.attention
        self.tables["coalition_budget"] = self.explainability.coalition_budget
        return self.explainability

    def table18(self) -> pd.DataFrame:
        assert self.experiment1 is not None
        capability = {
            "cvss_only": ("no", "no", "no", "no", "no"),
            "cvss_epss_calibrated": ("no", "no", "no", "no", "no"),
            "xgboost": ("no", "no", "no", "partial", "no"),
            "securebert_cve": ("no", "no", "no", "partial", "no"),
            "rag_llm": ("partial", "no", "no", "partial", "no"),
            "single_agent": ("partial", "no", "no", "partial", "no"),
            "flat_multi_agent": ("yes", "no", "no", "partial", "no"),
            "proposed": ("yes", "yes", "yes", "yes", "yes"),
        }
        frame = self.experiment1.table9.set_index("raw_key")
        rows = []
        for name, values in capability.items():
            rows.append(
                {
                    "System": self.config.evaluation.baseline_display[name],
                    "F1": round(float(frame.loc[name, "f1_mean"]), 3),
                    "Org. context": values[0],
                    "Adaptive": values[1],
                    "Integrity check": values[2],
                    "Explainability": values[3],
                    "Adv. robust": values[4],
                }
            )
        table = pd.DataFrame(rows)
        self.tables["table18"] = table
        return table

    def comparison(self) -> pd.DataFrame:
        rows: List[ComparisonRow] = []
        if self.environment is not None:
            corpus = self.environment.corpus
            measured = {
                "corpus_records": len(corpus.corpus),
                "confirmed_exploited": int(corpus.corpus["kev"].sum()),
                "uncertain_held_out": len(corpus.uncertain),
                "sampled_negatives": int((corpus.labelled["label"] == 0).sum()),
                "labelled_set": len(corpus.labelled),
                "positive_prevalence": float(corpus.labelled["label"].mean()),
                "training_split": len(corpus.train),
                "evaluation_split": len(corpus.evaluation),
                "unique_cwe": int(corpus.corpus["cwe"].nunique()),
                "unique_attack": int(corpus.corpus["attack_technique"].nunique()),
                "distinct_vendors": int(corpus.corpus["vendor"].nunique()),
            }
            for key, paper_value in PAPER_VALUES["table8"].items():
                rows.append(
                    ComparisonRow(
                        f"Table 8: {key}", paper_value, measured[key], note=DATASET_DEPENDENT_NOTE
                    )
                )
        if "table7" in self.tables:
            summary = dict(zip(self.tables["table7"]["Property"], self.tables["table7"]["Value"]))
            mapping = {
                "hosts": "Hosts in asset inventory",
                "subsystems": "Operational subsystems",
                "segments": "Network segments",
                "software_components": "Distinct software components",
                "pairs": "Vulnerability-asset pairs on these hosts",
                "patch_records": "Patch and change records",
                "alerts": "Security operations alerts captured",
                "analyst_decisions": "Analyst triage decisions",
            }
            for key, paper_value in PAPER_VALUES["table7"].items():
                value = summary.get(mapping[key])
                if value is None or value == "":
                    continue
                rows.append(
                    ComparisonRow(
                        f"Table 7: {key}", paper_value, float(value), note=DATASET_DEPENDENT_NOTE
                    )
                )
        if self.experiment1 is not None:
            frame = self.experiment1.table9.set_index("raw_key")
            for name, paper_value in PAPER_VALUES["table9_f1"].items():
                rows.append(
                    ComparisonRow(f"Table 9 F1: {name}", paper_value, float(frame.loc[name, "f1_mean"]))
                )
            for name, paper_value in PAPER_VALUES["table9_auc"].items():
                rows.append(
                    ComparisonRow(f"Table 9 AUC: {name}", paper_value, float(frame.loc[name, "auc_mean"]))
                )
            ablation = self.experiment1.table10.set_index("raw_key")
            for name, paper_value in PAPER_VALUES["table10_f1"].items():
                if name in ablation.index:
                    rows.append(
                        ComparisonRow(
                            f"Table 10 F1: {name}", paper_value, float(ablation.loc[name, "f1_mean"])
                        )
                    )
        if self.experiment2 is not None and len(self.experiment2.tau_sensitivity):
            best_tau = self.experiment2.tau_sensitivity.loc[
                self.experiment2.tau_sensitivity["accuracy_mean"].idxmax(), "tau"
            ]
            best_gamma = self.experiment2.gamma_sensitivity.loc[
                self.experiment2.gamma_sensitivity["accuracy_mean"].idxmax(), "gamma"
            ]
            rows.append(
                ComparisonRow("Optimal SW-UCB window tau", PAPER_VALUES["hyperparameters"]["tau"], float(best_tau))
            )
            rows.append(
                ComparisonRow(
                    "Optimal D-UCB discount gamma", PAPER_VALUES["hyperparameters"]["gamma"], float(best_gamma)
                )
            )
        if self.experiment3 is not None and len(self.experiment3.quality_curves):
            with_check, without_check = self.experiment3.degradation()
            rows.append(ComparisonRow("Decision-quality degradation with check", 0.105, round(with_check, 3)))
            rows.append(
                ComparisonRow("Decision-quality degradation without check", 0.413, round(without_check, 3))
            )
        if self.experiment5 is not None:
            table = self.experiment5.table16.set_index("system")
            for name, paper_value in PAPER_VALUES["table16_use_case_f1"].items():
                display = self.config.evaluation.baseline_display[name]
                if display in table.index:
                    rows.append(
                        ComparisonRow(
                            f"Table 16 use-case F1: {name}",
                            paper_value,
                            float(table.loc[display, "use_case_f1"]),
                        )
                    )
            rows.append(
                ComparisonRow(
                    "Analyst agreement (proposed)",
                    PAPER_VALUES["other"]["analyst_agreement"],
                    round(self.experiment5.agreement["proposed"], 3),
                )
            )
            rows.append(
                ComparisonRow(
                    "Analyst agreement (flat multi-agent)",
                    PAPER_VALUES["other"]["flat_analyst_agreement"],
                    round(self.experiment5.agreement["flat_multi_agent"], 3),
                )
            )
        if self.explainability is not None:
            fidelity = self.explainability.surrogate_fidelity
            primary = fidelity[fidelity["surrogate_family"] == "gradient_boosting"].set_index("agent")
            for agent, key in (
                ("Vulnerability Agent", "surrogate_r2_va"),
                ("Contextual Awareness Agent", "surrogate_r2_ca"),
            ):
                if agent in primary.index:
                    rows.append(
                        ComparisonRow(
                            f"Surrogate R2 ({agent})",
                            PAPER_VALUES["explainability"][key],
                            float(primary.loc[agent, "held_out_r2"]),
                        )
                    )
            if np.isfinite(self.explainability.counterfactual_validity):
                rows.append(
                    ComparisonRow(
                        "Counterfactual validity",
                        PAPER_VALUES["explainability"]["counterfactual_validity"],
                        round(self.explainability.counterfactual_validity, 3),
                    )
                )
        if "parse_telemetry" in self.tables and len(self.tables["parse_telemetry"]):
            telemetry = self.tables["parse_telemetry"].set_index("agent")
            for agent, key in (
                ("vulnerability_agent", "malformed_vulnerability"),
                ("supervisor_agent", "malformed_supervisor"),
            ):
                if agent in telemetry.index:
                    rows.append(
                        ComparisonRow(
                            f"Malformed JSON rate ({agent})",
                            PAPER_VALUES["parsing"][key],
                            round(float(telemetry.loc[agent, "malformed_json_rate"]), 4),
                        )
                    )
        if self.base_run is not None:
            rows.append(
                ComparisonRow(
                    "Isolation Forest threshold",
                    PAPER_VALUES["hyperparameters"]["isolation_threshold"],
                    round(self.base_run.pipeline.verifier_va.threshold, 3),
                )
            )
        frame = build_comparison(rows)
        self.tables["reproduction_comparison"] = frame
        return frame

    def comparison_summary(self) -> Dict[str, float]:
        frame = self.tables.get("reproduction_comparison")
        if frame is None:
            frame = self.comparison()
        return summarise_comparison(frame)

    def save_tables(self, directory: str | Path | None = None) -> Path:
        directory = Path(directory or Path(self.config.runtime.results_dir) / "tables")
        directory.mkdir(parents=True, exist_ok=True)
        for name, table in self.tables.items():
            table.to_csv(directory / f"{name}.csv", index=False)
        LOGGER.info("wrote %d tables to %s", len(self.tables), directory)
        return directory
