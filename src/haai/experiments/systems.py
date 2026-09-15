from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..agents.llm import AgentModels
from ..agents.schemas import ParseTelemetry
from ..baselines.models import (
    CVSSOnlyBaseline,
    CalibratedCVSSEPSSBaseline,
    RAGLLMBaseline,
    SecureBERTBaseline,
    SingleAgentBaseline,
    XGBoostBaseline,
)
from ..config import Config
from ..logging_utils import get_logger, timed
from ..pipeline.features import aggregate_to_records
from ..pipeline.orchestrator import AssessmentTrace, HierarchicalPipeline

LOGGER = get_logger()


@dataclass
class SystemRun:
    pipeline: HierarchicalPipeline
    trace: AssessmentTrace
    ablation_traces: Dict[str, AssessmentTrace]
    record_scores: Dict[str, pd.DataFrame]
    tuning_scores: Dict[str, pd.DataFrame]
    fitted_baselines: Dict[str, object]
    telemetry: ParseTelemetry
    models: AgentModels

    def labels(self, system: str) -> np.ndarray:
        return self.record_scores[system]["label"].to_numpy(dtype=int)

    def scores(self, system: str) -> np.ndarray:
        return self.record_scores[system]["score"].to_numpy(dtype=float)


def _record_frame(pairs: pd.DataFrame, values: np.ndarray, column: str = "score") -> pd.DataFrame:
    working = pairs.assign(**{column: values})
    return aggregate_to_records(working, column)


def run_systems(
    config: Config,
    models: AgentModels,
    tuning_pairs: pd.DataFrame,
    evaluation_pairs: pd.DataFrame,
    tuning_records: pd.DataFrame,
    evaluation_records: pd.DataFrame,
    rng: np.random.Generator,
    telemetry: ParseTelemetry | None = None,
    ablations: bool = True,
    generate_explanations: bool = False,
) -> SystemRun:
    telemetry = telemetry or ParseTelemetry()
    pipeline = HierarchicalPipeline(config, models, rng, telemetry)
    with timed("fitting the hierarchy on the tuning split", LOGGER):
        pipeline.fit(tuning_pairs)

    with timed("hierarchical assessment of the evaluation split", LOGGER):
        trace = pipeline.assess(
            evaluation_pairs,
            generate_explanations=generate_explanations,
            key="evaluation_vulnerability",
        )

    record_scores: Dict[str, pd.DataFrame] = {}
    tuning_scores: Dict[str, pd.DataFrame] = {}

    fitted: Dict[str, object] = {}
    structured = {
        "cvss_only": CVSSOnlyBaseline(),
        "cvss_epss_calibrated": CalibratedCVSSEPSSBaseline(),
        "xgboost": XGBoostBaseline(),
        "securebert_cve": SecureBERTBaseline(config),
    }
    for name, model in structured.items():
        with timed(f"baseline {name}", LOGGER):
            model.fit(tuning_records, config, rng)
            record_scores[name] = evaluation_records.loc[:, ["cve_id", "label"]].assign(
                score=model.score(evaluation_records)
            )
            tuning_scores[name] = tuning_records.loc[:, ["cve_id", "label"]].assign(
                score=model.score(tuning_records)
            )
            fitted[name] = model

    with timed("baseline rag_llm", LOGGER):
        rag = RAGLLMBaseline(config, models.vulnerability, telemetry, models.log)
        rag.fit(tuning_records, config, rng)
        record_scores["rag_llm"] = evaluation_records.loc[:, ["cve_id", "label"]].assign(
            score=rag.score(evaluation_records)
        )
        tuning_scores["rag_llm"] = tuning_records.loc[:, ["cve_id", "label"]].assign(
            score=rag.score(tuning_records)
        )
        fitted["rag_llm"] = rag

    with timed("baseline single_agent", LOGGER):
        single = SingleAgentBaseline(config, models.vulnerability, telemetry, models.log)
        record_scores["single_agent"] = _record_frame(
            evaluation_pairs, single.score(evaluation_pairs)
        )
        tuning_scores["single_agent"] = _record_frame(tuning_pairs, single.score(tuning_pairs))
        fitted["single_agent"] = single

    record_scores["flat_multi_agent"] = _record_frame(evaluation_pairs, trace.flat_risk)
    record_scores["proposed"] = _record_frame(evaluation_pairs, trace.joint_risk)

    tuning_trace = pipeline.assess(tuning_pairs, key="tuning_assessment")
    tuning_scores["flat_multi_agent"] = _record_frame(tuning_pairs, tuning_trace.flat_risk)
    tuning_scores["proposed"] = _record_frame(tuning_pairs, tuning_trace.joint_risk)

    ablation_traces: Dict[str, AssessmentTrace] = {}
    if ablations:
        ablation_traces["full"] = trace
        settings = {
            "no_supervisor": {"variant": "flat"},
            "no_mab": {"use_mab": False},
            "no_integrity_check": {"apply_integrity": False},
            "no_confidence_weighting": {"use_confidence_weighting": False},
        }
        for name, options in settings.items():
            with timed(f"ablation {name}", LOGGER):
                if options.get("variant") == "flat":
                    ablation_traces[name] = trace
                    continue
                ablation_traces[name] = pipeline.assess(
                    evaluation_pairs, key="evaluation_vulnerability", **options
                )

    return SystemRun(
        pipeline=pipeline,
        trace=trace,
        ablation_traces=ablation_traces,
        record_scores=record_scores,
        tuning_scores=tuning_scores,
        fitted_baselines=fitted,
        telemetry=telemetry,
        models=models,
    )


def ablation_record_scores(run: SystemRun, name: str, pairs: pd.DataFrame) -> pd.DataFrame:
    trace = run.ablation_traces[name]
    values = trace.flat_risk if name == "no_supervisor" else trace.joint_risk
    return _record_frame(pairs, values)
