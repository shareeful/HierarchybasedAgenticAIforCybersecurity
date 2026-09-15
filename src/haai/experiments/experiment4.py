from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from ..agents.llm import CallLog
from ..config import Config
from ..logging_utils import get_logger
from ..metrics import classification_metrics, select_threshold
from ..pipeline.features import aggregate_to_records
from ..seeds import SeedBook
from .systems import SystemRun

LOGGER = get_logger()


@dataclass
class Experiment4Result:
    table15: pd.DataFrame
    timing: pd.DataFrame
    accuracy: pd.DataFrame
    call_log: pd.DataFrame
    note: str


def run_experiment4(
    config: Config,
    run: SystemRun,
    evaluation_pairs: pd.DataFrame,
    n_runs: int | None = None,
) -> Experiment4Result:
    book = SeedBook(config.evaluation.master_seed)
    pipeline = run.pipeline
    for model in (pipeline.models.vulnerability, pipeline.models.contextual, pipeline.models.supervisor):
        model.cache = None

    sample = min(config.runtime.timing_sample, len(evaluation_pairs))
    rng = book.stream("experiment4", 0)
    order = rng.permutation(len(evaluation_pairs))[:sample]
    bounded = evaluation_pairs.iloc[np.sort(order)].reset_index(drop=True)

    timing_rows: List[dict] = []
    accuracy_rows: List[dict] = []
    threshold = None

    for fraction in config.evaluation.scalability_fractions:
        count = max(int(round(fraction * len(bounded))), config.runtime.inference_batch_size)
        count = min(count, len(bounded))
        subset = bounded.iloc[:count].reset_index(drop=True)
        log = CallLog()
        pipeline.log = log
        pipeline.vulnerability_agent.log = log
        pipeline.contextual_agent.log = log
        pipeline.supervisor_agent.log = log

        started = time.perf_counter()
        preprocessing_started = time.perf_counter()
        unique = subset.drop_duplicates(subset="cve_id").reset_index(drop=True)
        preprocessing = time.perf_counter() - preprocessing_started

        vulnerability_started = time.perf_counter()
        vulnerability = pipeline.vulnerability_agent.assess(unique)
        vulnerability_seconds = time.perf_counter() - vulnerability_started

        risk_map = dict(zip(unique["cve_id"], vulnerability.score))
        risk = subset["cve_id"].map(risk_map).to_numpy(dtype=float)

        contextual_started = time.perf_counter()
        contextual = pipeline.contextual_agent.assess(subset, risk)
        contextual_seconds = time.perf_counter() - contextual_started

        integrity_started = time.perf_counter()
        pipeline.verifier_va.verify(pipeline.integrity_vectors_va(unique, vulnerability), refit=False)
        pipeline.verifier_ca.verify(pipeline.integrity_vectors_ca(subset, contextual), refit=False)
        integrity_seconds = time.perf_counter() - integrity_started

        supervisor_started = time.perf_counter()
        decision = pipeline.supervisor_agent.decide(
            subset,
            risk,
            contextual.score,
            contextual.secondary,
            np.full(len(subset), 0.8),
            np.full(len(subset), 0.8),
            np.zeros(len(subset), dtype=bool),
            np.zeros(len(subset), dtype=bool),
            generate_explanations=False,
        )
        supervisor_seconds = time.perf_counter() - supervisor_started
        total = time.perf_counter() - started

        agentic = vulnerability_seconds + contextual_seconds + supervisor_seconds
        timing_rows.append(
            {
                "pairs": len(subset),
                "unique_records": len(unique),
                "preprocessing_seconds": round(preprocessing, 4),
                "vulnerability_seconds": round(vulnerability_seconds, 4),
                "contextual_seconds": round(contextual_seconds, 4),
                "integrity_seconds": round(integrity_seconds, 4),
                "supervisor_seconds": round(supervisor_seconds, 4),
                "total_seconds": round(total, 4),
                "seconds_per_pair": round(total / max(len(subset), 1), 5),
                "overhead_percent": round(100.0 * agentic / total, 2) if total > 0 else float("nan"),
                "llm_calls": log.calls,
                "cached_calls": log.cached_calls,
            }
        )

        records = aggregate_to_records(subset.assign(score=decision.joint_risk), "score")
        if threshold is None:
            threshold = select_threshold(
                records["label"].to_numpy(dtype=int), records["score"].to_numpy(dtype=float)
            )
        measured = classification_metrics(
            records["label"].to_numpy(dtype=int),
            records["score"].to_numpy(dtype=float),
            threshold,
        )
        accuracy_rows.append({"pairs": len(subset), "f1": round(measured.f1, 4), "auc": round(measured.auc, 4)})
        LOGGER.info("timed %d pairs in %.2f s (%d live model calls)", len(subset), total, log.calls)

    timing = pd.DataFrame(timing_rows)
    accuracy = pd.DataFrame(accuracy_rows)
    table15 = timing.loc[
        :, ["pairs", "total_seconds", "seconds_per_pair", "overhead_percent", "llm_calls"]
    ].rename(
        columns={
            "pairs": "Vulnerability-asset pairs",
            "total_seconds": "Wall-clock seconds",
            "seconds_per_pair": "Seconds per pair",
            "overhead_percent": "Agent-tier share (%)",
            "llm_calls": "Model calls",
        }
    )
    note = (
        f"timings measured on {sample} pairs with the completion cache disabled; every reported "
        f"number is wall-clock time for calls actually executed in this process"
    )
    return Experiment4Result(
        table15=table15,
        timing=timing,
        accuracy=accuracy,
        call_log=pipeline.log.as_frame(),
        note=note,
    )
