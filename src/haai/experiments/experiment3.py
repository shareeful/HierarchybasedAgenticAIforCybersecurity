from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..agents.supervisor_agent import joint_risk_eq23
from ..config import Config
from ..integrity.detector import (
    FixedThresholdDetector,
    IsolationForestVerifier,
    RollingZScoreDetector,
)
from ..integrity.poisoning import inject_attack_model, recorded_tampering
from ..logging_utils import get_logger
from ..metrics import classification_metrics, detection_metrics, select_threshold
from ..pipeline.features import aggregate_to_records
from ..seeds import SeedBook
from ..stats import format_mean_std
from .systems import SystemRun

LOGGER = get_logger()

TELEMETRY_COLUMNS: Tuple[str, ...] = ("cvss", "epss", "risk_score", "confidence")

DETECTORS = ("isolation_forest", "fixed_threshold", "rolling_zscore")


@dataclass
class Experiment3Result:
    table14: pd.DataFrame
    detection_curves: pd.DataFrame
    quality_curves: pd.DataFrame
    provenance: str

    def degradation(self) -> Tuple[float, float]:
        frame = self.quality_curves
        highest = frame["rate"].max()
        row = frame[frame["rate"] == highest].iloc[0]
        return float(row["degradation_with_check"]), float(row["degradation_without_check"])


def _build_detector(name: str, config: Config, rng: np.random.Generator):
    if name == "isolation_forest":
        return IsolationForestVerifier(config, "vulnerability_agent", rng)
    if name == "fixed_threshold":
        return FixedThresholdDetector(config, "vulnerability_agent")
    return RollingZScoreDetector(config, "vulnerability_agent")


def run_experiment3(
    config: Config,
    run: SystemRun,
    telemetry: pd.DataFrame,
    evaluation_pairs: pd.DataFrame,
    n_runs: int | None = None,
) -> Experiment3Result:
    n_runs = n_runs or config.evaluation.n_runs
    book = SeedBook(config.evaluation.master_seed)
    clean = telemetry.loc[:, list(TELEMETRY_COLUMNS)].to_numpy(dtype=float)
    recorded = recorded_tampering(telemetry, TELEMETRY_COLUMNS)

    detection_rows: List[dict] = []
    if recorded is not None:
        provenance = recorded.provenance
        reference = clean[~recorded.poisoned_mask]
        for name in DETECTORS:
            for index in range(n_runs):
                rng = book.stream(f"experiment3:{name}:recorded", index)
                detector = _build_detector(name, config, rng)
                detector.prime(reference) if hasattr(detector, "prime") else None
                detector.calibrate(reference)
                _, flags = detector.verify(recorded.vectors, refit=False)
                detection, false_positive = detection_metrics(flags, recorded.poisoned_mask)
                detection_rows.append(
                    {
                        "detector": name,
                        "rate": round(recorded.rate, 4),
                        "run": index,
                        "detection": detection,
                        "false_positive": false_positive,
                    }
                )
    else:
        provenance = "injected attack model (config.integrity); no tampering labels in telemetry"
        for name in DETECTORS:
            for rate in config.integrity.poisoning_rates:
                for index in range(n_runs):
                    rng = book.stream(f"experiment3:{name}:{rate}", index)
                    batch = inject_attack_model(config, clean, 1, 0, None, rate, rng)
                    detector = _build_detector(name, config, rng)
                    if hasattr(detector, "prime"):
                        detector.prime(clean)
                    detector.calibrate(clean)
                    _, flags = detector.verify(batch.vectors, refit=False)
                    detection, false_positive = detection_metrics(flags, batch.poisoned_mask)
                    detection_rows.append(
                        {
                            "detector": name,
                            "rate": rate,
                            "run": index,
                            "detection": detection,
                            "false_positive": false_positive,
                        }
                    )
    detection_frame = pd.DataFrame(detection_rows)
    detection_curves = (
        detection_frame.groupby(["detector", "rate"])
        .agg(
            detection_mean=("detection", "mean"),
            detection_std=("detection", "std"),
            false_positive_mean=("false_positive", "mean"),
        )
        .reset_index()
        .fillna(0.0)
    )

    trace = run.trace
    pipeline = run.pipeline
    unique = trace.unique
    baseline_records = aggregate_to_records(
        evaluation_pairs.assign(score=trace.joint_risk), "score"
    )
    baseline_threshold = select_threshold(
        baseline_records["label"].to_numpy(dtype=int), baseline_records["score"].to_numpy(dtype=float)
    )
    baseline = classification_metrics(
        baseline_records["label"].to_numpy(dtype=int),
        baseline_records["score"].to_numpy(dtype=float),
        baseline_threshold,
    ).f1

    quality_rows: List[dict] = []
    vectors = pipeline.integrity_vectors_va(unique, trace.vulnerability_unique)
    for rate in config.integrity.poisoning_rates:
        with_check: List[float] = []
        without_check: List[float] = []
        for index in range(max(n_runs // 2, 1)):
            rng = book.stream(f"experiment3:quality:{rate}", index)
            batch = inject_attack_model(config, vectors, 1, 0, 2, rate, rng)
            corrupted_risk = batch.vectors[:, 3]
            risk_map = dict(zip(unique["cve_id"], corrupted_risk))
            pair_risk = evaluation_pairs["cve_id"].map(risk_map).to_numpy(dtype=float)
            _, flags = pipeline.verifier_va.verify(batch.vectors, refit=False)
            flag_map = dict(zip(unique["cve_id"], flags))
            pair_flags = evaluation_pairs["cve_id"].map(flag_map).to_numpy(dtype=bool)

            unguarded = joint_risk_eq23(
                pair_risk, trace.contextual.score, trace.confidence_va, trace.confidence_ca
            )
            guarded = np.where(
                pair_flags,
                trace.contextual.score,
                unguarded,
            )
            for values, sink in ((guarded, with_check), (unguarded, without_check)):
                records = aggregate_to_records(evaluation_pairs.assign(score=values), "score")
                measured = classification_metrics(
                    records["label"].to_numpy(dtype=int),
                    records["score"].to_numpy(dtype=float),
                    baseline_threshold,
                )
                sink.append(baseline - measured.f1)
        quality_rows.append(
            {
                "rate": rate,
                "degradation_with_check": round(float(np.mean(with_check)), 4),
                "degradation_without_check": round(float(np.mean(without_check)), 4),
                "baseline_f1": round(baseline, 4),
            }
        )
    quality_curves = pd.DataFrame(quality_rows)

    table_rows: List[dict] = []
    for detector in DETECTORS:
        subset = detection_frame[detection_frame["detector"] == detector]
        if not len(subset):
            continue
        highest = subset["rate"].max()
        worst = subset[subset["rate"] == highest]
        table_rows.append(
            {
                "Detector": detector.replace("_", " ").title(),
                "poisoning_rate": highest,
                "detection_rate": format_mean_std(worst["detection"]),
                "false_positive_rate": format_mean_std(worst["false_positive"]),
            }
        )
    table14 = pd.DataFrame(table_rows)

    return Experiment3Result(
        table14=table14,
        detection_curves=detection_curves,
        quality_curves=quality_curves,
        provenance=provenance,
    )
