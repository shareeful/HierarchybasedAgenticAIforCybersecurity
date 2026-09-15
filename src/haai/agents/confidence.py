from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression



def vulnerability_arm(cwe: str, epss_quartile: int) -> Tuple[str, int]:
    return (str(cwe), int(epss_quartile))


def contextual_arm(zone: str, criticality: float, buckets: Tuple[float, ...]) -> Tuple[str, int]:
    return (str(zone), int(np.digitize(float(criticality), np.asarray(buckets, dtype=float))))


def epss_quartile_edges(values: np.ndarray, n_quartiles: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_quartiles + 1)[1:-1]
    return np.quantile(np.asarray(values, dtype=float), quantiles)


def assign_epss_quartile(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(values, dtype=float), edges)


def arm_reliability(policy, arm: Hashable) -> float:
    return float(np.clip(policy.reliability(arm), 0.0, 1.0))


@dataclass
class CalibrationReport:
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    bins: pd.DataFrame

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"metric": "Brier score", "uncalibrated": self.brier_before, "calibrated": self.brier_after},
                {"metric": "Expected calibration error", "uncalibrated": self.ece_before, "calibrated": self.ece_after},
            ]
        )


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> Tuple[float, pd.DataFrame]:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    rows = []
    total = 0.0
    for index in range(bins):
        mask = assignment == index
        if not mask.any():
            continue
        mean_confidence = float(confidence[mask].mean())
        accuracy = float(correct[mask].mean())
        weight = float(mask.mean())
        total += weight * abs(mean_confidence - accuracy)
        rows.append(
            {
                "bin_low": float(edges[index]),
                "bin_high": float(edges[index + 1]),
                "count": int(mask.sum()),
                "mean_confidence": mean_confidence,
                "empirical_accuracy": accuracy,
            }
        )
    return total, pd.DataFrame(rows)


class ConfidenceCalibrator:
    def __init__(self, name: str):
        self.name = name
        self.model: IsotonicRegression | None = None
        self.report: CalibrationReport | None = None

    def _blend(self, reported: np.ndarray, sequence_logprob: np.ndarray, reliability: np.ndarray) -> np.ndarray:
        reported = np.clip(np.asarray(reported, dtype=float), 0.0, 1.0)
        sequence = np.exp(np.clip(np.asarray(sequence_logprob, dtype=float), -20.0, 0.0))
        reliability = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
        return np.clip((reported + sequence + reliability) / 3.0, 0.0, 1.0)

    def fit(
        self,
        reported: np.ndarray,
        sequence_logprob: np.ndarray,
        reliability: np.ndarray,
        correct: np.ndarray,
    ) -> "ConfidenceCalibrator":
        raw = self._blend(reported, sequence_logprob, reliability)
        correct = np.asarray(correct, dtype=float)
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, correct)
        calibrated = self.model.predict(raw)
        ece_before, _ = expected_calibration_error(raw, correct)
        ece_after, bins = expected_calibration_error(calibrated, correct)
        self.report = CalibrationReport(
            brier_before=float(np.mean((raw - correct) ** 2)),
            brier_after=float(np.mean((calibrated - correct) ** 2)),
            ece_before=float(ece_before),
            ece_after=float(ece_after),
            bins=bins,
        )
        return self

    def transform(
        self, reported: np.ndarray, sequence_logprob: np.ndarray, reliability: np.ndarray
    ) -> np.ndarray:
        raw = self._blend(reported, sequence_logprob, reliability)
        if self.model is None:
            return raw
        return np.clip(self.model.predict(raw), 1e-3, 1.0)


def calibration_table(calibrators: Dict[str, ConfidenceCalibrator]) -> pd.DataFrame:
    rows = []
    for name, calibrator in calibrators.items():
        if calibrator.report is None:
            continue
        rows.append(
            {
                "agent": name,
                "brier_uncalibrated": round(calibrator.report.brier_before, 4),
                "brier_calibrated": round(calibrator.report.brier_after, 4),
                "ece_uncalibrated": round(calibrator.report.ece_before, 4),
                "ece_calibrated": round(calibrator.report.ece_after, 4),
            }
        )
    return pd.DataFrame(rows)
