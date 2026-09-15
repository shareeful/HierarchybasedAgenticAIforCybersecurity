from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    auc: float
    average_precision: float
    threshold: float
    prevalence: float
    support: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "average_precision": self.average_precision,
            "threshold": self.threshold,
            "prevalence": self.prevalence,
            "support": self.support,
        }


def select_threshold(labels: np.ndarray, scores: np.ndarray, n_grid: int = 200) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    unique = np.unique(scores)
    grid = unique if len(unique) <= n_grid else np.quantile(scores, np.linspace(0.001, 0.999, n_grid))
    best_threshold, best_f1 = float(grid[0]), -1.0
    for candidate in grid:
        predicted = (scores >= candidate).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            labels, predicted, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(candidate)
    return best_threshold


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> ClassificationMetrics:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    distinct = len(np.unique(labels)) > 1
    auc = float(roc_auc_score(labels, scores)) if distinct else float("nan")
    average_precision = float(average_precision_score(labels, scores)) if distinct else float("nan")
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted, average="binary", zero_division=0
    )
    return ClassificationMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc=auc,
        average_precision=average_precision,
        threshold=float(threshold),
        prevalence=float(labels.mean()),
        support=int(len(labels)),
    )


def _f1_from_operating_point(tpr: float, fpr: float, prevalence: float) -> float:
    positives = prevalence
    negatives = 1.0 - prevalence
    true_positive = tpr * positives
    false_positive = fpr * negatives
    false_negative = (1.0 - tpr) * positives
    denominator = 2.0 * true_positive + false_positive + false_negative
    return float(2.0 * true_positive / denominator) if denominator > 0 else 0.0


def maximum_f1_hull_bound(auc: float, prevalence: float, grid: int = 20_001) -> float:
    if not np.isfinite(auc) or not 0.0 < prevalence < 1.0 or auc < 0.5:
        return float("nan")
    fpr = np.linspace(0.0, 1.0, grid)
    tpr = np.clip(2.0 * auc - 1.0 + fpr, 0.0, 1.0)
    feasible = tpr >= fpr
    if not feasible.any():
        return float("nan")
    values = [
        _f1_from_operating_point(float(t), float(f), prevalence)
        for t, f in zip(tpr[feasible], fpr[feasible])
    ]
    return float(np.max(values))


def maximum_f1_binormal(auc: float, prevalence: float, grid: int = 20_001) -> float:
    if not np.isfinite(auc) or not 0.0 < prevalence < 1.0 or not 0.0 < auc < 1.0:
        return float("nan")
    separation = np.sqrt(2.0) * norm.ppf(auc)
    fpr = np.linspace(1e-6, 1.0 - 1e-6, grid)
    tpr = norm.cdf(norm.ppf(fpr) + separation)
    values = [_f1_from_operating_point(float(t), float(f), prevalence) for t, f in zip(tpr, fpr)]
    return float(np.max(values))


def roc_points(labels: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(np.asarray(labels).astype(int), np.asarray(scores, dtype=float))
    return fpr, tpr


def cumulative_regret(best_reward: np.ndarray, achieved: np.ndarray) -> np.ndarray:
    return np.cumsum(np.asarray(best_reward, dtype=float) - np.asarray(achieved, dtype=float))


def detection_metrics(flags: np.ndarray, poisoned: np.ndarray) -> Tuple[float, float]:
    flags = np.asarray(flags).astype(bool)
    poisoned = np.asarray(poisoned).astype(bool)
    detection = float(flags[poisoned].mean()) if poisoned.any() else float("nan")
    false_positive = float(flags[~poisoned].mean()) if (~poisoned).any() else float("nan")
    return detection, false_positive


def recovery_cycles(
    accuracy: np.ndarray, shift_cycle: int, pre_shift_window: int = 100, tolerance_sigma: float = 1.0
) -> int:
    accuracy = np.asarray(accuracy, dtype=float)
    start = max(shift_cycle - pre_shift_window, 0)
    if shift_cycle <= start:
        return 0
    pre_mean = float(np.mean(accuracy[start:shift_cycle]))
    pre_std = float(np.std(accuracy[start:shift_cycle]))
    target = pre_mean - tolerance_sigma * pre_std
    for offset in range(shift_cycle, len(accuracy)):
        window = accuracy[offset : offset + 20]
        if len(window) >= 5 and float(np.mean(window)) >= target:
            return offset - shift_cycle
    return len(accuracy) - shift_cycle
