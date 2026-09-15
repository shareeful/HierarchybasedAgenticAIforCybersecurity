from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from ..metrics import (
    classification_metrics,
    maximum_f1_binormal,
    maximum_f1_hull_bound,
    roc_points,
    select_threshold,
)
from ..seeds import SeedBook
from ..stats import format_mean_std, paired_wilcoxon
from .systems import SystemRun, ablation_record_scores

LOGGER = get_logger()


@dataclass
class Experiment1Result:
    table9: pd.DataFrame
    table10: pd.DataFrame
    per_run: pd.DataFrame
    ablation_per_run: pd.DataFrame
    roc: Dict[str, Tuple[np.ndarray, np.ndarray]]
    thresholds: Dict[str, float]
    significance: pd.DataFrame
    parse_telemetry: pd.DataFrame
    consistency: pd.DataFrame

    def improvement(self, config: Config) -> str:
        frame = self.table9.set_index("raw_key")
        others = [name for name in config.evaluation.baselines if name != "proposed"]
        best = max(others, key=lambda name: float(frame.loc[name, "f1_mean"]))
        delta = float(frame.loc["proposed", "f1_mean"]) - float(frame.loc[best, "f1_mean"])
        row = self.significance.set_index("comparison")
        key = f"proposed vs {best}"
        detail = row.loc[key, "result"] if key in row.index else ""
        return (
            f"Proposed F1 {frame.loc['proposed', 'f1']} AUC {frame.loc['proposed', 'auc']}; "
            f"improvement over {config.evaluation.baseline_display[best]} of {delta:+.3f} F1, {detail}"
        )


def _bootstrap_indices(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    picked: List[np.ndarray] = []
    for value in np.unique(labels):
        index = np.flatnonzero(labels == value)
        picked.append(rng.choice(index, size=len(index), replace=True))
    return np.concatenate(picked)


def run_experiment1(config: Config, run: SystemRun, evaluation_pairs: pd.DataFrame, n_runs: int | None = None) -> Experiment1Result:
    n_runs = n_runs or config.evaluation.n_runs
    book = SeedBook(config.evaluation.master_seed)

    thresholds: Dict[str, float] = {}
    for name in config.evaluation.baselines:
        tuning = run.tuning_scores.get(name)
        if tuning is None:
            thresholds[name] = 0.5
            continue
        thresholds[name] = select_threshold(
            tuning["label"].to_numpy(dtype=int), tuning["score"].to_numpy(dtype=float)
        )

    rows: List[dict] = []
    roc: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name in config.evaluation.baselines:
        frame = run.record_scores[name]
        labels = frame["label"].to_numpy(dtype=int)
        scores = frame["score"].to_numpy(dtype=float)
        roc[name] = roc_points(labels, scores)
        for index in range(n_runs):
            rng = book.stream(f"experiment1:{name}", index)
            picked = _bootstrap_indices(labels, rng)
            measured = classification_metrics(labels[picked], scores[picked], thresholds[name])
            rows.append({"system": name, "run": index, **measured.as_dict()})
    per_run = pd.DataFrame(rows)

    table_rows: List[dict] = []
    consistency_rows: List[dict] = []
    for name in config.evaluation.baselines:
        subset = per_run[per_run["system"] == name]
        auc_mean = float(subset["auc"].mean())
        prevalence = float(subset["prevalence"].mean())
        hull = maximum_f1_hull_bound(auc_mean, prevalence)
        binormal = maximum_f1_binormal(auc_mean, prevalence)
        observed = float(subset["f1"].mean())
        table_rows.append(
            {
                "System": config.evaluation.baseline_display[name],
                "raw_key": name,
                "precision": format_mean_std(subset["precision"]),
                "recall": format_mean_std(subset["recall"]),
                "f1": format_mean_std(subset["f1"]),
                "auc": format_mean_std(subset["auc"]),
                "average_precision": format_mean_std(subset["average_precision"]),
                "f1_mean": round(float(subset["f1"].mean()), 4),
                "auc_mean": round(auc_mean, 4),
                "threshold": round(thresholds[name], 4),
            }
        )
        consistency_rows.append(
            {
                "System": config.evaluation.baseline_display[name],
                "auc": round(auc_mean, 3),
                "prevalence": round(prevalence, 4),
                "observed_f1": round(observed, 3),
                "max_f1_concave_hull": round(hull, 3),
                "max_f1_proper_binormal": round(binormal, 3),
                "within_hull_bound": bool(observed <= hull + 1e-6) if np.isfinite(hull) else None,
                "within_binormal_shape": bool(observed <= binormal + 1e-6)
                if np.isfinite(binormal)
                else None,
            }
        )
    table9 = pd.DataFrame(table_rows)

    ablation_rows: List[dict] = []
    for name in config.evaluation.ablations:
        if name not in run.ablation_traces:
            continue
        frame = ablation_record_scores(run, name, evaluation_pairs)
        labels = frame["label"].to_numpy(dtype=int)
        scores = frame["score"].to_numpy(dtype=float)
        threshold = thresholds["proposed"] if name == "full" else select_threshold(labels, scores)
        for index in range(n_runs):
            rng = book.stream(f"experiment1:ablation:{name}", index)
            picked = _bootstrap_indices(labels, rng)
            measured = classification_metrics(labels[picked], scores[picked], threshold)
            ablation_rows.append({"variant": name, "run": index, **measured.as_dict()})
    ablation_per_run = pd.DataFrame(ablation_rows)

    table10_rows: List[dict] = []
    for name in config.evaluation.ablations:
        subset = ablation_per_run[ablation_per_run["variant"] == name]
        if not len(subset):
            continue
        full = ablation_per_run[ablation_per_run["variant"] == "full"]
        table10_rows.append(
            {
                "Variant": config.evaluation.ablation_display[name],
                "raw_key": name,
                "f1": format_mean_std(subset["f1"]),
                "auc": format_mean_std(subset["auc"]),
                "f1_mean": round(float(subset["f1"].mean()), 4),
                "auc_mean": round(float(subset["auc"].mean()), 4),
                "delta_f1": round(float(subset["f1"].mean() - full["f1"].mean()), 4),
            }
        )
    table10 = pd.DataFrame(table10_rows)

    significance_rows: List[dict] = []
    proposed = per_run[per_run["system"] == "proposed"].sort_values("run")["f1"].to_numpy()
    for name in config.evaluation.baselines:
        if name == "proposed":
            continue
        other = per_run[per_run["system"] == name].sort_values("run")["f1"].to_numpy()
        result = paired_wilcoxon(proposed, other)
        significance_rows.append(
            {
                "comparison": f"proposed vs {name}",
                "mean_delta_f1": round(float(np.mean(proposed - other)), 4),
                "result": result.format(),
                "p_value": result.p_value,
            }
        )
    significance = pd.DataFrame(significance_rows)

    return Experiment1Result(
        table9=table9,
        table10=table10,
        per_run=per_run,
        ablation_per_run=ablation_per_run,
        roc=roc,
        thresholds=thresholds,
        significance=significance,
        parse_telemetry=pd.DataFrame(run.telemetry.as_rows()),
        consistency=pd.DataFrame(consistency_rows),
    )


def label_noise_sensitivity(
    config: Config, run: SystemRun, rates: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10)
) -> pd.DataFrame:
    book = SeedBook(config.evaluation.master_seed)
    frame = run.record_scores["proposed"]
    labels = frame["label"].to_numpy(dtype=int)
    scores = frame["score"].to_numpy(dtype=float)
    baseline_threshold = select_threshold(labels, scores)
    rows = []
    for rate in rates:
        rng = book.stream(f"label_noise:{rate}", 0)
        flipped = labels.copy()
        if rate > 0:
            count = int(round(rate * len(labels)))
            index = rng.choice(len(labels), size=count, replace=False)
            flipped[index] = 1 - flipped[index]
        measured = classification_metrics(flipped, scores, baseline_threshold)
        rows.append(
            {
                "flip_rate": rate,
                "f1": round(measured.f1, 4),
                "auc": round(measured.auc, 4),
                "delta_f1": None,
            }
        )
    base_f1 = rows[0]["f1"]
    for row in rows:
        row["delta_f1"] = round(row["f1"] - base_f1, 4)
    return pd.DataFrame(rows)


def labelling_sensitivity(
    config: Config,
    pipeline,
    corpus,
    organisation,
    budget: int = 1_500,
    protocols: Tuple[str, ...] = ("as_reported", "uncertain_as_negative", "ratio_20"),
) -> pd.DataFrame:
    from ..data.corpus import alternative_labelling
    from ..pipeline.environment import make_pairs
    from ..pipeline.features import aggregate_to_records

    book = SeedBook(config.evaluation.master_seed)
    rows: List[dict] = []
    for protocol in protocols:
        rng = book.stream(f"labelling:{protocol}", 0)
        frame = (
            corpus.labelled
            if protocol == "as_reported"
            else alternative_labelling(corpus, config, protocol, rng)
        )
        frame = frame[frame["label"].notna()].reset_index(drop=True)
        if len(frame) > budget:
            index = np.sort(rng.permutation(len(frame))[:budget])
            frame = frame.iloc[index].reset_index(drop=True)
        pairs = make_pairs(config, frame, organisation)
        trace = pipeline.assess(pairs, key=f"labelling:{protocol}")
        records = aggregate_to_records(pairs.assign(score=trace.joint_risk), "score")
        labels = records["label"].to_numpy(dtype=int)
        scores = records["score"].to_numpy(dtype=float)
        if len(np.unique(labels)) < 2:
            LOGGER.warning("protocol %s produced a single-class evaluation set", protocol)
            continue
        threshold = select_threshold(labels, scores)
        measured = classification_metrics(labels, scores, threshold)
        rows.append(
            {
                "protocol": protocol,
                "records": measured.support,
                "prevalence": round(measured.prevalence, 4),
                "f1": round(measured.f1, 4),
                "auc": round(measured.auc, 4),
                "precision": round(measured.precision, 4),
                "recall": round(measured.recall, 4),
                "threshold_selected_within_protocol": round(threshold, 4),
            }
        )
    return pd.DataFrame(rows)
