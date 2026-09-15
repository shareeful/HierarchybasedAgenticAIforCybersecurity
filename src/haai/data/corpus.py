from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .feeds import FeedBundle, load_feeds

LOGGER = get_logger()


@dataclass
class ValidationReport:
    received: int
    missing_field_records: int
    outlier_records: int
    retained: int
    outlier_fields: Dict[str, int] = field(default_factory=dict)
    missing_fields: Dict[str, int] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        rows = [
            {"stage": "records received", "count": self.received},
            {"stage": "removed: incomplete enrichment", "count": self.missing_field_records},
            {"stage": "removed: statistical outlier z>3", "count": self.outlier_records},
            {"stage": "records retained", "count": self.retained},
        ]
        for name, count in sorted(self.missing_fields.items()):
            rows.append({"stage": f"  incomplete field: {name}", "count": count})
        for name, count in sorted(self.outlier_fields.items()):
            rows.append({"stage": f"  outlier field: {name}", "count": count})
        return pd.DataFrame(rows)


@dataclass
class LabelledCorpus:
    corpus: pd.DataFrame
    labelled: pd.DataFrame
    uncertain: pd.DataFrame
    train: pd.DataFrame
    finetune: pd.DataFrame
    reward_tuning: pd.DataFrame
    evaluation: pd.DataFrame
    validation: ValidationReport
    provenance: Dict[str, str]

    def characteristics(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ("Corpus records (complete enrichment)", len(self.corpus)),
                ("Disclosure year range", f"{int(self.corpus.year.min())}-{int(self.corpus.year.max())}"),
                ("Confirmed-exploited records", int(self.corpus["kev"].sum())),
                ("Uncertain records held out", len(self.uncertain)),
                ("Sampled negative records", int((~self.labelled["label"].astype(bool)).sum())),
                ("Labelled set", len(self.labelled)),
                ("Positive prevalence", f"{self.labelled['label'].mean() * 100:.1f}%"),
                ("Training split", f"{len(self.train)} records"),
                ("  of which LoRA fine-tuning", f"{len(self.finetune)} records"),
                ("  of which reward/threshold tuning", f"{len(self.reward_tuning)} records"),
                ("Evaluation split", f"{len(self.evaluation)} records"),
                ("Unique CWE categories", int(self.corpus["cwe"].nunique())),
                ("Unique ATT&CK techniques", int(self.corpus["attack_technique"].nunique())),
                ("Distinct vendors", int(self.corpus["vendor"].nunique())),
            ],
            columns=["Property", "Value"],
        )


def enrich(bundle: FeedBundle, config: Config) -> pd.DataFrame:
    frame = bundle.vulnerabilities.copy()
    frame = frame.merge(bundle.epss, on="cve_id", how="left")
    kev_ids = set(bundle.kev["cve_id"])
    frame["kev"] = frame["cve_id"].isin(kev_ids)
    frame = frame.merge(bundle.exploitdb, on="cve_id", how="left")
    frame["public_exploit_count"] = frame["public_exploit_count"].fillna(0).astype(int)
    frame["poc_available"] = frame["public_exploit_count"] > 0

    family = dict(zip(bundle.cwe["cwe"], bundle.cwe["cwe_family"]))
    frame["cwe_family"] = frame["cwe"].map(family).fillna("unmapped")
    frame["cwe_known"] = frame["cwe"].isin(set(bundle.cwe["cwe"]))

    def first_technique(cwe_list) -> str:
        for cwe in cwe_list:
            techniques = bundle.cwe_attack.get(cwe)
            if techniques:
                return techniques[0]
        return ""

    def all_techniques(cwe_list) -> Tuple[str, ...]:
        collected: List[str] = []
        for cwe in cwe_list:
            collected.extend(bundle.cwe_attack.get(cwe, ()))
        return tuple(dict.fromkeys(collected))

    frame["attack_technique"] = [first_technique(value) for value in frame["cwe_list"]]
    frame["attack_techniques"] = [all_techniques(value) for value in frame["cwe_list"]]
    frame["affected_software_breadth"] = [len(value) for value in frame["affected_software"]]
    frame["vendor_count"] = [len(value) for value in frame["vendor_list"]]
    return frame


def attach_priors(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    exploited = reference[reference["kev"]]
    cwe_rate = exploited["cwe"].value_counts() / max(len(exploited), 1)
    technique_rate = exploited["attack_technique"].value_counts() / max(len(exploited), 1)
    cwe_scale = float(cwe_rate.max()) if len(cwe_rate) else 1.0
    technique_scale = float(technique_rate.max()) if len(technique_rate) else 1.0
    frame = frame.copy()
    frame["cwe_risk_prior"] = (
        frame["cwe"].map(cwe_rate).fillna(0.0) / (cwe_scale if cwe_scale else 1.0)
    )
    frame["attack_technique_prior"] = (
        frame["attack_technique"].map(technique_rate).fillna(0.0)
        / (technique_scale if technique_scale else 1.0)
    )
    return frame


def _completeness_filter(
    frame: pd.DataFrame, required: Tuple[str, ...]
) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    mask = pd.Series(True, index=frame.index)
    per_field: Dict[str, int] = {}
    for name in required:
        if name not in frame.columns:
            continue
        if name in ("affected_software", "cwe_list", "vendor_list"):
            present = frame[name].map(lambda value: value is not None and len(value) > 0)
        elif name == "cwe":
            present = frame[name].astype(str).str.startswith("CWE-") & frame["cwe_known"]
        else:
            present = frame[name].notna()
        per_field[name] = int((~present).sum())
        mask &= present
    return frame[mask], int((~mask).sum()), per_field


def _outlier_filter(frame: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, int, Dict[str, int]]:
    scales = {"cvss": "linear", "epss": "logit", "affected_software_breadth": "log"}
    mask = pd.Series(True, index=frame.index)
    per_field: Dict[str, int] = {}
    for column, scale in scales.items():
        if column not in frame.columns:
            continue
        values = frame[column].astype(float).to_numpy()
        if scale == "logit":
            clipped = np.clip(values, 1e-6, 1 - 1e-6)
            values = np.log(clipped / (1.0 - clipped))
        elif scale == "log":
            values = np.log1p(np.clip(values, 0.0, None))
        std = values.std()
        if std <= 0:
            continue
        z = np.abs(values - values.mean()) / std
        field_mask = z <= threshold
        per_field[column] = int((~field_mask).sum())
        mask &= field_mask
    return frame[mask], int((~mask).sum()), per_field


def _stratified_negatives(
    positives: pd.DataFrame,
    candidates: pd.DataFrame,
    ratio: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    required = len(positives) * ratio
    key_levels = (
        ("year", "cwe_family", "vendor"),
        ("year", "cwe_family"),
        ("cwe_family",),
        ("year",),
    )
    positions = np.arange(len(candidates))
    taken = np.zeros(len(candidates), dtype=bool)
    selected: List[int] = []

    demand: Dict[tuple, int] = {}
    for stratum, count in positives.groupby(list(key_levels[0]), observed=True).size().items():
        key = tuple(stratum) if isinstance(stratum, tuple) else (stratum,)
        demand[key] = int(count) * ratio

    for columns in key_levels:
        if not demand:
            break
        buckets: Dict[tuple, np.ndarray] = {}
        for stratum, index in candidates.groupby(list(columns), observed=True).indices.items():
            key = tuple(stratum) if isinstance(stratum, tuple) else (stratum,)
            buckets[key] = np.asarray(index)
        remaining: Dict[tuple, int] = {}
        column_positions = [key_levels[0].index(c) for c in columns]
        for stratum, count in demand.items():
            key = tuple(stratum[i] for i in column_positions)
            available = buckets.get(key)
            if available is None:
                remaining[stratum] = count
                continue
            free = available[~taken[available]]
            if free.size == 0:
                remaining[stratum] = count
                continue
            take = min(count, free.size)
            picked = rng.choice(free, size=take, replace=False)
            taken[picked] = True
            selected.extend(picked.tolist())
            if take < count:
                remaining[stratum] = count - take
        demand = remaining

    shortfall = required - len(selected)
    if shortfall > 0:
        free = positions[~taken]
        if free.size:
            extra = rng.choice(free, size=min(shortfall, free.size), replace=False)
            selected.extend(extra.tolist())
    return candidates.iloc[sorted(selected[:required])]


def _stratified_split(
    frame: pd.DataFrame, fraction: float, rng: np.random.Generator, strata: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    first: List[int] = []
    for _, group in frame.groupby(strata, observed=True):
        index = np.array(group.index.to_numpy(), copy=True)
        rng.shuffle(index)
        cut = int(round(len(index) * fraction))
        first.extend(index[:cut].tolist())
    first_set = set(first)
    left = frame.loc[sorted(first_set)]
    right = frame.loc[[i for i in frame.index if i not in first_set]]
    return left, right


def build_labelled_corpus(
    config: Config,
    rng: np.random.Generator,
    root: Path,
    bundle: FeedBundle | None = None,
) -> LabelledCorpus:
    bundle = bundle if bundle is not None else load_feeds(root, config.corpus.year_range)
    frame = enrich(bundle, config)
    received = len(frame)

    frame, missing, missing_fields = _completeness_filter(frame, config.corpus.completeness_required)
    frame, outliers, outlier_fields = _outlier_filter(frame, config.corpus.outlier_z_threshold)
    frame = frame.reset_index(drop=True)
    frame = attach_priors(frame, frame)

    validation = ValidationReport(
        received=received,
        missing_field_records=missing,
        outlier_records=outliers,
        retained=len(frame),
        outlier_fields=outlier_fields,
        missing_fields=missing_fields,
    )
    LOGGER.info(
        "Phase 1 validation: %d received, %d incomplete, %d outliers, %d retained",
        received,
        missing,
        outliers,
        len(frame),
    )
    if len(frame) == 0:
        raise ValueError(
            "no records survived Phase 1 validation; check that the NVD, EPSS and CWE feeds "
            "cover the configured year range"
        )

    frame["risk_target"] = np.where(frame["kev"], 1.0, frame["epss"].astype(float))
    frame["epss_quartile"] = pd.qcut(
        frame["epss"].rank(method="first"), config.bandit.epss_quartiles, labels=False
    ).astype(int)

    positives = frame[frame["kev"]].copy()
    if len(positives) == 0:
        raise ValueError(
            "the KEV catalogue and the NVD feed do not intersect; the labelled set would be empty"
        )
    uncertain_mask = (~frame["kev"]) & (frame["epss"] >= config.corpus.uncertain_epss_floor)
    uncertain = frame[uncertain_mask].copy()
    negative_pool = frame[(~frame["kev"]) & (~uncertain_mask)].copy()

    negatives = _stratified_negatives(
        positives, negative_pool, config.corpus.negatives_per_positive, rng
    )
    positives = positives.assign(label=1)
    negatives = negatives.assign(label=0)
    labelled = (
        pd.concat([positives, negatives], ignore_index=True)
        .sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        .reset_index(drop=True)
    )
    LOGGER.info(
        "Labelled set: %d records (%d positive, %d negative, prevalence %.1f%%); %d uncertain held out",
        len(labelled),
        int(labelled["label"].sum()),
        int((labelled["label"] == 0).sum()),
        100 * labelled["label"].mean(),
        len(uncertain),
    )

    strata = ["label", "cwe_family"]
    train, evaluation = _stratified_split(labelled, config.corpus.train_fraction, rng, strata)
    finetune, reward_tuning = _stratified_split(
        train, config.corpus.finetune_fraction_of_train, rng, strata
    )

    provenance = {
        "nvd_years": ",".join(str(year) for year in bundle.years),
        "nvd_records": str(len(bundle.vulnerabilities)),
        "epss_records": str(len(bundle.epss)),
        "kev_records": str(len(bundle.kev)),
        "attack_techniques": str(len(bundle.attack)),
        "cwe_entries": str(len(bundle.cwe)),
        "exploitdb_cves": str(len(bundle.exploitdb)),
    }

    return LabelledCorpus(
        corpus=frame,
        labelled=labelled,
        uncertain=uncertain.assign(label=np.nan),
        train=train.reset_index(drop=True),
        finetune=finetune.reset_index(drop=True),
        reward_tuning=reward_tuning.reset_index(drop=True),
        evaluation=evaluation.reset_index(drop=True),
        validation=validation,
        provenance=provenance,
    )


def alternative_labelling(
    corpus: LabelledCorpus, config: Config, mode: str, rng: np.random.Generator
) -> pd.DataFrame:
    if mode == "uncertain_as_negative":
        forced = corpus.uncertain.assign(label=0)
        return pd.concat([corpus.labelled, forced], ignore_index=True)
    if mode == "ratio_20":
        positives = corpus.corpus[corpus.corpus["kev"]].assign(label=1)
        uncertain_ids = set(corpus.uncertain["cve_id"])
        pool = corpus.corpus[
            (~corpus.corpus["kev"]) & (~corpus.corpus["cve_id"].isin(uncertain_ids))
        ]
        negatives = _stratified_negatives(positives, pool, 20, rng).assign(label=0)
        return pd.concat([positives, negatives], ignore_index=True)
    raise ValueError(f"unknown labelling mode: {mode}")
