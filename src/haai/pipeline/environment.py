from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from ..config import Config
from ..data.corpus import LabelledCorpus, build_labelled_corpus
from ..data.feeds import FeedBundle, load_feeds
from ..data.organisation import (
    Organisation,
    build_exposure_pairs,
    exposure_ground_truth,
    load_organisation,
    load_outcomes,
    load_telemetry,
)
from ..data.registry import data_root, require_all
from ..data.usecase import UseCase, load_use_case
from ..logging_utils import get_logger, timed

LOGGER = get_logger()


@dataclass
class Environment:
    root: Path
    feeds: FeedBundle
    corpus: LabelledCorpus
    organisation: Organisation
    use_case: UseCase
    outcomes: pd.DataFrame
    telemetry: pd.DataFrame

    @property
    def assets(self) -> pd.DataFrame:
        return self.organisation.assets

    @property
    def product_disruption_rate(self) -> Dict[str, float]:
        merged = dict(self.organisation.product_disruption_rate)
        merged.update(self.use_case.organisation.product_disruption_rate)
        return merged

    def provenance(self) -> pd.DataFrame:
        rows = [{"item": key, "value": value} for key, value in self.corpus.provenance.items()]
        rows.append({"item": "data_root", "value": str(self.root)})
        rows.append({"item": "organisation_hosts", "value": str(len(self.organisation.assets))})
        rows.append({"item": "use_case_hosts", "value": str(len(self.use_case.organisation.assets))})
        rows.append({"item": "remediation_decisions", "value": str(len(self.outcomes))})
        rows.append({"item": "telemetry_observations", "value": str(len(self.telemetry))})
        rows.append({"item": "analyst_decisions", "value": str(len(self.use_case.decisions))})
        return pd.DataFrame(rows)


def build_environment(config: Config, root: str | Path | None = None) -> Environment:
    base = require_all(data_root(root or (config.runtime.data_dir or None)))
    with timed("loading vulnerability intelligence feeds", LOGGER):
        feeds = load_feeds(base, config.corpus.year_range)
    with timed("Phase 1 corpus construction", LOGGER):
        corpus = build_labelled_corpus(
            config, np.random.default_rng(config.evaluation.master_seed), base, feeds
        )
    with timed("loading organisational records", LOGGER):
        organisation = load_organisation(base, "organisation", config, "organisation")
        use_case = load_use_case(base, config, "usecase")
        outcomes = load_outcomes(base, "organisation")
        telemetry = load_telemetry(base, "organisation")
    return Environment(
        root=base,
        feeds=feeds,
        corpus=corpus,
        organisation=organisation,
        use_case=use_case,
        outcomes=outcomes,
        telemetry=telemetry,
    )


def make_pairs(
    config: Config,
    vulnerabilities: pd.DataFrame,
    organisation: Organisation,
) -> pd.DataFrame:
    limit = config.org.max_assets_per_cve or None
    pairs = build_exposure_pairs(config, vulnerabilities, organisation, limit)
    columns = [
        column
        for column in (
            "cve_id",
            "cvss",
            "epss",
            "poc_available",
            "cwe",
            "cwe_family",
            "cwe_risk_prior",
            "attack_technique",
            "attack_technique_prior",
            "affected_software",
            "affected_software_breadth",
            "vendor",
            "year",
            "kev",
            "label",
            "risk_target",
            "epss_quartile",
            "description",
        )
        if column in vulnerabilities.columns
    ]
    merged = pairs.merge(vulnerabilities.loc[:, columns], on="cve_id", how="inner")
    merged["exposure_ground_truth"] = exposure_ground_truth(
        config, merged, merged["epss"].to_numpy(dtype=float)
    )
    return merged.reset_index(drop=True)
