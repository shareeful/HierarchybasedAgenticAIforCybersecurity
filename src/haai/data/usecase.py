from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger
from .organisation import Organisation, as_bool, load_organisation
from .registry import read_table
from .spec import ALERTS, DECISIONS, OPERATOR_TARGETS, SCENARIOS, SUBSYSTEMS

LOGGER = get_logger()


@dataclass
class UseCase:
    organisation: Organisation
    subsystems: pd.DataFrame
    scenarios: pd.DataFrame
    alerts: pd.DataFrame
    decisions: pd.DataFrame
    operator_targets: pd.DataFrame

    def subsystem_table(self) -> pd.DataFrame:
        columns = ["subsystem", "hosts", "criticality", "segment"]
        if "description" in self.subsystems.columns:
            columns.append("description")
        return self.subsystems.loc[:, columns].reset_index(drop=True)

    def scenario_table(self) -> pd.DataFrame:
        return self.scenarios.reset_index(drop=True)

    def summary(self, pair_count: int) -> pd.DataFrame:
        window = ""
        if len(self.organisation.changes) and self.organisation.changes["change_date"].notna().any():
            span = self.organisation.changes["change_date"]
            months = (span.max() - span.min()).days / 30.44
            window = f"{months:.0f}"
        return pd.DataFrame(
            [
                ("Hosts in asset inventory", len(self.organisation.assets)),
                ("Operational subsystems", len(self.subsystems)),
                ("Network segments", int(self.organisation.assets["segment"].nunique())),
                ("Distinct software components", int(self.organisation.software["product"].nunique())),
                ("Vulnerability-asset pairs on these hosts", pair_count),
                ("Control families recorded", len(self.organisation.control_families)),
                ("Patch and change records", len(self.organisation.changes)),
                ("Change record window (months)", window),
                ("Attack scenarios defined", len(self.scenarios)),
                ("Attack scenarios executed", int(self.scenarios["executed"].sum())),
                ("Security operations alerts captured", len(self.alerts)),
                ("Analyst triage decisions", len(self.decisions)),
                ("Analysts contributing decisions", int(self.decisions["analyst_id"].nunique())
                 if "analyst_id" in self.decisions.columns else ""),
            ],
            columns=["Property", "Value"],
        )


def load_use_case(root: Path, config: Config, subdirectory: str = "usecase") -> UseCase:
    organisation = load_organisation(root, subdirectory, config, "use case")

    subsystems = read_table(root, SUBSYSTEMS.key, subdirectory).copy()
    subsystems["subsystem"] = subsystems["subsystem"].astype(str)
    subsystems["hosts"] = pd.to_numeric(subsystems["hosts"], errors="coerce").astype("Int64")
    subsystems["criticality"] = pd.to_numeric(subsystems["criticality"], errors="coerce")

    scenarios = read_table(root, SCENARIOS.key, subdirectory).copy()
    scenarios["executed"] = as_bool(scenarios["executed"])

    alerts = read_table(root, ALERTS.key, subdirectory).copy()
    alerts["asset_id"] = alerts["asset_id"].astype(str)
    alerts["raised_at"] = pd.to_datetime(alerts["raised_at"], errors="coerce", utc=True)
    for column in ("acknowledged_at", "resolved_at"):
        if column in alerts.columns:
            alerts[column] = pd.to_datetime(alerts[column], errors="coerce", utc=True)

    decisions = read_table(root, DECISIONS.key, subdirectory).copy()
    decisions["cve_id"] = decisions["cve_id"].astype(str)
    decisions["asset_id"] = decisions["asset_id"].astype(str)
    decisions["action"] = decisions["action"].astype(str).str.strip().str.lower()
    decisions["decided_at"] = pd.to_datetime(decisions["decided_at"], errors="coerce", utc=True)
    unknown = sorted(set(decisions["action"]) - set(config.bandit.actions))
    if unknown:
        raise ValueError(
            f"analyst_decisions.csv contains actions outside the configured set "
            f"{config.bandit.actions}: {unknown}"
        )
    if "urgency" in decisions.columns:
        decisions["urgency"] = pd.to_numeric(decisions["urgency"], errors="coerce")

    targets = read_table(root, OPERATOR_TARGETS.key, subdirectory).copy()
    targets["metric"] = targets["metric"].astype(str)
    targets["baseline"] = pd.to_numeric(targets["baseline"], errors="coerce")
    targets["target"] = pd.to_numeric(targets["target"], errors="coerce")

    missing_assets = set(decisions["asset_id"]) - set(organisation.assets["asset_id"])
    if missing_assets:
        raise ValueError(
            f"analyst_decisions.csv references {len(missing_assets)} asset ids absent from "
            f"{subdirectory}/assets.csv"
        )
    LOGGER.info(
        "use case: %d hosts, %d subsystems, %d alerts, %d recorded analyst decisions",
        len(organisation.assets),
        len(subsystems),
        len(alerts),
        len(decisions),
    )
    return UseCase(
        organisation=organisation,
        subsystems=subsystems.reset_index(drop=True),
        scenarios=scenarios.reset_index(drop=True),
        alerts=alerts.reset_index(drop=True),
        decisions=decisions.reset_index(drop=True),
        operator_targets=targets.reset_index(drop=True),
    )


def alert_pressure(use_case: UseCase) -> pd.Series:
    counts = use_case.alerts.groupby("asset_id").size()
    if not len(counts):
        return pd.Series(dtype=float)
    return counts / float(counts.max())


def response_times(use_case: UseCase) -> pd.DataFrame:
    alerts = use_case.alerts
    if "acknowledged_at" not in alerts.columns:
        return pd.DataFrame(columns=["asset_id", "acknowledge_seconds", "resolve_seconds"])
    acknowledge = (alerts["acknowledged_at"] - alerts["raised_at"]).dt.total_seconds()
    resolve = (
        (alerts["resolved_at"] - alerts["raised_at"]).dt.total_seconds()
        if "resolved_at" in alerts.columns
        else pd.Series(np.nan, index=alerts.index)
    )
    return pd.DataFrame(
        {
            "asset_id": alerts["asset_id"],
            "acknowledge_seconds": acknowledge,
            "resolve_seconds": resolve,
        }
    ).dropna(subset=["acknowledge_seconds"])
