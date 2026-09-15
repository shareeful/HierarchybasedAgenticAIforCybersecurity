from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import pandas as pd

PAPER_VALUES: Dict[str, Dict[str, float]] = {
    "table8": {
        "corpus_records": 118_000,
        "confirmed_exploited": 742,
        "uncertain_held_out": 6_880,
        "sampled_negatives": 7_420,
        "labelled_set": 8_162,
        "positive_prevalence": 0.091,
        "training_split": 6_530,
        "evaluation_split": 1_632,
        "unique_cwe": 168,
        "unique_attack": 71,
        "distinct_vendors": 1_240,
    },
    "table7": {
        "hosts": 57,
        "subsystems": 5,
        "segments": 5,
        "software_components": 214,
        "pairs": 2_940,
        "patch_records": 1_860,
        "alerts": 3_412,
        "analyst_decisions": 486,
    },
    "table9_f1": {
        "cvss_only": 0.587,
        "cvss_epss_calibrated": 0.672,
        "xgboost": 0.652,
        "securebert_cve": 0.727,
        "rag_llm": 0.743,
        "single_agent": 0.690,
        "flat_multi_agent": 0.756,
        "proposed": 0.846,
    },
    "table9_auc": {
        "cvss_only": 0.629,
        "cvss_epss_calibrated": 0.709,
        "xgboost": 0.678,
        "securebert_cve": 0.765,
        "rag_llm": 0.779,
        "single_agent": 0.723,
        "flat_multi_agent": 0.793,
        "proposed": 0.881,
    },
    "table10_f1": {
        "full": 0.846,
        "no_supervisor": 0.767,
        "no_mab": 0.783,
        "no_integrity_check": 0.832,
        "no_confidence_weighting": 0.815,
    },
    "table11_regret": {
        "static": 551.2,
        "dqn": 241.6,
        "discounted_ucb": 249.5,
        "thompson": 218.3,
        "sw_ucb": 190.4,
    },
    "table11_accuracy_post": {
        "static": 0.619,
        "dqn": 0.704,
        "discounted_ucb": 0.687,
        "thompson": 0.709,
        "sw_ucb": 0.778,
    },
    "table11_recovery": {
        "sw_ucb": 41,
        "thompson": 57,
        "discounted_ucb": 71,
        "dqn": 118,
    },
    "table14_detection": {
        0.05: 0.775, 0.10: 0.847, 0.15: 0.891, 0.20: 0.927, 0.25: 0.949, 0.30: 0.961,
    },
    "table14_quality": {
        0.05: 0.836, 0.10: 0.816, 0.15: 0.793, 0.20: 0.772, 0.25: 0.752, 0.30: 0.731,
    },
    "table16_use_case_f1": {
        "cvss_only": 0.559,
        "cvss_epss_calibrated": 0.641,
        "xgboost": 0.619,
        "securebert_cve": 0.698,
        "rag_llm": 0.716,
        "single_agent": 0.659,
        "flat_multi_agent": 0.727,
        "proposed": 0.821,
    },
    "hyperparameters": {"tau": 80, "gamma": 0.85, "isolation_threshold": 0.62},
    "explainability": {
        "surrogate_r2_va": 0.93,
        "surrogate_r2_ca": 0.91,
        "shap_deletion_top": 0.241,
        "shap_deletion_random": 0.052,
        "lime_deletion_top": 0.198,
        "lime_deletion_random": 0.049,
        "attention_deletion_top": 0.173,
        "attention_deletion_random": 0.046,
        "counterfactual_validity": 0.941,
    },
    "parsing": {
        "malformed_vulnerability": 0.014,
        "malformed_supervisor": 0.021,
        "schema_violations": 0.005,
    },
    "other": {
        "analyst_agreement": 0.794,
        "flat_analyst_agreement": 0.612,
        "cvss_analyst_agreement": 0.479,
        "overhead_percent_40k": 10.6,
        "anomaly_target": 0.241,
        "response_target": 0.314,
        "supervisor_label_analyst_agreement": 0.813,
    },
}


@dataclass
class ComparisonRow:
    quantity: str
    paper: float
    reproduced: float
    unit: str = ""
    note: str = ""

    @property
    def absolute_error(self) -> float:
        return abs(self.reproduced - self.paper)

    @property
    def relative_error(self) -> float:
        return self.absolute_error / abs(self.paper) if self.paper else float("nan")


def build_comparison(rows: Sequence[ComparisonRow]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "quantity": row.quantity,
                "paper": row.paper,
                "reproduced": row.reproduced,
                "absolute_error": round(row.absolute_error, 4),
                "relative_error": round(row.relative_error, 4),
                "unit": row.unit,
                "note": row.note,
            }
            for row in rows
        ]
    )
    return frame


DATASET_DEPENDENT_NOTE = (
    "depends entirely on the dataset supplied; comparable only when the reader supplies the "
    "same corpus and pilot records the paper used"
)


def summarise_comparison(frame: pd.DataFrame) -> Dict[str, float]:
    if len(frame) == 0:
        return {
            "quantities_compared": 0,
            "mean_absolute_error": float("nan"),
            "median_relative_error": float("nan"),
            "within_5_percent": 0,
            "within_10_percent": 0,
        }
    return {
        "quantities_compared": int(len(frame)),
        "mean_absolute_error": float(frame["absolute_error"].mean()),
        "median_relative_error": float(frame["relative_error"].median()),
        "within_5_percent": int((frame["relative_error"] <= 0.05).sum()),
        "within_10_percent": int((frame["relative_error"] <= 0.10).sum()),
    }
