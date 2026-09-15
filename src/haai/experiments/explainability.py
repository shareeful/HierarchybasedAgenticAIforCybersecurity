from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..logging_utils import get_logger, timed
from ..pipeline.features import matrix_for
from ..seeds import SeedBook
from ..xai.counterfactual import Counterfactual, counterfactual_frame, search_counterfactual
from ..xai.faithfulness import (
    DeletionResult,
    counterfactual_validity,
    deletion_test,
    model_deletion_test,
)
from ..xai.kernel_shap import kernel_shap
from ..xai.lime_local import lime_explain
from ..xai.shap_exact import exact_shapley
from ..xai.surrogate import ReferenceSurrogate, build_surrogate
from .systems import SystemRun

LOGGER = get_logger()


@dataclass
class ExplainabilityResult:
    faithfulness: pd.DataFrame
    surrogate_fidelity: pd.DataFrame
    attributions: pd.DataFrame
    counterfactuals: pd.DataFrame
    counterfactual_validity: float
    counterfactuals_proposed: int
    counterfactuals_verified: int
    attention: pd.DataFrame
    coalition_budget: pd.DataFrame


def _model_predictor(
    run: SystemRun, template: pd.DataFrame, fields: Sequence[str]
) -> Callable[[np.ndarray], np.ndarray]:
    agent = run.pipeline.vulnerability_agent

    def predict(matrix: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(matrix)
        block = pd.concat([template] * int(np.ceil(len(matrix) / len(template))), ignore_index=True)
        block = block.iloc[: len(matrix)].reset_index(drop=True)
        for position, name in enumerate(fields):
            if name == "poc_available":
                block[name] = matrix[:, position] > 0.5
            else:
                block[name] = matrix[:, position]
        return agent.assess(block).score

    return predict


def run_explainability(
    config: Config, run: SystemRun, evaluation_pairs: pd.DataFrame, sample_size: int = 64
) -> ExplainabilityResult:
    book = SeedBook(config.evaluation.master_seed)
    rng = book.stream("explainability", 0)
    fields = tuple(config.xai.vulnerability_fields)
    unique = run.trace.unique
    matrix = matrix_for(unique, fields)
    target = np.asarray(run.trace.vulnerability_unique.score, dtype=float)

    fidelity_rows: List[dict] = []
    surrogates: Dict[str, ReferenceSurrogate] = {}
    for family in config.xai.surrogate_families:
        surrogate = build_surrogate(config, family, fields, matrix, target, rng)
        surrogates[family] = surrogate
        fidelity_rows.append(
            {
                "agent": "Vulnerability Agent",
                "surrogate_family": family,
                "held_out_r2": round(surrogate.r2, 4),
                "training_records": len(matrix),
            }
        )

    contextual_fields = tuple(
        name for name in config.xai.contextual_fields if name in evaluation_pairs.columns
    )
    contextual_matrix = matrix_for(evaluation_pairs, contextual_fields)
    contextual_target = run.trace.contextual.score
    contextual_surrogates: Dict[str, ReferenceSurrogate] = {}
    for family in config.xai.surrogate_families:
        surrogate = build_surrogate(
            config, family, contextual_fields, contextual_matrix, contextual_target, rng
        )
        contextual_surrogates[family] = surrogate
        fidelity_rows.append(
            {
                "agent": "Contextual Awareness Agent",
                "surrogate_family": family,
                "held_out_r2": round(surrogate.r2, 4),
                "training_records": len(contextual_matrix),
            }
        )
    surrogate_fidelity = pd.DataFrame(fidelity_rows)

    primary = surrogates["gradient_boosting"]
    coalitions = 2 ** len(fields)
    affordable = max(config.xai.model_query_budget // coalitions, 1)
    instances = min(sample_size, affordable, len(matrix))
    picked = rng.choice(len(matrix), size=instances, replace=False)
    sample = matrix[picked]
    baseline = primary.marginal_expectation()

    predict = _model_predictor(run, unique.iloc[picked].reset_index(drop=True), fields)

    with timed(f"exact Shapley on the model over {instances} instances", LOGGER):
        model_shap = np.vstack(
            [
                _exact_shapley_on_model(predict, row, baseline, len(fields))
                for row in sample
            ]
        )
    surrogate_shap = np.vstack([exact_shapley(primary, row, baseline) for row in sample])
    kernel = np.vstack(
        [kernel_shap(primary, row, config.xai.kernel_shap_coalitions, rng, baseline) for row in sample]
    )
    lime_rows = []
    for row in sample:
        coefficients, _ = lime_explain(
            primary,
            row,
            config.xai.lime_neighbours,
            config.xai.lime_kernel_width,
            config.xai.lime_nonzero_coefficients,
            rng,
        )
        lime_rows.append(coefficients)
    lime = np.vstack(lime_rows)

    attention_frame = pd.DataFrame(columns=["field", "mean_attention"])
    if run.trace.vulnerability_unique.attention is not None:
        pooled = run.trace.vulnerability_unique.attention.mean(axis=0)
        attention_frame = pd.DataFrame(
            {
                "field": list(run.trace.vulnerability_unique.fields),
                "mean_attention": np.round(pooled, 5),
            }
        )

    results: List[DeletionResult] = [
        model_deletion_test(
            predict, sample, model_shap, baseline, config.xai.deletion_test_top_k, rng, "SHAP (exact, model)"
        ),
        deletion_test(
            primary, sample, kernel, config.xai.deletion_test_top_k, rng, "Kernel SHAP (surrogate)"
        ),
        deletion_test(primary, sample, lime, config.xai.deletion_test_top_k, rng, "LIME (surrogate)"),
    ]
    faithfulness = pd.DataFrame(
        [
            {
                "technique": item.technique,
                "evaluated_on": item.evaluated_on,
                "top_k_delta": round(item.top_k_delta, 5),
                "random_delta": round(item.random_delta, 5),
                "ratio": round(item.ratio(), 3),
                "instances": item.instances or instances,
            }
            for item in results
        ]
    )

    attributions = pd.DataFrame(model_shap, columns=[f"shap_{name}" for name in fields])
    attributions.insert(0, "cve_id", unique["cve_id"].to_numpy()[picked])
    for position, name in enumerate(fields):
        attributions[f"surrogate_shap_{name}"] = surrogate_shap[:, position]

    contextual_primary = contextual_surrogates["gradient_boosting"]
    zone_reach = dict(
        zip(evaluation_pairs["zone"].astype(str), evaluation_pairs["zone_reach"].astype(float))
    )
    counterfactuals: List[Counterfactual] = []
    field_index = {name: position for position, name in enumerate(contextual_fields)}
    cf_sample = rng.choice(len(evaluation_pairs), size=min(instances, len(evaluation_pairs)), replace=False)

    def recompute(row: Dict[str, float]) -> float:
        vector = np.zeros(len(contextual_fields))
        for name, position in field_index.items():
            vector[position] = float(row.get(name, 0.0))
        return float(contextual_primary.predict(vector.reshape(1, -1))[0])

    for position in cf_sample:
        record = evaluation_pairs.iloc[int(position)].to_dict()
        controls = tuple()
        candidate = search_counterfactual(config, record, controls, zone_reach, recompute)
        counterfactuals.append(candidate)

    verified = _verify_counterfactuals(
        run, config, evaluation_pairs, cf_sample, counterfactuals, contextual_fields
    )
    validity = counterfactual_validity(verified)

    coalition_budget = pd.DataFrame(
        [
            {
                "technique": "Exact Shapley on the model",
                "coalitions_per_instance": coalitions,
                "instances_explained": instances,
                "model_queries": coalitions * instances,
                "query_budget": config.xai.model_query_budget,
            },
            {
                "technique": "Kernel SHAP on the surrogate",
                "coalitions_per_instance": config.xai.kernel_shap_coalitions,
                "instances_explained": instances,
                "model_queries": 0,
                "query_budget": config.xai.model_query_budget,
            },
        ]
    )

    return ExplainabilityResult(
        faithfulness=faithfulness,
        surrogate_fidelity=surrogate_fidelity,
        attributions=attributions,
        counterfactuals=counterfactual_frame(verified),
        counterfactual_validity=validity,
        counterfactuals_proposed=int(sum(1 for item in verified if item.achieved)),
        counterfactuals_verified=int(sum(1 for item in verified if item.verified)),
        attention=attention_frame,
        coalition_budget=coalition_budget,
    )


def _exact_shapley_on_model(
    predict: Callable[[np.ndarray], np.ndarray],
    instance: np.ndarray,
    baseline: np.ndarray,
    n_features: int,
) -> np.ndarray:
    from itertools import combinations
    from math import factorial

    coalitions = []
    for size in range(n_features + 1):
        coalitions.extend(combinations(range(n_features), size))
    index = {coalition: position for position, coalition in enumerate(coalitions)}
    matrix = np.tile(baseline, (len(coalitions), 1))
    for position, coalition in enumerate(coalitions):
        for feature in coalition:
            matrix[position, feature] = instance[feature]
    values = predict(matrix)
    phi = np.zeros(n_features)
    for feature in range(n_features):
        others = [i for i in range(n_features) if i != feature]
        for size in range(len(others) + 1):
            weight = factorial(size) * factorial(n_features - size - 1) / factorial(n_features)
            for subset in combinations(others, size):
                with_feature = tuple(sorted(subset + (feature,)))
                phi[feature] += weight * (
                    values[index[with_feature]] - values[index[tuple(sorted(subset))]]
                )
    return phi


def _verify_counterfactuals(
    run: SystemRun,
    config: Config,
    pairs: pd.DataFrame,
    positions: np.ndarray,
    candidates: Sequence[Counterfactual],
    fields: Sequence[str],
) -> List[Counterfactual]:
    from ..xai.counterfactual import admissible_changes, apply_change

    agent = run.pipeline.contextual_agent
    rows = []
    keep: List[int] = []
    for offset, (position, candidate) in enumerate(zip(positions, candidates)):
        if not candidate.achieved:
            continue
        record = pairs.iloc[int(position)].to_dict()
        change = None
        zone_reach = {str(record.get("zone", "")): float(record.get("zone_reach", 0.0))}
        for name, delta in admissible_changes(config, tuple(), str(record.get("zone", "")), zone_reach):
            if name == candidate.change:
                change = delta
                break
        if change is None:
            continue
        rows.append(apply_change(record, change, len(config.org.control_families)))
        keep.append(offset)
    if not rows:
        return list(candidates)
    frame = pd.DataFrame(rows)
    risk = frame["risk_score_input"].to_numpy(dtype=float) if "risk_score_input" in frame else np.full(len(frame), 0.5)
    output = agent.assess(frame, risk)
    updated = list(candidates)
    for slot, offset in enumerate(keep):
        item = updated[offset]
        item.verified_exposure = float(output.score[slot])
        item.verified = bool(item.verified_exposure <= config.xai.counterfactual_medium_band)
    return updated
