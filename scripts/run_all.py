from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from haai import plots
from haai.data.registry import DatasetError
from haai.agents.llm import ModelUnavailable
from haai.experiments.experiment2 import POLICY_DISPLAY
from haai.logging_utils import get_logger
from haai.study import Study, configure

LOGGER = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full study over a supplied dataset")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--evaluation-sample", type=int, default=0)
    parser.add_argument("--explain-sample", type=int, default=64)
    parser.add_argument("--labelling-budget", type=int, default=1500)
    parser.add_argument("--skip-labelling-sensitivity", action="store_true")
    parser.add_argument("--skip-finetuning-datasets", action="store_true")
    arguments = parser.parse_args()

    overrides = {"runtime.results_dir": arguments.results_dir}
    if arguments.runs is not None:
        overrides["evaluation.n_runs"] = arguments.runs
    if arguments.evaluation_sample:
        overrides["runtime.evaluation_sample"] = arguments.evaluation_sample
    config = configure(**overrides)
    plots.apply_style()
    study = Study(config)

    try:
        study.build(arguments.data_dir)
    except DatasetError as error:
        print(str(error), file=sys.stderr)
        return 2

    if not arguments.skip_finetuning_datasets:
        study.build_finetuning_datasets(Path(arguments.results_dir) / "finetuning")

    try:
        study.run_base()
    except ModelUnavailable as error:
        print(str(error), file=sys.stderr)
        return 3

    experiment1 = study.run_experiment1()
    if not arguments.skip_labelling_sensitivity:
        study.run_labelling_sensitivity(arguments.labelling_budget)
    experiment2 = study.run_experiment2()
    experiment3 = study.run_experiment3()
    experiment5 = study.run_experiment5()
    explainability = study.run_explainability(arguments.explain_sample)
    experiment4 = study.run_experiment4()
    study.table18()
    comparison = study.comparison()

    auc = {
        name: float(experiment1.table9.set_index("raw_key").loc[name, "auc_mean"])
        for name in config.evaluation.baselines
    }
    figures = {
        "figure1_methodology": plots.figure1_methodology(config),
        "figure2_environment": plots.figure2_environment(config),
        "figure3_scores": plots.figure3_grouped_bars(experiment1.per_run, config),
        "figure4_roc": plots.figure4_roc(experiment1.roc, auc, config),
        "figure5_ablation": plots.figure5_ablation(
            experiment1.table10, experiment1.ablation_per_run, config
        ),
        "figure6_regret": plots.figure6_regret(experiment2.regret_curves, config, POLICY_DISPLAY),
        "figure7_policy_accuracy": plots.figure7_policy_accuracy(
            experiment2.accuracy_curves, config, POLICY_DISPLAY
        ),
        "figure8_hyperparameters": plots.figure8_hyperparameters(
            experiment2.tau_sensitivity, experiment2.gamma_sensitivity
        ),
        "figure9_detection": plots.figure9_detection(experiment3.detection_curves, config),
        "figure10_decision_quality": plots.figure10_decision_quality(experiment3.quality_curves),
        "figure11_inference_time": plots.figure11_inference_time(experiment4.timing),
        "figure12_scalability": plots.figure12_scalability_f1(experiment4.accuracy),
        "figure13_use_case": plots.figure13_use_case(
            experiment5.action_distribution, experiment5.confusion, experiment5.agreement
        ),
        "figure_agreement_by_subsystem": plots.figure_agreement_by_subsystem(
            experiment5.agreement_by_subsystem
        ),
        "figure_reward_profiles": plots.figure_reward_sensitivity(experiment2.table12),
        "figure_xai_faithfulness": plots.figure_xai_faithfulness(explainability.faithfulness),
        "figure_reproduction": plots.figure_reproduction_comparison(comparison),
    }
    for name, figure in figures.items():
        plots.save(figure, config, name)

    study.save_tables()
    summary = {
        "improvement": experiment1.improvement(config),
        "reproduction": study.comparison_summary(),
        "analyst_agreement": experiment5.agreement,
        "integrity_provenance": experiment3.provenance,
        "timing_note": experiment4.note,
        "counterfactual_validity": explainability.counterfactual_validity,
        "provenance": {row["item"]: row["value"] for row in study.tables["provenance"].to_dict("records")},
    }
    output = Path(arguments.results_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    config.save(output / "config.json")
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
