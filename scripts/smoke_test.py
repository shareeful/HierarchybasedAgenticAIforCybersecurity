from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

from haai import plots
from haai.agents.llm import ModelUnavailable
from haai.data.registry import DatasetError
from haai.experiments.experiment2 import POLICY_DISPLAY
from haai.logging_utils import get_logger
from haai.study import Study, configure

LOGGER = get_logger()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run every stage once on a bounded sample of the supplied dataset"
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--runs", type=int, default=3)
    arguments = parser.parse_args()

    config = configure(
        **{
            "evaluation.n_runs": arguments.runs,
            "runtime.evaluation_sample": arguments.sample,
            "runtime.results_dir": "results/smoke",
            "runtime.timing_sample": min(arguments.sample, 64),
            "xai.model_query_budget": 256,
        }
    )
    plots.apply_style()
    study = Study(config)
    failures = []

    def stage(name, function):
        try:
            value = function()
            LOGGER.info("PASS %s", name)
            return value
        except (DatasetError, ModelUnavailable) as error:
            print(f"\n{name} cannot run:\n{error}\n", file=sys.stderr)
            raise
        except Exception as error:
            failures.append((name, traceback.format_exc()))
            LOGGER.error("FAIL %s: %s", name, error)
            return None

    try:
        stage("build environment", lambda: study.build(arguments.data_dir))
        stage("finetuning datasets", lambda: study.build_finetuning_datasets("results/smoke/finetuning"))
        stage("load models", study.load_models)
        stage("base run", study.run_base)
    except (DatasetError, ModelUnavailable):
        return 2

    experiment1 = stage("experiment 1", lambda: study.run_experiment1(arguments.runs))
    experiment2 = stage("experiment 2", lambda: study.run_experiment2(arguments.runs))
    experiment3 = stage("experiment 3", lambda: study.run_experiment3(arguments.runs))
    experiment5 = stage("experiment 5", study.run_experiment5)
    explainability = stage("explainability", lambda: study.run_explainability(8))
    experiment4 = stage("experiment 4", lambda: study.run_experiment4(arguments.runs))
    stage("table 18", study.table18)
    comparison = stage("comparison", study.comparison)

    if experiment1 is not None:
        auc = {
            name: float(experiment1.table9.set_index("raw_key").loc[name, "auc_mean"])
            for name in config.evaluation.baselines
        }
        stage("figure 3", lambda: plots.save(plots.figure3_grouped_bars(experiment1.per_run, config), config, "figure3"))
        stage("figure 4", lambda: plots.save(plots.figure4_roc(experiment1.roc, auc, config), config, "figure4"))
        stage("figure 5", lambda: plots.save(
            plots.figure5_ablation(experiment1.table10, experiment1.ablation_per_run, config), config, "figure5"))
    if experiment2 is not None:
        stage("figure 6", lambda: plots.save(
            plots.figure6_regret(experiment2.regret_curves, config, POLICY_DISPLAY), config, "figure6"))
        stage("figure 8", lambda: plots.save(
            plots.figure8_hyperparameters(experiment2.tau_sensitivity, experiment2.gamma_sensitivity), config, "figure8"))
    if experiment3 is not None:
        stage("figure 9", lambda: plots.save(plots.figure9_detection(experiment3.detection_curves, config), config, "figure9"))
        stage("figure 10", lambda: plots.save(plots.figure10_decision_quality(experiment3.quality_curves), config, "figure10"))
    if experiment4 is not None:
        stage("figure 11", lambda: plots.save(plots.figure11_inference_time(experiment4.timing), config, "figure11"))
    if experiment5 is not None:
        stage("figure 13", lambda: plots.save(
            plots.figure13_use_case(experiment5.action_distribution, experiment5.confusion, experiment5.agreement),
            config, "figure13"))
    if explainability is not None:
        stage("figure faithfulness", lambda: plots.save(
            plots.figure_xai_faithfulness(explainability.faithfulness), config, "figure_faithfulness"))
    if comparison is not None and len(comparison):
        stage("figure reproduction", lambda: plots.save(
            plots.figure_reproduction_comparison(comparison), config, "figure_reproduction"))
    stage("save tables", study.save_tables)

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} STAGE(S) FAILED")
        for name, trace in failures:
            print("-" * 78)
            print(name)
            print(trace)
        return 1
    print("ALL STAGES PASSED")
    if experiment1 is not None:
        print("\nTable 9")
        print(experiment1.table9.drop(columns=["raw_key", "f1_mean", "auc_mean"]).to_string(index=False))
        print("\nMetric consistency")
        print(experiment1.consistency.to_string(index=False))
    if experiment5 is not None:
        print("\nAnalyst agreement:", experiment5.agreement)
    if comparison is not None:
        print("\nReproduction summary:", study.comparison_summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
