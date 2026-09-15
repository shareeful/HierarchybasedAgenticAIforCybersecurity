from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .config import Config

PALETTE = {
    "cvss_only": "#8c8c8c",
    "cvss_epss_calibrated": "#9467bd",
    "xgboost": "#8c564b",
    "securebert_cve": "#17becf",
    "rag_llm": "#e377c2",
    "single_agent": "#ff7f0e",
    "flat_multi_agent": "#2ca02c",
    "proposed": "#1f77b4",
    "static": "#8c8c8c",
    "dqn": "#e377c2",
    "discounted_ucb": "#ff7f0e",
    "thompson": "#2ca02c",
    "sw_ucb": "#1f77b4",
    "isolation_forest": "#1f77b4",
    "zscore": "#ff7f0e",
    "threshold": "#8c8c8c",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.autolayout": True,
        }
    )


def save(figure: Figure, config: Config, name: str) -> Path:
    directory = Path(config.runtime.results_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.{config.runtime.figure_format}"
    figure.savefig(path, bbox_inches="tight")
    return path


def figure3_grouped_bars(per_run: pd.DataFrame, config: Config) -> Figure:
    metrics = ["precision", "recall", "f1", "auc"]
    labels = ["Precision", "Recall", "F1", "AUC"]
    systems = list(config.evaluation.baselines)
    display = [config.evaluation.baseline_display[s] for s in systems]
    figure, axis = plt.subplots(figsize=(10, 4.2))
    width = 0.2
    positions = np.arange(len(systems))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        means = [per_run[per_run["system"] == s][metric].mean() for s in systems]
        errors = [per_run[per_run["system"] == s][metric].std() for s in systems]
        axis.bar(
            positions + (index - 1.5) * width, means, width, yerr=errors,
            capsize=2.5, label=label, alpha=0.9,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(display, rotation=18, ha="right")
    axis.set_ylabel("Score")
    axis.set_ylim(0.4, 1.0)
    axis.legend(ncol=4, loc="upper left")
    axis.set_title("Risk assessment scores across systems and metrics (mean$\\pm$std)")
    return figure


def figure4_roc(roc: Dict[str, Tuple[np.ndarray, np.ndarray]], auc: Dict[str, float], config: Config) -> Figure:
    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    for name in config.evaluation.baselines:
        if name not in roc:
            continue
        fpr, tpr = roc[name]
        axis.plot(
            fpr, tpr, color=PALETTE.get(name), linewidth=2.0 if name == "proposed" else 1.3,
            label=f"{config.evaluation.baseline_display[name]} ({auc.get(name, float('nan')):.3f})",
        )
    axis.plot([0, 1], [0, 1], color="#cccccc", linestyle=":", linewidth=1)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves for all systems (AUC-ROC in parentheses)")
    axis.legend(loc="lower right", fontsize=8)
    return figure


def figure5_ablation(table10: pd.DataFrame, ablation_per_run: pd.DataFrame, config: Config) -> Figure:
    order = list(config.evaluation.ablations)
    display = [config.evaluation.ablation_display[name] for name in order]
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    positions = np.arange(len(order))
    width = 0.36
    for offset, (metric, label) in enumerate((("f1", "F1"), ("auc", "AUC-ROC"))):
        means = [ablation_per_run[ablation_per_run["variant"] == n][metric].mean() for n in order]
        errors = [ablation_per_run[ablation_per_run["variant"] == n][metric].std() for n in order]
        axis.bar(positions + (offset - 0.5) * width, means, width, yerr=errors, capsize=3, label=label)
    full_f1 = ablation_per_run[ablation_per_run["variant"] == "full"]["f1"].mean()
    for index, name in enumerate(order):
        if name == "full":
            continue
        delta = ablation_per_run[ablation_per_run["variant"] == name]["f1"].mean() - full_f1
        axis.annotate(
            f"{delta:+.3f}", (positions[index] - 0.5 * width, 0.02 + ablation_per_run[ablation_per_run["variant"] == name]["f1"].mean()),
            ha="center", fontsize=8, color="#b22222",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(display, rotation=12, ha="right")
    axis.set_ylabel("Score")
    axis.legend(ncol=2)
    axis.set_title("Ablation study: F1 and AUC-ROC across configurations")
    return figure


def figure6_regret(curves: Dict[str, np.ndarray], config: Config, display: Dict[str, str]) -> Figure:
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for name, values in curves.items():
        axis.plot(values, color=PALETTE.get(name), linewidth=1.6, label=display.get(name, name))
    axis.set_xlabel("Accepted replay events")
    axis.set_ylabel("Cumulative regret against the empirical best action")
    axis.set_title("Cumulative regret under offline replay of the recorded ledger")
    axis.legend(fontsize=8)
    return figure


def figure7_policy_accuracy(curves: Dict[str, np.ndarray], config: Config, display: Dict[str, str]) -> Figure:
    figure, axis = plt.subplots(figsize=(6.6, 4.2))
    for name, values in curves.items():
        axis.plot(values, color=PALETTE.get(name), linewidth=1.5, label=display.get(name, name))
    axis.set_xlabel("Accepted replay events")
    axis.set_ylabel("Agreement with the empirical best action")
    axis.set_title("Policy accuracy under offline replay of the recorded ledger")
    axis.legend(fontsize=8, loc="lower right")
    return figure


def figure8_hyperparameters(tau: pd.DataFrame, gamma: pd.DataFrame) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    axes[0].errorbar(tau["tau"], tau["accuracy_mean"], yerr=tau["accuracy_std"], marker="o", color="#1f77b4", capsize=3)
    best_tau = tau.loc[tau["accuracy_mean"].idxmax()]
    axes[0].axvline(best_tau["tau"], color="#b22222", linestyle="--", linewidth=1)
    axes[0].annotate(f"$\\tau^*$ = {int(best_tau['tau'])}", (best_tau["tau"], best_tau["accuracy_mean"]),
                     xytext=(6, -14), textcoords="offset points", color="#b22222", fontsize=9)
    axes[0].set_xlabel("SW-UCB window size $\\tau$")
    axes[0].set_ylabel("Agreement with best action")
    axes[0].set_title("(a)")

    axes[1].errorbar(gamma["gamma"], gamma["accuracy_mean"], yerr=gamma["accuracy_std"], marker="o", color="#2ca02c", capsize=3)
    best_gamma = gamma.loc[gamma["accuracy_mean"].idxmax()]
    axes[1].axvline(best_gamma["gamma"], color="#b22222", linestyle="--", linewidth=1)
    axes[1].annotate(f"$\\gamma^*$ = {best_gamma['gamma']:.2f}", (best_gamma["gamma"], best_gamma["accuracy_mean"]),
                     xytext=(-52, -14), textcoords="offset points", color="#b22222", fontsize=9)
    axes[1].set_xlabel("Discounted UCB factor $\\gamma$")
    axes[1].set_ylabel("Agreement with best action")
    axes[1].set_title("(b)")
    return figure


def figure9_detection(curves: pd.DataFrame, config: Config) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    labels = {
        "isolation_forest": "Isolation Forest",
        "rolling_zscore": "Rolling z-score baseline",
        "fixed_threshold": "Fixed threshold baseline",
    }
    for name, group in curves.groupby("detector", observed=True):
        group = group.sort_values("rate")
        rates = group["rate"] * 100
        axes[0].errorbar(rates, group["detection_mean"], yerr=group["detection_std"], marker="o",
                         capsize=3, color=PALETTE.get(name), label=labels.get(name, name))
        axes[1].plot(rates, group["false_positive_mean"], marker="o",
                     color=PALETTE.get(name), label=labels.get(name, name))
    axes[0].set_xlabel("Poisoning rate (%)")
    axes[0].set_ylabel("Detection rate")
    axes[0].set_title("(a)")
    axes[0].legend(fontsize=8)
    axes[1].axhline(config.integrity.false_positive_ceiling, color="#b22222", linestyle="--", linewidth=1)
    axes[1].annotate(f"operational ceiling {config.integrity.false_positive_ceiling}",
                     (curves["rate"].min() * 100, config.integrity.false_positive_ceiling + 0.002),
                     fontsize=8, color="#b22222")
    axes[1].set_xlabel("Poisoning rate (%)")
    axes[1].set_ylabel("False positive rate")
    axes[1].set_title("(b)")
    axes[1].legend(fontsize=8)
    return figure


def figure10_decision_quality(curves: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(6.2, 4.0))
    frame = curves.sort_values("rate")
    axis.plot(frame["rate"] * 100, frame["degradation_with_check"], marker="o",
              color="#1f77b4", label="With integrity check")
    axis.plot(frame["rate"] * 100, frame["degradation_without_check"], marker="o",
              color="#b22222", label="Without integrity check")
    gap = float(frame["degradation_without_check"].iloc[-1] - frame["degradation_with_check"].iloc[-1])
    axis.annotate(
        f"$\\Delta$ = {gap:.3f}",
        (
            frame["rate"].iloc[-1] * 100,
            0.5 * (frame["degradation_with_check"].iloc[-1] + frame["degradation_without_check"].iloc[-1]),
        ),
        xytext=(-70, 0), textcoords="offset points", fontsize=9,
    )
    axis.set_xlabel("Poisoning rate (%)")
    axis.set_ylabel("F1 degradation against the clean run")
    axis.set_title("Decision degradation with and without integrity verification")
    axis.legend(fontsize=9)
    return figure


def figure11_inference_time(timing: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(6.8, 4.0))
    for column, colour, label in (
        ("vulnerability_seconds", "#1f77b4", "Vulnerability Agent"),
        ("contextual_seconds", "#2ca02c", "Contextual Awareness Agent"),
        ("supervisor_seconds", "#ff7f0e", "Supervisor Agent"),
        ("integrity_seconds", "#9467bd", "Integrity verification"),
    ):
        axis.plot(timing["pairs"], timing[column], marker="o", color=colour, label=label)
    axis.plot(timing["pairs"], timing["total_seconds"], marker="s", color="#444444", label="Total")
    axis.set_xlabel("Vulnerability-asset pairs assessed")
    axis.set_ylabel("Measured wall-clock seconds")
    axis.set_title("Measured inference time by pipeline stage")
    axis.legend(fontsize=8)
    return figure


def figure12_scalability_f1(accuracy: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(accuracy["pairs"], accuracy["f1"], marker="o", color="#1f77b4", label="F1")
    axis.plot(accuracy["pairs"], accuracy["auc"], marker="s", color="#2ca02c", label="AUC")
    axis.set_xlabel("Vulnerability-asset pairs assessed")
    axis.set_ylabel("Score")
    axis.set_title("Discrimination against assessed volume")
    axis.legend(fontsize=9)
    return figure


def figure13_use_case(distribution: pd.DataFrame, confusion: pd.DataFrame, agreement: Dict[str, float]) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    actions = list(distribution["action"])
    labels = [a.capitalize() for a in actions]
    positions = np.arange(len(actions))
    width = 0.2
    for index, (column, label) in enumerate(
        (("analyst", "Analyst (recorded)"), ("proposed", "Proposed"),
         ("flat_multi_agent", "Flat Multi-Agent"), ("static_cvss", "Static CVSS"))
    ):
        axes[0].bar(positions + (index - 1.5) * width, distribution[column], width, label=label)
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel(f"Decisions (of {int(agreement['matched_decisions'])})")
    axes[0].set_title("(a) Action distribution")
    axes[0].legend(fontsize=8)

    matrix = confusion.to_numpy()
    axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks(range(len(actions)))
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_yticks(range(len(actions)))
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Recommended by proposed approach")
    axes[1].set_ylabel("Recorded analyst decision")
    threshold = matrix.max() / 2 if matrix.max() else 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > 0:
                axes[1].text(j, i, int(matrix[i, j]), ha="center", va="center",
                             color="white" if matrix[i, j] > threshold else "black", fontsize=9)
    axes[1].set_title(
        f"(b) Agreement: {int(round(agreement['proposed'] * agreement['matched_decisions']))}"
        f"/{int(agreement['matched_decisions'])} = {100 * agreement['proposed']:.1f}%"
    )
    axes[1].grid(False)
    return figure


def figure_reward_sensitivity(table12: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(7.4, 3.8))
    positions = np.arange(len(table12))
    axis.bar(positions, table12["mean_reward"], color="#1f77b4", alpha=0.85)
    axis.set_ylabel("Mean recorded reward")
    axis.set_xticks(positions)
    axis.set_xticklabels(table12["profile"], rotation=16, ha="right")
    axis.set_title("Recorded reward under each composite-reward weight profile")
    return figure


def figure_xai_faithfulness(results: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    positions = np.arange(len(results))
    width = 0.36
    axis.bar(positions - width / 2, results["top_k_delta"], width, label="Top-3 ranked fields masked")
    axis.bar(positions + width / 2, results["random_delta"], width, label="3 random fields masked")
    axis.set_xticks(positions)
    axis.set_xticklabels(results["technique"], rotation=10, ha="right")
    axis.set_ylabel("Mean absolute score change")
    axis.set_title("Explanation faithfulness: deletion test")
    axis.legend(fontsize=9)
    return figure


def figure_reproduction_comparison(comparison: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    positions = np.arange(len(comparison))
    width = 0.36
    axis.bar(positions - width / 2, comparison["paper"], width, label="Paper reported", alpha=0.85)
    axis.bar(positions + width / 2, comparison["reproduced"], width, label="Reproduced", alpha=0.85)
    axis.set_xticks(positions)
    axis.set_xticklabels(comparison["quantity"], rotation=22, ha="right", fontsize=8)
    axis.set_ylabel("Value")
    axis.set_title("Reproduced values against the values reported in the paper")
    axis.legend(fontsize=9)
    return figure


def _box(axis, x, y, width, height, text, facecolor, fontsize=8.5, textcolor="#1a1a1a", weight="normal"):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=0.9, edgecolor="#4a4a4a", facecolor=facecolor,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              fontsize=fontsize, color=textcolor, fontweight=weight, wrap=True)
    return patch


def _arrow(axis, start, end, colour="#4a4a4a", style="-|>", linewidth=1.1, connection="arc3,rad=0.0"):
    axis.annotate("", xy=end, xytext=start,
                  arrowprops=dict(arrowstyle=style, color=colour, linewidth=linewidth,
                                  connectionstyle=connection, shrinkA=1, shrinkB=1))


def figure1_methodology(config: Config) -> Figure:
    figure, axis = plt.subplots(figsize=(11.0, 5.4))
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off"); axis.grid(False)
    phases = [
        ("Phase 1\nData collection\nand preparation",
         "NVD, EPSS, PoC feeds,\nMITRE ATT&CK, CISA KEV\n"
         "completeness check, z>3 outlier filter\n"
         "pairing rule S_aff \u2229 sw_vers(a) \u2260 \u2205", "#dbe9f6"),
        ("Phase 2\nAgent fine-tuning\nand initialisation",
         f"LoRA r={config.lora.rank}, α={config.lora.alpha} on q_proj/v_proj\n"
         "prompt–completion pairs (Tables 2–4)\n"
         "MAB policies initialised", "#e3f0da"),
        ("Phase 3\nRisk assessment\nand decision making",
         "parallel task-agent inference\nIsolation Forest integrity check\n"
         "confidence-weighted joint risk (Eq. 23)\naction utility (Eq. 24)", "#fdeadb"),
        ("Phase 4\nReward computing\nand policy updating",
         "four outcome signals\ncomposite reward (Eq. 26)\n"
         "agent-specific credit assignment\nsurrogate retraining", "#efe3f5"),
    ]
    width, gap = 0.215, 0.0325
    for index, (title, body, colour) in enumerate(phases):
        x = 0.02 + index * (width + gap)
        _box(axis, x, 0.60, width, 0.30, title, colour, fontsize=10, weight="bold")
        _box(axis, x, 0.24, width, 0.32, body, "#fbfbfb", fontsize=8)
        if index < len(phases) - 1:
            _arrow(axis, (x + width + 0.004, 0.75), (x + width + gap - 0.004, 0.75))
    _arrow(axis, (0.02 + 3 * (width + gap) + width / 2, 0.20), (0.02 + width / 2, 0.20),
           colour="#8a5a9a", connection="arc3,rad=0.30")
    axis.text(0.5, 0.05, "policy update closes the loop into the next assessment cycle",
              ha="center", fontsize=8.5, color="#8a5a9a", style="italic")
    axis.set_title("Four-phase methodology for hierarchical multi-AI-agent risk assessment", fontsize=11)
    return figure


def figure2_environment(config: Config) -> Figure:
    figure, axis = plt.subplots(figsize=(11.0, 6.4))
    axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off"); axis.grid(False)

    _box(axis, 0.02, 0.66, 0.28, 0.29,
         "Public vulnerability feeds\n(read from disk)\n\nNVD CVE JSON 2.0\nFIRST EPSS scores\n"
         "CISA KEV catalogue\nMITRE CWE and CAPEC\nMITRE ATT&CK\nExploit-DB index", "#dbe9f6", fontsize=8)
    _box(axis, 0.02, 0.24, 0.28, 0.36,
         "Organisational records\n(read from disk)\n\nasset inventory\nsoftware inventory\n"
         "control register\nchange records\nremediation ledger\nagent telemetry log",
         "#e3f0da", fontsize=8)
    axis.text(0.16, 0.195, "pairing rule: affected CPE product matches installed product",
              ha="center", fontsize=8, style="italic", color="#4a4a4a")

    _box(axis, 0.37, 0.78, 0.26, 0.15, "Supervisor Agent\nMistral-7B-Instruct + LoRA", "#fdeadb", fontsize=9, weight="bold")
    _box(axis, 0.355, 0.53, 0.125, 0.17, "Vulnerability\nAgent\nLlama 3.1 8B\n+ LoRA", "#fdeadb", fontsize=8)
    _box(axis, 0.505, 0.53, 0.125, 0.17, "Contextual\nAwareness Agent\nLlama 3.1 8B\n+ LoRA", "#fdeadb", fontsize=8)
    _arrow(axis, (0.4175, 0.705), (0.45, 0.775))
    _arrow(axis, (0.5675, 0.705), (0.545, 0.775))
    _box(axis, 0.37, 0.26, 0.26, 0.22,
         "Exposure reference (Eq. 28)\ncomputed from recorded fields\n\n"
         "software version match\nzone reachability\nasset criticality\n"
         "control coverage gap\nexploit probability", "#eeeeee", fontsize=8)

    _box(axis, 0.70, 0.62, 0.28, 0.31,
         "Recorded remediation ledger\n\naction taken\nexploited afterwards\n"
         "disruption observed\nanalyst acceptance\nresidual exposure", "#efe3f5", fontsize=8)
    _box(axis, 0.70, 0.40, 0.28, 0.15, "Composite reward\n(Eq. 26, evaluated on records)", "#efe3f5", fontsize=9, weight="bold")
    _box(axis, 0.70, 0.12, 0.28, 0.19,
         "Offline replay evaluation\nof the bandit policies\n(Experiment 2)", "#fce4e4", fontsize=8)

    _arrow(axis, (0.30, 0.80), (0.355, 0.64))
    _arrow(axis, (0.30, 0.42), (0.505, 0.58))
    _arrow(axis, (0.63, 0.855), (0.70, 0.78))
    _arrow(axis, (0.84, 0.615), (0.84, 0.555))
    _arrow(axis, (0.70, 0.475), (0.63, 0.86), colour="#8a5a9a", connection="arc3,rad=-0.35")
    _arrow(axis, (0.84, 0.395), (0.84, 0.315), colour="#b22222")

    axis.text(0.665, 0.585, "policy\nupdate", fontsize=8, color="#8a5a9a", ha="center")
    axis.text(0.5, 0.04, "every box on the left is a file the reader supplies; nothing in this diagram is generated",
              ha="center", fontsize=8.5, style="italic", color="#4a4a4a")
    axis.set_title("Data sources and the assessment hierarchy", fontsize=11)
    return figure


def figure_agreement_by_subsystem(by_subsystem) -> Figure:
    grouped = by_subsystem.sort_values("agreement").set_index("subsystem").rename(
        columns={"agreement": "mean", "decisions": "size"}
    )
    figure, axis = plt.subplots(figsize=(7.6, 3.8))
    axis.barh(
        [str(s).replace("_", " ") for s in grouped.index], grouped["mean"], color="#1f77b4", alpha=0.85
    )
    for position, (value, count) in enumerate(zip(grouped["mean"], grouped["size"])):
        axis.text(value + 0.01, position, f"{value:.0%} (n={int(count)})", va="center", fontsize=8)
    axis.set_xlim(0, 1.12)
    axis.set_xlabel("Agreement with recorded analyst decision")
    axis.set_title("Analyst agreement by subsystem")
    return figure
