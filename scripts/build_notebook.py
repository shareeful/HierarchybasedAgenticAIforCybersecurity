from __future__ import annotations

import json
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "Hierarchical_Agentic_AI_Risk_Assessment.ipynb"

CELLS: List[tuple] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text))


def code(text: str) -> None:
    CELLS.append(("code", text))


markdown(
    "# Hierarchical Agentic AI for Dynamic Cybersecurity Risk Assessment\n"
    "\n"
    "Reference implementation of *A Hierarchical Agentic AI based Dynamic Cybersecurity Risk "
    "Assessment with Explainable Informed Decision Making* (Sardar, Hassan, Islam, Papastergiou, "
    "Lekidis).\n"
    "\n"
    "**This notebook reads recorded data only. It generates nothing.** If the vulnerability feeds, "
    "the organisational records and the pilot records are not present, every stage stops with an "
    "error naming the missing file and where to obtain it. There is no fallback corpus, no "
    "simulated organisation and no generated analyst decisions.\n"
    "\n"
    "The three agents are real language models. The Vulnerability Agent and the Contextual "
    "Awareness Agent run Llama 3.1 8B Instruct with a LoRA adapter; the Supervisor Agent runs "
    "Mistral-7B Instruct with its own adapter (paper, sections 3.2 to 3.4). Without the weights, "
    "the agent tier stops with an error rather than substituting a statistical stand-in.\n"
    "\n"
    "## What you must supply\n"
    "\n"
    "| Directory | Files | Source |\n"
    "| --- | --- | --- |\n"
    "| `nvd/` | `nvdcve-2.0-<year>.json` | NVD JSON 2.0 data feeds |\n"
    "| `epss/` | `epss_scores-<date>.csv.gz` | FIRST EPSS |\n"
    "| `kev/` | `known_exploited_vulnerabilities.json` | CISA KEV catalogue |\n"
    "| `attack/` | `enterprise-attack.json` | MITRE ATT&CK STIX bundle |\n"
    "| `cwe/` | `cwec_*.xml` | MITRE CWE catalogue |\n"
    "| `capec/` | `capec_*.xml` | MITRE CAPEC catalogue |\n"
    "| `exploitdb/` | `files_exploits.csv` | Exploit-DB index |\n"
    "| `organisation/` | `assets.csv`, `software_inventory.csv`, `controls.csv`, "
    "`change_records.csv`, `remediation_outcomes.csv`, `agent_telemetry.csv` | your organisation |\n"
    "| `usecase/` | the same four inventory files plus `subsystems.csv`, `attack_scenarios.csv`, "
    "`soc_alerts.csv`, `analyst_decisions.csv`, `operator_targets.csv` | the pilot site |\n"
    "\n"
    "Run the dataset check below before anything else; it prints the exact schema of every file."
)

code(
    "import importlib.util, os, subprocess, sys\n"
    "\n"
    "try:\n"
    "    IN_COLAB = importlib.util.find_spec('google.colab') is not None\n"
    "except (ImportError, ModuleNotFoundError, ValueError):\n"
    "    IN_COLAB = False\n"
    "if IN_COLAB:\n"
    "    if not os.path.exists('new-rag-based-agentic-ai'):\n"
    "        subprocess.run(['git', 'clone', '--depth', '1',\n"
    "                        'https://github.com/mhassan720/new-rag-based-agentic-ai.git'], check=True)\n"
    "    os.chdir('new-rag-based-agentic-ai')\n"
    "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], check=True)\n"
    "print('working directory:', os.getcwd())"
)

markdown(
    "## 1. Point the code at your data\n"
    "\n"
    "Set `HAAI_DATA_DIR` to the directory holding the corpus described above. On Colab the usual "
    "route is Google Drive. Set `HAAI_MODEL_DIR` if the model weights are on disk rather than "
    "pulled from the model host, and `HAAI_ADAPTER_DIR` if you have trained adapters."
)

code(
    "import os\n"
    "\n"
    "IN_COLAB = globals().get('IN_COLAB', False)\n"
    "if IN_COLAB:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    os.environ.setdefault('HAAI_DATA_DIR', '/content/drive/MyDrive/haai-data')\n"
    "else:\n"
    "    os.environ.setdefault('HAAI_DATA_DIR', 'data')\n"
    "os.environ.setdefault('HAAI_CACHE_DIR', 'cache')\n"
    "os.environ.setdefault('HAAI_ADAPTER_DIR', 'adapters')\n"
    "print('dataset directory:', os.environ['HAAI_DATA_DIR'])"
)

code(
    "import subprocess, sys\n"
    "\n"
    "check = subprocess.run([sys.executable, 'scripts/check_dataset.py'], capture_output=True, text=True)\n"
    "print(check.stdout)\n"
    "print(check.stderr)\n"
    "DATA_READY = check.returncode == 0\n"
    "if not DATA_READY:\n"
    "    print('The dataset is incomplete. Every stage below will stop until the files above are present.')"
)

markdown(
    "## 2. Configuration\n"
    "\n"
    "Every constant used anywhere in the analysis lives in `src/haai/config.py`, each traceable to "
    "a section of the paper. Nothing is fitted to the paper's published results, and there is no "
    "calibration file."
)

code(
    "import sys\n"
    "sys.path.insert(0, 'src')\n"
    "\n"
    "import matplotlib\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "\n"
    "from haai import plots\n"
    "from haai.study import Study, configure\n"
    "\n"
    "pd.set_option('display.width', 200)\n"
    "pd.set_option('display.max_columns', 40)\n"
    "plots.apply_style()\n"
    "\n"
    "QUICK = True\n"
    "overrides = {'runtime.results_dir': 'results'}\n"
    "if QUICK:\n"
    "    overrides.update({\n"
    "        'evaluation.n_runs': 5,\n"
    "        'runtime.evaluation_sample': 400,\n"
    "        'runtime.timing_sample': 64,\n"
    "        'xai.model_query_budget': 512,\n"
    "        'org.max_assets_per_cve': 6,\n"
    "    })\n"
    "config = configure(**overrides)\n"
    "study = Study(config)\n"
    "print('runs per experiment:', config.evaluation.n_runs)"
)

markdown(
    "## 3. Phase 1: corpus construction and validation\n"
    "\n"
    "The corpus is assembled by joining the feeds you supplied: NVD records give CVSS, CWE and "
    "affected CPE products; EPSS gives exploitation probability; the KEV catalogue supplies the "
    "positive label; Exploit-DB supplies proof-of-concept availability; CWE and CAPEC together "
    "resolve each weakness to an ATT&CK technique. Records missing any required field are dropped, "
    "then values beyond three standard deviations on their natural scale are dropped. Negatives are "
    "sampled at the paper's 10:1 ratio, matched on year, CWE family and vendor."
)

code(
    "environment = study.build()\n"
    "display(study.tables['feed_summary'])\n"
    "display(study.tables['validation'])\n"
    "display(study.tables['table8'])"
)

code(
    "display(study.tables['provenance'])"
)

markdown(
    "## 4. The pilot site (Tables 5 to 7)\n"
    "\n"
    "Every row here is read from the pilot records you supplied. The analyst triage decisions in "
    "Table 7 are counted from `usecase/analyst_decisions.csv`; they are never produced by this code."
)

code(
    "display(study.tables['table5'])\n"
    "display(study.tables['table6'])\n"
    "display(study.tables['table7'])"
)

markdown(
    "## 5. LoRA instruction datasets\n"
    "\n"
    "Prompts follow Tables 2, 3 and 4 exactly. Targets come from recorded data: the KEV label for "
    "the Vulnerability Agent, the Equation 28 exposure reference for the Contextual Awareness "
    "Agent, and the recorded remediation action for the Supervisor Agent. Fine-tuning itself is "
    "`scripts/train_lora_agents.py`; the numeric terms of Equations 14, 16 and 18 are the live "
    "training objectives, computed from the model's own digit distribution."
)

code(
    "statistics = study.build_finetuning_datasets('results/finetuning')\n"
    "display(statistics)\n"
    "print(open('results/finetuning/vulnerability_agent.jsonl').readline()[:1200])"
)

code(
    "import subprocess, sys\n"
    "\n"
    "result = subprocess.run(\n"
    "    [sys.executable, 'scripts/train_lora_agents.py', '--dry-run',\n"
    "     '--dataset-dir', 'results/finetuning'],\n"
    "    capture_output=True, text=True,\n"
    ")\n"
    "print(result.stdout[-2000:])\n"
    "print(result.stderr[-2000:])"
)

markdown(
    "## 6. Load the agent language models\n"
    "\n"
    "This is the step that needs a GPU and the model weights. It raises `ModelUnavailable` with "
    "instructions if either is missing. Completions are cached on disk under `HAAI_CACHE_DIR`, so "
    "a second pass over the same prompts is free and exactly reproducible."
)

code(
    "from haai.agents.llm import ModelUnavailable\n"
    "\n"
    "try:\n"
    "    models = study.load_models()\n"
    "    base_run = study.run_base()\n"
    "    MODELS_READY = True\n"
    "    print('agent tier ready;', models.log.calls, 'model calls so far')\n"
    "except ModelUnavailable as error:\n"
    "    MODELS_READY = False\n"
    "    print(error)"
)

markdown(
    "## 7. Experiment 1: risk assessment accuracy (Tables 9, 10, 18; Figures 3 to 5)\n"
    "\n"
    "Baselines are real models: a calibrated logistic regression over CVSS and EPSS, gradient "
    "boosting over the six Equation 11 fields, SecureBERT fine-tuned on CVE descriptions, and a "
    "retrieval-augmented LLM using a transformer encoder for retrieval and the task model for "
    "judgement.\n"
    "\n"
    "One metric protocol throughout: the decision threshold is chosen on the tuning split, and "
    "precision, recall, F1 and AUC are all computed on the full evaluation split at that "
    "threshold. Dispersion comes from stratified bootstrap resampling of the evaluation split.\n"
    "\n"
    "The consistency table states, for each measured AUC and prevalence, the largest F1 attainable "
    "by any concave ROC curve with that area, and the largest attainable under a proper binormal "
    "ROC. An F1 above the first is arithmetically impossible; an F1 above the second is possible "
    "but implies an unusually shaped ROC curve."
)

code(
    "experiment1 = study.run_experiment1()\n"
    "display(experiment1.table9.drop(columns=['raw_key', 'f1_mean', 'auc_mean']))\n"
    "print(experiment1.improvement(config))"
)

code(
    "display(experiment1.consistency)"
)

code(
    "display(experiment1.table10.drop(columns=['raw_key']))\n"
    "display(experiment1.significance)"
)

code(
    "auc = {name: float(experiment1.table9.set_index('raw_key').loc[name, 'auc_mean'])\n"
    "       for name in config.evaluation.baselines}\n"
    "display(plots.figure3_grouped_bars(experiment1.per_run, config))\n"
    "display(plots.figure4_roc(experiment1.roc, auc, config))\n"
    "display(plots.figure5_ablation(experiment1.table10, experiment1.ablation_per_run, config))"
)

markdown(
    "## 8. Prompt parsing, confidence calibration and runtime failure handling\n"
    "\n"
    "`src/haai/agents/parsing.py` is the failure-handling layer: JSON repair for trailing commas, "
    "unquoted keys, single quotes and unclosed braces; one bounded retry; clamping of out-of-range "
    "values; imputation with a human-review flag when a field cannot be recovered; and detection of "
    "contradictions between the reported score and the intensifier vocabulary of the reasoning "
    "text. Every event is counted per agent.\n"
    "\n"
    "Confidence is calibrated by isotonic regression on the tuning split against whether the agent "
    "was actually right, blending the self-reported confidence, the sequence log-probability and "
    "the bandit reliability estimate for the arm."
)

code(
    "display(study.tables['parse_telemetry'])\n"
    "display(study.tables['calibration'])\n"
    "display(study.tables['supervisor_adherence'])"
)

markdown(
    "### Sensitivity to the labelling protocol\n"
    "\n"
    "The labelled set depends on a protocol choice: uncertain records are held out, and negatives "
    "are sampled at 10:1. This table re-runs the hierarchy under two alternatives, forcing the "
    "uncertain records to be negative, and sampling negatives at 20:1, so a reader can see how much "
    "of the headline result is a property of that choice. The threshold is reselected within each "
    "protocol, and the column says so."
)

code(
    "labelling = study.run_labelling_sensitivity(1000)\n"
    "display(labelling)"
)

markdown(
    "## 9. Experiment 2: adaptive decision policies (Tables 11, 12; Figures 6 to 8)\n"
    "\n"
    "The bandit policies are evaluated by unbiased offline replay of the recorded remediation "
    "ledger: each logged decision is presented in turn, the policy proposes an action, and the "
    "event is retained only when the proposal matches the action actually taken. Rewards are "
    "Equation 26 evaluated on the recorded outcome. Regret and accuracy are measured against the "
    "empirically best action for each context stratum. No threat shift is injected; the coverage "
    "table states how densely the ledger supports the comparison."
)

code(
    "experiment2 = study.run_experiment2()\n"
    "display(experiment2.coverage)\n"
    "display(experiment2.table11.drop(columns=['raw_key', 'recovery_mean']))\n"
    "display(experiment2.table12)"
)

code(
    "from haai.experiments.experiment2 import POLICY_DISPLAY\n"
    "\n"
    "display(plots.figure6_regret(experiment2.regret_curves, config, POLICY_DISPLAY))\n"
    "display(plots.figure7_policy_accuracy(experiment2.accuracy_curves, config, POLICY_DISPLAY))\n"
    "display(plots.figure8_hyperparameters(experiment2.tau_sensitivity, experiment2.gamma_sensitivity))\n"
    "display(plots.figure_reward_sensitivity(experiment2.table12))"
)

markdown(
    "## 10. Experiment 3: runtime integrity verification (Table 14; Figures 9 and 10)\n"
    "\n"
    "If the telemetry log carries a `tampered` column with positives, detection is measured against "
    "those recorded labels. Otherwise the configured attack model is injected into the recorded "
    "agent output vectors and that provenance is reported alongside the numbers, so a reader always "
    "knows which of the two they are looking at."
)

code(
    "experiment3 = study.run_experiment3()\n"
    "print('provenance:', experiment3.provenance)\n"
    "display(experiment3.table14)\n"
    "display(experiment3.quality_curves)"
)

code(
    "display(plots.figure9_detection(experiment3.detection_curves, config))\n"
    "display(plots.figure10_decision_quality(experiment3.quality_curves))"
)

markdown(
    "## 11. Experiment 5: the pilot use case (Tables 16, 17; Figure 13)\n"
    "\n"
    "The pipeline is run over the pilot inventory and its recommended actions are compared against "
    "the recorded analyst decisions, joined on the vulnerability-asset pair. The number of decisions "
    "that could be matched is reported next to the number recorded, so partial coverage is visible "
    "rather than hidden. Operator targets are reported only when the pilot records carry a measured "
    "`achieved` value; otherwise the notebook says the measurement is not derivable from the "
    "supplied data instead of inventing one."
)

code(
    "experiment5 = study.run_experiment5()\n"
    "display(study.tables['analyst_agreement'])\n"
    "display(experiment5.confusion)\n"
    "display(experiment5.agreement_by_subsystem)"
)

code(
    "display(experiment5.table16)\n"
    "display(experiment5.table17)\n"
    "display(experiment5.operator_targets)"
)

code(
    "display(plots.figure13_use_case(experiment5.action_distribution, experiment5.confusion,\n"
    "                               experiment5.agreement))\n"
    "display(plots.figure_agreement_by_subsystem(experiment5.agreement_by_subsystem))"
)

markdown(
    "## 12. Explainability (Table 13)\n"
    "\n"
    "Exact Shapley values over the six Equation 11 fields are computed against the language model "
    "itself: 2^6 = 64 masked queries per instance, bounded by `xai.model_query_budget`. The deletion "
    "test for those attributions therefore measures the model, not a surrogate of it. Kernel SHAP "
    "and LIME remain surrogate-based because their coalition counts are not affordable against a "
    "language model, and the table labels them as such.\n"
    "\n"
    "Counterfactuals are searched on the surrogate and then re-queried against the model; validity "
    "is the fraction that actually hold when the model is asked, so it is measured rather than true "
    "by construction. Table 13 reports held-out surrogate fidelity for each surrogate family."
)

code(
    "explainability = study.run_explainability(64)\n"
    "display(explainability.surrogate_fidelity)\n"
    "display(explainability.coalition_budget)\n"
    "display(explainability.faithfulness)"
)

code(
    "print('counterfactuals proposed:', explainability.counterfactuals_proposed)\n"
    "print('verified against the model:', explainability.counterfactuals_verified)\n"
    "print('validity:', explainability.counterfactual_validity)\n"
    "display(explainability.counterfactuals.head(10))\n"
    "display(explainability.attention)"
)

code(
    "display(plots.figure_xai_faithfulness(explainability.faithfulness))"
)

markdown(
    "## 13. Experiment 4: measured computational performance (Table 15; Figures 11 and 12)\n"
    "\n"
    "The completion cache is disabled for this experiment, so every number is wall-clock time for "
    "calls actually executed in this process. There is no per-call constant and no extrapolation "
    "beyond the volumes actually run."
)

code(
    "experiment4 = study.run_experiment4()\n"
    "print(experiment4.note)\n"
    "display(experiment4.table15)\n"
    "display(experiment4.call_log)"
)

code(
    "display(plots.figure11_inference_time(experiment4.timing))\n"
    "display(plots.figure12_scalability_f1(experiment4.accuracy))"
)

markdown(
    "## 14. Capability comparison and reproduction against the paper (Table 18)\n"
    "\n"
    "The comparison table places each measured quantity beside the value printed in the paper. Rows "
    "describing corpus and pilot size carry a note: they depend entirely on which dataset you "
    "supplied, and are only comparable if you supplied the same records the authors used."
)

code(
    "display(study.table18())\n"
    "comparison = study.comparison()\n"
    "display(comparison)\n"
    "print(study.comparison_summary())"
)

code(
    "display(plots.figure_reproduction_comparison(comparison))\n"
    "display(plots.figure1_methodology(config))\n"
    "display(plots.figure2_environment(config))"
)

code(
    "path = study.save_tables()\n"
    "print('tables written to', path)"
)

markdown(
    "## 15. What this notebook does not claim\n"
    "\n"
    "- The analyst agreement figure is an agreement rate against decisions **you** supplied. If your "
    "  `analyst_decisions.csv` was produced by anything other than human analysts, the number "
    "  measures self-consistency, not external validation.\n"
    "- Offline replay of a logged policy is unbiased only for the actions the log actually covers. "
    "  The coverage table in section 9 states how many context strata carry all four actions; where "
    "  coverage is thin, the policy comparison is correspondingly weak.\n"
    "- Operator outcome targets cannot be computed from pre-deployment records. They are reported "
    "  only when the pilot supplies a measured post-deployment value.\n"
    "- Where the paper's numbers are not reproduced, the comparison table shows the gap. Nothing in "
    "  this code is tuned to close it."
)


def build() -> Path:
    cells = []
    for kind, source in CELLS:
        lines = source.split("\n")
        payload = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": payload})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": payload,
                }
            )
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1))
    return NOTEBOOK


if __name__ == "__main__":
    path = build()
    payload = json.loads(path.read_text())
    codes = sum(1 for cell in payload["cells"] if cell["cell_type"] == "code")
    texts = sum(1 for cell in payload["cells"] if cell["cell_type"] == "markdown")
    print(f"wrote {path} with {codes} code cells and {texts} markdown cells")
