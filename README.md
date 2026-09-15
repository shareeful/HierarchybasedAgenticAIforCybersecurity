# Hierarchical Agentic AI for Dynamic Cybersecurity Risk Assessment

Reference implementation of *A Hierarchical Agentic AI based Dynamic Cybersecurity Risk Assessment
with Explainable Informed Decision Making* (Sardar, Hassan, Islam, Papastergiou, Lekidis).

**This code reads recorded data. It does not generate any.** There is no fallback corpus, no
simulated organisation, no synthesised analyst decisions and no shipped results. If the data is not
present, every entry point stops with an error naming the missing file, its required columns and
where to obtain it. If you have the data, everything runs; if you do not, nothing does.

The three agents are real language models, not statistical stand-ins. The Vulnerability Agent and
the Contextual Awareness Agent run **Llama 3.1 8B Instruct** with LoRA adapters; the Supervisor
Agent runs **Mistral-7B Instruct** with its own adapter, exactly as specified in sections 3.2–3.4
of the paper. Without the weights the agent tier raises `ModelUnavailable` rather than falling back
to anything.

---

## 1. Quick start

```bash
pip install -r requirements.txt          # analysis stack
pip install -r requirements-llm.txt      # torch, transformers, peft, accelerate, bitsandbytes

export HAAI_DATA_DIR=/path/to/corpus     # where your data lives
export HAAI_MODEL_DIR=/path/to/weights   # optional: local model weights
export HAAI_ADAPTER_DIR=adapters         # optional: trained LoRA adapters
export HAAI_CACHE_DIR=cache              # optional: on-disk completion cache

python scripts/check_dataset.py          # prints exactly what is present and what is missing
python scripts/run_all.py                # the full study
python scripts/smoke_test.py             # every stage once on a bounded sample
```

`run_all.py` accepts `--runs`, `--evaluation-sample`, `--explain-sample`, `--labelling-budget`,
`--skip-labelling-sensitivity` and `--skip-finetuning-datasets` for bounded runs.

`notebooks/Hierarchical_Agentic_AI_Risk_Assessment.ipynb` is the Colab entry point. It clones the
repository, mounts Drive, runs the dataset check, and then walks every experiment, table and figure
in order. It needs a GPU runtime for the agent tier.

Exit codes: `2` means the dataset is incomplete, `3` means the model weights are unavailable.

---

## 2. The dataset you must supply

Everything lives under `$HAAI_DATA_DIR` (default `./data`, which is git-ignored). `scripts/check_dataset.py`
prints this contract with the exact column list for each file.

### Public vulnerability intelligence

| Directory | Files | Source |
| --- | --- | --- |
| `nvd/` | `nvdcve-2.0-<year>.json[.gz|.zip]` | <https://nvd.nist.gov/vuln/data-feeds> |
| `epss/` | `epss_scores-<date>.csv[.gz]` | <https://epss.empiricalsecurity.com/> |
| `kev/` | `known_exploited_vulnerabilities.json` | <https://www.cisa.gov/known-exploited-vulnerabilities-catalog> |
| `attack/` | `enterprise-attack.json` | <https://github.com/mitre-attack/attack-stix-data> |
| `cwe/` | `cwec_*.xml[.zip]` | <https://cwe.mitre.org/data/downloads.html> |
| `capec/` | `capec_*.xml[.zip]` | <https://capec.mitre.org/data/downloads.html> |
| `exploitdb/` | `files_exploits.csv` | <https://gitlab.com/exploit-database/exploitdb> |

CWE supplies the weakness hierarchy and the CAPEC links; CAPEC supplies the ATT&CK taxonomy
mappings. The chain CVE → CWE → CAPEC → ATT&CK is how each record acquires a technique, and it is
computed, not assumed.

### Organisational records (`organisation/`)

| File | Required columns |
| --- | --- |
| `assets.csv` | `asset_id, zone, criticality, external_reachable, adjacent_zones` |
| `software_inventory.csv` | `asset_id, vendor, product, version` |
| `controls.csv` | `asset_id, control_family, active` |
| `change_records.csv` | `asset_id, product, change_date, change_type, caused_disruption` |
| `remediation_outcomes.csv` | `cve_id, asset_id, decision_date, action, exploited_after, disruption_observed, analyst_accepted, residual_exposure` |
| `agent_telemetry.csv` | `observed_at, agent, cve_id, asset_id, cvss, epss, risk_score, confidence` (optional `tampered`) |

### Pilot site records (`usecase/`)

The same four inventory files, plus:

| File | Required columns |
| --- | --- |
| `subsystems.csv` | `subsystem, hosts, criticality, segment` |
| `attack_scenarios.csv` | `scenario_id, name, entry_subsystem, target_subsystem, technique, executed` |
| `soc_alerts.csv` | `alert_id, asset_id, raised_at, severity, alert_type` |
| `analyst_decisions.csv` | `decision_id, cve_id, asset_id, decided_at, action` |
| `operator_targets.csv` | `metric, baseline, target` (optional `achieved`) |

`action` must be drawn from the configured action set (`patch`, `compensate`, `defer`, `accept`).

---

## 3. What each experiment measures, and on what

| Experiment | Method | Data it consumes |
| --- | --- | --- |
| 1 — accuracy (Tables 9, 10, 18) | seven baselines plus the hierarchy, one metric protocol | corpus splits, organisational inventory |
| 2 — adaptive policy (Tables 11, 12) | unbiased offline replay of the recorded ledger | `remediation_outcomes.csv` |
| 3 — integrity (Table 14) | recorded tampering labels, else a declared attack model | `agent_telemetry.csv` |
| 4 — performance (Table 15) | measured wall-clock, cache disabled | evaluation pairs |
| 5 — pilot use case (Tables 16, 17) | agreement against recorded triage decisions | `usecase/` |
| Explainability (Table 13) | exact Shapley against the model itself | evaluation pairs |
| Labelling sensitivity | the hierarchy re-run under two alternative labelling protocols | corpus splits |

### Metric protocol

One protocol, stated once and applied everywhere: the decision threshold is chosen on the **tuning
split**, and precision, recall, F1, AUC and average precision are all computed on the **full
evaluation split** at that threshold. There is no class-balanced downsample and no mixed protocol.
Dispersion comes from stratified bootstrap resampling of the evaluation split.

Because a suspicious F1/AUC pair was the single most-cited problem with the paper's Table 9, the
code reports the arithmetic directly (`metric_consistency.csv`):

- `max_f1_concave_hull` — the largest F1 attainable by **any** concave ROC curve with the measured
  AUC at the measured prevalence. Exceeding this is impossible.
- `max_f1_proper_binormal` — the largest F1 attainable under a proper binormal ROC with that AUC.
  Exceeding this is possible but implies an unusually shaped ROC curve.

For the paper's headline row (AUC 0.881 at 9.1% prevalence) these evaluate to **0.865** and
**0.509**. The paper's F1 of 0.846 is therefore *not* arithmetically impossible, as has been
claimed; it is below the hull bound. It does sit far above the binormal value, which means it
requires an ROC curve with a very steep initial segment. The code prints both numbers for every
system so a reader can judge that for themselves instead of taking either claim on trust.

### Sensitivity to the labelling protocol

The labelled set rests on a protocol choice: uncertain records (not in KEV but with EPSS above the
floor) are held out, and negatives are sampled at 10:1. `labelling_sensitivity.csv` re-runs the
hierarchy with the uncertain records forced negative, and with negatives at 20:1, so the dependence
of the headline number on that choice is visible rather than assumed away. The threshold is
reselected within each protocol and the column name says so.

### Offline policy evaluation

Experiment 2 does not inject a threat shift into a simulated environment. It replays the recorded
remediation ledger in order: each logged decision is presented, the policy proposes an action, and
the event is retained only when the proposal matches the action actually taken (the standard
unbiased offline estimator for logged bandit feedback). Reward is Equation 26 evaluated on the
recorded outcome fields. Regret and accuracy are measured against the empirically best action for
each context stratum. `replay_coverage.csv` reports how many strata carry all four actions — where
coverage is thin, the comparison is correspondingly weak, and the table says so.

### Explainability

Exact Shapley values over the six Equation 11 fields are computed **against the language model**:
2⁶ = 64 masked queries per instance, bounded by `xai.model_query_budget`. The deletion test on those
attributions therefore measures the model. Kernel SHAP (1,024 coalitions) and LIME remain
surrogate-based because those coalition counts are not affordable against an 8B model, and the
faithfulness table labels each row with `evaluated_on`.

Counterfactuals are searched on the surrogate and then **re-queried against the model**; validity is
the fraction that actually hold, reported alongside the number proposed. It is measured, not true by
construction.

Table 13 reports held-out R² for three surrogate families fitted to real agent outputs. No R² target
appears anywhere in the configuration.

### Timing

Experiment 4 disables the completion cache and reports wall-clock seconds per pipeline stage for
calls actually executed in this process. There is no per-call latency constant, and no
extrapolation past the volumes actually run. The call log separates executed from cached calls.

---

## 4. LoRA fine-tuning

```bash
python scripts/train_lora_agents.py --rebuild-datasets           # build data and train all three
python scripts/train_lora_agents.py --dry-run                    # report sizes and hyperparameters
python scripts/train_lora_agents.py --agents supervisor_agent    # one adapter
```

Hyperparameters follow the paper: rank 16, α 32, dropout 0.05, `q_proj`/`v_proj`, learning rate
3e-4 with linear decay, 10 epochs, sequence length 1,024, 4-bit quantised base weights.

Instruction targets come from recorded data — the KEV label for the Vulnerability Agent, the
Equation 28 exposure reference for the Contextual Awareness Agent, and the **recorded remediation
action** for the Supervisor Agent.

Equations 14, 16 and 18 are the live training objectives, not documentation. The numeric term is
differentiable: at each digit position of the target value the loss takes the probability-weighted
expectation over the model's digit-token distribution, assembles the number by place value, and
takes the squared error against the target. Where the tokenizer merges digits into a single token
the alignment check fails and that span falls back to token cross-entropy alone, so the loss is
never silently wrong. `train_lora_adapter` measures numeric mean absolute error before and after
training and records both in `training_report.json`.

---

## 5. Equation map

| Equations | Module |
| --- | --- |
| 1–10 — feed ingestion, enrichment, validation | `data/feeds.py`, `data/corpus.py` |
| 11 — vulnerability feature set | `baselines/models.py:STRUCTURED_FIELDS`, `config.xai.vulnerability_fields` |
| 12–13 — prompt construction | `agents/prompts.py` |
| 14 — vulnerability loss | `finetune/lora.py:vulnerability_loss` |
| 15 — contextual feature set | `config.xai.contextual_fields` |
| 16 — contextual loss | `finetune/lora.py:contextual_loss` |
| 17 — supervisor input package | `agents/supervisor_agent.py:SupervisorAgent.decide` |
| 18 — supervisor loss | `finetune/lora.py:supervisor_loss` |
| 19–22 — bandit policies and reliability | `bandits/`, `agents/confidence.py` |
| 23 — confidence-weighted joint risk | `agents/supervisor_agent.py:joint_risk_eq23` |
| 24 — action utility | `agents/supervisor_agent.py:action_utility` |
| 25 — trust policy and escalation | `agents/supervisor_agent.py:apply_trust_policy` |
| 26 — composite reward | `data/outcomes.py:composite_reward` |
| 27 — integrity verification | `integrity/detector.py` |
| 28 — exposure reference | `data/organisation.py:exposure_ground_truth` |
| 29–34 — post-action outcomes | read from `remediation_outcomes.csv`, not modelled |
| 35–37 — attribution and faithfulness | `xai/` |

Equations 29–34 describe a simulation of post-action outcomes. This implementation does not
simulate them; it reads what actually happened from the remediation ledger, which is why
Experiment 2 is an offline replay rather than a synthetic rollout.

---

## 6. Reproducibility

Every constant is in `src/haai/config.py`, traceable to a section of the paper, overridable as
`section.field`. There is no `calibration.json` and no fitting to the paper's published results —
the script that used to do that has been deleted.

Seeds derive from a single master seed through `numpy.random.SeedSequence`. Model decoding is greedy
at temperature 0, and completions are cached on disk keyed by a hash of `(model, prompt, token
budget)`, so a second pass over the same prompts is both free and byte-identical.

`results/` is git-ignored. Nothing in this repository is a stored result.

---

## 7. Limitations, stated plainly

- **The pilot use case is only as external as your data.** The agreement rate compares the system's
  recommendations against `usecase/analyst_decisions.csv`. If that file was not written by human
  analysts, the number measures self-consistency, not external validation. The code reports how many
  of the recorded decisions could be matched to a pair alongside how many were recorded, so partial
  coverage is visible.
- **Offline replay is unbiased only where the log has coverage.** `replay_coverage.csv` states how
  many context strata carry all four actions.
- **Operator outcome targets cannot be computed from pre-deployment records.** They are reported only
  when `operator_targets.csv` carries a measured `achieved` column; otherwise the table says the
  measurement is not derivable from the supplied data.
- **Corpus and pilot size comparisons against the paper are dataset-dependent.** Those rows in
  `reproduction_comparison.csv` carry a note saying so; they are meaningful only if you supplied the
  same records the authors used.
- **Integrity evaluation prefers recorded tampering labels.** When the telemetry log has none, the
  configured attack model is injected into the recorded agent outputs and the provenance string
  says exactly that.
- **Where the paper's numbers are not reproduced, the comparison table shows the gap.** Nothing here
  is tuned to close it.

---

## 8. Repository layout

```
src/haai/
  config.py              every constant, traced to a paper section
  study.py               orchestration of the whole study
  metrics.py             one metric protocol, plus the F1/AUC consistency bounds
  data/spec.py           the dataset contract
  data/registry.py       resolution, schema validation, hard failure when absent
  data/feeds.py          NVD, EPSS, KEV, ATT&CK, CWE, CAPEC, Exploit-DB parsers
  data/corpus.py         Phase 1 validation, labelling, splits
  data/organisation.py   inventories, CPE pairing, Equation 28
  data/usecase.py        pilot records including the recorded analyst decisions
  data/outcomes.py       recorded rewards and the offline replay evaluator
  agents/llm.py          transformers + PEFT inference, logprobs, attention, cache
  agents/prompts.py      Tables 2, 3 and 4 with field-span tracking
  agents/parsing.py      repair, retry, clamp, impute, contradiction detection
  agents/confidence.py   isotonic calibration with Brier and ECE
  agents/*_agent.py      the three agents
  baselines/models.py    CVSS, CVSS+EPSS, XGBoost, SecureBERT, RAG-LLM, single-agent, flat
  bandits/               Thompson, discounted UCB, sliding-window UCB, static, DQN
  integrity/             Isolation Forest verifier and two comparators
  xai/                   exact and kernel SHAP, LIME, attention pooling, counterfactuals
  finetune/              instruction datasets and the LoRA training loop
  experiments/           the five experiments, explainability, reproduction comparison
  pipeline/              environment assembly and the hierarchical orchestrator
scripts/
  check_dataset.py       report presence and schema of every required file
  run_all.py             the full study
  train_lora_agents.py   fine-tune the three adapters
  smoke_test.py          run every stage once on a bounded sample
  build_notebook.py      regenerate the Colab notebook
  run_notebook.py        execute the notebook's cells outside Jupyter
```
