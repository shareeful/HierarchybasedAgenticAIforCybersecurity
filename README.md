# HierVulEx — Hierarchical Multi-AI-Agent Cybersecurity Risk Assessment

A hierarchical multi-AI-agent system for cybersecurity risk assessment and explainable decision making. Three LLM-based agents are organised across two tiers: a Vulnerability Agent and Contextual Awareness Agent at the task tier, and a Supervisor Agent at the supervisory tier. Each agent employs a distinct Multi-Armed Bandit algorithm for adaptive decision making.

## Architecture

**Task Tier**
- **Vulnerability Agent** (Llama 3.1 8B Instruct + Thompson Sampling): Assesses real-world exploitation urgency using NVD, EPSS, proof-of-concept repositories, and MITRE ATT&CK.
- **Contextual Awareness Agent** (Llama 3.1 8B Instruct + Discounted UCB): Evaluates organisation-specific exposure by reasoning over network topology, asset criticality, and patch history.

**Supervisory Tier**
- **Supervisor Agent** (Mistral-7B Instruct + Sliding Window UCB): Synthesises both assessments, verifies agent integrity using Isolation Forest anomaly detection, selects a mitigation action, recommends a security control category, and generates dual-audience explanations.

## Files

| File | Purpose |
|------|---------|
| `data.py` | Data collection from NVD, EPSS, CISA KEV; feature vector construction; validation |
| `mab.py` | Thompson Sampling, Discounted UCB, Sliding Window UCB |
| `agents.py` | VulnerabilityAgent, ContextualAwarenessAgent, SupervisorAgent |
| `pipeline.py` | HierVulExPipeline orchestrator, reward computation, audit trail |
| `main.py` | CLI entry point |

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run with built-in example data (no GPU or model downloads required):

```bash
python main.py
```

Run with live data from NVD, EPSS, and CISA KEV:

```bash
python main.py --fetch-data --start-date 2024-01-01 --end-date 2024-12-31
```

Simulate post-action outcomes and update MAB policies:

```bash
python main.py --simulate-outcomes
```

Save results, audit trail, and pipeline state:

```bash
python main.py --simulate-outcomes \
  --output results.json \
  --save-audit audit.json \
  --save-state state.json
```

Load a saved state and continue learning:

```bash
python main.py --load-state state.json --simulate-outcomes
```

## Using Local LLM Models

Requires a GPU with at least 40 GB VRAM for all three models simultaneously.

```bash
python main.py --use-models \
  --va-model meta-llama/Llama-3.1-8B-Instruct \
  --ca-model meta-llama/Llama-3.1-8B-Instruct \
  --sa-model mistralai/Mistral-7B-Instruct-v0.3
```

Run `huggingface-cli login` before using gated models. Without `--use-models`, the system uses rule-based computation which is fully functional.

## Output Format

```json
{
  "status": "DECIDED",
  "cve_id": "CVE-2024-1234",
  "joint_risk": 0.8940,
  "adjusted_risk": 0.7152,
  "action": "compensate",
  "control_type": "input_validation",
  "va_flagged": false,
  "ca_flagged": false,
  "technical_explanation": "...",
  "plain_explanation": "...",
  "va_output": {
    "risk_score": 0.868,
    "shap_values": {"epss": 0.304, "poc_available": 0.217, "cvss": 0.174},
    "lime_explanation": "...",
    "confidence": 0.1
  },
  "ca_output": {
    "exposure_score": 0.920,
    "attention_weights": {"network_zone": 0.45, "asset_criticality": 0.35, "control_coverage": 0.20},
    "counterfactual": "...",
    "regression_risk": 0.1,
    "confidence": 0.5
  }
}
```

## Mitigation Actions

| Action | Adjusted Risk Threshold |
|--------|------------------------|
| `patch` | > 0.75 |
| `compensate` | 0.50 – 0.75 |
| `defer` | 0.30 – 0.50 |
| `accept` | < 0.30 |

## Security Control Categories

`input_validation` · `access_control` · `patch_management` · `network_segmentation` · `logging_and_monitoring`

## Data Sources

- [NVD](https://nvd.nist.gov) — CVE records and CVSS scores
- [EPSS](https://www.first.org/epss/) — Daily exploit probability scores  
- [CISA KEV](https://www.cisa.gov/known-exploited-vulnerabilities-catalog) — Ground-truth exploitation labels
- [MITRE ATT&CK](https://attack.mitre.org) — Adversary technique mappings
