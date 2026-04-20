# Hierarchical Multi-AI-Agent Cybersecurity Risk Assessment

A hierarchical multi-AI-agent approach for cybersecurity risk assessment and
informed decision making. Three LLM-based agents operate across two tiers to
deliver accurate, contextual, and explainable risk decisions.

## Architecture

**Task Tier**
- **Vulnerability Agent** (Llama 3.1 8B Instruct + Thompson Sampling)
- **Contextual Awareness Agent** (Llama 3.1 8B Instruct + Discounted UCB)

**Supervisory Tier**
- **Supervisor Agent** (Mistral-7B Instruct + Sliding Window UCB)

## Files

| File | Purpose |
|------|---------|
| `data_pipeline.py` | Phase 1: NVD/EPSS/CISA collection, feature vectors, validation |
| `agents.py` | All three agents, MAB policies, XAI, and integrity detection |
| `training.py` | Phase 2: LoRA fine-tuning for all three agents |
| `pipeline.py` | Phase 3 and 4: assessment pipeline, reward computation, policy update |
| `evaluate.py` | Four experiments linked to four research questions |
| `main.py` | Entry point for all execution modes |

## Requirements

```
Python >= 3.9
CUDA GPU with >= 24GB VRAM
```

Install:
```bash
pip install -r requirements.txt
```

## Usage

Full pipeline:
```bash
python main.py --mode full --nvd-api-key YOUR_KEY --start-date 2023-01-01 --end-date 2024-01-01
```

Quick test without fine-tuning:
```bash
python main.py --mode full --skip-training --max-records 200 --max-assess 50
```

Individual modes: `collect`, `train`, `assess`, `evaluate`

## Hugging Face Access

```bash
huggingface-cli login
```

## Outputs

| File | Contents |
|------|---------|
| `vulnerability_records.json` | Collected CVE records |
| `exposure_pairs.json` | Vulnerability-asset pairs |
| `audit_trail.json` | Full decision traces |
| `evaluation_results.json` | All four experiment results |
| `checkpoints/` | Fine-tuned model checkpoints |

## NVD API Key

Optional. Increases rate limit from 5 to 50 requests per 30 seconds.
Register at https://nvd.nist.gov/developers/request-an-api-key
