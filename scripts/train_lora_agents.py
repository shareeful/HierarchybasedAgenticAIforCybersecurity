from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from haai.agents.llm import ModelUnavailable
from haai.data.registry import DatasetError
from haai.finetune.dataset import InstructionDataset, InstructionExample, read_jsonl
from haai.finetune.lora import train_lora_adapter
from haai.logging_utils import get_logger
from haai.study import Study, configure

LOGGER = get_logger()

NUMERIC_FIELDS = {
    "vulnerability_agent": ("risk_score", "confidence"),
    "contextual_agent": ("exposure_score", "regression_risk", "confidence"),
    "supervisor_agent": ("joint_risk",),
}


def load_dataset(path: Path, name: str) -> InstructionDataset:
    rows = read_jsonl(path)
    examples = [
        InstructionExample(
            prompt=row["prompt"],
            completion=row["completion"],
            targets=row["targets"],
            weight=float(row.get("weight", 1.0)),
        )
        for row in rows
    ]
    return InstructionDataset(name, examples, NUMERIC_FIELDS[name])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune the LoRA adapters for the three agents on recorded data"
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--dataset-dir", type=str, default="results/finetuning")
    parser.add_argument("--adapter-dir", type=str, default="adapters")
    parser.add_argument(
        "--agents",
        nargs="*",
        default=["vulnerability_agent", "contextual_agent", "supervisor_agent"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--rebuild-datasets", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    overrides = {}
    if arguments.epochs is not None:
        overrides["lora.epochs"] = arguments.epochs
    config = configure(**overrides)

    dataset_dir = Path(arguments.dataset_dir)
    if arguments.rebuild_datasets or not dataset_dir.exists():
        study = Study(config)
        try:
            study.build(arguments.data_dir)
        except DatasetError as error:
            print(str(error), file=sys.stderr)
            return 2
        study.build_finetuning_datasets(dataset_dir)

    reports = []
    for name in arguments.agents:
        path = dataset_dir / f"{name}.jsonl"
        if not path.exists():
            print(f"missing instruction file {path}; rerun with --rebuild-datasets", file=sys.stderr)
            return 2
        dataset = load_dataset(path, name)
        if arguments.max_examples:
            dataset.examples = dataset.examples[: arguments.max_examples]
        base_model = (
            config.lora.supervisor_base_model
            if name == "supervisor_agent"
            else config.lora.task_tier_base_model
        )
        if arguments.dry_run:
            LOGGER.info(
                "%s: %d examples, base model %s, rank %d, alpha %d, targets %s, %d epochs",
                name,
                len(dataset),
                base_model,
                config.lora.rank,
                config.lora.alpha,
                ", ".join(config.lora.target_modules),
                config.lora.epochs,
            )
            reports.append(
                {
                    "agent": name,
                    "examples": len(dataset),
                    "base_model": base_model,
                    "dry_run": True,
                }
            )
            continue
        try:
            report = train_lora_adapter(
                config, dataset, base_model, Path(arguments.adapter_dir) / name
            )
        except ModelUnavailable as error:
            print(str(error), file=sys.stderr)
            return 3
        LOGGER.info(json.dumps(report.as_dict(), indent=2))
        reports.append(report.as_dict())

    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
