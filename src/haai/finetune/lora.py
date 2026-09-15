from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..agents.llm import ModelUnavailable, resolve_model
from ..config import Config
from ..logging_utils import get_logger
from .dataset import InstructionDataset

LOGGER = get_logger()


def _require():
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ModelUnavailable(
            "LoRA fine-tuning needs torch, transformers and peft; install requirements-llm.txt"
        ) from error
    return torch, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer


@dataclass
class NumericSpan:
    field: str
    target: float
    positions: Tuple[int, ...]
    place_values: Tuple[float, ...]


def locate_numeric_spans(
    tokenizer, completion: str, prompt_length: int, fields: Sequence[str], targets: Dict[str, float]
) -> List[NumericSpan]:
    spans: List[NumericSpan] = []
    encoding = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    for field in fields:
        if field not in targets:
            continue
        marker = f'"{field}":'
        start = completion.find(marker)
        if start == -1:
            marker = f'"{field}": '
            start = completion.find(marker)
            if start == -1:
                continue
        cursor = start + len(marker)
        while cursor < len(completion) and completion[cursor] in " ":
            cursor += 1
        end = cursor
        while end < len(completion) and (completion[end].isdigit() or completion[end] == "."):
            end += 1
        text = completion[cursor:end]
        if "." not in text:
            continue
        digit_positions: List[int] = []
        place_values: List[float] = []
        aligned = True
        for character_index in range(cursor, end):
            if not completion[character_index].isdigit():
                continue
            decimal_point = completion.index(".", cursor, end)
            exponent = character_index - decimal_point if character_index > decimal_point else 0
            place = 10.0 ** (-exponent) if exponent > 0 else 1.0
            match = None
            for token_index, (token_start, token_end) in enumerate(offsets):
                if token_start <= character_index < token_end:
                    match = (token_index, token_start, token_end)
                    break
            if match is None or match[2] - match[1] != 1:
                aligned = False
                break
            digit_positions.append(prompt_length + match[0])
            place_values.append(place)
        if digit_positions and aligned:
            spans.append(
                NumericSpan(field, float(targets[field]), tuple(digit_positions), tuple(place_values))
            )
    return spans


def digit_token_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    for digit in "0123456789":
        encoded = tokenizer.encode(digit, add_special_tokens=False)
        ids.append(encoded[-1] if encoded else 0)
    return ids


def expected_numeric_value(torch, logits, spans: Sequence[NumericSpan], digit_ids: Sequence[int]):
    values = []
    targets = []
    digits = torch.tensor(list(range(10)), dtype=logits.dtype, device=logits.device)
    index = torch.tensor(list(digit_ids), device=logits.device)
    for span in spans:
        total = torch.zeros((), dtype=logits.dtype, device=logits.device)
        for position, place in zip(span.positions, span.place_values):
            if position - 1 < 0 or position - 1 >= logits.shape[0]:
                continue
            distribution = torch.softmax(logits[position - 1].index_select(0, index), dim=-1)
            total = total + place * (distribution * digits).sum()
        values.append(total)
        targets.append(torch.tensor(span.target, dtype=logits.dtype, device=logits.device))
    if not values:
        return None, None
    return torch.stack(values), torch.stack(targets)


def vulnerability_loss(torch, token_loss, numeric_predicted, numeric_target, alpha: float, weight: float):
    if numeric_predicted is None:
        return weight * token_loss
    regression = torch.mean((numeric_predicted - numeric_target) ** 2)
    return weight * (alpha * regression + (1.0 - alpha) * token_loss)


def contextual_loss(torch, token_loss, numeric_predicted, numeric_target, beta: float, weight: float):
    if numeric_predicted is None:
        return weight * token_loss
    regression = torch.mean((numeric_predicted - numeric_target) ** 2)
    return weight * (beta * regression + (1.0 - beta) * token_loss)


def supervisor_loss(torch, token_loss, numeric_predicted, numeric_target, weight: float):
    if numeric_predicted is None:
        return weight * token_loss
    regression = torch.mean((numeric_predicted - numeric_target) ** 2)
    return weight * (token_loss + regression)


LOSS_BY_AGENT = {
    "vulnerability_agent": vulnerability_loss,
    "contextual_agent": contextual_loss,
    "supervisor_agent": supervisor_loss,
}


@dataclass
class TrainingReport:
    agent: str
    base_model: str
    adapter_path: str
    examples: int
    trainable_parameters: int
    total_parameters: int
    epochs: int
    losses: List[float]
    base_numeric_mae: float
    tuned_numeric_mae: float

    def as_dict(self) -> dict:
        return {
            "agent": self.agent,
            "base_model": self.base_model,
            "adapter_path": self.adapter_path,
            "examples": self.examples,
            "trainable_parameters": self.trainable_parameters,
            "total_parameters": self.total_parameters,
            "trainable_fraction": self.trainable_parameters / max(self.total_parameters, 1),
            "epochs": self.epochs,
            "final_loss": self.losses[-1] if self.losses else float("nan"),
            "base_numeric_mae": self.base_numeric_mae,
            "tuned_numeric_mae": self.tuned_numeric_mae,
            "improvement": self.base_numeric_mae - self.tuned_numeric_mae,
        }


def _encode(tokenizer, example, max_length: int):
    prompt_ids = tokenizer(example.prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(example.completion, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    return input_ids, labels, len(prompt_ids)


def _numeric_mae(torch, model, tokenizer, dataset, digit_ids, max_length: int, limit: int) -> float:
    errors: List[float] = []
    model.eval()
    with torch.no_grad():
        for example in dataset.examples[:limit]:
            input_ids, labels, prompt_length = _encode(tokenizer, example, max_length)
            spans = locate_numeric_spans(
                tokenizer, example.completion, prompt_length, dataset.numeric_fields, example.targets
            )
            if not spans:
                continue
            tensor = torch.tensor([input_ids], device=model.device)
            logits = model(input_ids=tensor).logits[0]
            predicted, target = expected_numeric_value(torch, logits, spans, digit_ids)
            if predicted is None:
                continue
            errors.append(float(torch.mean(torch.abs(predicted - target)).item()))
    return float(np.mean(errors)) if errors else float("nan")


def train_lora_adapter(
    config: Config,
    dataset: InstructionDataset,
    base_model: str,
    output_dir: str | Path,
    evaluation_limit: int = 64,
) -> TrainingReport:
    torch, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer = _require()
    reference = resolve_model(base_model)
    tokenizer = AutoTokenizer.from_pretrained(reference, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: Dict[str, object] = {}
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.float16
        if config.lora.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                LOGGER.warning("bitsandbytes unavailable; training in float16 without quantisation")
    model = AutoModelForCausalLM.from_pretrained(reference, **load_kwargs)
    if not torch.cuda.is_available():
        model.to("cpu")

    digit_ids = digit_token_ids(tokenizer)
    base_mae = _numeric_mae(
        torch, model, tokenizer, dataset, digit_ids, config.lora.max_sequence_length, evaluation_limit
    )

    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info(
        "%s: %d trainable of %d parameters (%.4f%%)",
        dataset.name,
        trainable,
        total,
        100.0 * trainable / max(total, 1),
    )

    optimiser = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.lora.learning_rate,
        weight_decay=config.lora.weight_decay,
    )
    steps = config.lora.epochs * max(len(dataset) // config.lora.batch_size, 1)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimiser, start_factor=1.0, end_factor=0.0, total_iters=max(steps, 1)
    )
    loss_function = LOSS_BY_AGENT[dataset.name]
    alpha = (
        config.lora.vulnerability_loss_alpha
        if dataset.name == "vulnerability_agent"
        else config.lora.contextual_loss_beta
    )

    model.train()
    history: List[float] = []
    order = np.arange(len(dataset))
    generator = np.random.default_rng(config.evaluation.master_seed)
    for epoch in range(config.lora.epochs):
        generator.shuffle(order)
        running = 0.0
        counted = 0
        optimiser.zero_grad()
        for step, position in enumerate(order):
            example = dataset.examples[int(position)]
            input_ids, labels, prompt_length = _encode(
                tokenizer, example, config.lora.max_sequence_length
            )
            tensor = torch.tensor([input_ids], device=model.device)
            label_tensor = torch.tensor([labels], device=model.device)
            output = model(input_ids=tensor, labels=label_tensor)
            spans = locate_numeric_spans(
                tokenizer, example.completion, prompt_length, dataset.numeric_fields, example.targets
            )
            predicted, target = expected_numeric_value(torch, output.logits[0], spans, digit_ids)
            if dataset.name == "supervisor_agent":
                loss = loss_function(torch, output.loss, predicted, target, example.weight)
            else:
                loss = loss_function(torch, output.loss, predicted, target, alpha, example.weight)
            loss = loss / config.lora.gradient_accumulation_steps
            loss.backward()
            running += float(loss.item()) * config.lora.gradient_accumulation_steps
            counted += 1
            if (step + 1) % config.lora.gradient_accumulation_steps == 0:
                optimiser.step()
                scheduler.step()
                optimiser.zero_grad()
        history.append(running / max(counted, 1))
        LOGGER.info("%s epoch %d/%d loss %.4f", dataset.name, epoch + 1, config.lora.epochs, history[-1])

    tuned_mae = _numeric_mae(
        torch, model, tokenizer, dataset, digit_ids, config.lora.max_sequence_length, evaluation_limit
    )

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

    report = TrainingReport(
        agent=dataset.name,
        base_model=base_model,
        adapter_path=str(directory),
        examples=len(dataset),
        trainable_parameters=trainable,
        total_parameters=total,
        epochs=config.lora.epochs,
        losses=history,
        base_numeric_mae=base_mae,
        tuned_numeric_mae=tuned_mae,
    )
    (directory / "training_report.json").write_text(json.dumps(report.as_dict(), indent=2))
    return report
