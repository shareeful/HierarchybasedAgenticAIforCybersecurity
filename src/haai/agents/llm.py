from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..logging_utils import get_logger

LOGGER = get_logger()

MODEL_DIR_VARIABLE = "HAAI_MODEL_DIR"
ADAPTER_DIR_VARIABLE = "HAAI_ADAPTER_DIR"


class ModelUnavailable(RuntimeError):
    pass


def _require_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ModelUnavailable(
            "the agent tier runs real language models; install the inference stack with "
            "'pip install -r requirements-llm.txt' (torch, transformers, peft, accelerate, "
            "bitsandbytes)"
        ) from error
    return torch, AutoModelForCausalLM, AutoTokenizer


def resolve_model(reference: str) -> str:
    base = os.environ.get(MODEL_DIR_VARIABLE)
    if base:
        candidate = Path(base) / Path(reference).name
        if candidate.exists():
            return str(candidate)
    local = Path(reference)
    if local.exists():
        return str(local)
    return reference


def resolve_adapter(name: str) -> Optional[str]:
    base = os.environ.get(ADAPTER_DIR_VARIABLE, "adapters")
    candidate = Path(base) / name
    if (candidate / "adapter_config.json").exists():
        return str(candidate)
    return None


@dataclass
class Completion:
    text: str
    prompt: str
    token_logprobs: np.ndarray
    generated_tokens: Tuple[str, ...]
    prompt_tokens: Tuple[str, ...]
    prompt_offsets: Tuple[Tuple[int, int], ...]
    attention: Optional[np.ndarray]
    seconds: float
    cached: bool = False

    def mean_logprob(self) -> float:
        if self.token_logprobs.size == 0:
            return float("nan")
        return float(self.token_logprobs.mean())

    def sequence_probability(self) -> float:
        if self.token_logprobs.size == 0:
            return float("nan")
        return float(np.exp(self.token_logprobs.mean()))

    def field_logprob(self, field_name: str) -> float:
        joined = ""
        marker = f'"{field_name}"'
        start = self.text.find(marker)
        if start == -1 or self.token_logprobs.size == 0:
            return self.mean_logprob()
        selected: List[float] = []
        cursor = 0
        for index, token in enumerate(self.generated_tokens):
            end = cursor + len(token)
            if cursor >= start and cursor <= start + len(marker) + 24:
                if index < len(self.token_logprobs):
                    selected.append(float(self.token_logprobs[index]))
            cursor = end
            joined += token
        if not selected:
            return self.mean_logprob()
        return float(np.mean(selected))


@dataclass
class CallLog:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    seconds: float = 0.0
    executed_seconds: float = 0.0
    per_agent: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def record(self, agent: str, completion: Completion) -> None:
        bucket = self.per_agent.setdefault(
            agent,
            {
                "calls": 0.0,
                "cached_calls": 0.0,
                "prompt_tokens": 0.0,
                "generated_tokens": 0.0,
                "seconds": 0.0,
                "executed_seconds": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["cached_calls"] += 1.0 if completion.cached else 0.0
        bucket["executed_seconds"] += 0.0 if completion.cached else completion.seconds
        bucket["prompt_tokens"] += len(completion.prompt_tokens)
        bucket["generated_tokens"] += len(completion.generated_tokens)
        bucket["seconds"] += completion.seconds
        self.calls += 1
        self.cached_calls += 1 if completion.cached else 0
        self.prompt_tokens += len(completion.prompt_tokens)
        self.generated_tokens += len(completion.generated_tokens)
        self.seconds += completion.seconds
        self.executed_seconds += 0.0 if completion.cached else completion.seconds

    def reset(self) -> None:
        self.calls = 0
        self.cached_calls = 0
        self.prompt_tokens = 0
        self.generated_tokens = 0
        self.seconds = 0.0
        self.executed_seconds = 0.0
        self.per_agent.clear()

    def as_frame(self):
        import pandas as pd

        rows = []
        for agent, bucket in sorted(self.per_agent.items()):
            rows.append(
                {
                    "agent": agent,
                    "calls": int(bucket["calls"]),
                    "cached_calls": int(bucket["cached_calls"]),
                    "prompt_tokens": int(bucket["prompt_tokens"]),
                    "generated_tokens": int(bucket["generated_tokens"]),
                    "seconds": round(bucket["seconds"], 4),
                    "executed_seconds": round(bucket["executed_seconds"], 4),
                    "seconds_per_executed_call": round(
                        bucket["executed_seconds"] / max(bucket["calls"] - bucket["cached_calls"], 1.0), 5
                    ),
                }
            )
        return pd.DataFrame(rows)



class CompletionCache:
    def __init__(self, path: str | Path | None = None):
        base = Path(path or os.environ.get("HAAI_CACHE_DIR", "cache")) / "completions"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / "completions.jsonl"
        self.entries: Dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.entries[payload["key"]] = payload
            LOGGER.info("loaded %d cached completions from %s", len(self.entries), self.path)
        self.handle = self.path.open("a", encoding="utf-8")

    @staticmethod
    def key(model: str, prompt: str, max_new_tokens: int) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(max_new_tokens).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(prompt.encode("utf-8"))
        return digest.hexdigest()

    def get(self, key: str) -> dict | None:
        return self.entries.get(key)

    def put(self, key: str, prompt: str, completion: "Completion") -> None:
        payload = {
            "key": key,
            "text": completion.text,
            "token_logprobs": completion.token_logprobs.tolist(),
            "generated_tokens": list(completion.generated_tokens),
            "prompt_tokens": list(completion.prompt_tokens),
            "prompt_offsets": [list(span) for span in completion.prompt_offsets],
            "seconds": completion.seconds,
        }
        self.entries[key] = payload
        self.handle.write(json.dumps(payload) + "\n")
        self.handle.flush()

    def restore(self, payload: dict, prompt: str) -> "Completion":
        return Completion(
            text=payload["text"],
            prompt=prompt,
            token_logprobs=np.asarray(payload["token_logprobs"], dtype=float),
            generated_tokens=tuple(payload["generated_tokens"]),
            prompt_tokens=tuple(payload["prompt_tokens"]),
            prompt_offsets=tuple(tuple(span) for span in payload["prompt_offsets"]),
            attention=None,
            seconds=float(payload["seconds"]),
        )

    def close(self) -> None:
        self.handle.close()


class LanguageModel:
    def __init__(
        self,
        reference: str,
        adapter: str | None = None,
        load_in_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device: str | None = None,
        seed: int = 0,
        cache: "CompletionCache | None" = None,
    ):
        torch, AutoModelForCausalLM, AutoTokenizer = _require_transformers()
        self.torch = torch
        self.cache = cache
        self.reference = resolve_model(reference)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.reference, use_fast=True)
        except Exception as error:
            raise ModelUnavailable(
                f"could not load the tokenizer for '{reference}'. Place the weights under "
                f"${MODEL_DIR_VARIABLE} or authenticate with the model host."
            ) from error
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        kwargs: Dict[str, object] = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
        if load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
            except ImportError:
                LOGGER.warning("bitsandbytes unavailable; loading %s in full precision", reference)
        elif torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.reference, **kwargs)
        except Exception as error:
            raise ModelUnavailable(
                f"could not load the weights for '{reference}'. Place them under "
                f"${MODEL_DIR_VARIABLE} or authenticate with the model host."
            ) from error
        if adapter is not None:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise ModelUnavailable(
                    "the LoRA adapter requires peft; install requirements-llm.txt"
                ) from error
            self.model = PeftModel.from_pretrained(self.model, adapter)
            LOGGER.info("attached LoRA adapter %s to %s", adapter, reference)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if "device_map" not in kwargs:
            self.model.to(self.device)

    def _encode(self, prompts: Sequence[str]):
        return self.tokenizer(
            list(prompts), return_tensors="pt", padding=True, truncation=True, max_length=4096
        )

    def generate(
        self,
        prompts: Sequence[str],
        agent: str,
        log: CallLog | None = None,
        need_attention: bool = False,
        max_new_tokens: int | None = None,
    ) -> List[Completion]:
        torch = self.torch
        prompts = list(prompts)
        budget = max_new_tokens or self.max_new_tokens
        results: List[Optional[Completion]] = [None] * len(prompts)
        pending: List[int] = []
        for position, prompt in enumerate(prompts):
            if self.cache is not None and not need_attention:
                key = self.cache.key(self.reference, prompt, budget)
                payload = self.cache.get(key)
                if payload is not None:
                    completion = self.cache.restore(payload, prompt)
                    completion.cached = True
                    results[position] = completion
                    if log is not None:
                        log.record(agent, completion)
                    continue
            pending.append(position)
        if not pending:
            return [item for item in results if item is not None]
        prompts_to_run = [prompts[position] for position in pending]
        encoded = self._encode(prompts_to_run)
        encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
        started = time.perf_counter()
        torch.manual_seed(self.seed)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=budget,
                do_sample=self.temperature > 0.0,
                temperature=self.temperature if self.temperature > 0.0 else None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                output_attentions=need_attention,
            )
        elapsed = time.perf_counter() - started
        prompt_length = encoded["input_ids"].shape[1]
        sequences = output.sequences[:, prompt_length:]
        transition = torch.stack(output.scores, dim=1).log_softmax(dim=-1)
        gathered = transition.gather(2, sequences.unsqueeze(-1)).squeeze(-1)
        completions: List[Completion] = []
        for row in range(sequences.shape[0]):
            tokens = sequences[row]
            mask = tokens != self.tokenizer.pad_token_id
            token_ids = tokens[mask].tolist()
            logprobs = gathered[row][mask].detach().float().cpu().numpy()
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            pieces = tuple(
                self.tokenizer.decode([identifier], skip_special_tokens=True) for identifier in token_ids
            )
            prompt_ids = encoded["input_ids"][row]
            keep = encoded["attention_mask"][row].bool()
            prompt_pieces = tuple(
                self.tokenizer.decode([identifier], skip_special_tokens=True)
                for identifier in prompt_ids[keep].tolist()
            )
            offsets = _offsets(prompts_to_run[row], prompt_pieces)
            attention = (
                _pool_attention(output.attentions, row, int(keep.sum()), prompt_length)
                if need_attention
                else None
            )
            completion = Completion(
                text=text,
                prompt=prompts_to_run[row],
                token_logprobs=logprobs,
                generated_tokens=pieces,
                prompt_tokens=prompt_pieces,
                prompt_offsets=offsets,
                attention=attention,
                seconds=elapsed / max(sequences.shape[0], 1),
            )
            position = pending[row]
            results[position] = completion
            completions.append(completion)
            if log is not None:
                log.record(agent, completion)
            if self.cache is not None and not need_attention:
                self.cache.put(self.cache.key(self.reference, prompts[position], budget), prompts[position], completion)
        return [item for item in results if item is not None]


def _offsets(prompt: str, pieces: Sequence[str]) -> Tuple[Tuple[int, int], ...]:
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        if not piece:
            offsets.append((cursor, cursor))
            continue
        position = prompt.find(piece, cursor)
        if position == -1:
            offsets.append((cursor, cursor))
            continue
        offsets.append((position, position + len(piece)))
        cursor = position + len(piece)
    return tuple(offsets)


def _pool_attention(attentions, row: int, prompt_tokens: int, prompt_length: int) -> np.ndarray:
    if not attentions:
        return np.zeros(prompt_tokens, dtype=float)
    pooled = np.zeros(prompt_length, dtype=float)
    steps = 0
    for step in attentions:
        if not step:
            continue
        for layer in step[-4:]:
            tensor = layer[row]
            weights = tensor.mean(dim=0)[-1, :prompt_length]
            pooled += weights.detach().float().cpu().numpy()
        steps += 1
    if steps:
        pooled /= steps
    offset = prompt_length - prompt_tokens
    trimmed = pooled[offset:] if offset > 0 else pooled
    total = trimmed.sum()
    return trimmed / total if total > 0 else trimmed


@dataclass
class AgentModels:
    vulnerability: LanguageModel
    contextual: LanguageModel
    supervisor: LanguageModel
    log: CallLog


def load_agent_models(config, seed: int = 0, cache: CompletionCache | None = None) -> AgentModels:
    task_reference = config.lora.task_tier_base_model
    supervisor_reference = config.lora.supervisor_base_model
    shared = LanguageModel(
        task_reference,
        adapter=resolve_adapter("vulnerability_agent"),
        load_in_4bit=config.lora.load_in_4bit,
        max_new_tokens=config.lora.max_new_tokens,
        temperature=config.lora.inference_temperature,
        seed=seed,
        cache=cache,
    )
    contextual = (
        shared
        if resolve_adapter("contextual_agent") is None
        else LanguageModel(
            task_reference,
            adapter=resolve_adapter("contextual_agent"),
            load_in_4bit=config.lora.load_in_4bit,
            max_new_tokens=config.lora.max_new_tokens,
            temperature=config.lora.inference_temperature,
            seed=seed,
            cache=cache,
        )
    )
    supervisor = LanguageModel(
        supervisor_reference,
        adapter=resolve_adapter("supervisor_agent"),
        load_in_4bit=config.lora.load_in_4bit,
        max_new_tokens=config.lora.max_new_tokens,
        temperature=config.lora.inference_temperature,
        seed=seed,
        cache=cache,
    )
    return AgentModels(shared, contextual, supervisor, CallLog())
